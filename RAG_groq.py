from groq.types.chat import ChatCompletionMessageParam
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
import sys
from pathlib import Path
# Ensure the local 'project' package is on the Python path for imports
project_path = Path(__file__).parent / "project"
sys.path.append(str(project_path))

import asyncio

from typing import cast

_retriever_cache: dict[tuple[str, str, int, int], object] = {}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_docs(docs_dir: Path) -> list[tuple[str, str]]:
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    exts = {".txt", ".md"}
    out: list[tuple[str, str]] = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            text = _read_text_file(p).strip()
            if text:
                out.append((str(p.relative_to(docs_dir)), text))
    return out


def _chunk_text(text: str, *, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - max(0, overlap_chars))
    return chunks


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No chunks to index. Add documents to your docs folder.")
        self._chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, *, k: int = 4) -> list[Chunk]:
        q = _tokenize(query)
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = ranked[: max(1, k)]
        return [self._chunks[i] for i in top]


def build_chunks(
    docs_dir: Path,

    chunk_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[Chunk]:
    if chunk_chars is None:
        chunk_chars = int(os.environ.get("RAG_CHUNK_CHARS", "1200"))
    if overlap_chars is None:
        overlap_chars = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))
    chunk_chars = max(1, chunk_chars)
    overlap_chars = max(0, overlap_chars)

    docs = _iter_docs(docs_dir)
    chunks: list[Chunk] = []
    for rel, text in docs:
        for piece in _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars):
            chunks.append(Chunk(source=rel, text=piece))
    return chunks


def fingerprint_docs_dir(docs_dir: Path) -> str:
    """Stable hash from doc paths + mtimes + sizes (no full file read)."""
    import hashlib

    h = hashlib.sha256()
    if not docs_dir.is_dir():
        return ""
    exts = {".txt", ".md"}
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = str(p.relative_to(docs_dir))
            st = p.stat()
            h.update(rel.encode())
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
    return h.hexdigest()


def make_retriever(docs_dir: Path, mode: str | None = None):
    """
    mode: 'bm25' (keyword) or 'vector' (embeddings + Weaviate).
    Default from env RAG_RETRIEVER (default: vector).

    Cache the retriever for the lifetime of the process to avoid expensive
    re-indexing and chunk-building on every incoming request.
    """
    docs_path = str(docs_dir.resolve())
    chunk_chars = int(os.environ.get("RAG_CHUNK_CHARS", "1200"))
    overlap_chars = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))
    m = (mode or os.environ.get("RAG_RETRIEVER", "vector")).strip().lower()
    cache_key = (docs_path, m, chunk_chars, overlap_chars)
    cached = _retriever_cache.get(cache_key)
    if cached is not None:
        return cached

    if m == "bm25":
        retriever = BM25Retriever(build_chunks(docs_dir))
    elif m == "vector":
        from vector_store import VectorRetriever

        retriever = VectorRetriever.from_docs(docs_dir)
    else:
        raise ValueError(f"Unknown retriever mode: {m!r} (use 'bm25' or 'vector')")

    _retriever_cache[cache_key] = retriever
    return retriever


def format_context(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] source: {c.source}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def answer_with_groq(*, model: str, api_key: str, query: str, context: str) -> tuple[str, dict]:
    """Backward-compatible single-turn call (system + one user message)."""
    messages = cast(
        list[ChatCompletionMessageParam],
        _default_rag_messages(query=query, context=context),
    )
    return answer_with_groq_messages(
        model=model, api_key=api_key, messages=messages, max_tokens=600
    )


def _default_rag_messages(*, query: str, context: str) -> list[ChatCompletionMessageParam]:
    return [
        { 
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}",
        },
    ]


def answer_with_groq_messages(
    *,
    model: str,
    api_key: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> tuple[str, dict]:
    """
    Send an arbitrary chat message list to Groq.

    Used by the context-engineering pipeline after token budgeting has built
    system / history / user slices. Each dict must have ``role`` and ``content``.
    """
    from context_pipeline.token_budget import enforce_request_token_limit, get_groq_request_limit

    client = Groq(api_key=api_key)
    safe_messages = enforce_request_token_limit(
        [dict(m) for m in messages],  # type: ignore[arg-type]
        model=model,
        max_output_tokens=max_tokens,
    )
    request_limit = get_groq_request_limit()

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=safe_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            }
            if isinstance(content, str):
                return content.strip(), usage
            if content is None:
                return "", usage
            return str(content).strip(), usage
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            if "rate_limit" not in err and "413" not in err and "too large" not in err:
                raise
            # Shrink payload and lower completion reserve, then retry
            request_limit = max(2048, int(request_limit * 0.75))
            max_tokens = max(128, int(max_tokens * 0.75))
            safe_messages = enforce_request_token_limit(
                safe_messages,
                model=model,
                max_output_tokens=max_tokens,
                request_limit=request_limit,
            )

    if last_exc is not None:
        raise last_exc
    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("Missing GROQ_API_KEY. Set it in your env or a .env file.", file=sys.stderr)
        return 2

    docs_dir = Path(os.environ.get("RAG_DOCS_DIR", "docs")).resolve()
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    top_k = int(os.environ.get("RAG_TOP_K", "4"))

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: python rag_groq.py "your question here"', file=sys.stderr)
        return 2

    # Use full LangChain LCEL chain with memory, token budgeting, and custom tools
    from project.langchain.chain import create_rag_chain

    # Resolve optional system prompt (fallback to built‑in if missing)
    system_prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"
    if not system_prompt_path.is_file():
        system_prompt_path = None

    chain = create_rag_chain(
        session_id="default_session",
        docs_dir=os.getenv("RAG_DOCS_DIR", "docs"),
        retriever_mode=os.getenv("RAG_RETRIEVER", "bm25"),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY", ""),
        system_prompt_path=str(system_prompt_path) if system_prompt_path else None,
    )
    response = chain.invoke({"question": query})
    print(response)


if __name__ == "__main__":
    raise SystemExit(main())

