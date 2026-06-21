import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from RAG_groq import answer_with_groq, make_retriever
from context_pipeline.agent import extract_answer_from_tool_syntax, looks_like_raw_tool_syntax, parse_text_tool_call

# Simple helper that delegates a user query to the RAG pipeline and returns the answer.
# It mirrors the logic used in the Flask app but is usable as a plain function.

def delegate_to_rag_agent(query: str, *, docs_dir: str | None = None, model: str | None = None, top_k: int = 4, retriever_mode: str | None = None) -> str:
    """Process *query* via the RAG pipeline and return the answer.

    Parameters
    ----------
    query: str
        The user question.
    docs_dir: str | None, optional
        Directory containing documents. Defaults to the environment variable ``RAG_DOCS_DIR`` or ``"docs"``.
    model: str | None, optional
        Groq model name. Defaults to ``GROQ_MODEL`` env var or ``"llama-3.1-8b-instant"``.
    top_k: int, optional
        Number of retrieved chunks. Hard‑capped at **2** to keep payload small.
    retriever_mode: str | None, optional
        Either ``"bm25"`` or ``"vector"``. Defaults to ``RAG_RETRIEVER`` env var ("vector").
    """
    # Resolve configuration from environment if not provided
    try:
        docs_path = Path(docs_dir or os.getenv("RAG_DOCS_DIR", "docs")).resolve()
    except (OSError, ValueError):
        docs_path = Path("docs").resolve()
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY – set it in .env or the environment.")

    # Enforce a safe upper bound on retrieved chunks
    top_k = min(top_k, 2)

    # Build the retriever (BM25 or vector) and fetch relevant chunks
    retriever = make_retriever(docs_path, mode=retriever_mode)
    hits = retriever.search(query, k=top_k)
    # Use only the first hit's text as raw context (no numbering)
    if hits:
        context = hits[0].text
    else:
        context = ""
    # Truncate context to a tighter budget (~400 chars ≈ 100 tokens)
    max_context_chars = 400
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."
    
    # Call Groq to generate the answer using the same prompt style as the web UI
    answer, _ = answer_with_groq(model=model_name, api_key=api_key, query=query, context=context)

    # Small models sometimes leak tool-call syntax instead of a plain answer
    if looks_like_raw_tool_syntax(answer):
        unwrapped = extract_answer_from_tool_syntax(answer)
        if unwrapped:
            return unwrapped
        parsed = parse_text_tool_call(answer)
        if parsed and parsed[0] in ("delegate_to_rag_agent", "search_knowledge_base"):
            sub_q = parsed[1].get("query") or query
            if sub_q != query:
                return delegate_to_rag_agent(sub_q, docs_dir=str(docs_path), model=model_name, top_k=top_k, retriever_mode=retriever_mode)

    return answer
