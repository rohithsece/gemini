import os
import sys
from pathlib import Path
import asyncio
from typing import Any, List

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

# Ensure the project root (the directory containing the 'project' folder) is in PYTHONPATH for imports.
project_root = Path(__file__).resolve().parents[1]  # .../project
workspace_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

# Load environment variables (API keys, etc.)
load_dotenv()

from context_pipeline.config import PipelineConfig
from context_pipeline.deduplication import dedupe_chunks
from context_pipeline.reranking import rerank_chunks, _get_cross_encoder
from context_pipeline.compression import compress_chunks_with_stats

# Import utilities from the main script (answer_with_groq and helper functions)
# To avoid circular imports, we import the functions directly via their file path.
import importlib.util

def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the RAG_groq module which contains answer_with_groq and retrieval helpers.
rag_groq_path = project_root.parent / "RAG_groq.py"
rag_module = _load_module_from_path("rag_groq", rag_groq_path)

# Extract needed functions
answer_with_groq = rag_module.answer_with_groq
make_retriever = rag_module.make_retriever
format_context = rag_module.format_context

class LangChainOrchestrator:
    """A lightweight orchestrator mimicking a LangChain pipeline.

    It performs retrieval using BM25, formats the prompt, and calls the Groq model.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", top_k: int = 5):
        self.model = model
        self.top_k = top_k
        self.last_pipeline_meta: dict[str, Any] | None = None

    def get_last_pipeline_meta(self) -> dict[str, Any] | None:
        return self.last_pipeline_meta

    async def _run_sync(self, user_query: str) -> str:
        """Synchronous part of the pipeline executed in a thread.

        Retrieves documents, formats context, and calls Groq.
        """
        # Retrieve relevant documents
        docs_path = project_root / "docs"
        cfg = PipelineConfig.from_env()
        retriever = make_retriever(project_root.parent / "docs")
        hits = retriever.search(user_query, k=self.top_k)

        unique = dedupe_chunks(hits, cfg)
        ranked = rerank_chunks(user_query, unique, getattr(retriever, "embedder", None), cfg, docs_path)
        compressed, compression_stats = compress_chunks_with_stats(user_query, ranked, cfg)

        # Store metrics for later inspection
        raw = len(hits)
        dedupe_keep_pct = round((len(unique) / raw) * 100, 1) if raw else 0.0
        self.last_pipeline_meta = {
            "retrieved_raw": raw,
            "after_dedupe": len(unique),
            "after_rerank": len(ranked),
            "final_chunks": len(compressed),
            "dedupe_keep_pct": dedupe_keep_pct,
            "dedupe_removed_pct": round((1 - len(unique) / raw) * 100, 1) if raw else 0.0,
            "compression_retained_pct": compression_stats.get("retained_pct", 100.0),
            "compression_reduction_pct": compression_stats.get("reduction_pct", 0.0),
            "compression_chars_before": compression_stats.get("chars_before"),
            "compression_chars_after": compression_stats.get("chars_after"),
            "rerank_mode": "Cross-Encoder" if _get_cross_encoder() is not None else ("Bi-Encoder" if getattr(retriever, "embedder", None) else "Lexical"),
        }

        context = format_context(compressed)
        # Call Groq synchronously via answer_with_groq
        response, usage = answer_with_groq(
            model=self.model,
            api_key=os.getenv("GROQ_API_KEY"),
            query=user_query,
            context=context,
        )
        return response

    async def run(self, user_query: str, chat_history: List[dict] | None = None) -> str:
        """Public async entry point.

        chat_history is currently unused but kept for API compatibility.
        """
        # Offload the synchronous heavy work to a thread to avoid blocking the event loop
        result = await asyncio.to_thread(self._run_sync, user_query)
        return result
