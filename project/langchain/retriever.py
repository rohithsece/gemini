"""Custom retriever implementation for the RAG pipeline.

This module defines `CustomContextRetriever`, a concrete `BaseRetriever`
that returns a list of `Document` objects. The implementation is lightweight
and returns an empty list by default; replace the stub with actual retrieval
logic (e.g., querying Weaviate) as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from RAG_groq import make_retriever


class CustomContextRetriever(BaseRetriever):
    """A minimal retriever wrapper that delegates to the local RAG retrievers.

    This wrapper is used by the LangGraph pipeline and supports both the
    `search` and `search_hybrid_oversampled` APIs expected by
    `context_pipeline.hybrid_retrieval`.
    """

    docs_dir: str = "."
    retriever_mode: str = "hybrid"
    top_k: int = 2
    _inner: Any = None

    def __init__(self, docs_dir: str = ".", retriever_mode: str = "hybrid", *, top_k: int = 2) -> None:
        super().__init__(docs_dir=docs_dir, retriever_mode=retriever_mode, top_k=top_k)
        self.docs_dir = docs_dir
        self.retriever_mode = retriever_mode
        self.top_k = top_k
        mode = "vector" if retriever_mode in ("vector", "hybrid") else "bm25"
        try:
            self._inner = make_retriever(Path(docs_dir), mode=mode)
        except Exception:
            # Fallback to BM25-only if vector retrieval is unavailable.
            self._inner = make_retriever(Path(docs_dir), mode="bm25")

    def search(self, query: str, *, k: int = 4) -> list:
        if self._inner is None:
            return []
        return self._inner.search(query, k=k)

    def search_hybrid_oversampled(self, query: str, *, candidate_limit: int, alpha: float | None = None) -> list:
        if self._inner is None:
            return []
        if hasattr(self._inner, "search_hybrid_oversampled"):
            return self._inner.search_hybrid_oversampled(query, candidate_limit=candidate_limit, alpha=alpha)
        return self.search(query, k=candidate_limit)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        chunks = self.search(query, k=self.top_k)
        return [Document(page_content=chunk.text, metadata={"source": chunk.source}) for chunk in chunks]

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
