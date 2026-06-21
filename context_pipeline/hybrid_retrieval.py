"""
Hybrid retrieval stage: dense vectors + sparse BM25 inside Weaviate.

WHY hybrid: pure embedding search can miss exact product SKUs, legal citations,
or rare acronyms; pure BM25 misses paraphrases. Combining both is standard in
modern retrieval stacks (Elasticsearch hybrid, Weaviate hybrid, Pinecone + SPLADE).

This module only *calls* Weaviate hybrid with an oversampled limit. The fusion
weights are controlled by ``alpha`` (Weaviate semantics: higher alpha leans
toward dense vector scores — see Weaviate hybrid search docs for your version).
"""

from __future__ import annotations

from RAG_groq import Chunk

from context_pipeline.config import PipelineConfig
from context_pipeline.logging_utils import log_stage

import re

def _compute_adaptive_alpha(query: str, base_alpha: float) -> float:
    """Dynamically adjust alpha based on query characteristics."""
    q_len = len(query.split())
    has_numbers = bool(re.search(r'\d+', query))
    is_short = q_len <= 3
    is_question = bool(re.match(r'^(how|what|why|when|where|who|explain|describe)\b', query.lower()))
    
    alpha = base_alpha
    if is_question:
        alpha = min(1.0, alpha + 0.2)
    elif is_short or has_numbers:
        alpha = max(0.0, alpha - 0.3)
        
    return round(alpha, 2)


def retrieve_hybrid_candidates(
    retriever: object,
    query: str,
    cfg: PipelineConfig,
) -> list[Chunk]:
    """
    Pull ``oversample * top_k_final`` chunks from the active retriever.

    For ``VectorRetriever`` we use Weaviate hybrid (already BM25+dense fused).
    For ``BM25Retriever`` (in-memory) we only have lexical scores — still useful
    for demos without Docker; we oversample the same way for a fair pipeline.
    """
    limit = max(cfg.top_k_final, cfg.retrieve_oversample * cfg.top_k_final)

    # Vector + Weaviate path
    if hasattr(retriever, "search_hybrid_oversampled"):
        adaptive_alpha = _compute_adaptive_alpha(query, cfg.hybrid_alpha)
        chunks = retriever.search_hybrid_oversampled(  # type: ignore[union-attr]
            query,
            candidate_limit=limit,
            alpha=adaptive_alpha,
        )
        log_stage(
            "hybrid_retrieval",
            mode="weaviate_hybrid",
            candidate_limit=limit,
            alpha=adaptive_alpha,
            base_alpha=cfg.hybrid_alpha,
            returned=len(chunks),
        )
        return chunks

    # BM25-only fallback (no Weaviate vectors)
    chunks = retriever.search(query, k=limit)  # type: ignore[union-attr]
    log_stage(
        "hybrid_retrieval",
        mode="bm25_only_fallback",
        candidate_limit=limit,
        returned=len(chunks),
        note="BM25-only retriever has no dense component; use Weaviate for true hybrid.",
    )
    return chunks
