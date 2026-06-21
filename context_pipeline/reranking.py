"""
Cross-chunk reranking after initial retrieval.

WHY rerank: the first-stage retriever optimizes *recall* (get anything relevant
into the candidate set). A second-stage *reranker* optimizes *precision* at the
top of the list — critical when only 3–4 chunks fit into the LLM window.

True cross-encoders (e.g. ``ms-marco-MiniLM``) concatenate query+document and
run a transformer classification — best quality, higher latency/cost.

Here we use a **lightweight bi-encoder** rerank: cosine similarity between the
same FastEmbed model used for indexing (query embedding vs chunk embedding).
This is much cheaper and still sharpens ordering for many domains.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from RAG_groq import Chunk

from context_pipeline.config import PipelineConfig
from context_pipeline.logging_utils import log_stage

if TYPE_CHECKING:
    from fastembed import TextEmbedding # type: ignore

_cross_encoder = None
_attempted_ce_load = False

def _get_cross_encoder():
    """Lazily load sentence-transformers CrossEncoder if enabled and installed."""
    global _cross_encoder, _attempted_ce_load
    if _cross_encoder is not None:
        return _cross_encoder
    if _attempted_ce_load:
        return None

    import os
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    enabled = os.environ.get("RAG_USE_CROSS_ENCODER", "false").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        _attempted_ce_load = True
        return None

    try:
        from sentence_transformers import CrossEncoder
        model_name = os.environ.get("RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder = CrossEncoder(model_name)
    except Exception as e:
        log_stage("cross_encoder_load_failed", error=str(e))

    _attempted_ce_load = True
    return _cross_encoder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _lexical_score(query: str, text: str) -> float:
    """Fallback when no embedder: normalized token overlap (cheap rerank)."""
    from rag_groq import _tokenize  # pyright: ignore[reportMissingImports] # reuse project tokenizer

    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q:
        return 0.0
    return len(q & t) / len(q)


def _calculate_custom_score(
    base_score: float,
    query: str,
    chunk: Chunk,
    docs_dir: "Path | None",
) -> float:
    import time
    from pathlib import Path
    
    # 1. Multiplicative Factor: Importance (keywords: critical, must, urgent, priority, important, warning)
    importance = 1.0
    text_lower = chunk.text.lower()
    if any(k in text_lower for k in ["critical", "must", "urgent", "priority", "important", "warning"]):
        importance = 1.25
        
    # 2. Multiplicative Factor: Recency & Freshness
    recency = 1.0
    freshness = 1.0
    if docs_dir and chunk.source:
        try:
            source_path = Path(chunk.source)
            file_path = source_path if source_path.is_absolute() else Path(docs_dir) / source_path
            if file_path.exists():
                try:
                    mtime = file_path.stat().st_mtime
                    age_seconds = max(0.0, time.time() - mtime)
                    # Recency: smooth exponential decay over 7 days (604800 seconds)
                    recency = math.exp(-age_seconds / 604800.0)
                    # Freshness: step boost if modified in the last 24 hours (86400 seconds)
                    if age_seconds < 86400.0:
                        freshness = 1.3
                except OSError:
                    pass
        except (OSError, ValueError):
            pass

    # 3. Additive Factor: Query overlap boost
    from RAG_groq import _tokenize
    q_tokens = set(_tokenize(query))
    c_tokens = set(_tokenize(chunk.text))
    overlap_boost = 0.0
    if q_tokens:
        overlap_ratio = len(q_tokens & c_tokens) / len(q_tokens)
        overlap_boost = 0.20 * overlap_ratio  # Additive boost up to 0.20

    # Fused Multiplicative (base * importance * recency * freshness) + Additive overlap_boost
    final_score = (base_score * importance * recency * freshness) + overlap_boost
    return final_score


def rerank_chunks(
    query: str,
    chunks: list[Chunk],
    embedder: "TextEmbedding | None",
    cfg: PipelineConfig,
    docs_dir: "Path | None" = None,
) -> list[Chunk]:
    """
    Score each chunk against ``query``, sort descending, keep top ``top_k_final``.

    If ``embedder`` is None (BM25-only retriever path), falls back to lexical score.
    Now incorporates Multiplicative factors (importance, recency, freshness)
    and an Additive query overlap boost.
    """
    if not chunks:
        log_stage("reranking", returned=0, mode="empty")
        return []

    scored: list[tuple[float, Chunk]] = []

    ce = _get_cross_encoder()
    if ce is not None:
        pairs = [[query, ch.text[: cfg.rerank_char_cap]] for ch in chunks]
        scores = ce.predict(pairs)
        for i, ch in enumerate(chunks):
            final_sc = _calculate_custom_score(float(scores[i]), query, ch, docs_dir)
            scored.append((final_sc, ch))
        mode = "true_cross_encoder"
    elif embedder is not None:
        qvec = list(embedder.embed([query]))
        qv = qvec[0].tolist() if hasattr(qvec[0], "tolist") else list(qvec[0])
        for ch in chunks:
            snippet = ch.text[: cfg.rerank_char_cap]
            cvec = list(embedder.embed([snippet]))
            cv = cvec[0].tolist() if hasattr(cvec[0], "tolist") else list(cvec[0])
            base_sc = _cosine(qv, cv)
            final_sc = _calculate_custom_score(base_sc, query, ch, docs_dir)
            scored.append((final_sc, ch))
        mode = "bi_encoder_cosine"
    else:
        for ch in chunks:
            base_sc = _lexical_score(query, ch.text)
            final_sc = _calculate_custom_score(base_sc, query, ch, docs_dir)
            scored.append((final_sc, ch))
        mode = "lexical_fallback"

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [c for _, c in scored[: cfg.top_k_final]]

    log_stage(
        "reranking",
        mode=mode,
        input_count=len(chunks),
        output_count=len(out),
        top_k_final=cfg.top_k_final,
        rerank_char_cap=cfg.rerank_char_cap,
        best_score=round(scored[0][0], 4) if scored else None,
    )
    return out
