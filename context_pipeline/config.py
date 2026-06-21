"""
Central configuration for the context-engineering pipeline.

All knobs are driven by environment variables so you can tune behaviour
without code changes — typical for staging vs production deployments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _i(name: str, default: str) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _b(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable snapshot of pipeline settings read once at run start.

    WHY a dataclass: passing a single object through stages avoids "parameter
    explosion" and makes logging reproducible (dump the dataclass fields).
    """

    # --- Hybrid retrieval (Weaviate) ---
    hybrid_alpha: float  # 0..1 blend inside Weaviate hybrid (dense vs sparse)
    retrieve_oversample: int  # fetch oversample * top_k candidates before dedupe/rerank

    # --- Dedupe ---
    dedupe_jaccard_threshold: float  # 1.0 = identical bag-of-words; lower = stricter

    # --- Rerank ---
    rerank_char_cap: int  # embed first N chars of each chunk for speed
    top_k_final: int  # chunks after rerank kept for later stages

    # --- Memory decay (chat turns) ---
    memory_decay_lambda: float  # exponent for age-based decay (see memory_decay.py)
    use_message_ts: bool  # if true, use ``ts`` field on messages when present

    # --- Compression ---
    compression_target_ratio: float  # aim to shrink each chunk toward this fraction
    compression_min_sentences: int  # always keep at least this many best sentences

    # --- Token budget ---
    model_context_tokens: int  # rough context window reserved for INPUT side
    max_output_tokens: int  # reserved for model completion — subtract from window
    frac_system: float  # fraction of INPUT budget for system string
    frac_history: float  # fraction for prior chat (after decay ordering)
    frac_documents: float  # fraction for retrieved + compressed context
    frac_query: float  # fraction for the latest user question text

    # --- Debug ---
    pipeline_debug: bool

    @staticmethod
    def from_env() -> "PipelineConfig":
        """Load from process environment (call after ``load_dotenv()``)."""
        return PipelineConfig(
            hybrid_alpha=_f("RAG_HYBRID_ALPHA", "0.75"),
            retrieve_oversample=max(1, _i("RAG_RETRIEVE_OVERSAMPLE", "2")),
            dedupe_jaccard_threshold=_f("RAG_DEDUPE_JACCARD", "0.88"),
            rerank_char_cap=max(128, _i("RAG_RERANK_CHAR_CAP", "256")),
            top_k_final=max(1, min(2, _i("RAG_TOP_K", "2"))),
            memory_decay_lambda=_f("RAG_MEMORY_DECAY_LAMBDA", "0.12"),
            use_message_ts=_b("RAG_MEMORY_USE_TS", "true"),
            compression_target_ratio=max(0.2, min(1.0, _f("RAG_COMPRESSION_TARGET", "0.45"))),
            compression_min_sentences=max(1, _i("RAG_COMPRESSION_MIN_SENT", "1")),
            model_context_tokens=_i("RAG_MODEL_CONTEXT_TOKENS", "3000"),
            max_output_tokens=max(128, min(512, _i("RAG_MAX_OUTPUT_TOKENS", "256"))),
            frac_system=_f("RAG_TOKEN_FRAC_SYSTEM", "0.10"),
            frac_history=_f("RAG_TOKEN_FRAC_HISTORY", "0.15"),
            frac_documents=_f("RAG_TOKEN_FRAC_DOCS", "0.65"),
            frac_query=_f("RAG_TOKEN_FRAC_QUERY", "0.10"),
            pipeline_debug=_b("RAG_PIPELINE_DEBUG", "true"),
        )

    def normalized_fracs(self) -> tuple[float, float, float, float]:
        """Return non-negative fractions scaled to sum to 1.0."""
        a, b, c, d = (
            max(0.0, self.frac_system),
            max(0.0, self.frac_history),
            max(0.0, self.frac_documents),
            max(0.0, self.frac_query),
        )
        s = a + b + c + d
        if s <= 0:
            return 0.12, 0.18, 0.58, 0.12
        return a / s, b / s, c / s, d / s
