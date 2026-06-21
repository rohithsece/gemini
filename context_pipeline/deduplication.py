"""
Near-duplicate removal for retrieved chunks.

WHY: oversampled hybrid search often returns overlapping windows from the same
long markdown file. Feeding duplicates wastes the token budget and can bias
the LLM toward repeating the same fact — dedup improves diversity per token.

We use Jaccard similarity on *word* sets (cheap, good for near-identical prose).
Cosine on embeddings would be slightly stronger but costs an extra encoder pass;
Jaccard is a common first-stage filter in production crawlers and RAG ingest.
"""

from __future__ import annotations

import re

from RAG_groq import Chunk

from context_pipeline.config import PipelineConfig
from context_pipeline.logging_utils import log_stage


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedupe_chunks(chunks: list[Chunk], cfg: PipelineConfig) -> list[Chunk]:
    """
    Greedy keep-list: scan in retrieval order, skip if Jaccard >= threshold vs any kept.

    Retrieval order matters: Weaviate already ranked strong items first, so we
    prefer keeping earlier hits and dropping later near-duplicates.
    """
    kept: list[Chunk] = []
    kept_sets: list[set[str]] = []
    removed = 0

    for ch in chunks:
        t = _tokens(ch.text)
        is_dup = False
        for prev in kept_sets:
            if jaccard(t, prev) >= cfg.dedupe_jaccard_threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
            continue
        kept.append(ch)
        kept_sets.append(t)

    log_stage(
        "deduplication",
        input_count=len(chunks),
        output_count=len(kept),
        removed=removed,
        jaccard_threshold=cfg.dedupe_jaccard_threshold,
    )
    return kept
