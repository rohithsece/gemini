"""
Context compression: shrink retrieved chunks before the LLM call.

WHY: retrieved passages often repeat boilerplate or include tangential sentences.
Compression increases **effective information density** per token — the model
sees more distinct facts within the same context window.

Strategy (simple, explainable, no extra ML deps):
1. Split chunk text into sentences with a regex on punctuation.
2. Score each sentence by lexical overlap with the user query (Jaccard on tokens).
3. Greedy-select highest-scoring sentences until a character budget is met.

BEFORE / AFTER (conceptual):
  BEFORE: 1200 chars with three redundant definitions of "Weaviate".
  AFTER: ~650 chars keeping the definition + config bullets most aligned to query.

This is **extractive** compression (select original sentences), not abstractive
summarization — common in latency-sensitive RAG microservices.
"""

from __future__ import annotations

import re

from RAG_groq import Chunk

from context_pipeline.config import PipelineConfig
from context_pipeline.logging_utils import log_stage


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(query: str, sentence: str) -> float:
    qt = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    st = set(re.findall(r"[a-zA-Z0-9_]+", sentence.lower()))
    if not qt:
        return 0.0
    return len(qt & st) / len(qt)


def compress_chunk_text(query: str, text: str, cfg: PipelineConfig) -> tuple[str, int, int]:
    """
    Return (compressed_text, char_before, char_after).

    Picks sentences in descending relevance score until ``target_ratio`` of the
    original character length is reached, but always keeps at least
    ``compression_min_sentences`` sentences.
    """
    before = len(text)
    sents = _split_sentences(text)
    if len(sents) <= cfg.compression_min_sentences:
        log_stage("context_compression", skipped=True, reason="too_few_sentences")
        return text, before, before

    target_chars = max(80, int(before * cfg.compression_target_ratio))
    order = sorted(range(len(sents)), key=lambda i: _score_sentence(query, sents[i]), reverse=True)

    picked: set[int] = set()
    body_chars = 0
    for i in order:
        if len(picked) >= cfg.compression_min_sentences and body_chars >= target_chars:
            break
        if i in picked:
            continue
        picked.add(i)
        body_chars += len(sents[i]) + 1

    # Ensure minimum sentence count even if budget tiny
    for i in order:
        if len(picked) >= cfg.compression_min_sentences:
            break
        picked.add(i)

    out = " ".join(sents[i] for i in sorted(picked)).strip()
    after = len(out)
    log_stage(
        "context_compression",
        chars_before=before,
        chars_after=after,
        ratio=round(after / before, 3) if before else None,
        sentences_total=len(sents),
        sentences_kept=len(picked),
    )
    return out, before, after


def compress_chunks_with_stats(query: str, chunks: list[Chunk], cfg: PipelineConfig) -> tuple[list[Chunk], dict[str, float | int | None]]:
    out: list[Chunk] = []
    total_before = 0
    total_after = 0
    for ch in chunks:
        new_text, before, after = compress_chunk_text(query, ch.text, cfg)
        total_before += before
        total_after += after
        out.append(Chunk(source=ch.source, text=new_text))

    retained_ratio = round(total_after / total_before, 3) if total_before else None
    stats = {
        "input_chunks": len(chunks),
        "output_chunks": len(out),
        "chars_before": total_before,
        "chars_after": total_after,
        "retained_pct": round(retained_ratio * 100, 1) if retained_ratio is not None else None,
        "reduction_pct": round((1 - retained_ratio) * 100, 1) if retained_ratio is not None else None,
    }
    return out, stats


def compress_chunks(query: str, chunks: list[Chunk], cfg: PipelineConfig) -> list[Chunk]:
    out, _ = compress_chunks_with_stats(query, chunks, cfg)
    return out
