"""
End-to-end Context Engineering orchestration.

Pipeline order (matches your specification)::

    User Query
    → Query Processing (normalize string)
    → Hybrid Retrieval (Weaviate oversample)
    → Deduplication (Jaccard)
    → Re-ranking (bi-encoder cosine or lexical fallback)
    → Memory Decay Scoring (trim chat history)
    → Context Compression (extractive sentences)
    → Token Budget Allocation (tiktoken / heuristic)
    → Final Prompt Assembly (Groq-compatible message list)
    → Groq LLM Response

Each function logs to stderr when ``RAG_PIPELINE_DEBUG`` is true via ``log_stage``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from RAG_groq import Chunk, format_context, make_retriever, answer_with_groq_messages

from context_pipeline.agent import RAGAgent, AgentResult
from context_pipeline.multi_agent import MultiAgentOrchestrator, orchestrator_to_agent_result
from context_pipeline.compression import compress_chunks, compress_chunks_with_stats
from context_pipeline.config import PipelineConfig
from context_pipeline.deduplication import dedupe_chunks
from context_pipeline.hybrid_retrieval import retrieve_hybrid_candidates
from context_pipeline.logging_utils import log_stage
from context_pipeline.memory_decay import trim_chat_messages
from context_pipeline.reranking import rerank_chunks, _get_cross_encoder
from context_pipeline.hybrid_retrieval import _compute_adaptive_alpha
from context_pipeline.token_budget import (
    build_messages_within_budget,
    compute_token_budget,
    count_tokens,
)
from context_pipeline.agent import sanitize_answer_for_ui


@dataclass
class PipelineResult:
    """Everything the web UI or CLI needs after one successful run."""

    answer: str
    context_display: str  # compressed, formatted for the "Retrieved context" drawer
    messages_sent: list[dict[str, str]]  # exact payload to Groq (for auditing)
    chunks_final: list[Chunk]
    meta: dict[str, Any] = field(default_factory=dict)


def _normalize_query(q: str) -> str:
    """Lightweight query processing — collapse whitespace, strip."""
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    log_stage("query_processing", length=len(q))
    return q


def run_context_pipeline(
    *,
    query: str,
    docs_dir: Path,
    retriever_mode: str,
    model: str,
    api_key: str,
    chat_messages: list[dict[str, Any]],
    cfg: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Execute the full advanced RAG path.

    :param query: latest user utterance (also duplicated inside ``chat_messages`` tail).
    :param chat_messages: full UI history; the **last** message should be the same user query.
    :param cfg: optional frozen config (defaults to ``PipelineConfig.from_env()``).
    """
    cfg = cfg or PipelineConfig.from_env()
    q = _normalize_query(query)

    retriever = make_retriever(docs_dir, mode=retriever_mode)
    embedder = getattr(retriever, "embedder", None)

    # --- Hybrid retrieval ---
    raw = retrieve_hybrid_candidates(retriever, q, cfg)

    # --- Dedupe ---
    unique = dedupe_chunks(raw, cfg)

    # --- Rerank ---
    ranked = rerank_chunks(q, unique, embedder, cfg, docs_dir)

    # --- Compression ---
    compressed, compression_stats = compress_chunks_with_stats(q, ranked, cfg)
    context_display = format_context(compressed)

    # --- Memory decay + token budget ---
    budget = compute_token_budget(cfg, model)
    # History = all completed turns except we strip the trailing user duplicate for packing
    prior = chat_messages[:-1] if chat_messages else []
    prior_trimmed = trim_chat_messages(
        prior,
        token_counter=lambda t: count_tokens(t, model),
        max_tokens=budget.history,
        cfg=cfg,
    )

    system_instruction = (
        "You are a production RAG assistant. "
        "The user message contains a CONTEXT section built by an engineering pipeline "
        "(hybrid retrieval → dedupe → rerank → compression → token budgeting). "
        "Answer ONLY from that CONTEXT. If facts are missing, reply that you do not know."
    )

    messages = build_messages_within_budget(
        system_instruction=system_instruction,
        history_messages=prior_trimmed,
        compressed_context=context_display,
        query=q,
        budget=budget,
        model=model,
    )

    log_stage("groq_request", model=model, message_count=len(messages))
    answer, usage = answer_with_groq_messages(
        model=model,
        api_key=api_key,
        messages=messages,
        max_tokens=cfg.max_output_tokens,
        temperature=0.2,
    )

    dedupe_keep_pct = round((len(unique) / len(raw)) * 100, 1) if raw else 0.0
    dedupe_removed_pct = round((1 - len(unique) / len(raw)) * 100, 1) if raw else 0.0
    compression_retained_pct = compression_stats.get("retained_pct", 100.0)
    compression_reduction_pct = compression_stats.get("reduction_pct", 0.0)

    meta = {
        "retrieved_raw": len(raw),
        "after_dedupe": len(unique),
        "after_rerank": len(ranked),
        "final_chunks": len(compressed),
        "dedupe_keep_pct": dedupe_keep_pct,
        "dedupe_removed_pct": dedupe_removed_pct,
        "compression_retained_pct": compression_retained_pct,
        "compression_reduction_pct": compression_reduction_pct,
        "compression_chars_before": compression_stats.get("chars_before"),
        "compression_chars_after": compression_stats.get("chars_after"),
        "query_tokens": len(q.split()),
        "query_normalized_len": len(q),
        "adaptive_alpha": _compute_adaptive_alpha(q, cfg.hybrid_alpha),
        "rerank_mode": "Cross-Encoder" if _get_cross_encoder() is not None else ("Bi-Encoder" if embedder else "Lexical"),
        "timestamp": time.time(),
        "usage": usage,
        "budget": {
            "system": budget.system,
            "history": budget.history,
            "documents": budget.documents,
            "query": budget.query,
            "input_total": budget.input_total,
        },
    }

    return PipelineResult(
        answer=answer,
        context_display=context_display,
        messages_sent=messages,
        chunks_final=compressed,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------

@dataclass
class AgentPipelineResult:
    """Result returned from the agent pipeline to the web layer."""
    answer: str
    agent_result: AgentResult
    context_display: str
    meta: dict[str, Any]


def run_agent_pipeline(
    *,
    query: str,
    docs_dir: Path,
    retriever_mode: str,
    model: str,
    api_key: str,
    chat_messages: list[dict[str, Any]],
    cfg: PipelineConfig | None = None,
) -> AgentPipelineResult:
    """
    Multi-Agent pipeline (Manager + RAG Agent + Data Agent).

    The MultiAgentOrchestrator acts as a Manager that decides whether to:
      - Delegate to the RAGAgent for document / knowledge-base questions.
      - Delegate to the DataAgent for math / data-analysis questions.
      - Call both in sequence for hybrid questions.

    The existing 'AI Agent' UI mode now transparently triggers this full
    multi-agent team without any UI changes.
    """
    cfg = cfg or PipelineConfig.from_env()
    q = _normalize_query(query)

    retriever = make_retriever(docs_dir, mode=retriever_mode)
    embedder  = getattr(retriever, "embedder", None)

    # Shared retriever function used by both Orchestrator and RAGAgent
    retrieval_stats: list[dict[str, float | int | str | None]] = []

    def retriever_fn(sub_query: str) -> str:
        raw = retrieve_hybrid_candidates(retriever, sub_query, cfg)
        unique = dedupe_chunks(raw, cfg)
        ranked = rerank_chunks(sub_query, unique, embedder, cfg, docs_dir)
        compressed, compression_stats = compress_chunks_with_stats(sub_query, ranked, cfg)

        dedupe_keep_pct = round((len(unique) / len(raw)) * 100, 1) if raw else 0.0
        dedupe_removed_pct = round((1 - len(unique) / len(raw)) * 100, 1) if raw else 0.0

        retrieval_stats.append({
            "query": sub_query,
            "retrieved_raw": len(raw),
            "after_dedupe": len(unique),
            "after_rerank": len(ranked),
            "final_chunks": len(compressed),
            "dedupe_keep_pct": dedupe_keep_pct,
            "dedupe_removed_pct": dedupe_removed_pct,
            "compression_retained_pct": compression_stats.get("retained_pct", 100.0),
            "compression_reduction_pct": compression_stats.get("reduction_pct", 0.0),
            "compression_chars_before": compression_stats.get("chars_before"),
            "compression_chars_after": compression_stats.get("chars_after"),
            "rerank_mode": "Cross-Encoder" if _get_cross_encoder() is not None else ("Bi-Encoder" if embedder else "Lexical"),
        })

        return format_context(compressed)

    orchestrator = MultiAgentOrchestrator(
        retriever_fn=retriever_fn,
        model=model,
        api_key=api_key,
        cfg=cfg,
    )

    prior = chat_messages[:-1] if chat_messages else []
    orch_result = orchestrator.run(q, prior)

    # Convert to AgentResult shape expected by the web layer
    result: AgentResult = orchestrator_to_agent_result(orch_result)

    # Use the last rag_result / data_result step as the context display
    display_kinds = {"rag_result", "data_result", "observation"}
    display_steps = [s for s in result.steps if s.kind in display_kinds]
    context_display = display_steps[-1].content if display_steps else ""

    total_raw = sum(s.get("retrieved_raw", 0) for s in retrieval_stats)
    total_dedupe = sum(s.get("after_dedupe", 0) for s in retrieval_stats)
    total_rerank = sum(s.get("after_rerank", 0) for s in retrieval_stats)
    total_final = sum(s.get("final_chunks", 0) for s in retrieval_stats)
    average_retained = round(
        sum(s.get("compression_retained_pct", 0.0) for s in retrieval_stats) / len(retrieval_stats),
        1,
    ) if retrieval_stats else 100.0
    average_reduction = round(
        sum(s.get("compression_reduction_pct", 0.0) for s in retrieval_stats) / len(retrieval_stats),
        1,
    ) if retrieval_stats else 0.0
    dedupe_keep_pct = round((total_dedupe / total_raw) * 100, 1) if total_raw else 0.0
    dedupe_removed_pct = round((1 - total_dedupe / total_raw) * 100, 1) if total_raw else 0.0
    rerank_mode = None
    if retrieval_stats:
        all_modes = {s.get("rerank_mode") for s in retrieval_stats}
        rerank_mode = all_modes.pop() if len(all_modes) == 1 else "Mixed"

    meta = {
        "agent_steps": [
            {"kind": s.kind, "label": s.label, "content": s.content}
            for s in result.steps
        ],
        "confidence": result.confidence,
        "sources_used": result.sources_used,
        "search_count": sum(1 for s in result.steps if s.kind == "delegation"),
        "retrieval_count": len(retrieval_stats),
        "retrieved_raw_total": total_raw,
        "after_dedupe_total": total_dedupe,
        "after_rerank_total": total_rerank,
        "final_chunks_total": total_final,
        "dedupe_keep_pct": dedupe_keep_pct,
        "dedupe_removed_pct": dedupe_removed_pct,
        "compression_retained_pct": average_retained,
        "compression_reduction_pct": average_reduction,
        "retrieval_stats": retrieval_stats,
        "rerank_mode": rerank_mode,
        "confidence": result.confidence,
        "sources_used": result.sources_used,
        "usage": result.usage,
        "rag_mode": "agent",
        "timestamp": time.time(),
    }

    clean_answer = sanitize_answer_for_ui(
        result.answer,
        query=q,
        model=model,
        api_key=api_key,
    )

    return AgentPipelineResult(
        answer=clean_answer,
        agent_result=result,
        context_display=context_display,
        meta=meta,
    )
