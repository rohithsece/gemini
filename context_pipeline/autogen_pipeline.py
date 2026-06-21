"""
AutoGen Agent pipeline integration.
Uses autogen-agentchat v0.7.5 with OpenAIChatCompletionClient (Groq-compatible)
and FunctionTool for RAG + data agent orchestration.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from dataclasses import dataclass
from RAG_groq import format_context, make_retriever
from context_pipeline.agent import AgentResult, AgentStep, sanitize_answer_for_ui
from context_pipeline.config import PipelineConfig
from context_pipeline.data_agent import DataAgent
from context_pipeline.hybrid_retrieval import retrieve_hybrid_candidates
from context_pipeline.deduplication import dedupe_chunks
from context_pipeline.reranking import rerank_chunks, _get_cross_encoder
from context_pipeline.compression import compress_chunks_with_stats

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False


@dataclass
class AutoGenPipelineResult:
    """Result returned from the autogen pipeline."""
    answer: str
    agent_result: AgentResult
    context_display: str
    meta: dict[str, Any]


async def _run_autogen(
    query: str,
    docs_dir: Path,
    retriever_mode: str,
    model: str,
    api_key: str,
    cfg: PipelineConfig,
    model_info: dict | None = None,
) -> tuple[str, list[AgentStep], list[dict[str, Any]], str]:
    """
    Async core: build agents, run the chat, collect results.
    # Ensure model_info has required fields for OpenAIChatCompletionClient
    if model_info is None:
        model_info = {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        }
    Returns (final_answer, steps, retrieval_stats, context_display).
    """
    # ------- retriever + stats collector -------
    retriever = make_retriever(docs_dir, mode=retriever_mode)
    embedder = getattr(retriever, "embedder", None)
    retrieval_stats: list[dict[str, Any]] = []

    def search_documents(query: str) -> str:
        """Search local documents for information relevant to the query."""
        raw = retrieve_hybrid_candidates(retriever, query, cfg)
        unique = dedupe_chunks(raw, cfg)
        ranked = rerank_chunks(query, unique, embedder, cfg, docs_dir)
        compressed, compression_stats = compress_chunks_with_stats(query, ranked, cfg)

        dedupe_keep_pct = round((len(unique) / len(raw)) * 100, 1) if raw else 0.0
        dedupe_removed_pct = round((1 - len(unique) / len(raw)) * 100, 1) if raw else 0.0

        retrieval_stats.append({
            "query": query,
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
            "rerank_mode": "Cross-Encoder" if _get_cross_encoder() is not None else (
                "Bi-Encoder" if embedder else "Lexical"
            ),
        })

        return format_context(compressed)

    # ------- data agent tool -------
    data_agent = DataAgent(model=model, api_key=api_key, cfg=cfg)

    def calculate_data(query: str, context_data: str = "") -> str:
        """Perform math or data analysis, optionally using the provided context_data."""
        res = data_agent.run(query, context_data)
        return res.answer

    # ------- model client (OpenAI-compatible → Groq) -------
    client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model_info=model_info,
    )

    system_message = (
        "You are a Manager Agent orchestrating a specialist AI team.\n"
        "You have access to two tools:\n"
        "  1. search_documents – searches a local knowledge base for information.\n"
        "  2. calculate_data  – performs math, statistics, and data analysis.\n\n"
        "Rules:\n"
        "- For document/knowledge questions, call search_documents first.\n"
        "- For math/calculation questions, call calculate_data.\n"
        "- Answer the user's question concisely once you have the results.\n"
        "- When done, reply with TERMINATE."
    )

    # ------- build assistant agent with tools -------
    assistant = AssistantAgent(
        name="Assistant",
        model_client=client,
        tools=[search_documents, calculate_data],
        system_message=system_message,
        reflect_on_tool_use=True,
    )

    # Termination: stop on "TERMINATE" or after 10 messages
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

    # Single-agent team (simplest structure for tool-use loops)
    team = RoundRobinGroupChat([assistant], termination_condition=termination)

    # ------- run the team -------
    steps: list[AgentStep] = []
    final_answer = ""
    context_display = ""

    task_msg = TextMessage(content=query, source="user")
    async for msg in team.run_stream(task=task_msg):
        # Messages are BaseAgentEvent / BaseChatMessage subclasses
        source = getattr(msg, "source", None)
        content = getattr(msg, "content", "")

        # ToolCallMessage / ToolCallResultMessage
        msg_type = type(msg).__name__

        if msg_type == "ToolCallRequestEvent":
            for call in getattr(msg, "content", []):
                fn_name = getattr(call, "name", "")
                fn_args = getattr(call, "arguments", "")
                steps.append(AgentStep(
                    kind="delegation",
                    label=f"AutoGen Tool: {fn_name}",
                    content=f"Arguments: {fn_args}",
                ))
        elif msg_type == "ToolCallExecutionEvent":
            for result in getattr(msg, "content", []):
                output = getattr(result, "content", "")
                tool_call_id = getattr(result, "call_id", "")
                kind = "observation"
                label = "Tool Result"
                steps.append(AgentStep(kind=kind, label=label, content=str(output)))
                if not context_display:
                    context_display = str(output)
        elif msg_type in ("TextMessage", "StopMessage") and source == "Assistant":
            text = str(content).replace("TERMINATE", "").strip()
            if text:
                final_answer = text

    if not final_answer:
        # Fallback: pick last assistant text from steps
        final_answer = "No response generated by AutoGen agent."

    return final_answer, steps, retrieval_stats, context_display


def run_autogen_pipeline(
    *,
    query: str,
    docs_dir: Path,
    retriever_mode: str,
    model: str,
    api_key: str,
    chat_messages: list[dict[str, Any]],
    cfg: PipelineConfig | None = None,
    model_info: dict | None = None,
) -> AutoGenPipelineResult:
    """
    Synchronous entry-point for the AutoGen multi-agent pipeline.
    Internally runs the async core on a fresh event loop.
    """
    if not _AUTOGEN_AVAILABLE:
        raise ImportError(
            "autogen-agentchat and autogen-ext[openai] are required. "
            "Run: pip install autogen-agentchat autogen-ext[openai]"
        )

    cfg = cfg or PipelineConfig.from_env()

    start_time = time.time()

    # Run async core synchronously (Flask is sync)
    final_answer, steps, retrieval_stats, context_display = asyncio.run(
        _run_autogen(query, docs_dir, retriever_mode, model, api_key, cfg, model_info=model_info)
    )

    clean_answer = sanitize_answer_for_ui(
        final_answer,
        query=query,
        model=model,
        api_key=api_key,
    )

    # ------- aggregate stats -------
    total_raw    = sum(s.get("retrieved_raw", 0) for s in retrieval_stats)
    total_dedupe = sum(s.get("after_dedupe", 0) for s in retrieval_stats)
    total_rerank = sum(s.get("after_rerank", 0) for s in retrieval_stats)
    total_final  = sum(s.get("final_chunks", 0) for s in retrieval_stats)

    average_retained = round(
        sum(s.get("compression_retained_pct", 0.0) for s in retrieval_stats) / len(retrieval_stats), 1
    ) if retrieval_stats else 100.0
    average_reduction = round(
        sum(s.get("compression_reduction_pct", 0.0) for s in retrieval_stats) / len(retrieval_stats), 1
    ) if retrieval_stats else 0.0

    dedupe_keep_pct    = round((total_dedupe / total_raw) * 100, 1) if total_raw else 0.0
    dedupe_removed_pct = round((1 - total_dedupe / total_raw) * 100, 1) if total_raw else 0.0

    rerank_mode = None
    if retrieval_stats:
        all_modes = {s.get("rerank_mode") for s in retrieval_stats}
        rerank_mode = all_modes.pop() if len(all_modes) == 1 else "Mixed"

    meta = {
        "agent_steps": [
            {"kind": s.kind, "label": s.label, "content": s.content}
            for s in steps
        ],
        "confidence": "high" if len(steps) > 0 else "medium",
        "sources_used": [s.label for s in steps if s.kind == "delegation"],
        "search_count": sum(1 for s in steps if s.kind == "delegation"),
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
        "rag_mode": "autogen",
        "timestamp": time.time(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    agent_result = AgentResult(
        answer=clean_answer,
        steps=steps,
        confidence=meta["confidence"],
        sources_used=meta["sources_used"],
        usage=meta["usage"],
    )

    return AutoGenPipelineResult(
        answer=clean_answer,
        agent_result=agent_result,
        context_display=context_display,
        meta=meta,
    )
