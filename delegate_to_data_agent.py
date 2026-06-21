"""Delegate helper for the Data / Coding agent (Python execution + math)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from context_pipeline.agent import extract_answer_from_tool_syntax, parse_text_tool_call
from context_pipeline.config import PipelineConfig
from context_pipeline.data_agent import DataAgent

load_dotenv()


def delegate_to_data_agent(
    query: str,
    *,
    context_data: str = "",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Run the Data (coding) agent on *query* and return a plain-text answer."""
    max_total_chars = 4000
    max_query_chars = 200
    if len(query) > max_query_chars:
        query = query[:max_query_chars] + "..."

    context_data = context_data or ""
    combined_len = len(query) + len(context_data)
    if combined_len > max_total_chars:
        allowed_context = max(0, max_total_chars - len(query))
        context_data = context_data[:allowed_context] + "..."

    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if "mixtral" in model_name.lower():
        model_name = "llama-3.1-8b-instant"

    final_api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not final_api_key:
        raise RuntimeError("Missing GROQ_API_KEY – set it in .env or the environment.")

    cfg = PipelineConfig.from_env()
    agent = DataAgent(model=model_name, api_key=final_api_key, cfg=cfg)
    answer = agent.run(query, context_data).answer

    # Unwrap leaked finalize_answer / tool syntax from small models
    unwrapped = extract_answer_from_tool_syntax(answer)
    if unwrapped:
        return unwrapped

    parsed = parse_text_tool_call(answer)
    if parsed and parsed[0] == "finalize_answer":
        inner = parsed[1].get("answer")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()

    return answer


# Alias used by comparison UI / orchestrator docs
delegate_to_coding_agent = delegate_to_data_agent
