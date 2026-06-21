"""
Multi-Agent Orchestrator (Manager).

Architecture
------------
                      ┌─────────────────────────────────┐
  User Query  ──────► │     MultiAgentOrchestrator       │
                      │  (ReAct loop with 3 sub-agents)  │
                      └────────────┬────────────────┬────┘
                                   │                │
                       ┌───────────▼───┐    ┌───────▼────────┐
                       │   RAGAgent    │    │   DataAgent     │
                       │ (doc search)  │    │ (math/analysis) │
                       └───────────────┘    └────────────────┘

The Manager receives the raw user query, decides which sub-agent(s) to use,
collects their results, and synthesises a final grounded answer.

Manager tools
-------------
1. delegate_to_rag_agent    — searches local documents via the RAG pipeline
2. delegate_to_data_agent   — performs math / data analysis via Python REPL
3. finalize_answer          — emits the final user-facing response

Smart features
--------------
* Automatic routing     : LLM decides which sub-agent to call based on intent.
* Multi-agent fusion    : can call both agents in sequence and merge results.
* Agent step tracking   : every delegation is visible in the UI trace.
* Confidence scoring    : inherited from individual sub-agent results.
* Usage aggregation     : token counts summed across all sub-agent calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import groq  # type: ignore
from groq import Groq  # type: ignore

from context_pipeline.agent import (
    RAGAgent,
    AgentResult,
    AgentStep,
    _parse_failed_generation,
    extract_answer_from_tool_syntax,
    looks_like_raw_tool_syntax,
    parse_all_text_tool_calls,
    parse_text_tool_call,
    sanitize_answer_for_ui,
)
from context_pipeline.config import PipelineConfig
from context_pipeline.data_agent import DataAgent, DataAgentResult
from context_pipeline.logging_utils import log_stage
from context_pipeline.token_budget import enforce_request_token_limit, truncate_text_to_tokens


# ---------------------------------------------------------------------------
# Manager tool schema
# ---------------------------------------------------------------------------

MANAGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_rag_agent",
            "description": (
                "Delegate a question to the RAG (document search) agent. "
                "Use this when the user is asking about information that may be "
                "in the uploaded documents / knowledge base (e.g. company policies, "
                "reports, manuals, articles stored in the docs/ folder). "
                "You may call this multiple times with refined queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused question to send to the RAG agent.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_data_agent",
            "description": (
                "Delegate a task to the Data Analysis / Math agent. "
                "Use this for arithmetic, statistics, algebra, unit conversions, "
                "data analysis, or any question that requires computation. "
                "Optionally pass context_data to provide numbers or tables for analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The math or analysis question.",
                    },
                    "context_data": {
                        "type": "string",
                        "description": (
                            "Optional: raw data (numbers, table, CSV text) for the "
                            "agent to analyse. Leave empty if not applicable."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Call this once you have gathered all the information you need "
                "and are ready to give the user a complete, well-structured final answer. "
                "Always call this to end the loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Complete final answer to the user's question.",
                    },
                    "sources_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of agents / queries used to gather information.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high = strong grounded answer, "
                            "medium = partial info, "
                            "low = mostly inference."
                        ),
                    },
                },
                "required": ["answer", "sources_used", "confidence"],
            },
        },
    },
]

MANAGER_SYSTEM = (
    "You are the Manager Agent — the orchestrator of a specialist AI team.\n\n"
    "YOUR TEAM:\n"
    "  • RAG Agent      — searches private local documents/knowledge base\n"
    "  • Data Agent     — performs math, statistics, and data analysis\n\n"
    "RULES:\n"
    "1. Analyse the user's question carefully before acting.\n"
    "2. For document/knowledge questions → delegate_to_rag_agent.\n"
    "3. For math/calculation/data questions → delegate_to_data_agent.\n"
    "4. For mixed questions → call both agents in sequence, then combine results.\n"
    "5. You may call each agent up to {max_calls} times with refined queries.\n"
    "6. Once you have enough information, call finalize_answer.\n"
    "7. Never fabricate facts. If no agent can answer, honestly say so.\n"
    "8. Synthesise agent results into a clear, concise final answer."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorStep:
    kind: str     # "delegation" | "rag_result" | "data_result" | "answer"
    label: str
    content: str


@dataclass
class OrchestratorResult:
    answer: str
    steps: list[OrchestratorStep] = field(default_factory=list)
    confidence: str = "medium"
    sources_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MultiAgentOrchestrator:
    """
    Manager that routes queries to the RAGAgent and/or DataAgent,
    then synthesises their results into one cohesive answer.
    """

    MAX_DELEGATIONS = 4   # total sub-agent calls allowed per query

    def __init__(
        self,
        *,
        retriever_fn: Callable[[str], str],
        model: str,
        api_key: str,
        cfg: PipelineConfig,
    ) -> None:
        self._retriever_fn = retriever_fn
        self.model = model
        self.api_key = api_key
        self.cfg = cfg
        self._client = Groq(api_key=api_key)

        # Sub-agents — they share the same model / key / cfg
        self._rag_agent  = RAGAgent(
            retriever_fn=retriever_fn,
            model=model,
            api_key=api_key,
            cfg=cfg,
        )
        self._data_agent = DataAgent(
            model=model,
            api_key=api_key,
            cfg=cfg,
        )

    def _handle_tool(
        self,
        fn: str,
        args: dict,
        tc_id: str,
        *,
        query: str,
        history: list[dict],
        steps: list[OrchestratorStep],
        usage: dict[str, int],
        delegation_count: int,
        messages: list[dict],
    ) -> tuple[OrchestratorResult | None, int]:
        """Execute one manager tool call. Returns (result, new_delegation_count)."""
        if fn == "delegate_to_rag_agent":
            delegation_count += 1
            sub_query = args.get("query", query)
            log_stage("orchestrator_rag_delegation", query=sub_query)

            steps.append(OrchestratorStep(
                "delegation",
                f"RAG Agent: {sub_query}",
                f"Delegating to RAG Agent with query: {sub_query}",
            ))

            rag_result: AgentResult = self._rag_agent.run(sub_query, history)

            if rag_result.usage:
                for k in usage:
                    usage[k] += rag_result.usage.get(k, 0)

            result_text = rag_result.answer
            steps.append(OrchestratorStep(
                "rag_result",
                f"RAG Result: {sub_query}",
                result_text,
            ))

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": truncate_text_to_tokens(result_text, 500, self.model),
            })
            return None, delegation_count

        if fn == "delegate_to_data_agent":
            delegation_count += 1
            sub_query = args.get("query", query)
            context_data = args.get("context_data", "")
            log_stage("orchestrator_data_delegation", query=sub_query)

            steps.append(OrchestratorStep(
                "delegation",
                f"Data Agent: {sub_query}",
                f"Delegating to Data Agent with query: {sub_query}",
            ))

            data_result: DataAgentResult = self._data_agent.run(sub_query, context_data)

            if data_result.usage:
                for k in usage:
                    usage[k] += data_result.usage.get(k, 0)

            result_text = data_result.answer
            steps.append(OrchestratorStep(
                "data_result",
                f"Data Result: {sub_query}",
                result_text,
            ))

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": truncate_text_to_tokens(result_text, 500, self.model),
            })
            return None, delegation_count

        if fn == "finalize_answer":
            answer = args.get("answer", "")
            confidence = args.get("confidence", "medium")
            sources = args.get("sources_used", [])

            steps.append(OrchestratorStep("answer", "Final answer", answer))
            log_stage(
                "orchestrator_done",
                steps=len(steps),
                delegations=delegation_count,
                confidence=confidence,
            )
            return OrchestratorResult(
                answer=self._clean_answer(answer, query),
                steps=steps,
                confidence=confidence,
                sources_used=sources,
                usage=usage,
            ), delegation_count

        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": f"Unknown tool: {fn}",
        })
        return None, delegation_count

    def _clean_answer(self, text: str, query: str) -> str:
        """Never surface raw delegate_to_* tool syntax to callers."""
        return sanitize_answer_for_ui(
            text,
            query=query,
            model=self.model,
            api_key=self.api_key,
        )

    # ------------------------------------------------------------------
    def run(self, query: str, history: list[dict]) -> OrchestratorResult:
        """Run the full multi-agent loop for *query*."""
        steps: list[OrchestratorStep] = []
        usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }
        delegation_count = 0

        system_text = MANAGER_SYSTEM.format(max_calls=self.MAX_DELEGATIONS)
        messages: list[dict] = [{"role": "system", "content": system_text}]

        # Inject recent history (last 2 turns)
        for m in history[-2:]:
            role = m.get("role", "")
            if role in ("user", "assistant"):
                content = truncate_text_to_tokens(m.get("content", "") or "", 200, self.model)
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": truncate_text_to_tokens(query, 200, self.model)})

        log_stage("orchestrator_start", query=query)

        max_completion = min(512, self.cfg.max_output_tokens)

        while True:
            at_limit = delegation_count >= self.MAX_DELEGATIONS
            tool_choice = "none" if at_limit else "auto"
            synthetic_tool_call: tuple[str, dict] | None = None
            safe_messages = enforce_request_token_limit(
                messages,
                model=self.model,
                max_output_tokens=max_completion,
            )

            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=safe_messages,    # type: ignore[arg-type]
                    tools=MANAGER_TOOLS,       # type: ignore[arg-type]
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    max_tokens=max_completion,
                    temperature=0.1,
                )
            except groq.BadRequestError as exc:
                synthetic_tool_call = _parse_failed_generation(exc)
                if synthetic_tool_call is None:
                    return OrchestratorResult(
                        answer=f"Manager Agent error: {exc}",
                        steps=steps,
                        usage=usage,
                    )
                resp = None  # type: ignore[assignment]

            if resp is not None:
                if resp.usage:
                    usage["prompt_tokens"]     += resp.usage.prompt_tokens
                    usage["completion_tokens"] += resp.usage.completion_tokens
                    usage["total_tokens"]      += resp.usage.total_tokens

            msg = resp.choices[0].message if resp is not None else None
            tool_calls_to_process: list[tuple[str, dict, str]] = []

            if synthetic_tool_call is not None:
                fn, args = synthetic_tool_call
                tc_id = "synthetic-0"
                tool_calls_to_process = [(fn, args, tc_id)]
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": fn, "arguments": json.dumps(args)},
                    }],
                })
            elif msg is None or not msg.tool_calls:
                parsed_calls = parse_all_text_tool_calls(msg.content if msg is not None else "")
                if parsed_calls:
                    tool_calls_to_process = []
                    tool_call_payload = []
                    for i, (fn, args) in enumerate(parsed_calls):
                        tc_id = f"synthetic-text-{i}"
                        tool_calls_to_process.append((fn, args, tc_id))
                        tool_call_payload.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": fn, "arguments": json.dumps(args)},
                        })
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_call_payload,
                    })
                else:
                    text = (msg.content if msg is not None else "") or "I could not determine an answer."
                    unwrapped = extract_answer_from_tool_syntax(text)
                    if unwrapped:
                        text = unwrapped
                    clean = self._clean_answer(text, query)
                    steps.append(OrchestratorStep("answer", "Final answer", clean))
                    log_stage("orchestrator_done", steps=len(steps), delegations=delegation_count)
                    return OrchestratorResult(
                        answer=clean,
                        steps=steps,
                        confidence="low" if delegation_count == 0 else "medium",
                        sources_used=[s.label for s in steps if s.kind == "delegation"],
                        usage=usage,
                    )
            else:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })
                tool_calls_to_process = [
                    (tc.function.name, _safe_json(tc.function.arguments), tc.id)
                    for tc in msg.tool_calls
                ]

            for fn, args, tc_id in tool_calls_to_process:
                result, delegation_count = self._handle_tool(
                    fn,
                    args,
                    tc_id,
                    query=query,
                    history=history,
                    steps=steps,
                    usage=usage,
                    delegation_count=delegation_count,
                    messages=messages,
                )
                if result is not None:
                    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Convert OrchestratorResult → AgentResult-compatible for the web layer
# ---------------------------------------------------------------------------

def orchestrator_to_agent_result(result: OrchestratorResult) -> AgentResult:
    """
    The web layer (web.py / pipeline.py) expects an ``AgentResult`` from the
    agent pipeline.  This helper converts an ``OrchestratorResult`` to that
    shape so the existing UI code works unchanged.
    """
    steps = [
        AgentStep(
            kind=s.kind,
            label=s.label,
            content=s.content,
        )
        for s in result.steps
    ]
    return AgentResult(
        answer=result.answer,
        steps=steps,
        confidence=result.confidence,
        sources_used=result.sources_used,
        usage=result.usage,
    )
