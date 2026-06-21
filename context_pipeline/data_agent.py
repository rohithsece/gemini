"""
Data Analysis / Math Agent.

This agent specialises in answering questions that require:
  - Mathematical calculations (arithmetic, statistics, algebra …)
  - Analysing structured data passed in as text (CSV rows, JSON, plain tables)
  - Generating step-by-step reasoning for numeric problems

It is given one tool:
  execute_python — a restricted Python REPL that evaluates an expression or a
                   short script and returns the printed output / result.

The agent is intentionally narrow: it does NOT search documents or browse the
web.  The MultiAgentOrchestrator (multi_agent.py) delegates to this agent only
when the incoming question is clearly math / data oriented.
"""

from __future__ import annotations

import io
import json
import math
import re
import statistics
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

import groq  # type: ignore
from groq import Groq  # type: ignore

from context_pipeline.config import PipelineConfig
from context_pipeline.logging_utils import log_stage
from context_pipeline.agent import (
    _parse_failed_generation,
    extract_answer_from_tool_syntax,
    looks_like_raw_tool_syntax,
    parse_text_tool_call,
)
from context_pipeline.token_budget import enforce_request_token_limit


# ---------------------------------------------------------------------------
# Safe execution environment
# ---------------------------------------------------------------------------

# Only these modules are available inside the sandbox.
_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format, "frozenset": frozenset,
        "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
        "int": int, "isinstance": isinstance, "issubclass": issubclass,
        "iter": iter, "len": len, "list": list, "map": map, "max": max,
        "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
        "print": print, "range": range, "repr": repr, "reversed": reversed,
        "round": round, "set": set, "slice": slice, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    },
    "math": math,
    "statistics": statistics,
}


def _execute_python(code: str, timeout_seconds: int = 10) -> str:
    """
    Execute *code* in a restricted namespace and return stdout + the last
    expression value (if any).  Exceptions are caught and returned as strings
    so the LLM can self-correct.
    """
    buf = io.StringIO()
    local_ns: dict[str, Any] = {}
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<data_agent>", "exec"), dict(_SAFE_GLOBALS), local_ns)  # noqa: S102
        output = buf.getvalue().strip()
        # If no print statements were used, try to return the last expression.
        if not output and local_ns:
            last = list(local_ns.values())[-1]
            output = repr(last)
        return output or "(no output)"
    except Exception:
        return f"ERROR:\n{traceback.format_exc(limit=5)}"


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

DATA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute a short Python script to perform calculations or analyse data. "
                "Available modules: math, statistics. "
                "Use print() to output your results. "
                "Do NOT import anything else — only the listed modules are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Valid Python code. Use print() for output. "
                            "Available: math, statistics built-ins."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Call this when you have computed the final answer and are ready to "
                "respond to the user. Always call this to end the loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Complete, clear answer including the numeric result and explanation.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

DATA_AGENT_SYSTEM = (
    "You are a precise Data Analysis and Mathematics Agent.\n\n"
    "RULES:\n"
    "1. ALWAYS use the execute_python tool to compute answers — never guess numbers.\n"
    "2. Write clear, concise Python code.  Use print() to show results.\n"
    "3. If the first execution has an error, fix the code and try again.\n"
    "4. Once you have the numeric result, call finalize_answer with a full explanation.\n"
    "5. Format numbers nicely (e.g. 2 decimal places for floats).\n"
    "6. If the question is not mathematical, call finalize_answer immediately and say "
    "   that this question should go to the document search agent instead."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DataAgentStep:
    kind: str    # "code" | "result" | "answer"
    label: str
    content: str


@dataclass
class DataAgentResult:
    answer: str
    steps: list[DataAgentStep] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DataAgent:
    """Specialised agent for math and data analysis tasks."""

    MAX_EXEC_CALLS = 5  # Maximum Python executions per query

    def __init__(self, *, model: str, api_key: str, cfg: PipelineConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    def run(self, query: str, context_data: str = "") -> DataAgentResult:
        """
        Run the data agent on *query*.

        :param query:        The math / analysis question from the Manager.
        :param context_data: Optional extra data (e.g. table extracted from docs)
                             the Manager passes along for the agent to analyse.
        """
        steps: list[DataAgentStep] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        exec_count = 0

        user_content = query
        if context_data.strip():
            user_content = f"{query}\n\nRelevant data:\n{context_data}"

        messages: list[dict] = [
            {"role": "system", "content": DATA_AGENT_SYSTEM},
            {"role": "user",   "content": user_content},
        ]

        log_stage("data_agent_start", query=query)

        max_completion = min(512, self.cfg.max_output_tokens)

        while True:
            at_limit = exec_count >= self.MAX_EXEC_CALLS
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
                    messages=safe_messages,     # type: ignore[arg-type]
                    tools=DATA_TOOLS,           # type: ignore[arg-type]
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    max_tokens=max_completion,
                    temperature=0.0,            # deterministic for math
                )
            except groq.BadRequestError as exc:
                synthetic_tool_call = _parse_failed_generation(exc)
                if synthetic_tool_call is None:
                    return DataAgentResult(
                        answer=f"Data Agent error: {exc}",
                        steps=steps,
                        usage=usage,
                    )
                resp = None

            if resp is not None and resp.usage:
                usage["prompt_tokens"]     += resp.usage.prompt_tokens
                usage["completion_tokens"] += resp.usage.completion_tokens
                usage["total_tokens"]      += resp.usage.total_tokens

            msg = resp.choices[0].message if resp is not None else None

            # ── No tool call → forced plain-text answer ──
            if synthetic_tool_call is None:
                if msg is None or not msg.tool_calls:
                    if msg is not None:
                        parsed_call = parse_text_tool_call(msg.content or "")
                        if parsed_call:
                            fn, args = parsed_call
                            messages.append({
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{
                                    "id": "synthetic-text-0",
                                    "type": "function",
                                    "function": {"name": fn, "arguments": json.dumps(args)},
                                }],
                            })
                            tool_calls_to_process = [(fn, args, "synthetic-text-0")]
                        else:
                            text = msg.content or "I could not compute an answer."
                            if looks_like_raw_tool_syntax(text):
                                leaked = parse_text_tool_call(text)
                                if leaked:
                                    fn, args = leaked
                                    messages.append({
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [{
                                            "id": "leaked-syntax-0",
                                            "type": "function",
                                            "function": {"name": fn, "arguments": json.dumps(args)},
                                        }],
                                    })
                                    tool_calls_to_process = [(fn, args, "leaked-syntax-0")]
                                else:
                                    unwrapped = extract_answer_from_tool_syntax(text)
                                    if unwrapped:
                                        text = unwrapped
                                    steps.append(DataAgentStep("answer", "Final answer", text))
                                    log_stage("data_agent_done", steps=len(steps), execs=exec_count)
                                    return DataAgentResult(answer=text, steps=steps, usage=usage)
                            else:
                                steps.append(DataAgentStep("answer", "Final answer", text))
                                log_stage("data_agent_done", steps=len(steps), execs=exec_count)
                                return DataAgentResult(answer=text, steps=steps, usage=usage)
                    else:
                        text = "I could not compute an answer."
                        steps.append(DataAgentStep("answer", "Final answer", text))
                        log_stage("data_agent_done", steps=len(steps), execs=exec_count)
                        return DataAgentResult(answer=text, steps=steps, usage=usage)
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
            else:
                fn, args = synthetic_tool_call
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "synthetic-0",
                        "type": "function",
                        "function": {"name": fn, "arguments": json.dumps(args)},
                    }],
                })
                tool_calls_to_process = [(fn, args, "synthetic-0")]

            for fn, args, tc_id in tool_calls_to_process:
                if fn == "execute_python":
                    exec_count += 1
                    code = args.get("code", "")
                    log_stage("data_agent_exec", iteration=exec_count)
                    steps.append(DataAgentStep("code", f"Execution #{exec_count}", code))

                    result = _execute_python(code)
                    steps.append(DataAgentStep("result", f"Result #{exec_count}", result))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    })

                elif fn == "finalize_answer":
                    answer = args.get("answer", "")
                    steps.append(DataAgentStep("answer", "Final answer", answer))
                    log_stage("data_agent_done", steps=len(steps), execs=exec_count)
                    return DataAgentResult(answer=answer, steps=steps, usage=usage)

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Unknown tool: {fn}",
                    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_finalize_from_content(text: str) -> dict[str, Any] | None:
    """Extract finalize_answer payload from plain-text Groq output."""
    parsed = parse_text_tool_call(text)
    if parsed and parsed[0] == "finalize_answer":
        return parsed[1]
    return None


def _parse_function_tool_response(text: str) -> dict[str, Any] | None:
    """Backward-compatible wrapper for finalize_answer text parsing."""
    result = _parse_finalize_from_content(text)
    if result and isinstance(result.get("answer"), str):
        return result
    return None


def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
