"""
Groq tool-calling agent loop (ReAct-style).

Mirrors context_pipeline/agent.py and data_agent.py:
  1. Send tools[] schema to Groq chat.completions.create(tools=...)
  2. If model returns tool_calls → execute each → append tool results to messages
  3. Repeat until finalize_answer or max iterations
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from api.tool_registry import TOOL_SCHEMAS, execute_tool

MAX_TOOL_ITERATIONS = 6

SYSTEM_PROMPT = """You are an assistant with tools for:
- RAG document search (search_knowledge_base)
- CRUD code generation (generate_crud_code)
- Query file save/run (save_query_file, run_query_file, list_query_files)
- Student records DB (read_student_records, create_student_record)
- Math/data (execute_python)

Use tools when needed. Always end by calling finalize_answer with a clear summary."""


@dataclass
class ToolStep:
    tool: str
    arguments: dict
    result: Any


@dataclass
class ToolAgentResult:
    answer: str
    steps: list[ToolStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


def _safe_args(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def run_tool_agent(query: str, *, model: str | None = None, max_iterations: int = MAX_TOOL_ITERATIONS) -> ToolAgentResult:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    model = model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    client = Groq(api_key=api_key)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    steps: list[ToolStep] = []
    tools_used: list[str] = []

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.2,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            text = (msg.content or "").strip()
            return ToolAgentResult(answer=text or "No response", steps=steps, tools_used=tools_used)

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

        for tc in msg.tool_calls:
            fn = tc.function.name
            args = _safe_args(tc.function.arguments)
            tools_used.append(fn)

            if fn == "finalize_answer":
                answer = args.get("answer", "")
                return ToolAgentResult(answer=answer, steps=steps, tools_used=tools_used)

            try:
                result = execute_tool(fn, args)
            except Exception as exc:
                result = {"error": str(exc)}

            steps.append(ToolStep(tool=fn, arguments=args, result=result))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return ToolAgentResult(
        answer="Reached max tool iterations.",
        steps=steps,
        tools_used=tools_used,
    )
