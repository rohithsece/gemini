"""
ReAct-style AI Agent that wraps the RAG pipeline as a callable tool.

The agent uses Groq function-calling to decide:
  1. search_knowledge_base — run the full RAG pipeline for a query
  2. finalize_answer       — emit the grounded answer with confidence + sources

Smart features
--------------
* Multi-hop retrieval  : searches up to MAX_SEARCH_CALLS times, refining the
                         query each iteration based on what it already found.
* Query decomposition  : the LLM naturally breaks compound questions into
                         multiple targeted sub-queries across tool calls.
* Source tracking      : every search query is recorded; the final answer
                         carries the list of queries that returned useful info.
* Confidence scoring   : high / medium / low based on retrieval hit quality.
* Agent scratchpad     : every step is captured in AgentStep for UI display.
* Graceful fallback    : if MAX_SEARCH_CALLS is reached with no tool_choice
                         the model is forced to answer with what it has.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import groq # type: ignore
from groq import Groq # type: ignore

from context_pipeline.config import PipelineConfig
from context_pipeline.logging_utils import log_stage
from context_pipeline.token_budget import compute_token_budget, enforce_request_token_limit, truncate_text_to_tokens


# ---------------------------------------------------------------------------
# Tool schema sent to Groq
# ---------------------------------------------------------------------------

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the document knowledge base using hybrid retrieval + reranking. "
                "Returns the most relevant context passages. "
                "Call this whenever you need facts to answer the user's question. "
                "You may call it multiple times with different or more focused queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused search query. Be specific. "
                            "For multi-part questions, break them into separate calls."
                        ),
                    }
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
                "Call this when you have gathered sufficient context and are ready "
                "to deliver the final answer. Always call this to end the loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Complete, grounded answer to the user's question.",
                    },
                    "sources_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search queries that returned useful information.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high = strong context found, "
                            "medium = partial context, "
                            "low = little or no relevant context."
                        ),
                    },
                },
                "required": ["answer", "sources_used", "confidence"],
            },
        },
    },
]

AGENT_SYSTEM = (
    "You are an intelligent RAG assistant with access to a private knowledge base.\n\n"
    "RULES:\n"
    "1. ALWAYS call search_knowledge_base first — never answer from memory.\n"
    "2. If the first search is insufficient, refine your query and search again "
    "(you may search up to {max_calls} times).\n"
    "3. For multi-part questions, make one search call per sub-topic.\n"
    "4. Once you have enough context, call finalize_answer.\n"
    "5. Never fabricate facts. If the knowledge base has no relevant info, "
    "say so honestly and set confidence to 'low'.\n"
    "6. Keep answers concise, grounded, and cite source filenames when possible."
)

@dataclass
class AgentStep:
    """One step in the agent's reasoning trace (shown in the UI)."""
    kind: str        # "search" | "observation" | "answer"
    label: str       # short display label
    content: str     # full text (observation result or final answer)


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    confidence: str          # "high" | "medium" | "low"
    sources_used: list[str]  # search queries that helped
    usage: dict              # cumulative Groq token counts

def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


_KNOWN_TOOLS = (
    "finalize_answer",
    "search_knowledge_base",
    "delegate_to_rag_agent",
    "delegate_to_data_agent",
    "execute_python",
)

# Groq small models sometimes truncate the leading character(s) of tool names
_TOOL_ALIASES: dict[str, str] = {
    "elegate_to_rag_agent": "delegate_to_rag_agent",
    "elegate_to_data_agent": "delegate_to_data_agent",
    "to_rag_agent": "delegate_to_rag_agent",
    "to_data_agent": "delegate_to_data_agent",
}

_TOOL_CALL_PATTERN = re.compile(
    r"(finalize_answer|search_knowledge_base|delegate_to_rag_agent|"
    r"delegate_to_data_agent|execute_python|elegate_to_rag_agent|"
    r"elegate_to_data_agent)\s*>\s*",
    re.IGNORECASE,
)


def _strip_tool_wrappers(text: str) -> str:
    """Remove outer parentheses/brackets models often wrap around tool syntax."""
    text = text.strip()
    while len(text) >= 2 and text[0] in "([{" and text[-1] in ")]}":
        text = text[1:-1].strip()
    return text


def looks_like_raw_tool_syntax(text: str) -> bool:
    """True when *text* appears to be a leaked tool invocation, not a user answer."""
    if not text or not isinstance(text, str):
        return False
    if parse_text_tool_call(text) is not None:
        return True
    lowered = text.lower()
    return any(name in lowered for name in _KNOWN_TOOLS) and ("{" in text or ">" in text)


def _normalize_tool_name(name: str) -> str:
    key = name.strip().lower()
    for alias, canonical in _TOOL_ALIASES.items():
        if key == alias.lower():
            return canonical
    return name


def _extract_json_object(text: str, start: int) -> tuple[str, int] | None:
    """Return a balanced ``{...}`` slice starting at *start*."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    return None


def _normalize_leaked_text(text: str) -> str:
    """Normalize smart quotes / code fences before parsing tool syntax."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"```[a-zA-Z]*", "", text)
    return text.strip()


def parse_all_text_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Parse every tool invocation embedded in *text* (models often emit several at once)."""
    if not text or not isinstance(text, str):
        return []

    text = _normalize_leaked_text(text)
    found: list[tuple[str, dict]] = []
    seen_spans: list[tuple[int, int]] = []

    for match in _TOOL_CALL_PATTERN.finditer(text):
        fn_name = _normalize_tool_name(match.group(1))
        json_start = match.end()
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1
        extracted = _extract_json_object(text, json_start)
        if not extracted:
            continue
        json_str, json_end = extracted
        args = _safe_json(json_str)
        if not args and fn_name not in (
            "search_knowledge_base",
            "delegate_to_rag_agent",
            "delegate_to_data_agent",
        ):
            continue
        span = (match.start(), json_end)
        if any(not (span[1] <= s0 or span[0] >= s1) for s0, s1 in seen_spans):
            continue
        seen_spans.append(span)
        found.append((fn_name, args))

    if found:
        return found

    single = parse_text_tool_call(text)
    return [single] if single else []


def _extract_finalize_answer_raw(text: str) -> str | None:
    """
    Extract the ``answer`` field from a leaked ``finalize_answer`` block.

    Groq models often emit JSON with *unescaped* quotes inside the answer string
    (e.g. referred to as the "NSE"), which breaks ``json.loads``.  We therefore
    scan for the ``"answer": "`` key and read until the next JSON key marker.
    """
    if not text:
        return None

    text = _normalize_leaked_text(text)

    for fn, args in parse_all_text_tool_calls(text):
        if fn == "finalize_answer":
            answer = args.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

    if not re.search(r"finalize_answer\s*>\s*\{", text, re.IGNORECASE):
        return None

    key_match = re.search(r'"answer"\s*:\s*"', text, re.IGNORECASE)
    if not key_match:
        return None

    ans_start = key_match.end()
    terminators = (
        '","confidence"',
        '", "confidence"',
        '","sources_used"',
        '", "sources_used"',
        '","answer"',
        '"}',
    )
    ans_end = len(text)
    for term in terminators:
        idx = text.find(term, ans_start)
        if idx != -1:
            ans_end = min(ans_end, idx)

    answer = text[ans_start:ans_end].replace('\\"', '"').strip()
    return answer or None


def extract_answer_from_tool_syntax(text: str) -> str | None:
    """Return inner answer text when the model leaked a finalize_answer tool call."""
    return _extract_finalize_answer_raw(text)


def sanitize_answer_for_ui(
    text: str,
    *,
    query: str,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Convert leaked tool-call text into a real user-facing answer (last-line defense)."""
    import os

    if not text:
        return text

    # Prefer finalize_answer prose — never re-run RAG when the model already wrote the answer
    finalized = _extract_finalize_answer_raw(text)
    if finalized:
        return finalized

    if not looks_like_raw_tool_syntax(text):
        return text

    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if "mixtral" in model_name.lower():
        model_name = "llama-3.1-8b-instant"
    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not key:
        return text

    try:
        from delegate_to_rag_agent import delegate_to_rag_agent
        from delegate_to_data_agent import delegate_to_data_agent

        resolved = resolve_leaked_tool_syntax(
            text,
            query=query,
            rag_executor=lambda sub_q: delegate_to_rag_agent(sub_q, model=model_name),
            data_executor=lambda sub_q, ctx: delegate_to_data_agent(
                sub_q,
                context_data=ctx,
                model=model_name,
                api_key=key,
            ),
        )
        if resolved and not looks_like_raw_tool_syntax(resolved):
            return resolved
    except Exception:
        pass

    # Minimal fallback: run the first RAG sub-query we can parse
    for fn, args in parse_all_text_tool_calls(text):
        if fn in ("delegate_to_rag_agent", "search_knowledge_base"):
            sub_q = str(args.get("query") or query).strip()
            if sub_q:
                try:
                    from delegate_to_rag_agent import delegate_to_rag_agent

                    return delegate_to_rag_agent(sub_q, model=model_name)
                except Exception:
                    break
    return text


def resolve_leaked_tool_syntax(
    text: str,
    *,
    query: str,
    rag_executor: Callable[[str], str],
    data_executor: Callable[[str, str], str],
) -> str:
    """
    Execute leaked tool syntax and return a user-facing answer.

    Handles single and multi-delegation blobs such as::

        (delegate_to_rag_agent>{"query": "IPO process"})
        (delegate_to_data_agent>{"query": "IPO benefits", "context_data": ""})
    """
    if not text or not looks_like_raw_tool_syntax(text):
        return text

    finalized = _extract_finalize_answer_raw(text)
    if finalized:
        return finalized

    calls = parse_all_text_tool_calls(text)
    if not calls:
        return text

    # finalize_answer always wins — even if delegations appear earlier in the blob
    for fn, args in calls:
        if fn == "finalize_answer":
            answer = args.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

    sections: list[str] = []
    for fn, args in calls:
        if fn == "finalize_answer":
            continue
        elif fn in ("delegate_to_rag_agent", "search_knowledge_base"):
            sub_q = str(args.get("query") or query).strip()
            if sub_q:
                sections.append(rag_executor(sub_q))
        elif fn == "delegate_to_data_agent":
            sub_q = str(args.get("query") or query).strip()
            ctx = str(args.get("context_data") or "")
            if sub_q:
                sections.append(data_executor(sub_q, ctx))

    if not sections:
        return text
    if len(sections) == 1:
        return sections[0]
    return "\n\n".join(f"{i}. {part}" for i, part in enumerate(sections, start=1))


def parse_text_tool_call(text: str) -> tuple[str, dict] | None:
    """
    Parse plain-text tool invocations emitted by smaller Groq models, e.g.:

      search_knowledge_base>{"query": "..."}
      (delegate_to_rag_agent>{"query": "..."})
      delegate_to_rag_agent({"query": "..."})
      <function=finalize_answer>{"answer": "..."}
    """
    if not text or not isinstance(text, str):
        return None

    candidates = [text.strip(), _strip_tool_wrappers(text.strip())]

    for candidate in candidates:
        if not candidate:
            continue

        # tool_name>{json}
        match = re.match(
            r"^(finalize_answer|search_knowledge_base|delegate_to_rag_agent|"
            r"delegate_to_data_agent|execute_python)\s*>\s*(.+)$",
            candidate,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            fn_name = _normalize_tool_name(match.group(1))
            args = _safe_json(match.group(2).strip())
            if args or fn_name in ("search_knowledge_base", "delegate_to_rag_agent", "delegate_to_data_agent"):
                return fn_name, args

        # tool_name({json})
        match = re.match(
            r"^(delegate_to_rag_agent|delegate_to_data_agent|search_knowledge_base|"
            r"finalize_answer|execute_python)\s*\(\s*(\{.*\})\s*\)\s*$",
            candidate,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            fn_name = _normalize_tool_name(match.group(1))
            args = _safe_json(match.group(2).strip())
            if args or fn_name in ("search_knowledge_base", "delegate_to_rag_agent", "delegate_to_data_agent"):
                return fn_name, args

        match = re.search(r"<function=(\w+)>(.*?)(?:</function>|$)", candidate, re.DOTALL)
        if match:
            return match.group(1), _safe_json(match.group(2).strip())

        for tool_name in _KNOWN_TOOLS:
            marker = f"<{tool_name}>"
            if marker not in candidate:
                continue
            payload = re.sub(rf"^.*?{re.escape(marker)}", "", candidate, flags=re.DOTALL).strip()
            payload = re.sub(r"</function>.*$", "", payload, flags=re.DOTALL).strip()
            if payload:
                args = _safe_json(payload)
                if args:
                    return tool_name, args

        # Search anywhere in the string (models embed tool syntax mid-sentence)
        for tool_name in _KNOWN_TOOLS:
            match = re.search(rf"{tool_name}\s*>\s*", candidate, re.IGNORECASE)
            if not match:
                continue
            extracted = _extract_json_object(candidate, match.end())
            if extracted:
                args = _safe_json(extracted[0])
                if args or tool_name in ("search_knowledge_base", "delegate_to_rag_agent", "delegate_to_data_agent"):
                    return tool_name, args

    return None


def _parse_failed_generation(exc: groq.BadRequestError) -> tuple[str, dict] | None:
    """
    When a small Groq model emits old-style tool syntax instead of JSON tool
    calls, the API returns a 400 with the raw generation in failed_generation.
    We parse it here so agent loops can execute the tool anyway.
    """
    try:
        body = exc.body or {}
        failed = body.get("error", {}).get("failed_generation", "")
        if not failed:
            return None
        return parse_text_tool_call(failed)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RAGAgent:
    MAX_SEARCH_CALLS = 3

    def __init__(
        self,
        *,
        retriever_fn: Callable[[str], str],
        model: str,
        api_key: str,
        cfg: PipelineConfig,
    ) -> None:
        self._retrieve = retriever_fn
        self.model = model
        self.cfg = cfg
        self._client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    def run(self, query: str, history: list[dict]) -> AgentResult:
        budget = compute_token_budget(self.cfg, self.model)
        steps: list[AgentStep] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        search_count = 0

        system_text = AGENT_SYSTEM.format(max_calls=self.MAX_SEARCH_CALLS)
        messages: list[dict] = [{"role": "system", "content": system_text}]

        # Inject recent history (last 2 turns) so agent has conversation context
        for m in history[-2:]:
            role = m.get("role", "")
            if role in ("user", "assistant"):
                content = truncate_text_to_tokens(m.get("content", "") or "", 200, self.model)
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": truncate_text_to_tokens(query, budget.query, self.model)})

        max_completion = min(512, self.cfg.max_output_tokens)

        while True:
            at_limit = search_count >= self.MAX_SEARCH_CALLS
            tool_choice = "none" if at_limit else "auto"

            # Some smaller Groq models emit old-style <function=name>{args} syntax
            # instead of proper JSON tool calls, causing a 400. We catch that,
            # parse the failed_generation field, and execute the tool manually.
            synthetic_tool_call: tuple[str, dict] | None = None
            safe_messages = enforce_request_token_limit(
                messages,
                model=self.model,
                max_output_tokens=max_completion,
            )
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=safe_messages,         # type: ignore[arg-type]
                    tools=AGENT_TOOLS,              # type: ignore[arg-type]
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    max_tokens=max_completion,
                    temperature=0.1,
                )
            except groq.BadRequestError as exc:
                synthetic_tool_call = _parse_failed_generation(exc)
                if synthetic_tool_call is None:
                    raise
                # Build a dummy response object we never read fields from
                resp = None  # type: ignore[assignment]

            if resp is not None:
                if resp.usage:
                    usage["prompt_tokens"]     += resp.usage.prompt_tokens
                    usage["completion_tokens"] += resp.usage.completion_tokens
                    usage["total_tokens"]      += resp.usage.total_tokens

            msg = resp.choices[0].message if resp is not None else None

            # ── No tool call → plain text final answer (forced when at limit) ──
            # ── Handle synthetic tool call (parsed from failed_generation) ──
            if synthetic_tool_call is not None:
                fn, args = synthetic_tool_call
                tool_calls_to_process = [(fn, args, "synthetic-0")]
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "synthetic-0",
                        "type": "function",
                        "function": {"name": fn, "arguments": json.dumps(args)},
                    }],
                })
            else:
                # ── No tool call → try parsing text-style tool syntax ──
                if not msg.tool_calls:  # type: ignore[union-attr]
                    parsed = parse_text_tool_call(msg.content or "")  # type: ignore[union-attr]
                    if parsed:
                        fn, args = parsed
                        tool_calls_to_process = [(fn, args, "synthetic-text-0")]
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": "synthetic-text-0",
                                "type": "function",
                                "function": {"name": fn, "arguments": json.dumps(args)},
                            }],
                        })
                    else:
                        text = msg.content or "I could not find relevant information."  # type: ignore[union-attr]
                        if looks_like_raw_tool_syntax(text):
                            leaked = parse_text_tool_call(text)
                            if leaked:
                                fn, args = leaked
                                tool_calls_to_process = [(fn, args, "leaked-syntax-0")]
                                messages.append({
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [{
                                        "id": "leaked-syntax-0",
                                        "type": "function",
                                        "function": {"name": fn, "arguments": json.dumps(args)},
                                    }],
                                })
                            else:
                                unwrapped = extract_answer_from_tool_syntax(text)
                                if unwrapped:
                                    text = unwrapped
                                steps.append(AgentStep("answer", "Final answer", text))
                                log_stage("agent_done", steps=len(steps), searches=search_count)
                                searched = [s.label for s in steps if s.kind == "search"]
                                return AgentResult(
                                    answer=text,
                                    steps=steps,
                                    confidence="low" if search_count == 0 else "medium",
                                    sources_used=searched,
                                    usage=usage,
                                )
                        else:
                            steps.append(AgentStep("answer", "Final answer", text))
                            log_stage("agent_done", steps=len(steps), searches=search_count)
                            searched = [s.label for s in steps if s.kind == "search"]
                            return AgentResult(
                                answer=text,
                                steps=steps,
                                confidence="low" if search_count == 0 else "medium",
                                sources_used=searched,
                                usage=usage,
                            )
                else:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",  # type: ignore[union-attr]
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls  # type: ignore[union-attr]
                        ],
                    })
                    tool_calls_to_process = [
                        (tc.function.name, _safe_json(tc.function.arguments), tc.id)
                        for tc in msg.tool_calls  # type: ignore[union-attr]
                    ]

            # ── Process each tool call ──
            for fn, args, tc_id in tool_calls_to_process:
                if fn == "search_knowledge_base":
                    search_count += 1
                    q = args.get("query", query)
                    log_stage("agent_search", query=q, iteration=search_count)

                    steps.append(AgentStep("search", q, f"Searching for: {q}"))

                    raw = self._retrieve(q)
                    obs = (
                        f"[Results for '{q}']\n{raw}"
                        if raw.strip()
                        else f"No relevant documents found for '{q}'."
                    )
                    steps.append(AgentStep("observation", q, obs))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": truncate_text_to_tokens(obs, min(600, budget.documents), self.model),
                    })

                elif fn == "finalize_answer":
                    answer     = args.get("answer", "")
                    confidence = args.get("confidence", "medium")
                    sources    = args.get("sources_used", [])

                    steps.append(AgentStep("answer", "Final answer", answer))
                    log_stage(
                        "agent_done",
                        steps=len(steps),
                        searches=search_count,
                        confidence=confidence,
                    )
                    return AgentResult(
                        answer=answer,
                        steps=steps,
                        confidence=confidence,
                        sources_used=sources,
                        usage=usage,
                    )

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Unknown tool: {fn}",
                    })
