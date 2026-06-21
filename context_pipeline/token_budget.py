# context_pipeline/token_budget.py
"""Utilities for managing token budgets in the RAG pipeline.

The pipeline needs to stay within the model's context window. This module
calculates how many tokens are available for each part of the prompt and
provides helpers to count tokens and assemble the final message list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any

# Optional tiktoken import – fall back to a naive word count if unavailable
try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


@dataclass(frozen=True)
class TokenBudget:
    """Token allocation for the different sections of a request.

    * ``system`` – tokens reserved for the system instruction.
    * ``history`` – tokens for prior chat messages.
    * ``documents`` – tokens for retrieved and compressed context.
    * ``query`` – tokens for the current user query.
    * ``input_total`` – total tokens available for *input* (everything
      except the model's output).
    """

    system: int
    history: int
    documents: int
    query: int
    input_total: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "system": self.system,
            "history": self.history,
            "documents": self.documents,
            "query": self.query,
            "input_total": self.input_total,
        }


def _get_encoder(model: str):
    """Return a tiktoken encoder for *model* or ``None`` if unavailable."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Some models use the same encoding (e.g., ``gpt-3.5-turbo``)
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _llama_safety_multiplier(model: str) -> float:
    """Groq Llama tokenizers often count higher than GPT encodings — pad estimates."""
    m = model.lower()
    if "llama" in m or "mixtral" in m or "gemma" in m:
        return 1.35
    return 1.0


def get_groq_request_limit() -> int:
    """Hard ceiling for a single Groq request (on_demand TPM for 8b-instant is 6000)."""
    try:
        return max(1024, int(os.environ.get("GROQ_MAX_REQUEST_TOKENS", "5500")))
    except ValueError:
        return 5500


def count_tokens(text: str, model: str) -> int:
    """Count the tokens used by *text* for the given *model*.

    If ``tiktoken`` cannot be imported we fall back to a conservative char
    estimate (Llama tends to use more tokens per character than GPT).
    """
    if not text:
        return 0
    encoder = _get_encoder(model)
    if encoder:
        return int(len(encoder.encode(text)) * _llama_safety_multiplier(model))
    # ~3 chars/token is safer for Llama than word-count or /4 heuristics
    return max(1, len(text) // 3)


def truncate_text_to_tokens(text: str, max_tokens: int, model: str) -> str:
    """Truncate *text* so ``count_tokens`` stays within *max_tokens*."""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text, model) <= max_tokens:
        return text
    # Binary search on character length (fast enough for RAG context sizes)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid], model) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    trimmed = text[:lo].rstrip()
    return trimmed + ("…" if lo < len(text) else "")


def count_messages_tokens(messages: List[Dict[str, Any]], model: str) -> int:
    """Estimate total prompt tokens for a Groq chat payload."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        if isinstance(content, str):
            total += count_tokens(content, model)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += count_tokens(str(tool_calls), model)
    # Per-message ChatML overhead
    total += max(0, len(messages) * 4)
    return total


def enforce_request_token_limit(
    messages: List[Dict[str, Any]],
    *,
    model: str,
    max_output_tokens: int,
    request_limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Shrink *messages* so prompt + reserved completion fit Groq's per-request cap."""
    limit = request_limit if request_limit is not None else get_groq_request_limit()
    safety = 96
    input_budget = max(512, limit - max_output_tokens - safety)
    if count_messages_tokens(messages, model) <= input_budget:
        return messages

    out = [dict(m) for m in messages]
    # Pass 1: cap individual bloated fields (history / tool observations)
    for i, msg in enumerate(out):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        per_msg_cap = input_budget // 3 if role in ("assistant", "tool") else input_budget // 6
        if role == "system":
            per_msg_cap = min(per_msg_cap, 400)
        if count_tokens(content, model) > per_msg_cap:
            out[i] = {**msg, "content": truncate_text_to_tokens(content, per_msg_cap, model)}

    if count_messages_tokens(out, model) <= input_budget:
        return out

    # Pass 2: drop oldest non-system messages until the latest user turn fits
    while len(out) > 2 and count_messages_tokens(out, model) > input_budget:
        drop_idx = next(
            (i for i, m in enumerate(out) if m.get("role") != "system"),
            None,
        )
        if drop_idx is None:
            break
        out.pop(drop_idx)

    # Pass 3: truncate the largest remaining non-user message
    while count_messages_tokens(out, model) > input_budget:
        candidates = [
            (i, count_tokens((m.get("content") or ""), model))
            for i, m in enumerate(out)
            if m.get("role") not in ("user", "system")
        ]
        if not candidates:
            break
        idx = max(candidates, key=lambda x: x[1])[0]
        content = out[idx].get("content", "") or ""
        current = count_tokens(content, model)
        target = max(32, int(current * 0.55))
        out[idx] = {**out[idx], "content": truncate_text_to_tokens(content, target, model)}
        if target <= 32:
            out.pop(idx)

    # Last resort: truncate system + latest user query
    if count_messages_tokens(out, model) > input_budget:
        for i, msg in enumerate(out):
            if msg.get("role") == "system":
                out[i] = {
                    **msg,
                    "content": truncate_text_to_tokens(msg.get("content", "") or "", 200, model),
                }
    if count_messages_tokens(out, model) > input_budget:
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                out[i] = {
                    **out[i],
                    "content": truncate_text_to_tokens(
                        out[i].get("content", "") or "",
                        max(64, input_budget // 4),
                        model,
                    ),
                }
                break

    return out


def compute_token_budget(cfg, model: str) -> TokenBudget:
    """Compute a token budget based on the pipeline configuration.

    The calculation mirrors the original implementation used throughout the
    repository:

    1. Derive a *safety* margin (128 tokens).
    2. Determine the usable input token window, capped by Groq's per-request
       limit (on_demand ``llama-3.1-8b-instant`` TPM is 6000):
       ``input_total = min(model_window, groq_limit - max_output - safety)``
    3. Split the ``input_total`` into four fractions using the normalized
       configuration values (system, history, documents, query).
    """
    safety = 128
    model_window = max(512, cfg.model_context_tokens - cfg.max_output_tokens - safety)
    groq_window = max(512, get_groq_request_limit() - cfg.max_output_tokens - safety)
    input_total = min(model_window, groq_window)
    fs, fh, fd, fq = cfg.normalized_fracs()
    # Allocate tokens – ensure the sum does not exceed ``input_total``.
    system = int(input_total * fs)
    history = int(input_total * fh)
    documents = int(input_total * fd)
    query = int(input_total * fq)
    # Adjust for rounding errors by adding any leftover to the system part.
    allocated = system + history + documents + query
    if allocated < input_total:
        system += input_total - allocated
    return TokenBudget(system=system, history=history, documents=documents, query=query, input_total=input_total)


def _truncate_history(messages: List[Dict[str, Any]], token_budget: int, model: str) -> List[Dict[str, Any]]:
    """Trim *messages* so their token count fits within *token_budget*.

    The function walks the list from the newest message backwards, accumulating
    token counts until the budget would be exceeded. The remaining oldest
    messages are discarded.
    """
    kept: List[Dict[str, Any]] = []
    used = 0
    # Iterate in reverse (most recent first)
    for msg in reversed(messages):
        content = msg.get("content", "")
        tokens = count_tokens(content, model)
        if used + tokens > token_budget:
            break
        kept.append(msg)
        used += tokens
    # Reverse again to restore original order
    return list(reversed(kept))


def build_messages_within_budget(
    *,
    system_instruction: str,
    history_messages: List[Dict[str, Any]],
    compressed_context: str,
    query: str,
    budget: TokenBudget,
    model: str,
) -> List[Dict[str, Any]]:
    """Assemble the final message list respecting the token budget.

    The message order follows the typical Groq/ChatML format:

    1. System instruction.
    2. Prior chat *history* (truncated if necessary).
    3. The retrieved *compressed_context* as an assistant message.
    4. The current user *query*.
    """
    system_instruction = truncate_text_to_tokens(system_instruction, budget.system, model)
    query = truncate_text_to_tokens(query, budget.query, model)

    messages: List[Dict[str, Any]] = []
    # System
    messages.append({"role": "system", "content": system_instruction})

    # History – truncate to fit the allocated budget
    if budget.history > 0:
        trimmed = _truncate_history(history_messages, budget.history, model)
        # Keep only role and content to avoid unsupported fields
        sanitized = [
            {
                "role": m.get("role", "assistant"),
                "content": truncate_text_to_tokens(m.get("content", "") or "", budget.history, model),
            }
            for m in trimmed
        ]
        messages.extend(sanitized)

    # Documents / context – treat as an assistant message
    if budget.documents > 0 and compressed_context:
        compressed_context = truncate_text_to_tokens(compressed_context, budget.documents, model)
        messages.append({"role": "assistant", "content": compressed_context})

    # Query – always included
    messages.append({"role": "user", "content": query})

    return enforce_request_token_limit(
        messages,
        model=model,
        max_output_tokens=int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "256")),
    )


