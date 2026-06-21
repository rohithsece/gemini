"""
Memory decay: older conversational turns matter less than recent ones.

WHY: long chat sessions would otherwise fill the entire context window with
stale user questions, starving the document retrieval channel. Products like
ChatGPT effectively summarize or drop old turns; decay is a transparent,
math-forward way to explain the same idea to an instructor.

DECAY FORMULA (per message)::

    weight = exp( -λ * age_hours )

where ``age_hours`` is (now - message_timestamp) / 3600.

- ``λ`` (lambda) controls how fast memory fades — larger λ forgets quicker.
- If timestamps are missing (plain UI messages), we approximate age using
  **position from the end of the list**: ``age_hours ≈ (distance_from_end) * 0.25``
  so the second-to-last message looks ~15 minutes older than the last, etc.

This is a smooth exponential prior used only for **trimming** history to fit
the token budget (we drop lowest-weight messages first while over budget).
"""

from __future__ import annotations

import math
import time
from typing import Any

from context_pipeline.config import PipelineConfig
from context_pipeline.logging_utils import log_stage


def _parse_ts(raw: Any) -> float | None:
    """Return epoch seconds or None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        try:
            from datetime import datetime

            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return None
    return None


def decay_weight_for_message(
    msg: dict[str, Any],
    *,
    index: int,
    total: int,
    now: float,
    cfg: PipelineConfig,
) -> float:
    """
    Compute a scalar weight in (0, 1] — higher means "more deserving of tokens".

    Recent timestamps (or positions near the end of the list) receive weights
    closer to 1.0; very old turns approach exp(-large) ≈ 0.
    """
    lam = max(1e-6, cfg.memory_decay_lambda)
    ts = _parse_ts(msg.get("ts")) if cfg.use_message_ts else None
    if ts is not None:
        age_h = max(0.0, (now - ts) / 3600.0)
    else:
        # Synthetic age: older toward the front of the list
        dist_from_end = max(0, total - 1 - index)
        age_h = dist_from_end * 0.25
    return math.exp(-lam * age_h)


def trim_chat_messages(
    messages: list[dict[str, Any]],
    *,
    token_counter: Any,
    max_tokens: int,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Drop whole messages (oldest with lowest decay weight first) until under budget.

    We never drop the **last** user message here — the pipeline passes the live
    ``query`` separately. This function only trims *prior* turns for history.
    """
    if not messages or max_tokens <= 0:
        return []

    now = time.time()
    total = len(messages)
    weights = [
        decay_weight_for_message(m, index=i, total=total, now=now, cfg=cfg)
        for i, m in enumerate(messages)
    ]

    kept_idx = list(range(len(messages)))

    def total_tokens(idxs: list[int]) -> int:
        s = 0
        for i in idxs:
            s += token_counter(messages[i].get("content", ""))
        return s

    while len(kept_idx) > 1 and total_tokens(kept_idx) > max_tokens:
        # Drop lowest decay weight first; on ties drop the older (smaller index) turn.
        drop = min(kept_idx, key=lambda i: (weights[i], i))
        kept_idx = [i for i in kept_idx if i != drop]

    kept_idx.sort()
    out = [messages[i] for i in kept_idx]

    log_stage(
        "memory_decay",
        input_messages=len(messages),
        output_messages=len(out),
        lambda_decay=cfg.memory_decay_lambda,
        budget_tokens=max_tokens,
        weights_sample=[round(weights[i], 4) for i in kept_idx[:5]],
    )
    return out
