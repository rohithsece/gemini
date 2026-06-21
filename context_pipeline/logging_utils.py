"""
Lightweight stage logging for demos, grading, and production log correlation.

We intentionally avoid structlog / loguru to keep dependencies minimal; prints
go to stderr so Flask access logs on stdout stay readable when piping output.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def log_stage(stage: str, **fields: Any) -> None:
    """
    Emit one JSON line per pipeline stage.

    WHY JSON lines: your instructor (or Datadog later) can grep ``stage`` keys
    without a special log parser. ``**fields`` holds counts, thresholds, etc.
    """
    payload = {"stage": stage, **fields}
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        line = json.dumps({"stage": stage, "note": "serialization_failed"}, default=str)
    try:
        print(line, file=sys.stderr, flush=True)
    except OSError:
        # Windows Flask/background processes can raise EINVAL on stderr writes.
        try:
            print(json.dumps(payload, ensure_ascii=True, default=str), file=sys.stdout, flush=True)
        except OSError:
            pass
