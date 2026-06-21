"""
Runnable demo for instructors / CI smoke tests.

Usage (from project root, with Weaviate + .env configured)::

    python -m context_pipeline.run_demo "What is Weaviate used for?"

Debug logs print to stderr; the final answer prints to stdout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from context_pipeline.config import PipelineConfig
from context_pipeline.pipeline import run_context_pipeline


def main() -> int:
    load_dotenv()
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print('Usage: python -m context_pipeline.run_demo "your question"', file=sys.stderr)
        return 2

    docs_dir = Path(os.environ.get("RAG_DOCS_DIR", "docs")).resolve()
    mode = os.environ.get("RAG_RETRIEVER", "vector").strip().lower()
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("Set GROQ_API_KEY", file=sys.stderr)
        return 2

    cfg = PipelineConfig.from_env()
    msgs = [{"role": "user", "content": q, "ts": __import__("time").time()}]

    out = run_context_pipeline(
        query=q,
        docs_dir=docs_dir,
        retriever_mode=mode,
        model=model,
        api_key=key,
        chat_messages=msgs,
        cfg=cfg,
    )
    print(out.answer)
    if cfg.pipeline_debug:
        print("\n--- meta ---", out.meta, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
