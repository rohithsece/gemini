"""Chat model wrapper for Groq LLM.

This module provides a minimal implementation of ``CustomGroqChatModel`` used by the
LangGraph pipeline (``project.langchain.graph``). The class follows the simple
interface expected by the pipeline – an ``invoke`` method that accepts a prompt
string and returns the LLM's answer.

A real implementation would read configuration (e.g., API key, model name) from
environment variables and use the ``groq`` SDK. For the purposes of this project
we provide a lightweight wrapper that gracefully falls back to a mock response if
the Groq SDK is unavailable or the API call fails. This ensures the pipeline can
run during development and testing without requiring external network access.
"""

from __future__ import annotations

import os
from typing import Any

# Attempt to import the Groq SDK; if unavailable we will use a mock.
try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore


class CustomGroqChatModel:
    """Simple wrapper around the Groq chat completion API.

    Parameters
    ----------
    model_name: str, optional
        The model identifier to use. Defaults to ``llama-3.1-8b-instant``.
    temperature: float, optional
        Sampling temperature; ``0.0`` yields deterministic output.
    """

    def __init__(self, model_name: str | None = None, temperature: float = 0.0) -> None:
        model = model_name or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        if "mixtral" in model.lower():
            model = "llama-3.1-8b-instant"
        self.model_name = model
        self.temperature = temperature
        # Initialise the Groq client if the SDK is present and an API key is set.
        if Groq is not None and os.getenv("GROQ_API_KEY"):
            self.client = Groq()
        else:
            self.client = None

    def invoke(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the generated answer.

        If the Groq client cannot be instantiated (missing SDK or API key) a
        deterministic mock string is returned so the rest of the pipeline can
        continue without raising an exception.
        """
        if self.client is None:
            # Mock response – useful for offline development and unit tests.
            return "[Mock Groq response] " + prompt[:75]
        try:
            from context_pipeline.config import PipelineConfig
            from context_pipeline.token_budget import (
                compute_token_budget,
                enforce_request_token_limit,
                truncate_text_to_tokens,
            )

            cfg = PipelineConfig.from_env()
            budget = compute_token_budget(cfg, self.model_name)
            prompt = truncate_text_to_tokens(prompt, budget.input_total, self.model_name)
            messages = enforce_request_token_limit(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                max_output_tokens=cfg.max_output_tokens,
            )
            from RAG_groq import answer_with_groq_messages

            answer, _ = answer_with_groq_messages(
                model=self.model_name,
                api_key=os.environ.get("GROQ_API_KEY", "").strip(),
                messages=messages,  # type: ignore[arg-type]
                max_tokens=cfg.max_output_tokens,
                temperature=self.temperature,
            )
            return answer
        except Exception as exc:
            # In production you might want to surface the error more clearly.
            return f"Error invoking Groq model: {exc}"
