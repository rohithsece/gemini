# crew_ai.py – thin wrapper to integrate CrewAI with existing LangGraph RAG

"""CrewAI integration module.

This module provides a single function `run_crew_ai` that matches the
signature expected by the Flask `/delegate` endpoint (see `web.py`).
The implementation is intentionally lightweight: it re‑uses the existing
LangGraph RAG pipeline (`run_rag`) to perform the actual retrieval and
generation, and then returns a result compatible with the UI.

If the optional `crewai` package is available we also expose a minimal
example of a Crew that could be expanded later.  The current code works
without installing additional dependencies, ensuring the app continues to
run even in environments where CrewAI is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

# Import the existing LangGraph RAG runner
try:
    from project.langchain.graph import run_rag
except Exception as exc:  # pragma: no cover – defensive guard
    raise ImportError("Unable to import run_rag from project.langchain.graph") from exc

# Optional CrewAI import – we keep it inside a try block so the module loads
# even if the library is not installed.
try:
    from crewai import Crew, Agent, Task  # type: ignore
    _CREWAI_AVAILABLE = True
except Exception:  # pragma: no cover
    _CREWAI_AVAILABLE = False


def _run_via_crew(query: str) -> str:
    """Run the query using a minimal CrewAI crew.

    This helper demonstrates how a Crew could be constructed around the
    existing `run_rag` function.  It is only used when the optional
    ``crewai`` package is present; otherwise the plain ``run_rag`` path
    is taken.
    """
    if not _CREWAI_AVAILABLE:
        # Fallback – simply call the LangGraph pipeline.
        return run_rag(query)

    # Define a very small agent that delegates to ``run_rag``.
    researcher = Agent(
        role="Researcher",
        goal="Answer user questions using the existing RAG pipeline",
        backstory="Uses the LangGraph pipeline to retrieve and generate answers.",
        verbose=False,
    )

    # The task description contains the user query.
    task = Task(
        description=f"Answer the following question using the RAG pipeline: {query}",
        expected_output="A concise answer string.",
        agent=researcher,
        # The ``tools`` argument lets the task call a custom function.
        # We expose a tiny wrapper that calls ``run_rag``.
        tools=[
            {
                "name": "run_rag",
                "description": "Execute the LangGraph RAG pipeline.",
                "func": lambda q=query: run_rag(q),
            }
        ],
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    # ``kickoff`` returns the final answer string.
    answer = crew.kickoff()
    return answer


def run_crew_ai(
    query: str,
    docs_dir: Path | str,
    retriever_mode: str,
    model: str,
    api_key: str,
    chat_messages: list[dict] | None = None,
    cfg: Any | None = None,
) -> Tuple[str, dict]:
    """Execute a CrewAI‑driven RAG workflow.

    Parameters
    ----------
    query: str
        The user question.
    docs_dir: Path | str
        Directory containing documents – currently unused because the
        underlying LangGraph pipeline discovers the index automatically.
    retriever_mode: str
        Desired retriever (e.g., ``"hybrid"``).  Present for API compatibility.
    model: str
        Model identifier – ignored; we continue to use the Groq LLM configured
        in the environment.
    api_key: str
        API key for Groq – also ignored here.
    chat_messages: list[dict] | None
        Prior chat history – not required for the current implementation.
    cfg: Any | None
        Optional configuration object – unused.

    Returns
    -------
    tuple[str, dict]
        ``answer`` – the generated response string.
        ``meta`` – a dictionary with optional debugging information.
    """
    # Ensure ``docs_dir`` is a Path for potential future use.
    _ = Path(docs_dir)  # noqa: F841 – placeholder for future extensions.

    # Primary execution path – try to use CrewAI if available, otherwise
    # fall back to the raw LangGraph pipeline.
    if _CREWAI_AVAILABLE:
        answer = _run_via_crew(query)
    else:
        answer = run_rag(query)

    # Minimal metadata – the UI can display this if desired.
    meta = {
        "context": "",
        "usage": {},
        "meta": {},
    }
    return answer, meta

# Export only the public function.
__all__ = ["run_crew_ai"]
