"""
Advanced Multi-Agent Orchestrator

Provides a higher-level orchestrator that classifies queries, runs the
RAG and Data agents (in parallel when appropriate), refines queries for
sub-agents, and synthesises their outputs into a single coherent result.

This is intentionally conservative: it relies on the existing RAGAgent and
DataAgent implementations and returns the same `OrchestratorResult` shape
used by `context_pipeline/multi_agent.py` so it can be swapped in easily.
"""

from __future__ import annotations

import re
import threading
from dataclasses import field, dataclass
from typing import Callable, List

from context_pipeline.agent import RAGAgent, AgentResult
from context_pipeline.data_agent import DataAgent, DataAgentResult
from context_pipeline.config import PipelineConfig
from context_pipeline.multi_agent import OrchestratorResult, OrchestratorStep


def _is_math_query(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    # heuristic: numeric tokens or math keywords
    if re.search(r"\d", text):
        return True
    math_kw = r"\b(sum|average|mean|median|std|stddev|variance|calculate|compute|convert|percentage|percent|ratio|difference|add|subtract|multiply|divide)\b"
    return re.search(math_kw, text, re.IGNORECASE) is not None


def _is_doc_query(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    doc_kw = r"\b(document|policy|manual|guide|report|readme|spec|procedure|how to|how-do|instruction|file|docs)\b"
    return re.search(doc_kw, text, re.IGNORECASE) is not None


class AdvancedMultiAgentOrchestrator:
    """Advanced orchestrator that routes queries to the RAG and Data agents.

    Constructor arguments mirror the existing `MultiAgentOrchestrator` so it
    can be used as a drop-in replacement in most call-sites.
    """

    MAX_DELEGATIONS = 6

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

        # instantiate sub-agents
        self._rag_agent = RAGAgent(retriever_fn=retriever_fn, model=model, api_key=api_key, cfg=cfg)
        self._data_agent = DataAgent(model=model, api_key=api_key, cfg=cfg)

    def run(self, query: str, history: List[dict]) -> OrchestratorResult:
        """Run the advanced orchestrator for *query*.

        Behaviour:
        - Classify the query (doc / math / mixed)
        - Delegate to the appropriate sub-agent(s). For mixed queries, run
          both agents in parallel to save latency.
        - Collect and synthesise results into an `OrchestratorResult`.
        """
        kind_doc = _is_doc_query(query)
        kind_math = _is_math_query(query)

        steps: List[OrchestratorStep] = []
        usage: dict = {}
        sources: List[str] = []

        rag_result: AgentResult | None = None
        data_result: DataAgentResult | None = None

        threads = []

        def _run_rag():
            nonlocal rag_result
            steps.append(OrchestratorStep("delegation", "RAG Agent", f"Delegating query to RAG: {query}"))
            rag_result = self._rag_agent.run(query, history)

        def _run_data():
            nonlocal data_result
            steps.append(OrchestratorStep("delegation", "Data Agent", f"Delegating query to DataAgent: {query}"))
            data_result = self._data_agent.run(query)

        # Decide delegation strategy
        if kind_doc and not kind_math:
            _run_rag()
        elif kind_math and not kind_doc:
            _run_data()
        else:
            # Mixed or unclear — run both in parallel up to MAX_DELEGATIONS
            t1 = threading.Thread(target=_run_rag, name="rag-thread")
            t2 = threading.Thread(target=_run_data, name="data-thread")
            t1.start()
            t2.start()
            threads.extend([t1, t2])
            for t in threads:
                t.join()

        # Collect usage and sources
        if rag_result:
            usage.update(rag_result.usage or {})
            sources.extend(rag_result.sources_used or [])
            steps.extend(OrchestratorStep(s.kind, s.label, s.content) for s in rag_result.steps)

        if data_result:
            # DataAgentResult has similar shape but different naming; try to adapt
            try:
                usage.update(data_result.usage or {})
            except Exception:
                pass
            try:
                # Data agent may not provide `sources_used` — add a short marker
                if getattr(data_result, "sources_used", None):
                    sources.extend(data_result.sources_used)
            except Exception:
                pass
            try:
                steps.extend(OrchestratorStep("data_result", "Data result", getattr(data_result, "answer", str(data_result))))
            except Exception:
                pass

        # Synthesize final answer
        parts: List[str] = []
        if rag_result and rag_result.answer:
            parts.append("Document search results:\n" + rag_result.answer)
        if data_result and getattr(data_result, "answer", None):
            parts.append("Data analysis:\n" + getattr(data_result, "answer"))

        if not parts:
            final_text = "I couldn't find a direct answer. Try rephrasing or provide more context."
            confidence = "low"
        else:
            final_text = "\n\n---\n\n".join(parts)
            # Simple confidence heuristic
            confidences = []
            if rag_result:
                confidences.append(rag_result.confidence)
            if data_result:
                try:
                    confidences.append(data_result.confidence)
                except Exception:
                    pass
            if "high" in confidences:
                confidence = "high"
            elif "medium" in confidences:
                confidence = "medium"
            else:
                confidence = "low"

        steps.append(OrchestratorStep("answer", "Synthesis", final_text))

        return OrchestratorResult(
            answer=final_text,
            steps=steps,
            confidence=confidence,
            sources_used=list(dict.fromkeys(sources)),
            usage=usage,
        )
