from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List, Any

from langgraph.graph import StateGraph

# Import existing pipeline components
from context_pipeline.hybrid_retrieval import retrieve_hybrid_candidates
from context_pipeline.deduplication import dedupe_chunks
from context_pipeline.reranking import rerank_chunks
from context_pipeline.compression import compress_chunks
from context_pipeline.config import PipelineConfig
from project.langchain.memory_layer import SQLiteChatMessageHistory
from project.langchain.prompt import get_rag_prompt
from project.langchain.retriever import CustomContextRetriever
from project.langchain.chat_model import CustomGroqChatModel  # Assuming this exists
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


@dataclass
class RAGState:
    """State that flows through the LangGraph pipeline.

    Attributes:
        query: User question.
        retrieved: List of Document objects after hybrid retrieval.
        deduped: List of Document after deduplication.
        reranked: List of Document after reranking.
        compressed: List of Document after compression.
        history: List of chat messages (BaseMessage) from SQLite.
        prompt: Final prompt string sent to the LLM.
        answer: LLM response text.
    """
    query: str
    retrieved: List[Document] | None = None
    deduped: List[Document] | None = None
    reranked: List[Document] | None = None
    compressed: List[Document] | None = None
    history: List[BaseMessage] | None = None
    prompt: str | None = None
    answer: str | None = None


def retrieve_node(state: RAGState) -> dict:
    cfg = PipelineConfig.from_env()
    retriever = CustomContextRetriever(docs_dir=".", retriever_mode="hybrid", top_k=cfg.top_k_final)
    chunks = retrieve_hybrid_candidates(retriever, state.query, cfg)
    return {"retrieved": chunks}


def dedupe_node(state: RAGState) -> dict:
    cfg = PipelineConfig.from_env()
    deduped = dedupe_chunks(state.retrieved or [], cfg)
    return {"deduped": deduped}


def rerank_node(state: RAGState) -> dict:
    cfg = PipelineConfig.from_env()
    retriever = CustomContextRetriever(docs_dir=".", retriever_mode="hybrid", top_k=cfg.top_k_final)
    embedder = getattr(getattr(retriever, "_inner", None), "embedder", None)
    reranked = rerank_chunks(state.query, state.deduped or [], embedder, cfg, docs_dir=pathlib.Path("."))
    return {"reranked": reranked}


def compress_node(state: RAGState) -> dict:
    cfg = PipelineConfig.from_env()
    compressed = compress_chunks(state.query, state.reranked or [], cfg)
    return {"compressed": compressed}


def history_node(state: RAGState) -> dict:
    # Load chat history with decay & token budgeting
    import os
    cfg = PipelineConfig.from_env()
    mem = SQLiteChatMessageHistory("default_session")
    from context_pipeline.token_budget import compute_token_budget
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if "mixtral" in model.lower():
        model = "llama-3.1-8b-instant"
    budget = compute_token_budget(cfg, model)
    history = mem.get_decayed_messages(model, budget.history)
    return {"history": history}


def prompt_node(state: RAGState) -> dict:
    prompt = get_rag_prompt(state.compressed or [], state.history or [], state.query)
    return {"prompt": prompt}


def llm_node(state: RAGState) -> dict:
    cfg = PipelineConfig.from_env()
    llm = CustomGroqChatModel()
    # assuming the LLM wrapper has an `invoke` that returns answer string
    answer = llm.invoke(state.prompt)
    return {"answer": answer}


def save_history_node(state: RAGState) -> dict:
    # Persist the new user & assistant messages
    mem = SQLiteChatMessageHistory("default_session")
    from langchain_core.messages import HumanMessage, AIMessage
    mem.add_user_message(state.query)
    mem.add_ai_message(state.answer)
    return {}


def build_graph() -> Any:
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("compress", compress_node)
    graph.add_node("history", history_node)
    graph.add_node("prompt", prompt_node)
    graph.add_node("llm", llm_node)
    graph.add_node("save", save_history_node)

    # Define edges – parallel retrieval and history loading, then sequential flow
    graph.add_edge("retrieve", "dedupe")
    graph.add_edge("dedupe", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "prompt")
    graph.add_edge("history", "prompt")
    graph.add_edge("prompt", "llm")
    graph.add_edge("llm", "save")

    # Entry points – start both retrieval and history
    graph.set_entry_point("retrieve")
    graph.set_entry_point("history")
    return graph.compile()

# Compiled graph – lazily built on first import
_compiled_graph = None

def _get_compiled():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

def run_rag(query: str) -> str:
    """Execute the RAG pipeline via LangGraph and return the answer.
    Guarantees a non‑empty string by falling back to the simple Groq answer
    function if the graph fails or returns no answer.
    """
    try:
        graph = _get_compiled()
        result = graph.invoke({"query": query})
        answer = result.get("answer")
    except Exception as exc:
        # Log the exception internally if needed (omitted for brevity)
        answer = None
    # Fallback to basic Groq answer generation if needed
    if not answer:
        try:
            answer = answer_with_groq(query)
        except Exception:
            answer = "[No answer generated]"
    return answer
