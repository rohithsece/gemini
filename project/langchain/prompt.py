from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


def get_rag_prompt(chunks: List[Document], history: List[BaseMessage], query: str) -> str:
    """Build a plain-text prompt from retrieved docs, chat history, and the user query."""
    import os
    from context_pipeline.config import PipelineConfig
    from context_pipeline.token_budget import compute_token_budget, truncate_text_to_tokens

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if "mixtral" in model.lower():
        model = "llama-3.1-8b-instant"
    cfg = PipelineConfig.from_env()
    budget = compute_token_budget(cfg, model)

    context = "\n\n".join(getattr(chunk, "page_content", str(chunk)) for chunk in chunks)
    context = truncate_text_to_tokens(context, budget.documents, model)
    history_text = "\n".join(
        f"{getattr(msg, 'type', 'user')}: {getattr(msg, 'content', str(msg))}" for msg in history
    )
    history_text = truncate_text_to_tokens(history_text, budget.history, model)
    query = truncate_text_to_tokens(query, budget.query, model)
    return (
        "You are a helpful RAG assistant. Use the provided context to answer the question."
        " Do not hallucinate; if the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"History:\n{history_text}\n\n"
        f"Question:\n{query}"
    )
