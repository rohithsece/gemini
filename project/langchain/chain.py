import os
from typing import Dict, Any, List
from pathlib import Path
from langchain_core.runnables import RunnableLambda, RunnableParallel
from project.langchain.retriever import CustomContextRetriever
from project.langchain.chat_model import CustomGroqChatModel
from project.langchain.memory_layer import SQLiteChatMessageHistory
from project.langchain.prompt import get_rag_prompt
from context_pipeline.config import PipelineConfig
from context_pipeline.token_budget import compute_token_budget
from langchain_core.documents import Document
from functools import lru_cache

def format_retrieved_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def create_rag_chain(
    session_id: str = "default_session",
    docs_dir: str = "docs",
    retriever_mode: str = "bm25",
    top_k: int = 5,
    model_name: str = "llama-3.1-8b-instant",
    api_key: str = "",
    system_prompt_path: str | None = None,
):
    retriever = CustomContextRetriever(docs_dir=docs_dir, retriever_mode=retriever_mode, top_k=top_k)
    history = SQLiteChatMessageHistory(session_id)
    llm = CustomGroqChatModel(model_name=model_name)
    
    @lru_cache(maxsize=256)
    def _cached_retrieve(q: str) -> List[Document]:
        return retriever.invoke(q)

    def _load_history(_: Any) -> Dict[str, Any]:
        cfg = PipelineConfig.from_env()
        budget = compute_token_budget(cfg, model_name)
        chat_history = history.get_decayed_messages(model_name, budget.history)
        return {"chat_history": chat_history, "budget": budget}

    def load_context_and_history(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        parallel = RunnableParallel(
            docs=RunnableLambda(lambda _: _cached_retrieve(question)),
            hist=RunnableLambda(_load_history),
        )
        out = parallel.invoke({})
        docs = out["docs"]
        hist_info = out["hist"]
        context_str = format_retrieved_docs(docs)
        return {
            "context": context_str,
            "chat_history": hist_info["chat_history"],
            "question": question,
        }

    def generate_response(inputs: Dict[str, Any]) -> str:
        question = inputs["question"]
        context = inputs["context"]
        chat_history = inputs["chat_history"]
        
        prompt = get_rag_prompt(
            chunks=[Document(page_content=context)],
            history=chat_history,
            query=question
        )
        
        answer = llm.invoke(prompt)
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer

    return RunnableLambda(load_context_and_history) | RunnableLambda(generate_response)
