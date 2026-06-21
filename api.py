import os
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import the pipeline function
from context_pipeline.autogen_pipeline import run_autogen_pipeline, AutoGenPipelineResult

app = FastAPI(title="AutoGen RAG API", version="0.1.0")

class RunRequest(BaseModel):
    query: str = Field(..., description="User query to answer")
    docs_dir: str = Field(..., description="Path to the documents directory")
    retriever_mode: str = Field("bm25", description="Retriever mode, e.g., 'bm25'")
    model: str = Field(..., description="LLM model name to use (Groq compatible)")
    api_key: str = Field(..., description="API key for the LLM provider")
    model_info: dict | None = Field(None, description="Optional model_info for non‑OpenAI models")
    # Optional additional chat messages for context (like conversation history)
    chat_messages: List[dict[str, Any]] = Field(
        default_factory=list,
        description="List of prior chat message dicts (role/content) if needed",
    )

class RunResponse(BaseModel):
    answer: str
    context_display: str
    meta: dict[str, Any]

@app.post("/run", response_model=RunResponse)
def run_endpoint(request: RunRequest):
    # Validate docs_dir exists
    docs_path = Path(request.docs_dir)
    if not docs_path.is_dir():
        raise HTTPException(status_code=400, detail="docs_dir does not exist or is not a directory")

    try:
        result: AutoGenPipelineResult = run_autogen_pipeline(
            query=request.query,
            docs_dir=docs_path,
            retriever_mode=request.retriever_mode,
            model=request.model,
            api_key=request.api_key,
            chat_messages=request.chat_messages,
            cfg=None,
            model_info=request.model_info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RunResponse(
        answer=result.answer,
        context_display=result.context_display,
        meta=result.meta,
    )
