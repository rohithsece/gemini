"""FastAPI routes for tool listing, direct invocation, and agent loop."""

from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.tool_agent import run_tool_agent
from api.tool_registry import TOOL_SCHEMAS, execute_tool, get_tool_schema, list_tool_names
from student_management.core import create_demo_user, init_db

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    create_demo_user()
    yield


app = FastAPI(
    title="Tool Calling API",
    description="Groq function-calling tools used by this RAG + CRUD project",
    version="1.0.0",
    lifespan=lifespan,
)


class ToolCallBody(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentBody(BaseModel):
    query: str


@app.get("/")
def tools_root():
    return {
        "service": "Tool Calling API",
        "docs": "/docs",
        "list_tools": "/tools",
        "call_tool": "POST /call",
        "run_agent": "POST /agent",
        "concepts": {
            "schema": "TOOL_SCHEMAS in api/tool_registry.py (Groq tools= format)",
            "dispatch": "execute_tool() in api/tool_registry.py",
            "agent_loop": "run_tool_agent() in api/tool_agent.py",
            "langchain_tools": "project/langchain/tools.py (@tool decorator)",
            "rag_agent_tools": "context_pipeline/agent.py AGENT_TOOLS",
        },
    }


@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {"name": s["function"]["name"], "description": s["function"]["description"]}
            for s in TOOL_SCHEMAS
        ]
    }


@app.get("/tools/{name}")
def get_tool(name: str):
    schema = get_tool_schema(name)
    if not schema:
        raise HTTPException(status_code=404, detail="Tool not found")
    return schema


@app.post("/call")
def call_tool(body: ToolCallBody):
    if body.name not in list_tool_names():
        raise HTTPException(status_code=404, detail=f"Unknown tool: {body.name}")
    try:
        result = execute_tool(body.name, body.arguments)
        return {"tool": body.name, "arguments": body.arguments, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent")
def agent_endpoint(body: AgentBody):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        result = run_tool_agent(query)
        return {
            "answer": result.answer,
            "tools_used": result.tools_used,
            "steps": [
                {"tool": s.tool, "arguments": s.arguments, "result": s.result}
                for s in result.steps
            ],
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
