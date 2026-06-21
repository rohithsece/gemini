"""
Tool registry — schemas (Groq/OpenAI format) + Python handlers.

Same concept as context_pipeline/agent.py AGENT_TOOLS + dispatch loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from delegate_to_rag_agent import delegate_to_rag_agent
from student_management.core import get_connection
from student_management.query_codegen import (
    build_crud_prompt,
    extract_python_code,
    infer_filename,
    is_crud_query,
    save_query_file,
)
from student_management.query_runner import list_queries, run_query, save_query
from RAG_groq import answer_with_groq
from context_pipeline.data_agent import _execute_python

ToolHandler = Callable[[dict[str, Any]], Any]

# ---------------------------------------------------------------------------
# Groq function-calling schemas (tools the LLM can invoke)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search indexed documents via the RAG pipeline and return grounded context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_crud_code",
            "description": (
                "Generate Python CRUD query code for student_management and save it "
                "under student_management/queries/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language CRUD request, e.g. create student record query.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_query_file",
            "description": "Save a Python query file to student_management/queries/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["filename", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query_file",
            "description": "Execute a saved query module from student_management/queries/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "e.g. create_record.py"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_query_files",
            "description": "List all .py query files in student_management/queries/.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_student_records",
            "description": "Read academic records for a student from SQLite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "integer", "description": "Demo ID: 10045"},
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_student_record",
            "description": "Insert a course/grade record for a student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "integer"},
                    "course": {"type": "string"},
                    "grade": {"type": "string"},
                },
                "required": ["student_id", "course", "grade"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run a short Python snippet in a safe sandbox (math/statistics).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": "Return the final answer to the user after using other tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "tools_used": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations (called when LLM or API client invokes a tool)
# ---------------------------------------------------------------------------

def _tool_search_knowledge_base(args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "Error: query is required"
    return delegate_to_rag_agent(query)


def _tool_generate_crud_code(args: dict) -> dict:
    description = (args.get("description") or "").strip()
    if not description:
        return {"error": "description is required"}
    if not is_crud_query(description):
        return {"error": "Not recognized as a CRUD query", "description": description}
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    raw, _ = answer_with_groq(
        model=model, api_key=api_key, query=build_crud_prompt(description), context=""
    )
    code = extract_python_code(raw)
    filename = infer_filename(description)
    path = save_query_file(code, filename)
    return {"code": code, "filename": path.name, "saved_file": str(path)}


def _tool_save_query_file(args: dict) -> dict:
    path = save_query(args["filename"], args["code"])
    return {"message": f"Saved {path.name}", "path": str(path)}


def _tool_run_query_file(args: dict) -> dict:
    try:
        return {"output": run_query(args["filename"])}
    except FileNotFoundError:
        return {"error": "File not found"}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_list_query_files(_args: dict) -> list[str]:
    return list_queries()


def _tool_read_student_records(args: dict) -> list[dict]:
    student_id = int(args["student_id"])
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, course, grade FROM records WHERE student_id = ? ORDER BY id",
        (student_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _tool_create_student_record(args: dict) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (student_id, course, grade) VALUES (?, ?, ?)",
        (int(args["student_id"]), args["course"], args["grade"]),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return {"message": "Record created", "id": row_id}


def _tool_execute_python(args: dict) -> str:
    return _execute_python(args.get("code") or "")


def _tool_finalize_answer(args: dict) -> dict:
    return {"answer": args.get("answer", ""), "tools_used": args.get("tools_used", [])}


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "search_knowledge_base": _tool_search_knowledge_base,
    "generate_crud_code": _tool_generate_crud_code,
    "save_query_file": _tool_save_query_file,
    "run_query_file": _tool_run_query_file,
    "list_query_files": _tool_list_query_files,
    "read_student_records": _tool_read_student_records,
    "create_student_record": _tool_create_student_record,
    "execute_python": _tool_execute_python,
    "finalize_answer": _tool_finalize_answer,
}


def list_tool_names() -> list[str]:
    return list(TOOL_HANDLERS.keys())


def get_tool_schema(name: str) -> dict | None:
    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return schema
    return None


def execute_tool(name: str, arguments: dict[str, Any] | str | None) -> Any:
    """Dispatch a tool call — same role as the agent loop in context_pipeline/agent.py."""
    if name not in TOOL_HANDLERS:
        raise ValueError(f"Unknown tool: {name}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    arguments = arguments or {}
    result = TOOL_HANDLERS[name](arguments)
    if isinstance(result, (dict, list)):
        return result
    return {"result": str(result)}
