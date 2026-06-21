"""ASGI entrypoint: FastAPI student API + Flask RAG UI on one port."""

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from student_management.fastapi_app import app as student_api
from api.fastapi_tools import app as tools_api

# Flask app is created at import time in web.py
from web import app as flask_app

root = FastAPI(
    title="RAG Model Server",
    description="Flask UI + FastAPI student CRUD API",
    version="1.0.0",
)


@root.get("/api")
def api_index():
    return {
        "ui": "/",
        "code_generator": "/code",
        "query_explorer": "/query-explorer",
        "student_api": "/api/students",
        "student_docs": "/api/students/docs",
        "tools_api": "/api/tools",
        "tools_docs": "/api/tools/docs",
        "tool_agent": "POST /api/tools/agent",
    }


root.mount("/api/tools", tools_api)
root.mount("/api/students", student_api)
root.mount("/", WSGIMiddleware(flask_app))

# uvicorn: uvicorn asgi:root --host 0.0.0.0 --port 7860
