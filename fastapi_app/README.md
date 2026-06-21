# FastAPI Tool Call Service

This small FastAPI application demonstrates how the Antigravity agent could expose its internal tool calls over HTTP.

## Project structure
```
fastapi_app/
├─ main.py          # FastAPI application code
├─ requirements.txt # Dependencies
└─ README.md        # This file
```

## Prerequisites
- Python 3.9+ installed
- `pip` available in your environment

## Installation
```bash
pip install -r requirements.txt
```

## Running the server
```bash
uvicorn main:app --reload
```
The service will be available at `http://127.0.0.1:8000`.

## API usage
POST `/tool-call` with JSON payload:
```json
{
  "tool_name": "view_file",
  "arguments": {
    "AbsolutePath": "file:///path/to/file",
    "StartLine": 1,
    "EndLine": 10
  }
}
```
The response will contain a status and either the tool output in `data` or an `error_message`.

## Extending
Add more handlers in `main.py` and register them in the `TOOL_HANDLERS` dictionary.
