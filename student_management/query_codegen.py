"""Generate and save CRUD query modules for student_management."""

import re
from pathlib import Path

QUERIES_DIR = Path(__file__).parent / "queries"

SCHEMA = """
Database: student_database.db
Table students: id INTEGER PRIMARY KEY, name TEXT, password_hash TEXT
Table records: id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, course TEXT, grade TEXT
Demo student_id: 10045
"""

EXAMPLE = """
import sqlite3
from document_store import save_document

def create_record(conn: sqlite3.Connection, student_id: int, course: str, grade: str) -> str:
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO records (student_id, course, grade) VALUES (?, ?, ?)",
            (student_id, course, grade),
        )
        conn.commit()
        result = f"[CREATE_RECORD] Success: student_id={student_id}, course='{course}', grade='{grade}'"
    except Exception as e:
        result = f"[CREATE_RECORD] Failure: {e}"
    save_document(result, operation="create_record")
    return result

def main(conn: sqlite3.Connection):
    # Expose a main function that runs the CRUD function with parameters specified or inferred from the user query
    return create_record(conn, student_id=10045, course="Biology 101", grade="A-")
"""


def is_crud_query(description: str) -> bool:
    d = description.lower()
    crud = {
        "create", "read", "update", "delete", "insert", "select", "crud",
        "get", "list", "fetch", "add", "remove", "show", "find", "modify", "edit"
    }
    domain = {
        "student", "record", "records", "grade", "course", "sqlite", "query",
        "database", "db", "academic"
    }
    return "crud" in d or (any(w in d for w in crud) and any(w in d for w in domain))


def build_crud_prompt(description: str) -> str:
    return (
        "Write a complete Python module for the student management CRUD system.\n"
        f"Schema:{SCHEMA}\n"
        "Rules:\n"
        "- Use sqlite3 with parameterized queries (?, not f-strings for values).\n"
        "- You MUST explicitly import sqlite3 (`import sqlite3`) and save_document (`from document_store import save_document`) at the top of the file.\n"
        "- Expose one main function named after the operation (create_record, read_records, update_record, delete_record, or a clear snake_case name).\n"
        "- Function signature should accept conn: sqlite3.Connection and relevant params (student_id, course, grade, record_id, etc.).\n"
        "- Log human-readable results via save_document(result, operation='operation_name').\n"
        "- Return useful data where appropriate (e.g. list of dicts/rows from read, or a string result message).\n"
        "- You MUST also expose a `main(conn: sqlite3.Connection)` function. Inside `main`, call the primary CRUD function with the exact parameter values (e.g., student_id, course, grade, record_id) specified or implied in the user query description. If any parameter is not specified, use a reasonable default. The `main` function should return the result of that call.\n"
        f"Example style:\n{EXAMPLE}\n"
        f"Request: {description}\n"
        "Return ONLY a markdown fenced python code block. No explanation."
    )


def extract_python_code(raw: str) -> str:
    match = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n```", raw)
    return match.group(1).strip() if match else raw.strip()


def infer_filename(description: str) -> str:
    d = description.lower()
    if any(w in d for w in ("create", "insert", "add")):
        base = "create_record"
    elif any(w in d for w in ("read", "get", "list", "select", "fetch")):
        base = "read_records"
    elif any(w in d for w in ("update", "edit", "modify")):
        base = "update_record"
    elif any(w in d for w in ("delete", "remove")):
        base = "delete_record"
    else:
        words = re.findall(r"[a-z]+", d)
        base = "_".join(words[:4]) if words else "generated_query"
    return sanitize_filename(f"{base}.py")


def sanitize_filename(name: str) -> str:
    name = "".join(c for c in name if c.isalnum() or c in "._-")
    if not name.endswith(".py"):
        name += ".py"
    return name


def save_query_file(code: str, filename: str) -> Path:
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    path = QUERIES_DIR / sanitize_filename(filename)
    path.write_text(code, encoding="utf-8")
    return path
