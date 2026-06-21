import sqlite3
from .auth import AuthDB
from .document_store import save_document

def create_student(conn: sqlite3.Connection, student_id: int, name: str, password: str) -> None:
    """Create a new student record.

    Uses :class:`AuthDB` to hash the password and insert the student.
    The operation outcome (success/failure) is saved as a human‑readable document.
    """
    auth = AuthDB(conn)
    try:
        auth.add_student(student_id, name, password)
        result = f"[CREATE] Success: Student ID {student_id}, Name '{name}' added."
    except Exception as e:
        result = f"[CREATE] Failure: {e}"
    save_document(result, operation="create_student")
