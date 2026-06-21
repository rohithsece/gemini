import sqlite3
from .document_store import save_document

def read_student(conn: sqlite3.Connection, student_id: int) -> None:
    """Read and return student information.

    Retrieves the student's name and returns a formatted string. The result is
    saved as a document using :func:`save_document`.
    """
    cur = conn.cursor()
    cur.execute("SELECT name FROM students WHERE id = ?", (student_id,))
    row = cur.fetchone()
    if row:
        result = f"[READ] Success: Student ID {student_id}, Name: {row['name']}"
    else:
        result = f"[READ] Failure: No student found with ID {student_id}"
    save_document(result, operation="read_student")
