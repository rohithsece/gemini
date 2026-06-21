import sqlite3
from .auth import AuthDB, hash_password
from .document_store import save_document

def update_student(conn: sqlite3.Connection, student_id: int, name: str | None = None, password: str | None = None) -> None:
    """Update a student's name and/or password.

    If *name* is provided, the student's name is updated.
    If *password* is provided, the password hash is updated using the hashing utility.
    The outcome is saved as a document.
    """
    cur = conn.cursor()
    try:
        if name:
            cur.execute("UPDATE students SET name = ? WHERE id = ?", (name, student_id))
        if password:
            new_hash = hash_password(password)
            cur.execute("UPDATE students SET password_hash = ? WHERE id = ?", (new_hash, student_id))
        conn.commit()
        result = f"[UPDATE] Success: Student ID {student_id} updated."
    except Exception as e:
        result = f"[UPDATE] Failure: {e}"
    save_document(result, operation="update_student")
