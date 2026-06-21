import sqlite3
from .document_store import save_document

def delete_student(conn: sqlite3.Connection, student_id: int) -> None:
    """Delete a student record.

    Removes the student with the given ``student_id`` from the ``students`` table.
    The outcome (success or failure) is saved as a document.
    """
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        if cur.rowcount:
            result = f"[DELETE] Success: Student ID {student_id} removed."
        else:
            result = f"[DELETE] Failure: No student found with ID {student_id}."
    except Exception as e:
        result = f"[DELETE] Failure: {e}"
    save_document(result, operation="delete_student")
