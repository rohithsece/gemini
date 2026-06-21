# delete_record.py
"""
CRUD: Delete a student record.
"""

import sqlite3
from document_store import save_document

def delete_record(conn: sqlite3.Connection, record_id: int, student_id: int) -> None:
    """Delete a specific record belonging to a student.

    Parameters
    ----------
    conn: sqlite3.Connection
        Active DB connection.
    record_id: int
        ID of the record to delete.
    student_id: int
        ID of the student that owns the record (ensures ownership).
    """
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM records WHERE id = ? AND student_id = ?",
            (record_id, student_id),
        )
        conn.commit()
        if cur.rowcount:
            result = f"[DELETE_RECORD] Success: record_id={record_id} deleted for student_id={student_id}."
        else:
            result = f"[DELETE_RECORD] Failure: No matching record found for record_id={record_id} and student_id={student_id}."
    except Exception as e:
        result = f"[DELETE_RECORD] Failure: {e}"
    save_document(result, operation="delete_record")
