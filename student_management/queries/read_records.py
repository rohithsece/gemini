import sqlite3
from document_store import save_document

def read_records(conn: sqlite3.Connection, student_id: int) -> list:
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT * FROM records WHERE student_id = ?",
            (student_id,),
        )
        rows = cur.fetchall()
        result = [dict(zip([desc[0] for desc in cur.description], row)) for row in rows]
        result = f"[READ_RECORDS] Success: student_id={student_id}, records={result}"
    except Exception as e:
        result = f"[READ_RECORDS] Failure: {e}"
    save_document(result, operation="read_records")
    return result

def main(conn: sqlite3.Connection):
    return read_records(conn, student_id=10045)

print(main(sqlite3.connect("student_database.db")))