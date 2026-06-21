import datetime
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("code_history.db")


def init_code_history_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()


init_code_history_db()


def add_code_entry(description: str, code: str) -> int:
    ts = datetime.datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO code_history (description, code, timestamp) VALUES (?, ?, ?)",
            (description, code, ts),
        )
        conn.commit()
        return cur.lastrowid


def get_code_entry(entry_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, description, code, timestamp FROM code_history WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "description": row[1], "code": row[2], "timestamp": row[3]}


def get_code_entries() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, description, code, timestamp FROM code_history ORDER BY id DESC"
        ).fetchall()
        return [
            {"id": r[0], "description": r[1], "code": r[2], "timestamp": r[3]} for r in rows
        ]


def update_code_entry(entry_id: int, description: str = None, code: str = None) -> bool:
    fields, params = [], []
    if description is not None:
        fields.append("description = ?")
        params.append(description)
    if code is not None:
        fields.append("code = ?")
        params.append(code)
    if not fields:
        return False
    params.append(entry_id)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"UPDATE code_history SET {', '.join(fields)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()
        return cur.rowcount > 0


def delete_code_entry(entry_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM code_history WHERE id = ?", (entry_id,))
        conn.commit()
        return cur.rowcount > 0
