import sqlite3
import time
from pathlib import Path

DB_PATH = Path("chat_memory.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            )
        ''')
        # Dynamic schema migration: add metadata column if not present
        cursor = conn.execute("PRAGMA table_info(messages)")
        cols = [c[1] for c in cursor.fetchall()]
        if "metadata" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
        conn.commit()

def save_message(session_id: str, role: str, content: str, ts: float = None, metadata: str = None):
    if ts is None:
        ts = time.time()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            'INSERT INTO messages (session_id, role, content, ts, metadata) VALUES (?, ?, ?, ?, ?)',
            (session_id, role, content, ts, metadata)
        )
        conn.commit()

def get_messages(session_id: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            'SELECT role, content, ts, metadata FROM messages WHERE session_id = ? ORDER BY ts ASC',
            (session_id,)
        )
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1], "ts": r[2], "metadata": r[3]} for r in rows]

def clear_messages(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        conn.commit()

# Initialize DB on import
init_db()
