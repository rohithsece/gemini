import sqlite3
from pathlib import Path
import datetime

# Database file for chat history (same folder as this script)
DB_PATH = Path(__file__).with_name('chat_history.db')

def init_history_db():
    """Create the history table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()

def add_interaction(user_msg: str, assistant_msg: str):
    """Insert a user/assistant pair with current timestamp."""
    ts = datetime.datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            'INSERT INTO history (user_message, assistant_message, timestamp) VALUES (?, ?, ?)',
            (user_msg, assistant_msg, ts)
        )
        conn.commit()

def get_history(limit: int | None = None):
    """Return a list of dicts with the stored interactions ordered by id.
    If limit is provided, returns the most recent `limit` rows.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if limit is not None:
            cursor.execute('SELECT id, user_message, assistant_message, timestamp FROM history ORDER BY id DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            # reverse to chronological order
            rows = rows[::-1]
        else:
            cursor.execute('SELECT id, user_message, assistant_message, timestamp FROM history ORDER BY id')
            rows = cursor.fetchall()
        return [
            {
                'id': r[0],
                'user_message': r[1],
                'assistant_message': r[2],
                'timestamp': r[3]
            } for r in rows
        ]
init_history_db()
