import sqlite3
from pathlib import Path

# Define path to the SQLite database file
DB_PATH = Path(__file__).with_name('items.db')

def init_db():
    """Initialize the database and create the items table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT
            )
        ''')
        conn.commit()

# Call initialization on import
init_db()

def add_item(name: str, description: str = None) -> int:
    """Insert a new item and return its generated id."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO items (name, description) VALUES (?, ?)",
            (name, description)
        )
        conn.commit()
        return cursor.lastrowid

def get_item(item_id: int) -> dict | None:
    """Retrieve a single item by id. Returns None if not found."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT id, name, description FROM items WHERE id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "description": row[2]}
        return None

def get_items() -> list[dict]:
    """Return all items as a list of dictionaries."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT id, name, description FROM items ORDER BY id ASC")
        rows = cursor.fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2]} for r in rows]

def update_item(item_id: int, name: str = None, description: str = None) -> bool:
    """Update fields of an existing item. Returns True if the item existed and was updated."""
    fields = []
    params = []
    if name is not None:
        fields.append("name = ?")
        params.append(name)
    if description is not None:
        fields.append("description = ?")
        params.append(description)
    if not fields:
        return False
    params.append(item_id)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            f"UPDATE items SET {', '.join(fields)} WHERE id = ?",
            tuple(params)
        )
        conn.commit()
        return cursor.rowcount > 0

def delete_item(item_id: int) -> bool:
    """Delete an item by id. Returns True if an item was deleted."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0
