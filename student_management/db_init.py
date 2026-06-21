import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / 'student_database.db'

def get_connection():
    """Return a SQLite connection to the student database, creating the file if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the required tables (students, records) if they do not exist.
    Also, insert a sample student for demo purposes.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Students table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    # Records table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            course TEXT NOT NULL,
            grade TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    # Insert a demo student if not already present
    cur.execute("SELECT 1 FROM students WHERE id = ?", (10045,))
    if not cur.fetchone():
        # Use the AuthDB helper to add student with hashed password
        from .auth import AuthDB
        auth = AuthDB(conn)
        auth.add_student(10045, 'John Doe', 'SecurePassword123')
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
