import hashlib
import sqlite3
import getpass

# Helper functions for password hashing and verification

def hash_password(password: str) -> str:
    """Return a SHA-256 hash of the given password string."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(stored_hash: str, provided_password: str) -> bool:
    """Compare stored hash with hash of provided password."""
    return stored_hash == hash_password(provided_password)


class AuthDB:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cursor = self.conn.cursor()
        self._ensure_tables()

    def _ensure_tables(self):
        # Ensure the students table exists (id, name, password_hash)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add_student(self, student_id: int, name: str, password: str):
        pwd_hash = hash_password(password)
        self.cursor.execute(
            "INSERT OR REPLACE INTO students (id, name, password_hash) VALUES (?, ?, ?)",
            (student_id, name, pwd_hash),
        )
        self.conn.commit()

    def authenticate(self, student_id: int, password: str) -> tuple[bool, str]:
        self.cursor.execute("SELECT name, password_hash FROM students WHERE id = ?", (student_id,))
        row = self.cursor.fetchone()
        if row and verify_password(row[1], password):
            return True, row[0]  # success, return student name
        return False, ""
