"""Shared DB and auth helpers for student management."""

import os
import sqlite3
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "student_database.db"
SECRET_KEY = os.environ.get("STUDENT_API_SECRET", "student-mgmt-dev-secret-change-me")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            course TEXT NOT NULL,
            grade TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()


def create_demo_user():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM students WHERE id = ?", (10045,))
    if not cur.fetchone():
        pwd_hash = generate_password_hash("SecurePassword123")
        cur.execute(
            "INSERT INTO students (id, name, password_hash) VALUES (?, ?, ?)",
            (10045, "John Doe", pwd_hash),
        )
        conn.commit()
    conn.close()


def login_student(student_id: int, password: str) -> tuple[bool, str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, password_hash FROM students WHERE id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    if row and check_password_hash(row["password_hash"], password):
        return True, row["name"]
    return False, ""
