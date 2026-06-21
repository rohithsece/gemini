import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).parent / "code_reviews.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the review history table if not exists."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                code TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                overall_score INTEGER NOT NULL,
                bug_report TEXT NOT NULL,
                quality_report TEXT NOT NULL,
                security_report TEXT NOT NULL,
                overall_summary TEXT NOT NULL
            )
        """)
        conn.commit()

def save_review(
    filename: str,
    code: str,
    overall_score: int,
    bug_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    security_report: Dict[str, Any],
    overall_summary: str
) -> int:
    """Saves a review execution into the SQLite DB and returns its primary key ID."""
    timestamp = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reviews (
                filename, code, timestamp, overall_score,
                bug_report, quality_report, security_report, overall_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                code,
                timestamp,
                overall_score,
                json.dumps(bug_report),
                json.dumps(quality_report),
                json.dumps(security_report),
                overall_summary
            )
        )
        conn.commit()
        return cursor.lastrowid

def get_all_reviews() -> List[Dict[str, Any]]:
    """Retrieves all reviews, sorted by timestamp descending, without the full source code (for list display)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, timestamp, overall_score, overall_summary FROM reviews ORDER BY timestamp DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_review_by_id(review_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single detailed review by its ID, parsing internal JSON fields."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        data = dict(row)
        # Parse JSON columns back to dicts
        try:
            data["bug_report"] = json.loads(data["bug_report"])
        except Exception:
            data["bug_report"] = {}
            
        try:
            data["quality_report"] = json.loads(data["quality_report"])
        except Exception:
            data["quality_report"] = {}
            
        try:
            data["security_report"] = json.loads(data["security_report"])
        except Exception:
            data["security_report"] = {}
            
        return data

def delete_review(review_id: int) -> bool:
    """Deletes a review record from the SQLite database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
        conn.commit()
        return cursor.rowcount > 0
