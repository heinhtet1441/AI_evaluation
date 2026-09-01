import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
import pandas as pd

DB_ENGINE = os.environ.get("DB_ENGINE", "sqlite").lower()
DB_PATH = os.environ.get("IDEA_DB_PATH", "idea_evaluation.db")
DATABASE_URL = os.environ.get("DATABASE_URL", None)
FEEDBACK_FILE = "feedback_store.json"

@contextmanager
def _connect():
    if DB_ENGINE == "postgres" and DATABASE_URL:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.DictCursor)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

def init_db():
    with _connect() as conn:
        cursor = conn.cursor()
        if DB_ENGINE == "postgres":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS idea_submissions (
                    id SERIAL PRIMARY KEY,
                    source_type VARCHAR(50) NOT NULL,
                    idea_text TEXT NOT NULL,
                    heuristic_json TEXT,
                    llm_rubric_json TEXT,
                    learned_score REAL,
                    overall_score REAL,
                    submitted_at VARCHAR(100) NOT NULL
                );
                CREATE TABLE IF NOT EXISTS idea_feedback (
                    id SERIAL PRIMARY KEY,
                    idea_text TEXT NOT NULL,
                    human_score REAL NOT NULL,
                    submitted_at VARCHAR(100) NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    version_dir VARCHAR(255) NOT NULL,
                    metrics_json TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at VARCHAR(100) NOT NULL
                );
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS idea_submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    idea_text TEXT NOT NULL,
                    heuristic_json TEXT,
                    llm_rubric_json TEXT,
                    learned_score REAL,
                    overall_score REAL,
                    submitted_at TEXT NOT NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS idea_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idea_text TEXT NOT NULL,
                    human_score REAL NOT NULL,
                    submitted_at TEXT NOT NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_dir TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    is_active INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
            """)

def register_model_version(version_dir: str, metrics: dict):
    with _connect() as conn:
        cursor = conn.cursor()
        # Deactivate previous active versions
        if DB_ENGINE == "postgres":
            cursor.execute("UPDATE model_versions SET is_active = FALSE")
            cursor.execute(
                "INSERT INTO model_versions (version_dir, metrics_json, is_active, created_at) VALUES (%s, %s, TRUE, %s)",
                (version_dir, json.dumps(metrics), datetime.now().isoformat())
            )
        else:
            cursor.execute("UPDATE model_versions SET is_active = 0")
            cursor.execute(
                "INSERT INTO model_versions (version_dir, metrics_json, is_active, created_at) VALUES (?, ?, 1, ?)",
                (version_dir, json.dumps(metrics), datetime.now().isoformat())
            )

def get_all_model_versions() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query("SELECT * FROM model_versions ORDER BY id DESC", conn)

def save_submission(source_type: str, result: dict):
    with _connect() as conn:
        cursor = conn.cursor()
        query = """INSERT INTO idea_submissions
                   (source_type, idea_text, heuristic_json, llm_rubric_json,
                    learned_score, overall_score, submitted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)"""
        if DB_ENGINE == "postgres":
            query = query.replace("?", "%s")
        cursor.execute(
            query,
            (
                source_type,
                result["extracted_text_full"],
                json.dumps(result["heuristic"]),
                json.dumps(result["llm_rubric"]) if result.get("llm_rubric") else None,
                result.get("learned_score"),
                result["overall_score"],
                datetime.now().isoformat(),
            ),
        )

def get_reference_corpus(limit: int = 500) -> list[str]:
    with _connect() as conn:
        cursor = conn.cursor()
        query = f"SELECT idea_text FROM idea_submissions ORDER BY id DESC LIMIT {limit}"
        cursor.execute(query)
        rows = cursor.fetchall()
    return [r["idea_text"] for r in rows]

def get_all_submissions() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query("SELECT * FROM idea_submissions ORDER BY id DESC", conn)

def save_feedback(idea_text: str, human_score: float):
    with _connect() as conn:
        cursor = conn.cursor()
        query = "INSERT INTO idea_feedback (idea_text, human_score, submitted_at) VALUES (?, ?, ?)"
        if DB_ENGINE == "postgres":
            query = query.replace("?", "%s")
        cursor.execute(query, (idea_text, human_score, datetime.now().isoformat()))

def get_feedback_df() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query("SELECT idea_text, human_score FROM idea_feedback", conn)

def get_feedback_count() -> int:
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS n FROM idea_feedback")
        row = cursor.fetchone()
    return row["n"] if row else 0

# --- Bulk Feedback & JSON Fallback Extensions for Admin API ---

def get_total_feedback_count() -> int:
    db_count = get_feedback_count()
    json_count = 0
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                data = json.load(f)
                json_count = len(data)
        except Exception:
            json_count = 0
    return max(db_count, json_count)

def get_latest_metrics() -> dict:
    return {"status": "active", "total_records": get_total_feedback_count()}

def save_bulk_feedback(rows: list) -> int:
    # 1. Save to Active DB Engine (PostgreSQL / SQLite)
    now_str = datetime.now().isoformat()
    with _connect() as conn:
        cursor = conn.cursor()
        for r in rows:
            idea_text = r.get("idea_text", "")
            human_score = float(r.get("human_score", 0.0))
            query = "INSERT INTO idea_feedback (idea_text, human_score, submitted_at) VALUES (?, ?, ?)"
            if DB_ENGINE == "postgres":
                query = query.replace("?", "%s")
            cursor.execute(query, (idea_text, human_score, now_str))

    # 2. Sync to JSON File Fallback
    existing = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.extend(rows)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    return len(rows)
