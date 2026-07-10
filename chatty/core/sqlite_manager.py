import os
import sqlite3
from typing import Optional, Dict, List
import uuid
from datetime import datetime




class SQLiteManager:
    """Simple SQLite-backed manager for storing prompts per vector DB.

    Stores entries in a single DB file with table `prompts`:
      (db_name TEXT PRIMARY KEY, system TEXT, assistant TEXT, updated_at TIMESTAMP)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_table()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    db_name TEXT PRIMARY KEY,
                    system TEXT,
                    assistant TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def set_prompts(self, db_name: str, system: Optional[str], assistant: Optional[str]) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM prompts WHERE db_name = ?", (db_name,))
            exists = cur.fetchone() is not None
            if exists:
                if system is not None and assistant is not None:
                    cur.execute(
                        "UPDATE prompts SET system = ?, assistant = ?, updated_at = CURRENT_TIMESTAMP WHERE db_name = ?",
                        (system, assistant, db_name),
                    )
                elif system is not None:
                    cur.execute(
                        "UPDATE prompts SET system = ?, updated_at = CURRENT_TIMESTAMP WHERE db_name = ?",
                        (system, db_name),
                    )
                elif assistant is not None:
                    cur.execute(
                        "UPDATE prompts SET assistant = ?, updated_at = CURRENT_TIMESTAMP WHERE db_name = ?",
                        (assistant, db_name),
                    )
                else:
                    # nothing to update
                    return
            else:
                cur.execute(
                    "INSERT INTO prompts (db_name, system, assistant) VALUES (?, ?, ?)",
                    (db_name, system or "", assistant or ""),
                )
            conn.commit()

    def get_prompts(self, db_name: str) -> Optional[Dict[str, str]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT system, assistant, updated_at FROM prompts WHERE db_name = ?", (db_name,))
            row = cur.fetchone()
            if not row:
                return None
            return {"system": row[0], "assistant": row[1], "updated_at": row[2]}

    def delete_prompts(self, db_name: str) -> bool:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM prompts WHERE db_name = ?", (db_name,))
            conn.commit()
            return cur.rowcount > 0

    def list_prompts(self) -> List[Dict[str, str]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT db_name, system, assistant, updated_at FROM prompts ORDER BY db_name")
            rows = cur.fetchall()
            return [
                {"db_name": r[0], "system": r[1], "assistant": r[2], "updated_at": r[3]} for r in rows
            ]


class ChatSessionManager:
    """Manage chat sessions and message history stored in a SQLite DB.

    Tables:
      sessions(session_id TEXT PRIMARY KEY, db_name TEXT, created_at TIMESTAMP, last_active TIMESTAMP)
      messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp TIMESTAMP)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    db_name TEXT,
                    created_at TIMESTAMP,
                    last_active TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP
                )
                """
            )
            conn.commit()

    def create_session(self, db_name: str, session_id: Optional[str] = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, db_name, created_at, last_active) VALUES (?, ?, ?, ?)",
                (session_id, db_name, now, now),
            )
            conn.commit()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            return cur.fetchone() is not None

    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute("UPDATE sessions SET last_active = ? WHERE session_id = ?", (now, session_id))
            conn.commit()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
            rows = cur.fetchall()
            return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    def list_sessions(self, db_name: Optional[str] = None) -> List[Dict[str, str]]:
        with self._conn() as conn:
            cur = conn.cursor()
            if db_name:
                cur.execute("SELECT session_id, db_name, created_at, last_active FROM sessions WHERE db_name = ?", (db_name,))
            else:
                cur.execute("SELECT session_id, db_name, created_at, last_active FROM sessions")
            rows = cur.fetchall()
            return [
                {"session_id": r[0], "db_name": r[1], "created_at": r[2], "last_active": r[3]} for r in rows
            ]


class APILogManager:
    """Log API calls to a separate SQLite DB file."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_table()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    endpoint TEXT,
                    method TEXT,
                    payload TEXT,
                    response_status INTEGER,
                    latency_ms REAL
                )
                """
            )
            conn.commit()

    def log(self, endpoint: str, method: str, payload: str, response_status: int, latency_ms: float) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO api_logs (timestamp, endpoint, method, payload, response_status, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (now, endpoint, method, payload, response_status, latency_ms),
            )
            conn.commit()

