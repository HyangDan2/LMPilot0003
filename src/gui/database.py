from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ChatRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_message_id INTEGER,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id_id ON messages(session_id, id)"
            )
            conn.commit()

    def create_session(self, title: str = "New Chat") -> int:
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
            conn.commit()
            return int(cur.lastrowid)

    def get_session_title(self, session_id: int) -> str:
        with self._connect() as conn:
            cur = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
            return str(row["title"]) if row is not None else ""

    def update_session_title(self, session_id: int, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id),
            )
            conn.commit()

    def add_message(self, session_id: int, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            conn.commit()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC, id DESC"
            )
            return [dict(row) for row in cur.fetchall()]

    def get_messages(self, session_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_recent_messages(self, session_id: int, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT role, content, created_at
                FROM (
                    SELECT id, role, content, created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id
                """,
                (session_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def count_messages(self, session_id: int) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) AS message_count FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            return int(row["message_count"]) if row is not None else 0

    def get_session_summary(self, session_id: int) -> str:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT content FROM session_summaries WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            return str(row["content"]) if row is not None else ""

    def upsert_session_summary(
        self,
        session_id: int,
        content: str,
        source_message_id: int | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_summaries (session_id, content, source_message_id, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    content = excluded.content,
                    source_message_id = excluded.source_message_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, content, source_message_id),
            )
            conn.commit()

    def delete_session(self, session_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
