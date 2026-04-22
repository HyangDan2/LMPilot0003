from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RagSearchResult:
    chunk_id: int
    source_type: str
    source_id: str
    source_label: str
    chunk_index: int
    content: str
    score: float


class RagStore:
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
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_source
                ON rag_chunks(source_type, source_id, chunk_index)
                """
            )
            conn.commit()

    def replace_source_chunks(
        self,
        source_type: str,
        source_id: str,
        source_label: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM rag_chunks WHERE source_type = ? AND source_id = ?",
                (source_type, source_id),
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                _validate_embedding(embedding)
                conn.execute(
                    """
                    INSERT INTO rag_chunks (
                        source_type,
                        source_id,
                        source_label,
                        chunk_index,
                        content,
                        embedding_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_type,
                        source_id,
                        source_label,
                        index,
                        chunk,
                        json.dumps([float(value) for value in embedding]),
                    ),
                )
            conn.commit()

    def clear_source_type(self, source_type: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks WHERE source_type = ?", (source_type,))
            conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RagSearchResult]:
        if top_k <= 0:
            return []
        _validate_embedding(query_embedding)

        scored: list[RagSearchResult] = []
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT id, source_type, source_id, source_label, chunk_index, content, embedding_json
                FROM rag_chunks
                """
            )
            for row in cur.fetchall():
                try:
                    embedding = [float(value) for value in json.loads(row["embedding_json"])]
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                score = cosine_similarity(query_embedding, embedding)
                if score < min_score:
                    continue
                scored.append(
                    RagSearchResult(
                        chunk_id=int(row["id"]),
                        source_type=str(row["source_type"]),
                        source_id=str(row["source_id"]),
                        source_label=str(row["source_label"]),
                        chunk_index=int(row["chunk_index"]),
                        content=str(row["content"]),
                        score=score,
                    )
                )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in str(text).splitlines()).strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        if end < len(normalized):
            boundary = normalized.rfind("\n\n", start, end)
            if boundary <= start:
                boundary = normalized.rfind("\n", start, end)
            if boundary <= start:
                boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_rag_context(results: list[RagSearchResult], max_chars: int = 4000) -> str:
    if max_chars <= 0:
        return ""

    blocks: list[str] = []
    total_chars = 0
    for index, result in enumerate(results, start=1):
        block = (
            f"[{index}] {result.source_label} "
            f"(score={result.score:.3f}, chunk={result.chunk_index})\n{result.content}"
        )
        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            block = block[:remaining].rstrip()
        blocks.append(block)
        total_chars += len(block)
        if total_chars >= max_chars:
            break
    return "\n\n".join(blocks)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _validate_embedding(embedding: list[float]) -> None:
    if not embedding:
        raise ValueError("embedding must not be empty")
    for value in embedding:
        if not isinstance(value, int | float):
            raise ValueError("embedding values must be numeric")
