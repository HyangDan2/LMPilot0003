from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilePayload:
    path: Path
    data: bytes
    size_bytes: int
    sha256: str


def compute_content_hash(data: bytes | str) -> str:
    """Return a stable SHA-256 hash for bytes or normalized text."""

    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_file_bytes(path: Path) -> FilePayload:
    """Load one source file and attach basic identity facts."""

    resolved = path.expanduser().resolve()
    data = resolved.read_bytes()
    return FilePayload(
        path=resolved,
        data=data,
        size_bytes=len(data),
        sha256=compute_content_hash(data),
    )
