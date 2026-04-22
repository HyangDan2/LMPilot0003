from __future__ import annotations

import hashlib
import re
from pathlib import Path


def stable_doc_id(path: Path) -> str:
    """Create a stable compact id from a file path."""

    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{slugify(path.stem)}-{digest}"


def slugify(value: str, fallback: str = "untitled") -> str:
    """Return a filesystem-friendly ASCII slug."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:80] or fallback


def relative_or_absolute(path: Path, base: Path) -> str:
    """Return a readable path relative to base when possible."""

    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())

