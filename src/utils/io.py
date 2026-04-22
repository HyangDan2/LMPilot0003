from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the resolved path."""

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def save_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with stable formatting."""

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent folders first."""

    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

