from __future__ import annotations

import re
import unicodedata


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Normalize extracted text without changing its factual content."""

    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()
