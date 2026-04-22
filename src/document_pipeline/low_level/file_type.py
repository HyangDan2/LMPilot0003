from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pptx", ".docx", ".xlsx", ".pdf"}


@dataclass(frozen=True)
class DetectedFileType:
    extension: str
    mime_type: str
    family: str
    confidence: float


def detect_file_type(path: Path) -> DetectedFileType:
    """Detect the document family used to select an extraction adapter."""

    extension = path.suffix.lower()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if extension in {".pptx", ".docx", ".xlsx"}:
        family = extension.lstrip(".")
        confidence = 0.95
    elif extension == ".pdf":
        family = "pdf"
        confidence = 0.90
    else:
        family = "unknown"
        confidence = 0.0
    return DetectedFileType(
        extension=extension,
        mime_type=mime_type,
        family=family,
        confidence=confidence,
    )
