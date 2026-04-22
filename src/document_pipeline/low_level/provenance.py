from __future__ import annotations

from pathlib import Path

from src.document_pipeline.schemas import Provenance


def file_provenance(path: Path, section_path: list[str] | None = None) -> Provenance:
    """Create baseline file-level provenance for a block."""

    return Provenance(
        source_path=str(path),
        location_type="file",
        section_path=list(section_path or []),
    )
