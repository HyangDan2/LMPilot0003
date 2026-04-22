from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.schemas import ParsedDocument
from src.utils.paths import stable_doc_id


class ParserError(Exception):
    """Raised when a source document cannot be parsed."""


class DocumentParser(ABC):
    """Base class for file parsers."""

    file_type: str

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Parse a file into a ParsedDocument."""

    def doc_id(self, path: Path) -> str:
        return stable_doc_id(path)

    def title_from_path(self, path: Path) -> str:
        return path.stem.replace("_", " ").replace("-", " ").strip() or path.name

