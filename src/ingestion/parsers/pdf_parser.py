from __future__ import annotations

from pathlib import Path

from src.ingestion.parsers.base import DocumentParser, ParserError
from src.models.schemas import ParsedDocument, Section


class PdfParser(DocumentParser):
    """MVP PDF parser using pypdf text extraction, with one section per page."""

    file_type = "pdf"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except Exception as exc:
            raise ParserError("PDF parsing requires pypdf.") from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise ParserError(f"Failed to open PDF {path}: {exc}") from exc

        doc_id = self.doc_id(path)
        sections: list[Section] = []
        pages_text: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages_text.append(text)
            sections.append(
                Section(
                    section_id=f"{doc_id}-page-{index}",
                    title=f"Page {index}",
                    level=1,
                    text=text,
                    page_or_slide=index,
                    metadata={"parser": "pdf"},
                )
            )

        return ParsedDocument(
            doc_id=doc_id,
            file_path=str(path.resolve()),
            file_type=self.file_type,
            title=self.title_from_path(path),
            text="\n\n".join(pages_text).strip(),
            sections=sections,
            assets=[],
            metadata={"page_count": len(reader.pages)},
        )

