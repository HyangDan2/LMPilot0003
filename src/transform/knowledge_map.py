from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.models.schemas import ParsedDocument


@dataclass(frozen=True)
class KnowledgeMap:
    """Readable and structured planning context derived from parsed documents."""

    documents: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"documents": self.documents}


def build_knowledge_map(documents: list[ParsedDocument]) -> KnowledgeMap:
    """Build a compact structured map from normalized documents."""

    mapped_documents: list[dict[str, Any]] = []
    for document in documents:
        mapped_documents.append(
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "file_path": document.file_path,
                "file_type": document.file_type,
                "metadata": document.metadata,
                "section_count": len(document.sections),
                "asset_count": len(document.assets),
                "sections": [
                    {
                        "section_id": section.section_id,
                        "title": section.title,
                        "level": section.level,
                        "page_or_slide": section.page_or_slide,
                        "preview": preview_text(section.text),
                        "metadata": section.metadata,
                    }
                    for section in document.sections
                ],
                "assets": [asset.to_dict() for asset in document.assets],
            }
        )
    return KnowledgeMap(mapped_documents)


def render_knowledge_map_markdown(knowledge_map: KnowledgeMap) -> str:
    """Render a human-readable markdown knowledge map."""

    lines = ["# Knowledge Map", ""]
    for document in knowledge_map.documents:
        lines.extend(
            [
                f"## {document['title']}",
                "",
                f"- Document ID: `{document['doc_id']}`",
                f"- File path: `{document['file_path']}`",
                f"- File type: `{document['file_type']}`",
                f"- Section count: {document['section_count']}",
                f"- Asset count: {document['asset_count']}",
            ]
        )
        metadata = document.get("metadata") or {}
        if metadata:
            lines.append(f"- Metadata: `{asdict_safe(metadata)}`")
        lines.extend(["", "### Section Previews", ""])
        for section in document.get("sections", []):
            ref = section["section_id"]
            location = section.get("page_or_slide")
            suffix = f" (page/slide: {location})" if location is not None else ""
            lines.extend(
                [
                    f"- `{ref}` {section['title']}{suffix}",
                    f"  {section.get('preview') or '(no extractable text)'}",
                ]
            )
        assets = document.get("assets", [])
        if assets:
            lines.extend(["", "### Asset Summaries", ""])
            for asset in assets:
                lines.append(
                    f"- `{asset.get('asset_id')}` {asset.get('kind')} "
                    f"on {asset.get('page_or_slide')}: {asset.get('caption') or '(no caption)'}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def preview_text(text: str, max_chars: int = 500) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def asdict_safe(value: Any) -> str:
    return str(value).replace("\n", " ")[:500]

