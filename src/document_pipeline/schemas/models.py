from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Provenance:
    """Source location for one extracted evidence unit."""

    source_path: str
    location_type: str = "file"
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    section_path: list[str] = field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceInfo:
    """Stable identity and file facts for a source document."""

    path: str
    filename: str
    extension: str
    mime_type: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentMetadata:
    """Common metadata with optional format-specific details."""

    title: str = ""
    author: str = ""
    created_at: str | None = None
    modified_at: str | None = None
    page_count: int | None = None
    slide_count: int | None = None
    sheet_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssetRef:
    """Reusable non-text asset discovered in a document."""

    asset_id: str
    document_id: str
    type: str
    source_path: str
    stored_path: str = ""
    mime_type: str = ""
    sha256: str = ""
    width: int | None = None
    height: int | None = None
    caption: str = ""
    provenance: Provenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict() if self.provenance else None
        return payload


@dataclass(frozen=True)
class ExtractedBlock:
    """Canonical text/table/image/navigation block used by downstream layers."""

    block_id: str
    document_id: str
    type: str
    order: int
    provenance: Provenance
    role: str = ""
    text: str = ""
    normalized_text: str = ""
    level: int | None = None
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    markdown: str = ""
    asset_ids: list[str] = field(default_factory=list)
    parent_block_id: str | None = None
    child_block_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class ExtractedDocument:
    """JSON-ready evidence package for one source document."""

    schema_version: str
    document_id: str
    source: SourceInfo
    metadata: DocumentMetadata
    blocks: list[ExtractedBlock] = field(default_factory=list)
    assets: list[AssetRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source": self.source.to_dict(),
            "metadata": self.metadata.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "assets": [asset.to_dict() for asset in self.assets],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EvidenceChunk:
    """Retrieval-ready chunk that keeps block-level provenance."""

    chunk_id: str
    document_id: str
    block_ids: list[str]
    text: str
    token_estimate: int
    provenance_refs: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance_refs"] = [ref.to_dict() for ref in self.provenance_refs]
        return payload


@dataclass(frozen=True)
class DocumentMap:
    """Lightweight navigation map for extracted documents."""

    documents: list[dict[str, Any]]
    blocks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"documents": list(self.documents), "blocks": list(self.blocks)}


@dataclass(frozen=True)
class OutputPlanSection:
    """One planned section for a generated report."""

    section_id: str
    title: str
    purpose: str
    source_chunk_ids: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    max_chars: int = 1200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutputPlan:
    """Grounded plan used before writing a final report."""

    schema_version: str
    title: str
    goal: str
    sections: list[OutputPlanSection]
    source_document_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "goal": self.goal,
            "sections": [section.to_dict() for section in self.sections],
            "source_document_ids": list(self.source_document_ids),
        }


@dataclass(frozen=True)
class SelectedEvidenceBlock:
    """One extracted block selected as grounding evidence for a report."""

    document_id: str
    block_id: str
    source_filename: str
    role: str
    text: str
    provenance: Provenance
    score: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class SelectedEvidence:
    """Prompt-sized evidence package selected from extracted blocks."""

    query: str
    max_input_chars: int
    blocks: list[SelectedEvidenceBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "max_input_chars": self.max_input_chars,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class ChunkSummary:
    """LLM or fallback summary for one context-safe chunk batch."""

    summary_id: str
    chunk_ids: list[str]
    summary: str
    key_points: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    fallback: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SectionSummary:
    """Reduced summary aligned to one output-plan section."""

    section_id: str
    title: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    chunk_summary_ids: list[str] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMReportResult:
    """Final report plus inspectable orchestration artifacts."""

    markdown: str
    chunk_summaries: list[ChunkSummary] = field(default_factory=list)
    section_summaries: list[SectionSummary] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    used_llm: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "chunk_summaries": [summary.to_dict() for summary in self.chunk_summaries],
            "section_summaries": [summary.to_dict() for summary in self.section_summaries],
            "attempts": list(self.attempts),
            "used_llm": self.used_llm,
            "fallback_reason": self.fallback_reason,
        }
