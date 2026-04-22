from __future__ import annotations

from typing import Any

from src.document_pipeline.schemas import (
    AssetRef,
    DocumentMetadata,
    ExtractedBlock,
    ExtractedDocument,
    Provenance,
    SourceInfo,
)


def documents_from_payload(payload: dict[str, Any]) -> list[ExtractedDocument]:
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Expected a documents list.")
    return [document_from_dict(item) for item in documents if isinstance(item, dict)]


def document_from_dict(payload: dict[str, Any]) -> ExtractedDocument:
    return ExtractedDocument(
        schema_version=str(payload.get("schema_version", "0.1")),
        document_id=str(payload.get("document_id", "")),
        source=source_from_dict(_dict(payload.get("source"))),
        metadata=metadata_from_dict(_dict(payload.get("metadata"))),
        blocks=[block_from_dict(item) for item in payload.get("blocks", []) if isinstance(item, dict)],
        assets=[asset_from_dict(item) for item in payload.get("assets", []) if isinstance(item, dict)],
        warnings=[str(item) for item in payload.get("warnings", [])],
    )


def source_from_dict(payload: dict[str, Any]) -> SourceInfo:
    return SourceInfo(
        path=str(payload.get("path", "")),
        filename=str(payload.get("filename", "")),
        extension=str(payload.get("extension", "")),
        mime_type=str(payload.get("mime_type", "")),
        size_bytes=int(payload.get("size_bytes", 0)),
        sha256=str(payload.get("sha256", "")),
    )


def metadata_from_dict(payload: dict[str, Any]) -> DocumentMetadata:
    return DocumentMetadata(
        title=str(payload.get("title", "")),
        author=str(payload.get("author", "")),
        created_at=_optional_str(payload.get("created_at")),
        modified_at=_optional_str(payload.get("modified_at")),
        page_count=_optional_int(payload.get("page_count")),
        slide_count=_optional_int(payload.get("slide_count")),
        sheet_count=_optional_int(payload.get("sheet_count")),
        extra=_dict(payload.get("extra")),
    )


def block_from_dict(payload: dict[str, Any]) -> ExtractedBlock:
    return ExtractedBlock(
        block_id=str(payload.get("block_id", "")),
        document_id=str(payload.get("document_id", "")),
        type=str(payload.get("type", "")),
        order=int(payload.get("order", 0)),
        provenance=provenance_from_dict(_dict(payload.get("provenance"))),
        role=str(payload.get("role", "")),
        text=str(payload.get("text", "")),
        normalized_text=str(payload.get("normalized_text", "")),
        level=_optional_int(payload.get("level")),
        headers=[str(item) for item in payload.get("headers", [])],
        rows=[[str(cell) for cell in row] for row in payload.get("rows", []) if isinstance(row, list)],
        markdown=str(payload.get("markdown", "")),
        asset_ids=[str(item) for item in payload.get("asset_ids", [])],
        parent_block_id=_optional_str(payload.get("parent_block_id")),
        child_block_ids=[str(item) for item in payload.get("child_block_ids", [])],
        metadata=_dict(payload.get("metadata")),
    )


def asset_from_dict(payload: dict[str, Any]) -> AssetRef:
    provenance_payload = payload.get("provenance")
    return AssetRef(
        asset_id=str(payload.get("asset_id", "")),
        document_id=str(payload.get("document_id", "")),
        type=str(payload.get("type", "")),
        source_path=str(payload.get("source_path", "")),
        stored_path=str(payload.get("stored_path", "")),
        mime_type=str(payload.get("mime_type", "")),
        sha256=str(payload.get("sha256", "")),
        width=_optional_int(payload.get("width")),
        height=_optional_int(payload.get("height")),
        caption=str(payload.get("caption", "")),
        provenance=provenance_from_dict(provenance_payload) if isinstance(provenance_payload, dict) else None,
        metadata=_dict(payload.get("metadata")),
    )


def provenance_from_dict(payload: dict[str, Any]) -> Provenance:
    bbox = payload.get("bbox")
    return Provenance(
        source_path=str(payload.get("source_path", "")),
        location_type=str(payload.get("location_type", "file")),
        page=_optional_int(payload.get("page")),
        slide=_optional_int(payload.get("slide")),
        sheet=_optional_str(payload.get("sheet")),
        cell_range=_optional_str(payload.get("cell_range")),
        bbox=tuple(float(item) for item in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
        section_path=[str(item) for item in payload.get("section_path", [])],
        char_start=_optional_int(payload.get("char_start")),
        char_end=_optional_int(payload.get("char_end")),
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
