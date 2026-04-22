from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.document_pipeline.schemas import AssetRef, ChunkSummary, DocumentMap, DocumentMetadata, EvidenceChunk
from src.document_pipeline.schemas import ExtractedBlock, ExtractedDocument, OutputPlan, Provenance, SectionSummary
from src.document_pipeline.schemas import SelectedEvidence, SourceInfo


def pipeline_output_dir(working_folder: Path) -> Path:
    """Return the automatic artifact folder for document pipeline outputs."""

    return working_folder.expanduser().resolve() / "llm_result" / "document_pipeline"


def save_extracted_documents(working_folder: Path, documents: list[ExtractedDocument]) -> Path:
    path = pipeline_output_dir(working_folder) / "extracted_documents.json"
    _write_json(path, {"documents": [document.to_dict() for document in documents]})
    return path


def save_single_document(working_folder: Path, document: ExtractedDocument) -> Path:
    path = pipeline_output_dir(working_folder) / "documents" / f"{document.document_id}.json"
    _write_json(path, document.to_dict())
    return path


def save_document_map(working_folder: Path, doc_map: DocumentMap) -> Path:
    path = pipeline_output_dir(working_folder) / "document_map.json"
    _write_json(path, doc_map.to_dict())
    return path


def save_chunks(working_folder: Path, chunks: list[EvidenceChunk]) -> Path:
    path = pipeline_output_dir(working_folder) / "chunks.json"
    _write_json(path, {"chunks": [chunk.to_dict() for chunk in chunks]})
    return path


def save_output_plan(working_folder: Path, output_plan: OutputPlan) -> Path:
    path = pipeline_output_dir(working_folder) / "output_plan.json"
    _write_json(path, output_plan.to_dict())
    return path


def save_selected_evidence(working_folder: Path, selected_evidence: SelectedEvidence) -> Path:
    path = pipeline_output_dir(working_folder) / "selected_evidence.json"
    _write_json(path, selected_evidence.to_dict())
    return path


def save_chunk_summaries(working_folder: Path, summaries: list[ChunkSummary]) -> Path:
    path = pipeline_output_dir(working_folder) / "llm_chunk_summaries.json"
    _write_json(path, {"chunk_summaries": [summary.to_dict() for summary in summaries]})
    return path


def save_section_summaries(working_folder: Path, summaries: list[SectionSummary]) -> Path:
    path = pipeline_output_dir(working_folder) / "llm_section_summaries.json"
    _write_json(path, {"section_summaries": [summary.to_dict() for summary in summaries]})
    return path


def save_report_attempts(working_folder: Path, attempts: list[dict[str, Any]]) -> Path:
    path = pipeline_output_dir(working_folder) / "llm_report_attempts.json"
    _write_json(path, {"attempts": list(attempts)})
    return path


def save_generated_markdown(working_folder: Path, markdown: str) -> Path:
    path = pipeline_output_dir(working_folder) / "generated_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def save_manifest(working_folder: Path, documents: list[ExtractedDocument]) -> Path:
    path = pipeline_output_dir(working_folder) / "extraction_manifest.json"
    payload = {
        "schema_version": "0.1",
        "document_count": len(documents),
        "documents": [
            _manifest_document_entry(document)
            for document in documents
        ],
    }
    _write_json(path, payload)
    return path


def load_extracted_documents_payload(working_folder: Path) -> dict[str, Any]:
    path = pipeline_output_dir(working_folder) / "extracted_documents.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("extracted_documents.json must contain a JSON object.")
    return payload


def load_extracted_documents(working_folder: Path) -> list[ExtractedDocument]:
    return documents_from_payload(load_extracted_documents_payload(working_folder))


def load_document_map_payload(working_folder: Path) -> dict[str, Any]:
    path = pipeline_output_dir(working_folder) / "document_map.json"
    return _read_json_object(path, "document_map.json")


def load_chunks_payload(working_folder: Path) -> dict[str, Any]:
    path = pipeline_output_dir(working_folder) / "chunks.json"
    return _read_json_object(path, "chunks.json")


def load_manifest_payload(working_folder: Path) -> dict[str, Any]:
    path = pipeline_output_dir(working_folder) / "extraction_manifest.json"
    return _read_json_object(path, "extraction_manifest.json")


def load_output_plan_payload(working_folder: Path) -> dict[str, Any]:
    path = pipeline_output_dir(working_folder) / "output_plan.json"
    return _read_json_object(path, "output_plan.json")


def load_selected_evidence_payload(working_folder: Path) -> dict[str, Any]:
    path = pipeline_output_dir(working_folder) / "selected_evidence.json"
    return _read_json_object(path, "selected_evidence.json")


def load_report_attempts_payload(working_folder: Path) -> dict[str, Any]:
    path = pipeline_output_dir(working_folder) / "llm_report_attempts.json"
    return _read_json_object(path, "llm_report_attempts.json")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _manifest_document_entry(document: ExtractedDocument) -> dict[str, Any]:
    path = Path(document.source.path)
    mtime_ns = None
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        pass
    return {
        "document_id": document.document_id,
        "path": document.source.path,
        "filename": document.source.filename,
        "extension": document.source.extension,
        "size_bytes": document.source.size_bytes,
        "mtime_ns": mtime_ns,
        "sha256": document.source.sha256,
        "block_count": len(document.blocks),
        "asset_count": len(document.assets),
        "warnings": list(document.warnings),
    }


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
