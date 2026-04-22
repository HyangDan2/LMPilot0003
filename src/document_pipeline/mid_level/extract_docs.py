from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.scanner import scan_supported_files
from src.utils.paths import stable_doc_id

from src.document_pipeline.adapters import LegacyParserAdapter
from src.document_pipeline.low_level import detect_file_type, read_file_bytes, validate_extracted_document
from src.document_pipeline.schemas import ExtractedDocument, SourceInfo


SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class ExtractionContext:
    working_folder: Path
    asset_output_dir: Path | None = None


def extract_single_doc(path: Path, context: ExtractionContext) -> ExtractedDocument:
    """Compose low-level primitives and an adapter for one document."""

    payload = read_file_bytes(path)
    detected = detect_file_type(path)
    adapter = LegacyParserAdapter()
    parsed = adapter.parse(payload.path)
    document_id = stable_doc_id(payload.path)
    source = SourceInfo(
        path=str(payload.path),
        filename=payload.path.name,
        extension=detected.extension,
        mime_type=detected.mime_type,
        size_bytes=payload.size_bytes,
        sha256=payload.sha256,
    )
    document = ExtractedDocument(
        schema_version=SCHEMA_VERSION,
        document_id=document_id,
        source=source,
        metadata=adapter.extract_metadata(payload.path, parsed),
        blocks=adapter.extract_blocks(payload.path, document_id, parsed),
        assets=adapter.extract_assets(payload.path, document_id, parsed),
        warnings=[] if detected.confidence > 0 else [f"Unknown file type: {payload.path.suffix}"],
    )
    return validate_extracted_document(document)


def extract_docs(working_folder: Path, context: ExtractionContext | None = None) -> list[ExtractedDocument]:
    """Extract every supported file in a working folder into the intermediate schema."""

    context = context or ExtractionContext(working_folder=working_folder)
    files = scan_supported_files(working_folder)
    return [extract_single_doc(path, context) for path in files]
