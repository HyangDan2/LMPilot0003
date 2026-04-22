from __future__ import annotations

from src.document_pipeline.schemas import ExtractedDocument


class ExtractionValidationError(ValueError):
    """Raised when an extracted document violates the intermediate contract."""


def validate_extracted_document(document: ExtractedDocument) -> ExtractedDocument:
    """Small schema guard used before mid-level outputs are persisted."""

    if not document.document_id:
        raise ExtractionValidationError("document_id is required.")
    if not document.source.path:
        raise ExtractionValidationError("source.path is required.")
    seen_block_ids: set[str] = set()
    for block in document.blocks:
        if not block.block_id:
            raise ExtractionValidationError("block_id is required.")
        if block.block_id in seen_block_ids:
            raise ExtractionValidationError(f"duplicate block_id: {block.block_id}")
        seen_block_ids.add(block.block_id)
        if block.document_id != document.document_id:
            raise ExtractionValidationError(f"block {block.block_id} has mismatched document_id.")
        if not block.provenance.source_path:
            raise ExtractionValidationError(f"block {block.block_id} is missing provenance.")
    return document
