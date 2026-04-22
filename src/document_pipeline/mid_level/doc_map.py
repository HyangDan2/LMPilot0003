from __future__ import annotations

from src.document_pipeline.schemas import DocumentMap, ExtractedDocument


def build_doc_map(documents: list[ExtractedDocument]) -> DocumentMap:
    """Build a navigation map without summarizing or rewriting evidence."""

    document_entries = [
        {
            "document_id": document.document_id,
            "title": document.metadata.title,
            "path": document.source.path,
            "block_count": len(document.blocks),
            "asset_count": len(document.assets),
        }
        for document in documents
    ]
    block_entries = [
        {
            "document_id": document.document_id,
            "block_id": block.block_id,
            "type": block.type,
            "role": block.role,
            "order": block.order,
            "text_preview": block.normalized_text[:240],
            "provenance": block.provenance.to_dict(),
        }
        for document in documents
        for block in document.blocks
    ]
    return DocumentMap(documents=document_entries, blocks=block_entries)
