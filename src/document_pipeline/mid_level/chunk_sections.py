from __future__ import annotations

from src.document_pipeline.low_level import compute_content_hash
from src.document_pipeline.schemas import EvidenceChunk, ExtractedDocument


def chunk_sections(documents: list[ExtractedDocument], max_chars: int = 2400) -> list[EvidenceChunk]:
    """Create simple retrieval chunks while preserving block provenance."""

    chunks: list[EvidenceChunk] = []
    for document in documents:
        current_block_ids: list[str] = []
        current_texts: list[str] = []
        current_refs = []
        current_len = 0
        for block in document.blocks:
            text = block.normalized_text or block.text
            if not text:
                continue
            if current_texts and current_len + len(text) > max_chars:
                chunks.append(_make_chunk(document.document_id, current_block_ids, current_texts, current_refs))
                current_block_ids = []
                current_texts = []
                current_refs = []
                current_len = 0
            current_block_ids.append(block.block_id)
            current_texts.append(text)
            current_refs.append(block.provenance)
            current_len += len(text)
        if current_texts:
            chunks.append(_make_chunk(document.document_id, current_block_ids, current_texts, current_refs))
    return chunks


def _make_chunk(document_id, block_ids, texts, provenance_refs) -> EvidenceChunk:
    text = "\n\n".join(texts)
    chunk_hash = compute_content_hash(f"{document_id}:{','.join(block_ids)}:{text}")[:16]
    return EvidenceChunk(
        chunk_id=f"chunk_{chunk_hash}",
        document_id=document_id,
        block_ids=list(block_ids),
        text=text,
        token_estimate=max(1, len(text) // 4),
        provenance_refs=list(provenance_refs),
    )
