from __future__ import annotations

from src.document_pipeline.schemas import DocumentMap, EvidenceChunk, ExtractedDocument


def generate_markdown_report(
    documents: list[ExtractedDocument],
    doc_map: DocumentMap | None = None,
    chunks: list[EvidenceChunk] | None = None,
    title: str = "Generated Document Report",
    max_blocks: int = 40,
    max_chunks: int = 20,
    max_preview_chars: int = 500,
) -> str:
    """Create a deterministic markdown evidence report from extracted artifacts."""

    lines = [
        f"# {title}",
        "",
        f"Generated from {len(documents)} extracted document(s).",
        "",
        "## Source Documents",
        "",
        "| Document | Type | Blocks | Assets |",
        "|---|---:|---:|---:|",
    ]
    if documents:
        for document in documents:
            lines.append(
                "| "
                f"{_escape_table(document.source.filename)} | "
                f"{_escape_table(document.source.extension)} | "
                f"{len(document.blocks)} | "
                f"{len(document.assets)} |"
            )
    else:
        lines.append("| none |  | 0 | 0 |")

    lines.extend(["", "## Document Structure", ""])
    map_blocks = doc_map.blocks if doc_map is not None else []
    if map_blocks:
        for block in map_blocks[:max_blocks]:
            preview = _single_line(str(block.get("text_preview", "")), max_preview_chars)
            lines.append(
                f"- `{block.get('document_id', '')}` / `{block.get('block_id', '')}` "
                f"({block.get('role') or block.get('type', 'block')}): {preview}"
            )
        if len(map_blocks) > max_blocks:
            lines.append(f"- ... {len(map_blocks) - max_blocks} more block(s) omitted.")
    else:
        for document in documents:
            lines.append(f"- {document.source.filename}: {len(document.blocks)} block(s)")

    lines.extend(["", "## Evidence Chunks", ""])
    chunks = chunks or []
    if chunks:
        for index, chunk in enumerate(chunks[:max_chunks], start=1):
            lines.extend(
                [
                    f"### Chunk {index}",
                    "",
                    f"- chunk_id: `{chunk.chunk_id}`",
                    f"- document_id: `{chunk.document_id}`",
                    f"- source_blocks: {', '.join(f'`{block_id}`' for block_id in chunk.block_ids)}",
                    "",
                    _quote_block(_single_line(chunk.text, max_preview_chars)),
                    "",
                ]
            )
        if len(chunks) > max_chunks:
            lines.append(f"_ {len(chunks) - max_chunks} more chunk(s) omitted._")
    else:
        lines.append("_No chunks are available yet. Run `/chunk_sections` for retrieval-ready chunks._")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `llm_result/document_pipeline/extracted_documents.json`",
            "- `llm_result/document_pipeline/extraction_manifest.json`",
            "- `llm_result/document_pipeline/document_map.json`",
            "- `llm_result/document_pipeline/chunks.json`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _single_line(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _quote_block(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|")
