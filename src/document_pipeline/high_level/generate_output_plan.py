from __future__ import annotations

from src.document_pipeline.schemas import DocumentMap, ExtractedDocument, OutputPlan, SelectedEvidence
from .write_output_plan import SUMMARY_SUBSECTIONS

DESCRIBES_HEADING = "What the Document Explicitly Describes"
METHODS_HEADING = "Main Methods or Components Explicitly Mentioned"
QUANTITATIVE_HEADING = "Quantitative Values Explicitly Present"
LIMITATIONS_HEADING = "Explicit Limitations or Constraints"
UNSPECIFIED_HEADING = "Unclear or Not Specified in Selected Evidence"


def generate_output_plan(
    output_plan: OutputPlan,
    documents: list[ExtractedDocument],
    doc_map: DocumentMap | None = None,
    selected_evidence: SelectedEvidence | None = None,
) -> str:
    """Write a deterministic markdown fallback from output plan and selected blocks."""

    lines = [f"# {output_plan.title}", ""]
    for section in output_plan.sections:
        if section.section_id == "summary":
            lines.extend(_summary_section(section.title, selected_evidence))
        elif section.section_id == "source_documents":
            lines.extend(_source_documents_section(documents, section.title))
        elif section.section_id == "open_issues":
            lines.extend(_open_issues_section(section.title, documents, selected_evidence))
        else:
            lines.extend(_summary_section(section.title, selected_evidence))
    return "\n".join(lines).rstrip() + "\n"


def _summary_section(title: str, selected_evidence: SelectedEvidence | None) -> list[str]:
    lines = [f"## {title}", ""]
    blocks = selected_evidence.blocks if selected_evidence is not None else []
    blocks_by_heading = _classify_summary_blocks(blocks)
    if not blocks:
        blocks_by_heading[UNSPECIFIED_HEADING] = [
            "No selected evidence blocks are available, so no document facts can be stated yet."
        ]
    for heading in SUMMARY_SUBSECTIONS:
        lines.extend([f"### {heading}", ""])
        entries = blocks_by_heading.get(heading, [])
        if entries:
            lines.extend(f"- {entry}" for entry in entries[:4])
        else:
            lines.append("- Not explicitly stated in the selected evidence.")
        lines.append("")
    return lines


def _classify_summary_blocks(blocks) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {heading: [] for heading in SUMMARY_SUBSECTIONS}
    for block in blocks[:8]:
        source = f"{block.source_filename} / {block.block_id}".strip(" /")
        text = _single_line(block.text, 320)
        heading = _summary_heading_for_text(text)
        grouped[heading].append(f"{text} (`{source}`)")
    if len(blocks) > 8:
        grouped[UNSPECIFIED_HEADING].append(f"{len(blocks) - 8} additional evidence block(s) were omitted from this fallback summary.")
    return grouped


def _summary_heading_for_text(text: str) -> str:
    lowered = text.lower()
    if any(char.isdigit() for char in text):
        return QUANTITATIVE_HEADING
    if any(term in lowered for term in ("limit", "constraint", "risk", "issue", "failure", "uncertain", "리스크", "문제", "한계", "제약")):
        return LIMITATIONS_HEADING
    if any(
        term in lowered
        for term in (
            "method",
            "component",
            "architecture",
            "algorithm",
            "system",
            "workflow",
            "process",
            "design",
            "implementation",
            "모듈",
            "방법",
            "시스템",
            "설계",
        )
    ):
        return METHODS_HEADING
    return DESCRIBES_HEADING


def _source_documents_section(documents: list[ExtractedDocument], title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Source | Type | Blocks | Assets |",
        "|---|---:|---:|---:|",
    ]
    if not documents:
        lines.append("| none |  | 0 | 0 |")
    for document in documents:
        lines.append(
            "| "
            f"{_escape_table(document.source.filename)} | "
            f"{_escape_table(document.source.extension)} | "
            f"{len(document.blocks)} | "
            f"{len(document.assets)} |"
        )
    lines.append("")
    return lines


def _open_issues_section(
    title: str,
    documents: list[ExtractedDocument],
    selected_evidence: SelectedEvidence | None,
) -> list[str]:
    lines = [f"## {title}", ""]
    gaps = []
    if not documents:
        gaps.append("No supported source documents were extracted from the attached folder.")
    if documents and (selected_evidence is None or not selected_evidence.blocks):
        gaps.append("Documents were extracted, but no evidence blocks were selected.")
    for document in documents:
        if document.warnings:
            gaps.extend(f"{document.source.filename}: {warning}" for warning in document.warnings)
    if not gaps:
        gaps.append("Review cited evidence before using this engineering report externally.")
        gaps.append("Treat any topic not explicitly stated in the selected evidence as unspecified.")
    lines.extend(f"- {gap}" for gap in gaps)
    lines.append("")
    return lines


def _single_line(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|")
