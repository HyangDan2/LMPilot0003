from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

from src.document_pipeline.schemas import ExtractedBlock, ExtractedDocument
from .markdown_format import sentence_per_line_markdown


PromptMessages = list[dict[str, str]]
SummaryModelCaller = Callable[[PromptMessages, int], str]

SUMMARY_SYSTEM_PROMPT = (
    "You are a careful document summarization assistant. "
    "Write only the final summary for end users. "
    "Use plain natural language grounded only in the provided material. "
    "Do not explain your task. "
    "Do not mention prompts, chunks, token limits, budgets, or instructions. "
    "Do not expose reasoning, planning, or analysis."
)


@dataclass(frozen=True)
class SummaryBudget:
    per_doc_input_chars: int = 12000
    per_doc_output_tokens: int = 480
    consolidate_input_chars: int = 6000
    consolidate_output_tokens: int = 720
    workspace_input_chars: int = 12000
    workspace_output_tokens: int = 3000
    block_excerpt_chars: int = 900

    def estimate_tokens(self, text: str) -> int:
        return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class DocumentSummaryArtifact:
    doc_id: str
    source_path: str
    block_count: int
    chunk_count: int
    estimated_input_tokens: int
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "block_count": self.block_count,
            "chunk_count": self.chunk_count,
            "estimated_input_tokens": self.estimated_input_tokens,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class WorkspaceSummarySections:
    overall_summary: str
    features: list[str]
    next_action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_summary": self.overall_summary,
            "features": list(self.features),
            "next_action": self.next_action,
        }

    def to_text(self) -> str:
        lines = [
            "Overall Summary:",
            self.overall_summary.strip() or "(missing)",
            "",
            "Features:",
        ]
        lines.extend(f"{index}. {feature}" for index, feature in enumerate(self.features, start=1))
        if not self.features:
            lines.append("(missing)")
        lines.extend(
            [
                "",
                "Next Action:",
                self.next_action.strip() or "(missing)",
            ]
        )
        return "\n".join(lines)


def summarize_documents_hierarchically(
    documents: list[ExtractedDocument],
    call_model: SummaryModelCaller,
    budget: SummaryBudget | None = None,
    progress: Callable[[str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[DocumentSummaryArtifact], WorkspaceSummarySections]:
    active_budget = budget or SummaryBudget()
    ordered_documents = sorted(documents, key=lambda document: (document.source.path, document.document_id))
    document_summaries: list[DocumentSummaryArtifact] = []

    for index, document in enumerate(ordered_documents, start=1):
        _raise_if_cancelled(cancel_requested)
        if progress is not None:
            progress(f"Summarizing document {index}/{len(ordered_documents)}: {document.source.filename}")
        document_summaries.append(
            summarize_document(document, call_model=call_model, budget=active_budget, cancel_requested=cancel_requested)
        )

    _raise_if_cancelled(cancel_requested)
    if progress is not None:
        progress("Combining document summaries into a workspace summary.")
    workspace_summary = summarize_workspace(
        document_summaries,
        call_model=call_model,
        budget=active_budget,
        cancel_requested=cancel_requested,
    )
    return document_summaries, workspace_summary


def summarize_document(
    document: ExtractedDocument,
    call_model: SummaryModelCaller,
    budget: SummaryBudget,
    cancel_requested: Callable[[], bool] | None = None,
) -> DocumentSummaryArtifact:
    chunks = _group_block_texts(document, budget)
    chunk_summaries: list[str] = []

    for chunk in chunks:
        _raise_if_cancelled(cancel_requested)
        chunk_summaries.append(
            call_model(
                _document_chunk_prompt(document=document, excerpt_group=chunk),
                budget.per_doc_output_tokens,
            ).strip()
        )

    if len(chunk_summaries) == 1:
        summary = chunk_summaries[0]
    else:
        _raise_if_cancelled(cancel_requested)
        summary = call_model(
            _document_consolidation_prompt(document, chunk_summaries, budget),
            budget.consolidate_output_tokens,
        ).strip()

    estimated_input_tokens = sum(budget.estimate_tokens(chunk) for chunk in chunks)
    return DocumentSummaryArtifact(
        doc_id=document.document_id,
        source_path=document.source.path,
        block_count=len(document.blocks),
        chunk_count=len(chunks),
        estimated_input_tokens=estimated_input_tokens,
        summary=summary,
    )


def summarize_workspace(
    document_summaries: list[DocumentSummaryArtifact],
    call_model: SummaryModelCaller,
    budget: SummaryBudget,
    cancel_requested: Callable[[], bool] | None = None,
) -> WorkspaceSummarySections:
    if not document_summaries:
        return WorkspaceSummarySections(
            overall_summary="No extracted documents are available.",
            features=[],
            next_action="Run extraction before requesting a summary.",
        )

    ordered_entries = sorted(document_summaries, key=lambda item: (item.source_path, item.doc_id))
    groups = _group_summary_entries(ordered_entries, budget.workspace_input_chars)
    group_summaries: list[WorkspaceSummarySections] = []

    for group in groups:
        _raise_if_cancelled(cancel_requested)
        group_summaries.append(
            _parse_workspace_summary_sections(
                call_model(
                    _workspace_group_prompt(group),
                    budget.workspace_output_tokens,
                ).strip()
            )
        )

    if len(group_summaries) == 1:
        return group_summaries[0]

    _raise_if_cancelled(cancel_requested)
    return _parse_workspace_summary_sections(
        call_model(
            _workspace_final_prompt(group_summaries, budget),
            budget.workspace_output_tokens,
        ).strip()
    )


def render_workspace_summary_markdown(
    document_summaries: list[DocumentSummaryArtifact],
    workspace_summary: WorkspaceSummarySections,
) -> str:
    lines = [
        "# Workspace Summary",
        "",
        "## Overall Summary",
        "",
        workspace_summary.overall_summary.strip(),
        "",
        "## Features",
        "",
    ]
    rendered_features = workspace_summary.features or ["No distinct features were identified."]
    for index, feature in enumerate(rendered_features, start=1):
        lines.extend([f"{index}. {feature.strip()}", ""])
    lines.extend(
        [
            "## Next Action",
            "",
            workspace_summary.next_action.strip(),
        ]
    )
    return sentence_per_line_markdown("\n".join(lines).rstrip() + "\n")


def _group_block_texts(document: ExtractedDocument, budget: SummaryBudget) -> list[str]:
    pieces: list[str] = []
    for block in sorted(document.blocks, key=lambda item: item.order):
        text = _block_excerpt(block, budget.block_excerpt_chars)
        if not text:
            continue
        pieces.append(_format_block(block, text))
    if not pieces:
        pieces = ["No extractable text blocks were found in this document."]
    return _pack_lines(pieces, budget.per_doc_input_chars)


def _group_summary_entries(entries: list[DocumentSummaryArtifact], max_chars: int) -> list[str]:
    lines = [
        "\n".join(
            [
                f"Document name: {Path(entry.source_path).name}",
                f"Document path: {entry.source_path}",
                f"Block count: {entry.block_count}",
                "Summary:",
                entry.summary.strip(),
            ]
        )
        for entry in entries
    ]
    return _pack_lines(lines, max_chars)


def _pack_lines(lines: list[str], max_chars: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_size = 0

    for line in lines:
        size = len(line) + 2
        if current and current_size + size > max_chars:
            groups.append("\n\n".join(current))
            current = [line]
            current_size = size
            continue
        current.append(line)
        current_size += size

    if current:
        groups.append("\n\n".join(current))
    return groups or [""]


def _format_block(block: ExtractedBlock, text: str) -> str:
    location = _block_location(block)
    block_type = block.type or "block"
    role = block.role or "content"
    return (
        f"Block {block.order} ({block_type}, {role}, {location})\n"
        f"{text}"
    )


def _block_excerpt(block: ExtractedBlock, max_chars: int) -> str:
    text = (block.normalized_text or block.text or "").strip()
    if not text and block.markdown.strip():
        text = block.markdown.strip()
    if not text and block.rows:
        row_preview = [" | ".join(cell.strip() for cell in row if cell.strip()) for row in block.rows]
        text = "\n".join(row for row in row_preview if row)
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[: max_chars - 3].rstrip() + "..." if len(compact) > max_chars else compact


def _block_location(block: ExtractedBlock) -> str:
    provenance = block.provenance
    if provenance.slide is not None:
        return f"slide {provenance.slide}"
    if provenance.page is not None:
        return f"page {provenance.page}"
    if provenance.sheet:
        return f"sheet {provenance.sheet}"
    return provenance.location_type


def _document_chunk_prompt(
    document: ExtractedDocument,
    excerpt_group: str,
) -> PromptMessages:
    title = document.metadata.title or document.source.filename
    return [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f'Please read the following excerpts from the document "{title}". '
                "Write a detailed factual summary in natural language. "
                "Focus on the main topics, notable claims, capabilities, limitations, and uncertainties that are directly supported by the text.\n\n"
                f"Document path: {document.source.path}\n\n"
                "Document excerpts:\n"
                f"{excerpt_group}"
            ),
        },
    ]


def _document_consolidation_prompt(
    document: ExtractedDocument,
    chunk_summaries: list[str],
    budget: SummaryBudget,
) -> PromptMessages:
    joined = "\n\n".join(summary.strip() for summary in chunk_summaries)
    title = document.metadata.title or document.source.filename
    return [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f'The notes below each summarize different excerpts from the document "{title}". '
                "Please merge them into one coherent summary that reads naturally and avoids repetition. "
                "Keep the result focused on the content of the document itself, and preserve specific important details when they are supported by the notes.\n\n"
                f"Document path: {document.source.path}\n\n"
                "Summary notes:\n"
                f"{joined[: budget.consolidate_input_chars]}"
            ),
        },
    ]


def _workspace_group_prompt(group: str) -> PromptMessages:
    return [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Below are summaries of documents from the same workspace. "
                "Please write a detailed workspace-level summary in natural language using exactly these sections: "
                '"Overall Summary", "Features", and "Next Action". '
                "Make Overall Summary the longest section and keep it near 60% of the response. "
                "In Features, list exactly 3 numbered items totaling about 30% of the response. "
                "In Next Action, write one short practical recommendation grounded in the documents, totaling about 10% of the response. "
                "Highlight the main themes, important facts, and meaningful differences across the documents.\n\n"
                "Document summaries:\n"
                f"{group}"
            ),
        },
    ]


def _workspace_final_prompt(group_summaries: list[WorkspaceSummarySections], budget: SummaryBudget) -> PromptMessages:
    joined = "\n\n".join(summary.to_text().strip() for summary in group_summaries)
    return [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Below are draft workspace summaries created from different subsets of the same workspace. "
                "Please combine them into one final workspace summary that reads naturally and stays focused on the source material. "
                'Use exactly these sections: "Overall Summary", "Features", and "Next Action". '
                "Make Overall Summary the longest section and keep it near 60% of the response. "
                "In Features, list exactly 3 numbered items totaling about 30% of the response. "
                "In Next Action, write one short practical recommendation grounded in the documents, totaling about 10% of the response.\n\n"
                "Draft workspace summaries:\n"
                f"{joined[: budget.workspace_input_chars]}"
            ),
        },
    ]


def _parse_workspace_summary_sections(text: str) -> WorkspaceSummarySections:
    normalized = text.strip()
    if not normalized:
        return WorkspaceSummarySections(
            overall_summary="",
            features=[],
            next_action="",
        )

    sections = _extract_titled_sections(normalized)
    overall_summary = _clean_section_body(sections.get("overall summary", ""))
    features_text = _clean_section_body(sections.get("features", ""))
    next_action = _clean_section_body(sections.get("next action", ""))

    if not sections:
        return WorkspaceSummarySections(
            overall_summary=normalized,
            features=[],
            next_action="",
        )

    if not overall_summary:
        overall_summary = normalized
    features = _parse_feature_items(features_text)
    if next_action and next_action.lower().startswith("next action:"):
        next_action = next_action.split(":", 1)[1].strip()
    return WorkspaceSummarySections(
        overall_summary=overall_summary,
        features=features[:3],
        next_action=next_action,
    )


def _extract_titled_sections(text: str) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"(?im)^(?:##\s*)?(Overall Summary|Features|Next Action)\s*:?\s*$",
            text,
        )
    )
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip().lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[title] = text[start:end].strip()
    return sections


def _clean_section_body(text: str) -> str:
    return text.strip().strip("-").strip()


def _parse_feature_items(text: str) -> list[str]:
    if not text:
        return []

    matches = list(re.finditer(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text))
    if not matches:
        compact = _clean_section_body(text)
        return [compact] if compact else []

    items: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item = text[start:end].strip()
        if item:
            items.append(item)
    return items


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RuntimeError("Summarization stopped.")
