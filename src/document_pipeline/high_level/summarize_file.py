from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any, Callable

from src.document_pipeline.mid_level import ExtractionContext, build_doc_map, extract_single_doc
from src.document_pipeline.schemas import DocumentMap, ExtractedDocument, OutputPlan, SelectedEvidence

from .detail_summary import DetailSummaryResult, detail_summaries_markdown, generate_detail_summaries
from .generate_report import ReportLLMClient
from .markdown_format import sentence_per_line_markdown
from .recursive_summary import RecursiveSummaryResult, run_recursive_summary, should_use_recursive_summary
from .select_evidence import format_selected_evidence, select_evidence_blocks
from .write_output_plan import SUMMARY_SUBSECTIONS, write_output_plan

ProgressCallback = Callable[[str, str], None]

DEFAULT_FILE_SUMMARY_GOAL = "Summarize this file as a concise engineering summary."


@dataclass(frozen=True)
class SummarizeFileResult:
    document: ExtractedDocument
    doc_map: DocumentMap
    output_plan: OutputPlan
    selected_evidence: SelectedEvidence
    markdown: str
    used_llm: bool = False
    fallback_reason: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    saved_files: list[Path] = field(default_factory=list)
    mode: str = "one-shot"
    evidence_group_count: int = 0
    recursive_merge_levels: int = 0
    selected_evidence_group_count: int = 0
    final_prompt_chars: int = 0
    detail_summary_count: int = 0
    detail_used_llm: bool = False
    detail_fallback_reason: str = ""


def summarize_file_pipeline(
    working_folder: Path,
    file_path: Path,
    goal: str = DEFAULT_FILE_SUMMARY_GOAL,
    llm_client: ReportLLMClient | None = None,
    llm_input_chars: int = 12000,
    generate_detail: bool = False,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> SummarizeFileResult:
    """Extract and summarize one source file without touching folder-level report artifacts."""

    root = working_folder.expanduser().resolve()
    source_path = file_path.expanduser().resolve()
    query = goal.strip() or DEFAULT_FILE_SUMMARY_GOAL
    total_started = perf_counter()
    timings: dict[str, float] = {}

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[1/8] Extracting selected file...\n")
    started = perf_counter()
    document = extract_single_doc(source_path, ExtractionContext(working_folder=root))
    timings["extraction"] = _elapsed(started)
    _emit(progress, "status", f"Extracted {document.source.filename} in {_format_seconds(timings['extraction'])}.\n")

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[2/8] Building document map...\n")
    started = perf_counter()
    doc_map = build_doc_map([document])
    timings["mapping"] = _elapsed(started)
    _emit(progress, "status", f"Mapped {len(doc_map.blocks)} block(s) in {_format_seconds(timings['mapping'])}.\n")

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[3/8] Writing file summary plan...\n")
    started = perf_counter()
    output_plan = write_output_plan([document], doc_map, goal=query)
    timings["planning"] = _elapsed(started)
    _emit(progress, "status", f"Created {len(output_plan.sections)} section(s) in {_format_seconds(timings['planning'])}.\n")

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[4/8] Selecting representative file evidence...\n")
    started = perf_counter()
    selected_evidence = select_evidence_blocks([document], output_plan, query, llm_input_chars)
    timings["evidence_selection"] = _elapsed(started)
    _emit(
        progress,
        "status",
        f"Selected {len(selected_evidence.blocks)} evidence block(s) in {_format_seconds(timings['evidence_selection'])}.\n",
    )

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[5/8] Building ranked file evidence groups...\n")
    started = perf_counter()
    recursive_result = _recursive_summary_for_file(
        [document],
        llm_client,
        query,
        llm_input_chars,
        {block.block_id for block in selected_evidence.blocks},
        progress,
        cancel_event,
    )
    timings["recursive_summary"] = _elapsed(started)
    if recursive_result.mode == "ranked-groups":
        _emit(
            progress,
            "status",
            (
                f"Using ranked-groups mode: {len(recursive_result.groups)} raw group(s), "
                f"{recursive_result.selected_group_count} selected group(s).\n"
            ),
        )
    else:
        _emit(progress, "status", f"Ranked evidence grouping not needed for {len(document.blocks)} block(s).\n")

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[6/8] Preparing grouped file evidence context...\n")
    prompt_preview = _file_summary_prompt(
        output_plan,
        document,
        selected_evidence,
        query,
        llm_input_chars,
        recursive_result.final_summary,
    )
    _emit(progress, "status", f"File-summary prompt preview prepared: {len(prompt_preview)} char(s).\n")

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[7/8] Generating grounded file summary...\n")
    started = perf_counter()
    markdown, used_llm, fallback_reason, attempts = _generate_file_summary(
        output_plan=output_plan,
        document=document,
        selected_evidence=selected_evidence,
        llm_client=llm_client,
        query=query,
        max_input_chars=llm_input_chars,
        recursive_summary=recursive_result.final_summary,
        progress=progress,
        cancel_event=cancel_event,
    )
    timings["llm_generation"] = _elapsed(started)

    _check_cancelled(cancel_event)
    started = perf_counter()
    if generate_detail:
        _emit(progress, "status", "[detail] Generating detail summaries...\n")
    detail_result = _detail_summaries_for_file(
        [document],
        llm_client,
        query,
        generate_detail,
        progress,
        cancel_event,
    )
    timings["detail_summaries"] = _elapsed(started)

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[8/8] Saving file summary artifacts...\n")
    started = perf_counter()
    saved_files = _save_file_summary_artifacts(
        root=root,
        document=document,
        doc_map=doc_map,
        output_plan=output_plan,
        selected_evidence=selected_evidence,
        recursive_result=recursive_result,
        detail_result=detail_result,
        prompt_preview=prompt_preview,
        attempts=attempts,
        markdown=markdown,
    )
    timings["saving"] = _elapsed(started)
    timings["total"] = _elapsed(total_started)
    _emit(progress, "status", f"Saved {len(saved_files)} artifact(s) in {_format_seconds(timings['saving'])}.\n")
    _emit(progress, "status", _format_timings(timings))

    return SummarizeFileResult(
        document=document,
        doc_map=doc_map,
        output_plan=output_plan,
        selected_evidence=selected_evidence,
        markdown=markdown,
        used_llm=used_llm,
        fallback_reason=fallback_reason,
        timings=timings,
        saved_files=saved_files,
        mode=recursive_result.mode,
        evidence_group_count=len(recursive_result.groups),
        recursive_merge_levels=recursive_result.merge_level_count,
        selected_evidence_group_count=recursive_result.selected_group_count,
        final_prompt_chars=len(prompt_preview),
        detail_summary_count=detail_result.summary_count,
        detail_used_llm=detail_result.used_llm,
        detail_fallback_reason=detail_result.fallback_reason,
    )


def _generate_file_summary(
    *,
    output_plan: OutputPlan,
    document: ExtractedDocument,
    selected_evidence: SelectedEvidence,
    llm_client: ReportLLMClient | None,
    query: str,
    max_input_chars: int,
    recursive_summary: str,
    progress: ProgressCallback | None,
    cancel_event: Event | None,
) -> tuple[str, bool, str, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    if llm_client is None:
        markdown = _fallback_file_summary(output_plan, document, selected_evidence)
        _emit(progress, "status", "LLM client is not configured. Using deterministic file summary.\n")
        _emit(progress, "markdown", markdown)
        attempts.append({"stage": "file_summary", "status": "fallback", "error": "LLM client is not configured."})
        return markdown, False, "LLM client is not configured.", attempts

    try:
        prompt = _file_summary_prompt(output_plan, document, selected_evidence, query, max_input_chars, recursive_summary)
        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise single-file engineering summaries from extracted evidence. "
                    "Return Markdown only. Do not invent unsupported facts. Cite block IDs and the source filename."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        stream_chat_completion = getattr(llm_client, "stream_chat_completion", None)
        if callable(stream_chat_completion):
            parts: list[str] = []
            for chunk in stream_chat_completion(messages):
                _check_cancelled(cancel_event)
                if getattr(chunk, "kind", "") != "final":
                    continue
                text = getattr(chunk, "text", "")
                if not text:
                    continue
                parts.append(text)
                _emit(progress, "markdown", text)
            content = "".join(parts)
        else:
            content = llm_client.chat_completion(messages)
            _emit(progress, "markdown", content)
        markdown = content.strip()
        if not markdown:
            raise ValueError("File summary response was empty.")
        if not markdown.startswith("#"):
            markdown = f"# File Summary: {document.source.filename}\n\n{markdown}"
        markdown = sentence_per_line_markdown(markdown)
        attempts.append(
            {
                "stage": "file_summary",
                "status": "succeeded",
                "selected_evidence_block_count": len(selected_evidence.blocks),
                "final_prompt_chars": len(prompt),
                "grouped_evidence_context_chars": len(recursive_summary),
                "llm_calls": 1,
            }
        )
        return markdown, True, "", attempts
    except Exception as exc:
        markdown = _fallback_file_summary(output_plan, document, selected_evidence)
        _emit(progress, "status", f"File summary LLM failed. Using deterministic fallback: {exc}\n")
        _emit(progress, "markdown", markdown)
        attempts.append({"stage": "file_summary", "status": "fallback", "error": str(exc)})
        return markdown, False, str(exc), attempts


def _recursive_summary_for_file(
    documents: list[ExtractedDocument],
    llm_client: ReportLLMClient | None,
    query: str,
    llm_input_chars: int,
    selected_block_ids: set[str],
    progress: ProgressCallback | None,
    cancel_event: Event | None,
) -> RecursiveSummaryResult:
    if not should_use_recursive_summary(documents, llm_input_chars):
        return RecursiveSummaryResult(mode="one-shot")
    block_count = sum(len(document.blocks) for document in documents)
    _emit(progress, "status", f"Large file detected: {block_count} block(s). Using ranked evidence grouping.\n")
    return run_recursive_summary(
        documents,
        llm_client,
        query,
        llm_input_chars,
        progress=progress,
        cancel_event=cancel_event,
        selected_block_ids=selected_block_ids,
    )


def _detail_summaries_for_file(
    documents: list[ExtractedDocument],
    llm_client: ReportLLMClient | None,
    query: str,
    generate_detail: bool,
    progress: ProgressCallback | None,
    cancel_event: Event | None,
) -> DetailSummaryResult:
    if not generate_detail:
        return DetailSummaryResult(enabled=False)
    return generate_detail_summaries(
        documents,
        llm_client,
        enabled=True,
        query=query,
        progress=progress,
        cancel_event=cancel_event,
    )


def _file_summary_prompt(
    output_plan: OutputPlan,
    document: ExtractedDocument,
    selected_evidence: SelectedEvidence,
    query: str,
    max_input_chars: int,
    recursive_summary: str = "",
) -> str:
    summary_subsections = ", ".join(SUMMARY_SUBSECTIONS)
    instructions = (
        "Write a single-file engineering summary as Markdown only.\n\n"
        "Requirements:\n"
        "- Start with exactly one H1 title in the form: File Summary: <filename>.\n"
        "- Use exactly these H2 sections in this order: Summary, Source Details, Open Issues and Next Actions.\n"
        f"- Under Summary, use H3 subsections when evidence supports them: {summary_subsections}.\n"
        "- If a Summary subsection lacks evidence, write: Not explicitly stated in the selected evidence.\n"
        "- Use only the selected evidence packet from this one file.\n"
        "- Summarize only what the selected evidence explicitly states.\n"
        "- Do not infer architecture, databases, proprietary engines, implementation details, risks, constraints, or recommendations unless explicitly stated in cited evidence.\n"
        "- Do not add recommendations unless the selected evidence itself contains recommendations or requested next actions.\n"
        "- In Source Details, include a compact table for filename, type, blocks, and assets.\n"
        "- In Open Issues and Next Actions, list only extraction warnings, missing selected evidence, or explicitly stated unresolved items.\n"
        "- Cite source filename and block ID for concrete claims.\n"
        "- Write each sentence on its own line in normal paragraphs.\n"
        "- Do not add extra H2 sections.\n"
    )
    context = (
        f"\nSummary query:\n{query}\n\n"
        "Source file:\n"
        f"- {document.source.filename} ({document.source.extension}): "
        f"{len(document.blocks)} block(s), {len(document.assets)} asset(s), document_id={document.document_id}\n"
        f"Selected evidence block count: {len(selected_evidence.blocks)}\n"
    )
    recursive_section = ""
    if recursive_summary.strip():
        recursive_section = (
            "\nTop ranked file evidence groups for broad coverage:\n"
            f"{_truncate(recursive_summary.strip(), max(1000, max_input_chars // 2))}\n"
        )
    fixed = instructions + context + recursive_section + "\nSelected evidence for citation checks:\n"
    evidence_budget = max(800, max_input_chars - len(fixed))
    return fixed + _truncate(format_selected_evidence(selected_evidence), evidence_budget)


def _fallback_file_summary(
    output_plan: OutputPlan,
    document: ExtractedDocument,
    selected_evidence: SelectedEvidence,
) -> str:
    lines = [f"# File Summary: {document.source.filename}", "", "## Summary", ""]
    blocks = selected_evidence.blocks
    for heading in SUMMARY_SUBSECTIONS:
        lines.extend([f"### {heading}", ""])
        matching = _fallback_entries_for_heading(heading, blocks)
        if matching:
            lines.extend(matching[:4])
        else:
            lines.append("- Not explicitly stated in the selected evidence.")
        lines.append("")
    lines.extend(
        [
            "## Source Details",
            "",
            "| Source | Type | Blocks | Assets |",
            "|---|---:|---:|---:|",
            f"| {document.source.filename} | {document.source.extension} | {len(document.blocks)} | {len(document.assets)} |",
            "",
            "## Open Issues and Next Actions",
            "",
        ]
    )
    if document.warnings:
        lines.extend(f"- {warning}" for warning in document.warnings)
    else:
        lines.append("- Review cited evidence before using this file summary externally.")
        lines.append("- Treat any topic not explicitly stated in the selected evidence as unspecified.")
    lines.append("")
    return sentence_per_line_markdown("\n".join(lines))


def _fallback_entries_for_heading(heading: str, blocks) -> list[str]:
    entries: list[str] = []
    for block in blocks:
        text = " ".join(block.text.split())
        if not text:
            continue
        if not _fallback_block_matches_heading(heading, text):
            continue
        source = f"{block.source_filename} / {block.block_id}".strip(" /")
        entries.append(f"- {_truncate(text, 320)} (`{source}`)")
    return entries


def _fallback_block_matches_heading(heading: str, text: str) -> bool:
    lowered = text.lower()
    if heading == "What the Document Explicitly Describes":
        return True
    if heading == "Main Methods or Components Explicitly Mentioned":
        return any(
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
                "model",
                "protocol",
                "interface",
                "module",
                "방법",
                "모듈",
                "시스템",
                "설계",
            )
        )
    if heading == "Quantitative Values Explicitly Present":
        return any(char.isdigit() for char in text)
    if heading == "Explicit Limitations or Constraints":
        return any(
            term in lowered
            for term in ("limit", "constraint", "risk", "issue", "failure", "uncertain", "한계", "제약", "리스크", "문제")
        )
    if heading == "Unclear or Not Specified in Selected Evidence":
        return False
    return False


def _save_file_summary_artifacts(
    *,
    root: Path,
    document: ExtractedDocument,
    doc_map: DocumentMap,
    output_plan: OutputPlan,
    selected_evidence: SelectedEvidence,
    recursive_result: RecursiveSummaryResult,
    detail_result: DetailSummaryResult,
    prompt_preview: str,
    attempts: list[dict[str, Any]],
    markdown: str,
) -> list[Path]:
    output_dir = root / "llm_result" / "document_pipeline" / "file_summaries" / document.document_id
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write_json(output_dir / "extracted_document.json", document.to_dict()),
        _write_json(output_dir / "document_map.json", doc_map.to_dict()),
        _write_json(output_dir / "output_plan.json", output_plan.to_dict()),
        _write_json(output_dir / "selected_evidence.json", selected_evidence.to_dict()),
        _write_json(
            output_dir / "evidence_groups.json",
            {
                "groups": [group.to_dict() for group in recursive_result.groups],
                "ranked_groups": [group.to_dict() for group in recursive_result.ranked_groups],
            },
        ),
        _write_json(output_dir / "selected_evidence_groups.json", recursive_result.to_dict()),
        _write_json(
            output_dir / "group_summaries.json",
            {
                "groups": [group.to_dict() for group in recursive_result.groups],
                "group_summaries": [summary.to_dict() for summary in recursive_result.group_summaries],
            },
        ),
        _write_json(output_dir / "recursive_summary_levels.json", recursive_result.to_dict()),
        _write_json(output_dir / "detail_summaries.json", detail_result.to_dict()),
        _write_json(output_dir / "summary_attempts.json", {"attempts": attempts}),
    ]
    detail_markdown_path = output_dir / "detail_summaries.md"
    detail_markdown_path.write_text(detail_summaries_markdown(detail_result), encoding="utf-8")
    files.append(detail_markdown_path)
    prompt_path = output_dir / "final_prompt_preview.txt"
    prompt_path.write_text(prompt_preview, encoding="utf-8")
    files.append(prompt_path)
    summary_path = output_dir / "generated_summary.md"
    summary_path.write_text(markdown, encoding="utf-8")
    files.append(summary_path)
    return files


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def _emit(progress: ProgressCallback | None, kind: str, text: str) -> None:
    if progress is not None and text:
        progress(kind, text)


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Slash tool cancelled.")


def _elapsed(started: float) -> float:
    return perf_counter() - started


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _format_timings(timings: dict[str, float]) -> str:
    labels = [
        ("extraction", "extraction"),
        ("mapping", "mapping"),
        ("planning", "planning"),
        ("evidence_selection", "evidence selection"),
        ("recursive_summary", "ranked evidence grouping"),
        ("llm_generation", "LLM generation"),
        ("detail_summaries", "detail summaries"),
        ("saving", "saving"),
        ("total", "total"),
    ]
    lines = ["Timings:"]
    lines.extend(f"- {label}: {_format_seconds(timings.get(key, 0.0))}" for key, label in labels)
    return "\n".join(lines) + "\n"


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."
