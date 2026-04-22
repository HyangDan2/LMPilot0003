from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Callable

from src.ingestion.scanner import scan_supported_files

from src.document_pipeline.mid_level import ExtractionContext, build_doc_map, extract_single_doc
from src.document_pipeline.schemas import DocumentMap, ExtractedDocument, OutputPlan, SelectedEvidence
from src.document_pipeline.storage import (
    load_extracted_documents,
    load_manifest_payload,
    save_document_map,
    save_extracted_documents,
    save_generated_markdown,
    save_manifest,
    save_output_plan,
    save_report_attempts,
    save_selected_evidence,
)

from .generate_report import ReportLLMClient, final_markdown_prompt, generate_report
from .detail_summary import DetailSummaryResult, detail_summaries_markdown, generate_detail_summaries
from .recursive_summary import RecursiveSummaryResult, run_recursive_summary, should_use_recursive_summary
from .select_evidence import select_evidence_blocks
from .write_output_plan import DEFAULT_REPORT_GOAL, write_output_plan

ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class GenerateReportResult:
    documents: list[ExtractedDocument]
    doc_map: DocumentMap
    output_plan: OutputPlan
    selected_evidence: SelectedEvidence
    markdown: str
    used_llm: bool = False
    fallback_reason: str = ""
    extraction_cache_used: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    prerequisite_steps: list[str] = field(default_factory=list)
    saved_files: list[Path] = field(default_factory=list)
    mode: str = "one-shot"
    evidence_group_count: int = 0
    recursive_merge_levels: int = 0
    selected_evidence_group_count: int = 0
    final_prompt_chars: int = 0
    detail_summary_count: int = 0
    detail_used_llm: bool = False
    detail_fallback_reason: str = ""


def generate_report_pipeline(
    working_folder: Path,
    goal: str = DEFAULT_REPORT_GOAL,
    llm_client: ReportLLMClient | None = None,
    llm_input_chars: int = 12000,
    force_refresh: bool = False,
    generate_detail: bool = False,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> GenerateReportResult:
    """Run the full report pipeline from source files and save every artifact.

    This function intentionally does not depend on slash-tool context state. It
    always extracts documents, builds a map, writes an output plan, selects
    evidence blocks, and then generates the final report from the attached
    folder.
    """

    root = working_folder.expanduser().resolve()
    total_started = perf_counter()
    timings: dict[str, float] = {}
    _check_cancelled(cancel_event)
    _emit(progress, "status", "[1/8] Extracting documents...\n")
    started = perf_counter()
    documents, extraction_cache_used = _load_or_extract_documents(root, force_refresh, progress, cancel_event)
    timings["extraction"] = _elapsed(started)
    _check_cancelled(cancel_event)
    cache_note = " from cache" if extraction_cache_used else ""
    _emit(progress, "status", f"Extracted {len(documents)} document(s){cache_note} in {_format_seconds(timings['extraction'])}.\n")
    _emit(progress, "status", "[2/8] Building document map...\n")
    started = perf_counter()
    doc_map = build_doc_map(documents)
    timings["mapping"] = _elapsed(started)
    _check_cancelled(cancel_event)
    _emit(progress, "status", f"Mapped {len(doc_map.blocks)} block(s) in {_format_seconds(timings['mapping'])}.\n")
    _emit(progress, "status", "[3/8] Writing output plan...\n")
    started = perf_counter()
    output_plan = write_output_plan(documents, doc_map, goal=goal)
    timings["planning"] = _elapsed(started)
    _check_cancelled(cancel_event)
    _emit(progress, "status", f"Created {len(output_plan.sections)} output-plan section(s) in {_format_seconds(timings['planning'])}.\n")
    _emit(progress, "status", "[4/8] Selecting representative evidence...\n")
    started = perf_counter()
    selected_evidence = select_evidence_blocks(documents, output_plan, goal, llm_input_chars)
    timings["evidence_selection"] = _elapsed(started)
    _emit(
        progress,
        "status",
        f"Selected {len(selected_evidence.blocks)} evidence block(s) in {_format_seconds(timings['evidence_selection'])}.\n",
    )
    _check_cancelled(cancel_event)
    _emit(progress, "status", "[5/8] Building ranked evidence groups...\n")
    started = perf_counter()
    recursive_result = _recursive_summary_for_documents(
        documents,
        llm_client,
        goal,
        llm_input_chars,
        {block.block_id for block in selected_evidence.blocks},
        progress,
        cancel_event,
    )
    timings["recursive_summary"] = _elapsed(started)
    mode = recursive_result.mode
    if mode == "ranked-groups":
        _emit(
            progress,
            "status",
            (
                f"Using ranked-groups mode: {len(recursive_result.groups)} raw group(s), "
                f"{recursive_result.selected_group_count} selected group(s).\n"
            ),
        )
    else:
        _emit(progress, "status", f"Ranked evidence grouping not needed for {sum(len(document.blocks) for document in documents)} block(s).\n")

    _emit(progress, "status", "[6/8] Preparing grouped evidence context...\n")
    prompt_preview = final_markdown_prompt(
        output_plan,
        documents,
        selected_evidence,
        goal,
        llm_input_chars,
        recursive_result.final_summary,
    )
    _emit(progress, "status", f"Final prompt preview prepared: {len(prompt_preview)} char(s).\n")

    _emit(progress, "status", "[7/8] Generating final grounded Markdown report...\n")
    started = perf_counter()
    report = generate_report(
        output_plan,
        documents,
        doc_map,
        selected_evidence,
        llm_client=llm_client,
        report_query=goal,
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
    detail_result = _detail_summaries_for_documents(
        documents,
        llm_client,
        goal,
        generate_detail,
        progress,
        cancel_event,
    )
    timings["detail_summaries"] = _elapsed(started)

    _check_cancelled(cancel_event)
    _emit(progress, "status", "[8/8] Saving report artifacts...\n")
    started = perf_counter()
    saved_files = [
        save_extracted_documents(root, documents),
        save_manifest(root, documents),
        save_document_map(root, doc_map),
        save_output_plan(root, output_plan),
        save_selected_evidence(root, selected_evidence),
        _save_json(root, "evidence_groups.json", {
            "groups": [group.to_dict() for group in recursive_result.groups],
            "ranked_groups": [group.to_dict() for group in recursive_result.ranked_groups],
        }),
        _save_json(root, "selected_evidence_groups.json", recursive_result.to_dict()),
        _save_json(root, "group_summaries.json", {
            "groups": [group.to_dict() for group in recursive_result.groups],
            "group_summaries": [summary.to_dict() for summary in recursive_result.group_summaries],
        }),
        _save_json(root, "recursive_summary_levels.json", recursive_result.to_dict()),
        _save_text(root, "final_prompt_preview.txt", prompt_preview),
        _save_json(root, "detail_summaries.json", detail_result.to_dict()),
        _save_text(root, "detail_summaries.md", detail_summaries_markdown(detail_result)),
        save_report_attempts(root, report.attempts),
        save_generated_markdown(root, report.markdown),
    ]
    timings["saving"] = _elapsed(started)
    timings["total"] = _elapsed(total_started)
    _emit(progress, "status", f"Saved {len(saved_files)} artifact(s) in {_format_seconds(timings['saving'])}.\n")
    _emit(progress, "status", _format_timings(timings))
    return GenerateReportResult(
        documents=documents,
        doc_map=doc_map,
        output_plan=output_plan,
        selected_evidence=selected_evidence,
        markdown=report.markdown,
        used_llm=report.used_llm,
        fallback_reason=report.fallback_reason,
        extraction_cache_used=extraction_cache_used,
        timings=timings,
        prerequisite_steps=[
            "extract_docs",
            "build_doc_map",
            "write_output_plan",
            "select_evidence_blocks",
            "generate_report",
        ],
        saved_files=saved_files,
        mode=mode,
        evidence_group_count=len(recursive_result.groups),
        recursive_merge_levels=recursive_result.merge_level_count,
        selected_evidence_group_count=recursive_result.selected_group_count,
        final_prompt_chars=len(prompt_preview),
        detail_summary_count=detail_result.summary_count,
        detail_used_llm=detail_result.used_llm,
        detail_fallback_reason=detail_result.fallback_reason,
    )


def _recursive_summary_for_documents(
    documents: list[ExtractedDocument],
    llm_client: ReportLLMClient | None,
    goal: str,
    llm_input_chars: int,
    selected_block_ids: set[str],
    progress: ProgressCallback | None,
    cancel_event: Event | None,
) -> RecursiveSummaryResult:
    if not should_use_recursive_summary(documents, llm_input_chars):
        return RecursiveSummaryResult(mode="one-shot")
    block_count = sum(len(document.blocks) for document in documents)
    _emit(progress, "status", f"Large document set detected: {block_count} block(s). Using ranked evidence grouping.\n")
    return run_recursive_summary(
        documents,
        llm_client,
        goal,
        llm_input_chars,
        progress=progress,
        cancel_event=cancel_event,
        selected_block_ids=selected_block_ids,
    )


def _detail_summaries_for_documents(
    documents: list[ExtractedDocument],
    llm_client: ReportLLMClient | None,
    goal: str,
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
        query=goal,
        progress=progress,
        cancel_event=cancel_event,
    )


def _save_json(root: Path, filename: str, payload: object) -> Path:
    path = root / "llm_result" / "document_pipeline" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _save_text(root: Path, filename: str, text: str) -> Path:
    path = root / "llm_result" / "document_pipeline" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _emit(progress: ProgressCallback | None, kind: str, text: str) -> None:
    if progress is not None and text:
        progress(kind, text)


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Slash tool cancelled.")


def _load_or_extract_documents(
    root: Path,
    force_refresh: bool,
    progress: ProgressCallback | None,
    cancel_event: Event | None,
) -> tuple[list[ExtractedDocument], bool]:
    files = scan_supported_files(root)
    _emit(progress, "status", f"Found {len(files)} supported source file(s).\n")
    if not force_refresh:
        cached = _load_cached_documents(root, files)
        if cached is not None:
            _emit(progress, "status", "Reusing unchanged extracted_documents.json.\n")
            return cached, True

    context = ExtractionContext(working_folder=root)
    documents: list[ExtractedDocument] = []
    for path in files:
        _check_cancelled(cancel_event)
        documents.append(extract_single_doc(path, context))
    return documents, False


def _load_cached_documents(root: Path, files: list[Path]) -> list[ExtractedDocument] | None:
    try:
        manifest = load_manifest_payload(root)
        documents = load_extracted_documents(root)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not _manifest_matches_files(manifest, files):
        return None
    return documents


def _manifest_matches_files(manifest: dict, files: list[Path]) -> bool:
    entries = manifest.get("documents")
    if not isinstance(entries, list):
        return False
    if len(entries) != len(files):
        return False
    by_path: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            return False
        by_path[str(Path(path_value).expanduser().resolve())] = entry
    for path in files:
        resolved = str(path.expanduser().resolve())
        entry = by_path.get(resolved)
        if entry is None:
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        if entry.get("size_bytes") != stat.st_size:
            return False
        if entry.get("mtime_ns") != stat.st_mtime_ns:
            return False
    return True


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
