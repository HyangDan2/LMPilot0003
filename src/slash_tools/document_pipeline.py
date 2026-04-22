from __future__ import annotations

from pathlib import Path

from src.document_pipeline.high_level import generate_markdown_report, generate_report_pipeline, summarize_file_pipeline
from src.document_pipeline.low_level import detect_file_type, normalize_text, read_file_bytes
from src.document_pipeline.mid_level import ExtractionContext, build_doc_map
from src.document_pipeline.mid_level import extract_docs as extract_docs_mid_level
from src.document_pipeline.mid_level import extract_single_doc as extract_single_doc_mid_level
from src.document_pipeline.storage import (
    load_extracted_documents_payload,
    save_document_map,
    save_extracted_documents,
    save_generated_markdown,
    save_manifest,
    save_single_document,
)
from src.gui.llm_client import OpenAICompatibleClient

from .context import SlashToolContext
from .errors import SlashToolError
from .path_safety import require_working_folder, resolve_workspace_path
from .results import SlashToolResult
from .serialization import documents_from_payload
from .workspace_state import load_workspace_state


def detect_file_type_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if len(args) != 1:
        raise SlashToolError("Usage: /detect_file_type <path>")
    path = resolve_workspace_path(root, args[0])
    detected = detect_file_type(path)
    return _result(
        "/detect_file_type",
        "\n".join(
            [
                "File type detected.",
                "",
                f"- path: {_relative_to_root(path, root)}",
                f"- extension: {detected.extension}",
                f"- mime_type: {detected.mime_type}",
                f"- family: {detected.family}",
                f"- confidence: {detected.confidence}",
            ]
        ),
    )


def read_file_info_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if len(args) != 1:
        raise SlashToolError("Usage: /read_file_info <path>")
    path = resolve_workspace_path(root, args[0])
    payload = read_file_bytes(path)
    return _result(
        "/read_file_info",
        "\n".join(
            [
                "File info.",
                "",
                f"- path: {_relative_to_root(payload.path, root)}",
                f"- size_bytes: {payload.size_bytes}",
                f"- sha256: {payload.sha256}",
            ]
        ),
    )


def normalize_text_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    if not args:
        raise SlashToolError("Usage: /normalize_text <text>")
    return _result("/normalize_text", normalize_text(" ".join(args)))


def extract_single_doc_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if len(args) != 1:
        raise SlashToolError("Usage: /extract_single_doc <path>")
    context.reset_for_folder(root) if not _same_working_folder(context.working_folder, root) else None
    path = resolve_workspace_path(root, args[0])
    document = extract_single_doc_mid_level(path, ExtractionContext(working_folder=root))
    context.documents = [document]
    saved_document = save_single_document(root, document)
    save_extracted_documents(root, context.documents)
    manifest = save_manifest(root, context.documents)
    saved_files = [_relative_to_root(saved_document, root), _relative_to_root(manifest, root)]
    return _result(
        "/extract_single_doc",
        "\n".join(
            [
                "Extracted 1 document.",
                "",
                _document_summary(document, root),
            ]
        ),
        saved_files=saved_files,
        next_actions=["/build_doc_map", "/generate_report", "/workspace_status"],
    )


def extract_docs_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if args:
        raise SlashToolError("Usage: /extract_docs")
    context.reset_for_folder(root) if not _same_working_folder(context.working_folder, root) else None
    documents = extract_docs_mid_level(root, ExtractionContext(working_folder=root))
    context.documents = documents
    context.doc_map = None
    docs_path = save_extracted_documents(root, documents)
    manifest = save_manifest(root, documents)
    saved_files = [_relative_to_root(docs_path, root), _relative_to_root(manifest, root)]
    lines = [
        f"Extracted {len(documents)} document(s).",
        "",
        "Documents:",
    ]
    lines.extend([f"- {_document_summary(document, root)}" for document in documents] or ["- none"])
    return _result(
        "/extract_docs",
        "\n".join(lines),
        saved_files=saved_files,
        next_actions=["/build_doc_map", "/generate_report", "/workspace_status"],
    )


def build_doc_map_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if args:
        raise SlashToolError("Usage: /build_doc_map")
    _ensure_documents(root, context)
    doc_map = build_doc_map(context.documents)
    context.doc_map = doc_map
    saved_path = save_document_map(root, doc_map)
    return _result(
        "/build_doc_map",
        "\n".join(
            [
                "Built document map.",
                "",
                f"- documents: {len(doc_map.documents)}",
                f"- blocks: {len(doc_map.blocks)}",
            ]
        ),
        saved_files=[_relative_to_root(saved_path, root)],
        next_actions=["/generate_report", "/workspace_status"],
    )


def workspace_status_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if args:
        raise SlashToolError("Usage: /workspace_status")
    state = load_workspace_state(root)
    return SlashToolResult(
        text=state.to_text(),
        tool_name="/workspace_status",
        next_actions=state.next_actions,
    )


def generate_markdown_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if args:
        raise SlashToolError("Usage: /generate_markdown")
    _ensure_documents(root, context)
    if context.doc_map is None:
        context.doc_map = build_doc_map(context.documents)
        save_document_map(root, context.doc_map)
    markdown = generate_markdown_report(context.documents, context.doc_map)
    saved_path = save_generated_markdown(root, markdown)
    return _result(
        "/generate_markdown",
        "\n".join(
            [
                "Generated markdown report.",
                "",
                f"- documents: {len(context.documents)}",
                f"- blocks: {sum(len(document.blocks) for document in context.documents)}",
            ]
        ),
        saved_files=[_relative_to_root(saved_path, root)],
        next_actions=["/workspace_status", "Ask a normal question about the generated report"],
    )


def generate_report_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    goal, llm_input_chars, use_llm, force_refresh, generate_detail = _parse_generate_report_args(args)
    context.reset_for_folder(root) if not _same_working_folder(context.working_folder, root) else None
    llm_client = _make_llm_client(context) if use_llm else None
    result = generate_report_pipeline(
        root,
        goal=goal,
        llm_client=llm_client,
        llm_input_chars=llm_input_chars,
        force_refresh=force_refresh,
        generate_detail=generate_detail,
        progress=progress,
        cancel_event=context.cancel_event,
    )
    context.documents = result.documents
    context.doc_map = result.doc_map
    saved_files = [_relative_to_root(path, root) for path in result.saved_files]
    lines = [
        "Generated report.",
        "",
        "Automatically ran:",
        *[f"- {step}" for step in result.prerequisite_steps],
        "",
        f"- documents: {len(result.documents)}",
        f"- mode: {result.mode}",
        f"- evidence_blocks: {len(result.selected_evidence.blocks)}",
        f"- evidence_groups: {result.evidence_group_count}",
        f"- selected_evidence_groups: {result.selected_evidence_group_count}",
        f"- recursive_merge_levels: {result.recursive_merge_levels}",
        f"- final_prompt_chars: {result.final_prompt_chars}",
        f"- detail_summaries: {result.detail_summary_count}",
        f"- detail_llm_used: {'yes' if result.detail_used_llm else 'no'}",
        f"- plan_sections: {len(result.output_plan.sections)}",
        f"- llm_input_chars: {llm_input_chars}",
        f"- llm_used: {'yes' if result.used_llm else 'no'}",
        f"- extraction_cache_used: {'yes' if result.extraction_cache_used else 'no'}",
        f"- goal: {result.output_plan.goal}",
        "",
        "Timings:",
        *[f"- {label}: {value:.2f}s" for label, value in result.timings.items()],
    ]
    if result.fallback_reason:
        lines.append(f"- fallback_reason: {result.fallback_reason}")
    if result.detail_fallback_reason:
        lines.append(f"- detail_fallback_reason: {result.detail_fallback_reason}")
    return _result(
        "/generate_report",
        "\n".join(lines),
        saved_files=saved_files,
        next_actions=["/workspace_status", "Ask a normal question about the generated report"],
    )


def summarize_file_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    file_arg, goal, llm_input_chars, use_llm, generate_detail = _parse_summarize_file_args(args)
    path = resolve_workspace_path(root, file_arg)
    llm_client = _make_llm_client(context) if use_llm else None
    result = summarize_file_pipeline(
        root,
        path,
        goal=goal,
        llm_client=llm_client,
        llm_input_chars=llm_input_chars,
        generate_detail=generate_detail,
        progress=progress,
        cancel_event=context.cancel_event,
    )
    context.documents = [result.document]
    context.doc_map = result.doc_map
    saved_files = [_relative_to_root(path, root) for path in result.saved_files]
    lines = [
        "Generated file summary.",
        "",
        f"- file: {_relative_to_root(path, root)}",
        f"- mode: {result.mode}",
        f"- blocks: {len(result.document.blocks)}",
        f"- evidence_blocks: {len(result.selected_evidence.blocks)}",
        f"- evidence_groups: {result.evidence_group_count}",
        f"- selected_evidence_groups: {result.selected_evidence_group_count}",
        f"- recursive_merge_levels: {result.recursive_merge_levels}",
        f"- final_prompt_chars: {result.final_prompt_chars}",
        f"- detail_summaries: {result.detail_summary_count}",
        f"- detail_llm_used: {'yes' if result.detail_used_llm else 'no'}",
        f"- llm_input_chars: {llm_input_chars}",
        f"- llm_used: {'yes' if result.used_llm else 'no'}",
        f"- goal: {result.output_plan.goal}",
        "",
        "Timings:",
        *[f"- {label}: {value:.2f}s" for label, value in result.timings.items()],
    ]
    if result.fallback_reason:
        lines.append(f"- fallback_reason: {result.fallback_reason}")
    if result.detail_fallback_reason:
        lines.append(f"- detail_fallback_reason: {result.detail_fallback_reason}")
    return _result(
        "/summarize_file",
        "\n".join(lines),
        saved_files=saved_files,
        next_actions=["/workspace_status", "Ask a normal question about the file summary"],
    )


def _ensure_documents(root: Path, context: SlashToolContext) -> None:
    if not _same_working_folder(context.working_folder, root):
        context.reset_for_folder(root)
    if context.documents:
        return
    try:
        context.documents = documents_from_payload(load_extracted_documents_payload(root))
    except FileNotFoundError as exc:
        raise SlashToolError("Run /extract_docs first, or restore extracted_documents.json.") from exc
    except (OSError, ValueError) as exc:
        raise SlashToolError(f"Could not load extracted_documents.json: {exc}") from exc


def _parse_generate_report_args(args: list[str]) -> tuple[str, int, bool, bool, bool]:
    goal = "Generate a concise engineering report from the attached workspace documents."
    llm_input_chars = 12000
    use_llm = True
    force_refresh = False
    generate_detail = False
    query_parts: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--no-llm":
            use_llm = False
            index += 1
            continue
        if token == "--fresh":
            force_refresh = True
            index += 1
            continue
        if token == "--generate-detail":
            if index + 1 >= len(args):
                raise SlashToolError("Usage: /generate_report [--no-llm] [--fresh] [--generate-detail true|false] [--llm-input-chars N] [query...]")
            generate_detail = _parse_bool_option("--generate-detail", args[index + 1])
            index += 2
            continue
        if token == "--llm-input-chars":
            if index + 1 >= len(args):
                raise SlashToolError("Usage: /generate_report [--no-llm] [--fresh] [--generate-detail true|false] [--llm-input-chars N] [query...]")
            try:
                llm_input_chars = int(args[index + 1])
            except ValueError as exc:
                raise SlashToolError("--llm-input-chars must be an integer.") from exc
            if llm_input_chars < 800:
                raise SlashToolError("--llm-input-chars must be at least 800.")
            index += 2
        elif token == "--goal":
            if index + 1 >= len(args):
                raise SlashToolError("Usage: /generate_report [--no-llm] [--fresh] [--generate-detail true|false] [--llm-input-chars N] [query...]")
            goal = " ".join(args[index + 1 :]).strip()
            break
        else:
            query_parts.extend(args[index:])
            break
    if query_parts:
        goal = " ".join(query_parts).strip()
    return goal, llm_input_chars, use_llm, force_refresh, generate_detail


def _parse_summarize_file_args(args: list[str]) -> tuple[str, str, int, bool, bool]:
    if not args:
        raise SlashToolError("Usage: /summarize_file <path> [--no-llm] [--generate-detail true|false] [--llm-input-chars N] [query...]")
    llm_input_chars = 12000
    use_llm = True
    generate_detail = False
    file_arg = ""
    query_parts: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--no-llm":
            use_llm = False
            index += 1
            continue
        if token == "--llm-input-chars":
            if index + 1 >= len(args):
                raise SlashToolError("Usage: /summarize_file <path> [--no-llm] [--generate-detail true|false] [--llm-input-chars N] [query...]")
            try:
                llm_input_chars = int(args[index + 1])
            except ValueError as exc:
                raise SlashToolError("--llm-input-chars must be an integer.") from exc
            if llm_input_chars < 800:
                raise SlashToolError("--llm-input-chars must be at least 800.")
            index += 2
            continue
        if token == "--generate-detail":
            if index + 1 >= len(args):
                raise SlashToolError("Usage: /summarize_file <path> [--no-llm] [--generate-detail true|false] [--llm-input-chars N] [query...]")
            generate_detail = _parse_bool_option("--generate-detail", args[index + 1])
            index += 2
            continue
        if token.startswith("--"):
            raise SlashToolError(f"Unknown /summarize_file option: {token}")
        if not file_arg:
            file_arg = token
        else:
            query_parts.append(token)
        index += 1
    if not file_arg:
        raise SlashToolError("Usage: /summarize_file <path> [--no-llm] [--generate-detail true|false] [--llm-input-chars N] [query...]")
    goal = " ".join(query_parts).strip() or "Summarize this file as a concise engineering summary."
    return file_arg, goal, llm_input_chars, use_llm, generate_detail


def _parse_bool_option(option: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SlashToolError(f"{option} must be true or false.")


def _make_llm_client(context: SlashToolContext):
    settings = context.llm_settings
    if settings is None or not getattr(settings, "base_url", "").strip() or not getattr(settings, "model", "").strip():
        return None
    client = OpenAICompatibleClient(settings)
    context.active_llm_client = client
    return client


def _same_working_folder(left: Path | None, right: Path) -> bool:
    if left is None:
        return False
    return left.expanduser().resolve() == right.expanduser().resolve()


def _document_summary(document, root: Path) -> str:
    return (
        f"{_relative_to_root(Path(document.source.path), root)}: "
        f"{len(document.blocks)} block(s), {len(document.assets)} asset(s)"
    )


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _result(
    tool_name: str,
    text: str,
    saved_files: list[str] | None = None,
    next_actions: list[str] | None = None,
) -> SlashToolResult:
    result = SlashToolResult(
        text=_with_saved_and_next(text, saved_files or [], next_actions or []),
        tool_name=tool_name,
        saved_files=list(saved_files or []),
        next_actions=list(next_actions or []),
    )
    return result


def _with_saved_and_next(text: str, saved_files: list[str], next_actions: list[str]) -> str:
    lines = [text.rstrip()]
    if saved_files:
        lines.extend(["", "Saved:"])
        lines.extend(f"- {path}" for path in saved_files)
    if next_actions:
        lines.extend(["", "Suggested next:"])
        lines.extend(f"- {action}" for action in next_actions)
    return "\n".join(lines)
