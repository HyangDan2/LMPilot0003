from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from src.document_pipeline.high_level import (
    SummaryBudget,
    generate_markdown_report,
    render_workspace_summary_markdown,
    summarize_documents_hierarchically,
)
from src.document_pipeline.low_level import detect_file_type, normalize_text, read_file_bytes
from src.document_pipeline.mid_level import ExtractionContext, build_doc_map
from src.document_pipeline.mid_level import extract_docs as extract_docs_mid_level
from src.document_pipeline.mid_level import extract_single_doc as extract_single_doc_mid_level
from src.document_pipeline.storage import (
    load_extracted_documents_payload,
    pipeline_scope_name_from_path,
    save_document_summaries,
    save_document_map,
    save_extracted_documents,
    save_generated_markdown,
    save_manifest,
    save_single_document,
    save_workspace_summary,
)
from src.gui.llm_client import LLMClientError, OpenAICompatibleClient
from src.document_pipeline.schemas import ExtractedDocument

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
    scope_name = _pipeline_scope_name_for_path(path)
    saved_document = save_single_document(root, document, scope_name)
    save_extracted_documents(root, context.documents, scope_name)
    manifest = save_manifest(root, context.documents, scope_name)
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
        next_actions=["/build_doc_map", "/generate_markdown", "/workspace_status"],
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
        next_actions=["/build_doc_map", "/generate_markdown", "/workspace_status"],
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
    saved_path = save_document_map(root, doc_map, _pipeline_scope_name_for_documents(context.documents))
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
        next_actions=["/generate_markdown", "/workspace_status"],
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
        save_document_map(root, context.doc_map, _pipeline_scope_name_for_documents(context.documents))
    markdown = generate_markdown_report(context.documents, context.doc_map)
    saved_path = save_generated_markdown(root, markdown, _pipeline_scope_name_for_documents(context.documents))
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


def summarize_doc_command(
    args: list[str], working_folder: str | Path | None, context: SlashToolContext, progress=None
) -> SlashToolResult:
    root = require_working_folder(working_folder)
    if len(args) > 1:
        raise SlashToolError("Usage: /summarize_doc [path]")
    target_path = resolve_workspace_path(root, args[0]) if args else None
    documents, auto_extract_message = _resolve_documents_for_summary(root, context, target_path, progress)
    if not documents:
        if target_path is not None:
            raise SlashToolError(f"No extracted content is available for {target_path.name}.")
        raise SlashToolError("No extracted documents are available in the attached folder.")
    client = _get_summary_client(context)
    budget = SummaryBudget()
    _emit_progress(progress, "status", "Preparing hierarchical summaries...")

    try:
        document_summaries, workspace_summary = summarize_documents_hierarchically(
            documents,
            call_model=lambda messages, max_tokens: _chat_summary(client, messages, max_tokens),
            budget=budget,
            progress=lambda message: _emit_progress(progress, "status", message),
            cancel_requested=context.cancel_requested,
        )
    except LLMClientError as exc:
        raise SlashToolError(f"Could not summarize documents: {exc}") from exc
    except RuntimeError as exc:
        raise SlashToolError(str(exc)) from exc

    _emit_progress(progress, "status", "Saving summary artifacts...")
    run_name = _summary_run_name(target_path)
    summaries_payload = {
        "schema_version": "0.1",
        "summary_method": "hierarchical",
        "run_name": run_name,
        "document_count": len(document_summaries),
        "target_path": _relative_to_root(target_path, root) if target_path is not None else None,
        "workspace_summary": workspace_summary,
        "budgets": {
            "per_doc_input_chars": budget.per_doc_input_chars,
            "per_doc_output_tokens": budget.per_doc_output_tokens,
            "consolidate_input_chars": budget.consolidate_input_chars,
            "consolidate_output_tokens": budget.consolidate_output_tokens,
            "workspace_input_chars": budget.workspace_input_chars,
            "workspace_output_tokens": budget.workspace_output_tokens,
            "block_excerpt_chars": budget.block_excerpt_chars,
        },
        "documents": [item.to_dict() for item in document_summaries],
    }
    markdown = render_workspace_summary_markdown(document_summaries, workspace_summary)
    summaries_path = save_document_summaries(root, summaries_payload, run_name)
    workspace_path = save_workspace_summary(root, markdown, run_name)
    saved_files = [_relative_to_root(summaries_path, root), _relative_to_root(workspace_path, root)]
    lines = ["Generated hierarchical document summaries."]
    if auto_extract_message:
        lines.extend(["", auto_extract_message])
    lines.extend(
        [
            "",
            f"- documents: {len(document_summaries)}",
            f"- blocks: {sum(len(document.blocks) for document in documents)}",
        ]
    )
    if target_path is not None:
        lines.append(f"- target_path: {_relative_to_root(target_path, root)}")
    lines.append(f"- output_run: {run_name}")
    lines.append(f"- workspace_summary: {workspace_summary}")
    return _result(
        "/summarize_doc",
        "\n".join(lines),
        saved_files=saved_files,
        next_actions=["/workspace_status", "Ask a normal question about the generated summary"],
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


def _resolve_documents_for_summary(
    root: Path,
    context: SlashToolContext,
    target_path: Path | None,
    progress,
) -> tuple[list[ExtractedDocument], str | None]:
    if not _same_working_folder(context.working_folder, root):
        context.reset_for_folder(root)

    if target_path is not None:
        target_document = _find_document_by_path(context.documents, target_path)
        if target_document is not None:
            context.documents = [target_document]
            context.doc_map = None
            return context.documents, None

    if context.documents:
        if target_path is None:
            return context.documents, None
        target_document = _find_document_by_path(context.documents, target_path)
        if target_document is not None:
            context.documents = [target_document]
            context.doc_map = None
            return context.documents, None

    _emit_progress(progress, "status", "Loading extracted documents...")
    target_scope_name = _pipeline_scope_name_for_path(target_path) if target_path is not None else None
    try:
        if target_scope_name is not None:
            try:
                loaded_documents = documents_from_payload(load_extracted_documents_payload(root, target_scope_name))
            except FileNotFoundError:
                loaded_documents = documents_from_payload(load_extracted_documents_payload(root))
        else:
            loaded_documents = documents_from_payload(load_extracted_documents_payload(root))
    except FileNotFoundError:
        loaded_documents = []
    except (OSError, ValueError) as exc:
        raise SlashToolError(f"Could not load extracted_documents.json: {exc}") from exc

    if loaded_documents:
        context.documents = loaded_documents
        if target_path is None:
            return context.documents, None
        target_document = _find_document_by_path(context.documents, target_path)
        if target_document is not None:
            context.documents = [target_document]
            context.doc_map = None
            return context.documents, None

    if target_path is not None:
        _emit_progress(progress, "status", f"No extracted artifact for {_relative_to_root(target_path, root)}. Extracting that file now...")
        extracted_document = extract_single_doc_mid_level(target_path, ExtractionContext(working_folder=root))
        context.documents = [extracted_document]
        context.doc_map = None
        save_single_document(root, extracted_document, target_scope_name)
        save_extracted_documents(root, context.documents, target_scope_name)
        save_manifest(root, context.documents, target_scope_name)
        return context.documents, f"Auto-extracted {_relative_to_root(target_path, root)} before summarizing."

    _emit_progress(progress, "status", "No extracted artifacts found. Running document extraction now...")
    documents = extract_docs_mid_level(root, ExtractionContext(working_folder=root))
    context.documents = documents
    context.doc_map = None
    save_extracted_documents(root, documents)
    save_manifest(root, documents)
    return context.documents, "Auto-ran /extract_docs before summarizing."


def _same_working_folder(left: Path | None, right: Path) -> bool:
    if left is None:
        return False
    return left.expanduser().resolve() == right.expanduser().resolve()


def _find_document_by_path(documents: list[ExtractedDocument], target_path: Path) -> ExtractedDocument | None:
    target = target_path.expanduser().resolve()
    for document in documents:
        try:
            candidate = Path(document.source.path).expanduser().resolve()
        except OSError:
            continue
        if candidate == target:
            return document
    return None


def _get_summary_client(context: SlashToolContext) -> OpenAICompatibleClient:
    if isinstance(context.active_llm_client, OpenAICompatibleClient):
        return context.active_llm_client
    settings = context.llm_settings
    if settings is None:
        raise SlashToolError("LLM settings are not configured. Set an OpenAI-compatible connection first.")
    if not getattr(settings, "base_url", "").strip():
        raise SlashToolError("LLM Base URL is missing. Configure an OpenAI-compatible connection first.")
    if not getattr(settings, "model", "").strip():
        raise SlashToolError("LLM Model Name is missing. Configure an OpenAI-compatible connection first.")
    return OpenAICompatibleClient(settings)


def _chat_summary(
    client: OpenAICompatibleClient,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    original_max_tokens = client.settings.max_tokens
    try:
        client.settings.max_tokens = max_tokens
        return client.chat_completion(messages)
    finally:
        client.settings.max_tokens = original_max_tokens


def _emit_progress(progress, kind: str, text: str) -> None:
    if progress is None:
        return
    progress(kind, text.rstrip() + "\n")


def _summary_run_name(target_path: Path | None) -> str:
    base_name = target_path.name if target_path is not None else "workspace"
    stem = Path(base_name).stem or "workspace"
    slug = _slugify_summary_name(stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{timestamp}"


def _slugify_summary_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    normalized = normalized.strip("._-")
    return normalized or "workspace"


def _pipeline_scope_name_for_documents(documents: list[ExtractedDocument]) -> str | None:
    if len(documents) != 1:
        return None
    return _pipeline_scope_name_for_path(Path(documents[0].source.path))


def _pipeline_scope_name_for_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return pipeline_scope_name_from_path(path)


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
