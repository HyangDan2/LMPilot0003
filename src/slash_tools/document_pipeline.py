from __future__ import annotations

from pathlib import Path

from src.document_pipeline.high_level import generate_markdown_report
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
