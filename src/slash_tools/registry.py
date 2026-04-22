from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .context import SlashToolContext
from .document_pipeline import (
    build_doc_map_command,
    detect_file_type_command,
    extract_docs_command,
    extract_single_doc_command,
    generate_markdown_command,
    normalize_text_command,
    read_file_info_command,
    workspace_status_command,
)
from .errors import SlashToolError
from .help import help_command
from .results import SlashToolResult, error_result


SlashProgressCallback = Callable[[str, str], None]
SlashHandler = Callable[[list[str], str | Path | None, SlashToolContext, SlashProgressCallback | None], SlashToolResult]


@dataclass(frozen=True)
class SlashTool:
    name: str
    summary: str
    usage: str
    handler: SlashHandler


SLASH_TOOLS: dict[str, SlashTool] = {
    "/help": SlashTool("/help", "Show available local slash tools.", "/help", help_command),
    "/detect_file_type": SlashTool(
        "/detect_file_type",
        "Detect file extension, MIME type, family, and confidence.",
        "/detect_file_type <path>",
        detect_file_type_command,
    ),
    "/read_file_info": SlashTool(
        "/read_file_info",
        "Show file size and SHA-256 hash.",
        "/read_file_info <path>",
        read_file_info_command,
    ),
    "/normalize_text": SlashTool(
        "/normalize_text",
        "Normalize whitespace and control characters.",
        "/normalize_text <text>",
        normalize_text_command,
    ),
    "/extract_single_doc": SlashTool(
        "/extract_single_doc",
        "Extract one document and auto-save JSON artifacts.",
        "/extract_single_doc <path>",
        extract_single_doc_command,
    ),
    "/extract_docs": SlashTool(
        "/extract_docs",
        "Extract all supported documents in the attached folder and auto-save JSON artifacts.",
        "/extract_docs",
        extract_docs_command,
    ),
    "/build_doc_map": SlashTool(
        "/build_doc_map",
        "Build and auto-save a structural document map.",
        "/build_doc_map",
        build_doc_map_command,
    ),
    "/workspace_status": SlashTool(
        "/workspace_status",
        "Show which document-pipeline artifacts are available.",
        "/workspace_status",
        workspace_status_command,
    ),
    "/generate_markdown": SlashTool(
        "/generate_markdown",
        "Generate and auto-save a deterministic markdown evidence report.",
        "/generate_markdown",
        generate_markdown_command,
    ),
}


def run_slash_command(
    command_text: str,
    working_folder: str | Path | None,
    context: SlashToolContext,
    progress: SlashProgressCallback | None = None,
) -> SlashToolResult | None:
    stripped = command_text.strip()
    if not stripped.startswith("/"):
        return None
    try:
        parts = shlex.split(stripped)
    except ValueError as exc:
        return error_result(f"malformed slash command: {exc}")
    if not parts:
        return None
    command = parts[0].lower()
    tool = SLASH_TOOLS.get(command)
    if tool is None:
        return error_result(f"unknown slash command '{command}'. Run /help to see available commands.", command)
    try:
        result = tool.handler(parts[1:], working_folder, context, progress)
        context.last_tool_name = result.tool_name
        context.last_tool_summary = result.history_text
        context.saved_files = list(result.saved_files)
        return result
    except SlashToolError as exc:
        return error_result(str(exc), command)
    except Exception as exc:
        return error_result(f"unexpected failure while running {command}: {exc}", command)
