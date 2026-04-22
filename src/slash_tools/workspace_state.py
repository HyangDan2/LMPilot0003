from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.document_pipeline.storage import pipeline_output_dir, summary_output_dir


@dataclass(frozen=True)
class WorkspaceState:
    working_folder: str
    extracted_documents_path: str | None = None
    extraction_manifest_path: str | None = None
    document_map_path: str | None = None
    generated_markdown_path: str | None = None
    document_summaries_path: str | None = None
    workspace_summary_path: str | None = None
    document_count: int = 0
    next_actions: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "Workspace status",
            "",
            "Attached folder:",
            self.working_folder,
            "",
            "Artifacts:",
            f"- extracted_documents.json: {_found(self.extracted_documents_path, self.document_count, 'document(s)')}",
            f"- extraction_manifest.json: {_found(self.extraction_manifest_path)}",
            f"- document_map.json: {_found(self.document_map_path)}",
            f"- generated_report.md: {_found(self.generated_markdown_path)}",
            f"- document_summaries.json: {_found(self.document_summaries_path)}",
            f"- workspace_summary.md: {_found(self.workspace_summary_path)}",
        ]
        if self.next_actions:
            lines.extend(["", "Suggested next:"])
            lines.extend(f"- {action}" for action in self.next_actions)
        return "\n".join(lines)


def load_workspace_state(working_folder: Path) -> WorkspaceState:
    root = working_folder.expanduser().resolve()
    output_dir = pipeline_output_dir(root)
    summary_dir = summary_output_dir(root)
    extracted_path = output_dir / "extracted_documents.json"
    manifest_path = output_dir / "extraction_manifest.json"
    doc_map_path = output_dir / "document_map.json"
    report_path = output_dir / "generated_report.md"
    latest_summary_dir = _latest_summary_run_dir(summary_dir)
    document_summaries_path = latest_summary_dir / "document_summaries.json" if latest_summary_dir is not None else None
    workspace_summary_path = latest_summary_dir / "workspace_summary.md" if latest_summary_dir is not None else None
    document_count = _document_count(extracted_path)
    next_actions = _next_actions(
        has_documents=extracted_path.exists() and document_count > 0,
        has_map=doc_map_path.exists(),
        has_report=report_path.exists(),
        has_workspace_summary=workspace_summary_path is not None and workspace_summary_path.exists(),
    )
    return WorkspaceState(
        working_folder=str(root),
        extracted_documents_path=_path_if_exists(extracted_path, root),
        extraction_manifest_path=_path_if_exists(manifest_path, root),
        document_map_path=_path_if_exists(doc_map_path, root),
        generated_markdown_path=_path_if_exists(report_path, root),
        document_summaries_path=_path_if_exists(document_summaries_path, root),
        workspace_summary_path=_path_if_exists(workspace_summary_path, root),
        document_count=document_count,
        next_actions=next_actions,
    )


def _path_if_exists(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _document_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    documents = payload.get("documents") if isinstance(payload, dict) else None
    return len(documents) if isinstance(documents, list) else 0


def _next_actions(
    has_documents: bool,
    has_map: bool,
    has_report: bool,
    has_workspace_summary: bool,
) -> list[str]:
    if not has_documents:
        return ["/extract_docs"]
    if not has_workspace_summary:
        return ["/summarize_doc", "/build_doc_map", "/generate_markdown"]
    if not has_map:
        return ["/build_doc_map", "/generate_markdown"]
    if not has_report:
        return ["/generate_markdown"]
    return [
        "Ask a normal question about the generated summary",
        "Ask a normal question about the generated report",
        "Re-run /extract_docs if source files changed",
    ]


def _found(path: str | None, count: int | None = None, unit: str = "") -> str:
    if path is None:
        return "missing"
    if count is None:
        return f"found ({path})"
    return f"found, {count} {unit} ({path})"


def _latest_summary_run_dir(summary_dir: Path) -> Path | None:
    if not summary_dir.exists():
        return None
    candidates = [child for child in summary_dir.iterdir() if child.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda child: (child.stat().st_mtime_ns, child.name))
