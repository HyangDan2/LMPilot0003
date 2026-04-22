from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MAX_ARTIFACT_READ_CHARS = 20000
MAX_ARTIFACT_LIST_ENTRIES = 200

ARTIFACT_TAG_RE = re.compile(
    r"\[(read_file|read_output|list_file|list_outputs)\](.*?)\[/\1\]",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ArtifactRequest:
    command: str
    requested_path: str


@dataclass(frozen=True)
class ArtifactToolResult:
    request: ArtifactRequest
    ok: bool
    text: str


ARTIFACT_ACCESS_INSTRUCTION = """Generated artifact access:
If you need previously generated output files, request them with one of these tags:
[read_output] document_pipeline/generated_report.md [/read_output]
[list_outputs] document_pipeline [/list_outputs]
Qwen-style aliases are also supported for generated outputs, such as:
[read_file] llm/document_pipeline/generated_report.md [/read_file]
[read_file] llm_output/document_pipeline/workspace_20260422_231501/workspace_summary.md [/read_file]
Only generated files under llm_result/ and llm_output/ are available through these tags.
Do not say you lack authority to read generated artifacts; request them with the tag when needed."""


def extract_artifact_requests(text: str) -> list[ArtifactRequest]:
    requests: list[ArtifactRequest] = []
    for match in ARTIFACT_TAG_RE.finditer(text):
        command = match.group(1).strip().lower()
        requested_path = match.group(2).strip()
        if requested_path:
            requests.append(ArtifactRequest(command=command, requested_path=requested_path))
    return requests


def execute_artifact_requests(
    working_folder: str | Path,
    requests: list[ArtifactRequest],
    max_read_chars: int = MAX_ARTIFACT_READ_CHARS,
    max_list_entries: int = MAX_ARTIFACT_LIST_ENTRIES,
) -> list[ArtifactToolResult]:
    return [
        execute_artifact_request(working_folder, request, max_read_chars, max_list_entries)
        for request in requests
    ]


def execute_artifact_request(
    working_folder: str | Path,
    request: ArtifactRequest,
    max_read_chars: int = MAX_ARTIFACT_READ_CHARS,
    max_list_entries: int = MAX_ARTIFACT_LIST_ENTRIES,
) -> ArtifactToolResult:
    try:
        target = resolve_output_artifact_path(working_folder, request.requested_path)
        if request.command in {"list_file", "list_outputs"}:
            return _list_artifact(target, request, max_list_entries)
        return _read_artifact(target, request, max_read_chars)
    except Exception as exc:
        return ArtifactToolResult(
            request=request,
            ok=False,
            text=f"Artifact request failed for `{request.requested_path}`: {exc}",
        )


def format_artifact_results(results: list[ArtifactToolResult]) -> str:
    blocks: list[str] = []
    for result in results:
        status = "ok" if result.ok else "error"
        blocks.append(
            "\n".join(
                [
                    f"[artifact_result status={status}]",
                    f"request: {result.request.command} {result.request.requested_path}",
                    result.text.rstrip(),
                    "[/artifact_result]",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_artifact_followup_messages(
    messages: list[dict],
    assistant_request_text: str,
    results: list[ArtifactToolResult],
) -> list[dict]:
    followup = [dict(message) for message in messages]
    followup.append({"role": "assistant", "content": assistant_request_text})
    followup.append(
        {
            "role": "user",
            "content": (
                "The generated artifact request(s) were executed locally.\n"
                "Use the artifact result content below to answer the user's original request.\n"
                "Do not emit another artifact tag unless another generated artifact is required.\n\n"
                f"{format_artifact_results(results)}"
            ),
        }
    )
    return followup


def resolve_output_artifact_path(working_folder: str | Path, requested_path: str) -> Path:
    root = Path(working_folder).expanduser().resolve()
    normalized = _normalize_artifact_request_path(requested_path)
    target = (root / normalized).expanduser().resolve()
    allowed_roots = [(root / "llm_result").resolve(), (root / "llm_output").resolve()]
    if not any(target == artifact_root or artifact_root in target.parents for artifact_root in allowed_roots):
        raise ValueError("Generated artifact access is limited to llm_result/ and llm_output/.")
    return target


def _normalize_artifact_request_path(requested_path: str) -> Path:
    raw = requested_path.strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if raw.startswith("/"):
        raise ValueError("Absolute paths are not allowed.")
    if raw.startswith("llm/"):
        raw = "llm_result/" + raw[len("llm/") :]
    elif raw == "llm":
        raw = "llm_result"
    elif raw.startswith("llm_output/") or raw == "llm_output":
        raw = raw
    elif raw.startswith("document_pipeline/") or raw == "document_pipeline":
        raw = "llm_result/" + raw
    elif not raw.startswith("llm_result/") and raw != "llm_result":
        raw = "llm_result/" + raw
    path = Path(raw)
    if ".." in path.parts:
        raise ValueError("Path traversal is not allowed.")
    return path


def _read_artifact(path: Path, request: ArtifactRequest, max_read_chars: int) -> ArtifactToolResult:
    if not path.exists():
        raise FileNotFoundError(path.name)
    if not path.is_file():
        raise IsADirectoryError(path.name)
    content = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(content) > max_read_chars
    if truncated:
        content = content[:max_read_chars].rstrip() + "\n\n[truncated]"
    return ArtifactToolResult(
        request=request,
        ok=True,
        text=f"path: {path}\n\n{content}",
    )


def _list_artifact(path: Path, request: ArtifactRequest, max_list_entries: int) -> ArtifactToolResult:
    if not path.exists():
        raise FileNotFoundError(path.name)
    if path.is_file():
        entries = [path.name]
    else:
        children = sorted(path.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))
        entries = [f"{child.name}/" if child.is_dir() else child.name for child in children[:max_list_entries]]
        if len(children) > max_list_entries:
            entries.append(f"[{len(children) - max_list_entries} more item(s) omitted]")
    return ArtifactToolResult(
        request=request,
        ok=True,
        text="path: " + str(path) + "\n\n" + "\n".join(f"- {entry}" for entry in entries),
    )
