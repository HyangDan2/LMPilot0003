from __future__ import annotations

from pathlib import Path

from .errors import SlashToolError


def require_working_folder(working_folder: str | Path | None) -> Path:
    if working_folder is None or str(working_folder).strip() == "":
        raise SlashToolError("Attach a folder before using this slash tool.")
    root = Path(working_folder).expanduser().resolve()
    if not root.exists():
        raise SlashToolError(f"Attached folder does not exist: {root}")
    if not root.is_dir():
        raise SlashToolError(f"Attached path is not a folder: {root}")
    return root


def resolve_workspace_path(working_folder: Path, user_path: str) -> Path:
    if not user_path.strip():
        raise SlashToolError("A file path is required.")
    relative = Path(user_path)
    if relative.is_absolute():
        target = relative.expanduser().resolve()
    else:
        target = (working_folder / relative).expanduser().resolve()
    if target != working_folder and working_folder not in target.parents:
        raise SlashToolError("Path is outside the attached working folder.")
    return target
