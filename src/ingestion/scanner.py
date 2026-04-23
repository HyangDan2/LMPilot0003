from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {".pptx", ".docx", ".xlsx", ".pdf"}
DEFAULT_EXCLUDED_DIR_NAMES = {"HD2docpipe", "__pycache__", ".git", ".venv", "node_modules"}


def scan_supported_files(
    root: Path,
    excluded_dirs: set[Path] | None = None,
    excluded_dir_names: set[str] | None = None,
) -> list[Path]:
    """Recursively scan root for supported document files."""

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Working directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Working path is not a directory: {root}")

    excluded_roots = {path.expanduser().resolve() for path in excluded_dirs or set()}
    excluded_names = excluded_dir_names or DEFAULT_EXCLUDED_DIR_NAMES
    files: list[Path] = []
    for path in root.rglob("*"):
        resolved = path.resolve()
        if any(resolved == excluded or excluded in resolved.parents for excluded in excluded_roots):
            continue
        if any(part in excluded_names for part in path.relative_to(root).parts[:-1]):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$"):
            files.append(resolved)
    return sorted(files, key=lambda path: str(path).lower())
