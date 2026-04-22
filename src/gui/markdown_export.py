from __future__ import annotations

import re
from typing import Any


ROLE_LABELS = {
    "user": "You",
    "assistant": "Assistant",
    "tool": "Tool",
    "system": "System",
}


def format_chat_markdown(title: str, messages: list[dict[str, Any]]) -> str:
    heading = _clean_heading(title) or "Chat Export"
    parts = [f"# {heading}"]
    for message in messages:
        role = ROLE_LABELS.get(str(message.get("role", "")), "System")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"## {role}\n\n{content}")
    return "\n\n".join(parts).rstrip() + "\n"


def safe_markdown_filename(title: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9._ -]+", "_", title.strip())
    filename = re.sub(r"\s+", " ", filename).strip(" .")
    return filename or "chat-export"


def _clean_heading(title: str) -> str:
    return title.replace("\n", " ").strip()
