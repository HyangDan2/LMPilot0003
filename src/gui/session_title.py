from __future__ import annotations

import re

from .token_handler import normalize_prompt_text


DEFAULT_SESSION_TITLE = "New Chat"
DEFAULT_ATTACHMENT_PROMPT = "Please review the attached file(s)."
TITLE_WORD_RE = re.compile(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣'_-]*")


def derive_session_title(text: str, max_words: int = 8, max_chars: int = 60) -> str:
    text = normalize_prompt_text(text)
    words = TITLE_WORD_RE.findall(text)
    title = " ".join(words[:max_words]).strip()
    if not title:
        return DEFAULT_SESSION_TITLE
    if len(title) > max_chars:
        title = title[:max_chars].rstrip()
    return title or DEFAULT_SESSION_TITLE


def derive_session_title_from_input(
    user_text: str,
    attachment_filenames: list[str] | None = None,
    max_chars: int = 60,
) -> str:
    normalized_text = normalize_prompt_text(user_text)
    if normalized_text and normalized_text != DEFAULT_ATTACHMENT_PROMPT:
        return derive_session_title(normalized_text, max_chars=max_chars)

    filenames = [filename.strip() for filename in attachment_filenames or [] if filename.strip()]
    if not filenames:
        return DEFAULT_SESSION_TITLE
    if len(filenames) == 1:
        return derive_session_title(f"Review {filenames[0]}", max_chars=max_chars)
    return derive_session_title(f"Review {filenames[0]} and {len(filenames) - 1} more files", max_chars=max_chars)
