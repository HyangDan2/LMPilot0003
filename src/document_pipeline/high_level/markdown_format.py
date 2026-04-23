from __future__ import annotations

import re


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9가-힣])")


def sentence_per_line_markdown(markdown: str) -> str:
    """Split plain Markdown paragraphs so each sentence starts on a new line."""

    lines = markdown.splitlines()
    formatted: list[str] = []
    paragraph: list[str] = []
    in_code_fence = False

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = " ".join(line.strip() for line in paragraph if line.strip())
        formatted.extend(_split_sentences(text))
        paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            in_code_fence = not in_code_fence
            formatted.append(line)
            continue
        if in_code_fence:
            formatted.append(line)
            continue
        if not stripped:
            flush_paragraph()
            formatted.append(line)
            continue
        list_lines = _split_list_item_sentences(line)
        if list_lines is not None:
            flush_paragraph()
            formatted.extend(list_lines)
            continue
        if _is_structural_markdown(stripped):
            flush_paragraph()
            formatted.append(line)
            continue
        paragraph.append(line)
    flush_paragraph()
    return "\n".join(formatted).rstrip() + "\n"


def _is_structural_markdown(stripped: str) -> bool:
    return (
        stripped.startswith("#")
        or stripped.startswith("|")
        or stripped.startswith(">")
    )


def _split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(text) if sentence.strip()]
    return sentences or [text]


def _split_list_item_sentences(line: str) -> list[str] | None:
    match = re.match(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$", line)
    if match is None:
        return None
    prefix, body = match.groups()
    sentences = _split_sentences(body.strip())
    if len(sentences) <= 1:
        return [line]
    continuation_prefix = " " * len(prefix)
    return [f"{prefix}{sentences[0]}", *[f"{continuation_prefix}{sentence}" for sentence in sentences[1:]]]
