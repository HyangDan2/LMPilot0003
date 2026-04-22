from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Protocol

from src.document_pipeline.schemas import ExtractedBlock, ExtractedDocument


DETAIL_SUMMARY_BATCH_SIZE = 10
DETAIL_SUMMARY_MAX_ITEMS = 100
DETAIL_SUMMARY_MAX_ITEM_CHARS = 3000
DETAIL_SUMMARY_FALLBACK_CHARS = 700

ProgressCallback = Callable[[str, str], None]


class DetailSummaryLLMClient(Protocol):
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class DetailSummaryGroup:
    item_id: str
    document_id: str
    filename: str
    location_type: str
    location_label: str
    sort_key: tuple[int, str]
    block_ids: list[str]
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "location_type": self.location_type,
            "location_label": self.location_label,
            "block_ids": list(self.block_ids),
            "char_count": self.char_count,
            "text": self.text,
        }


@dataclass(frozen=True)
class DetailSummaryItem:
    item_id: str
    document_id: str
    filename: str
    location_type: str
    location_label: str
    block_ids: list[str]
    summary: str
    key_points: list[str] = field(default_factory=list)
    used_llm: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "location_type": self.location_type,
            "location_label": self.location_label,
            "block_ids": list(self.block_ids),
            "summary": self.summary,
            "key_points": list(self.key_points),
            "used_llm": self.used_llm,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class DetailSummaryResult:
    enabled: bool
    groups: list[DetailSummaryGroup] = field(default_factory=list)
    summaries: list[DetailSummaryItem] = field(default_factory=list)
    used_llm: bool = False
    fallback_reason: str = ""
    truncated: bool = False
    max_items: int = DETAIL_SUMMARY_MAX_ITEMS
    batch_size: int = DETAIL_SUMMARY_BATCH_SIZE

    @property
    def summary_count(self) -> int:
        return len(self.summaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "used_llm": self.used_llm,
            "fallback_reason": self.fallback_reason,
            "summary_count": self.summary_count,
            "group_count": len(self.groups),
            "truncated": self.truncated,
            "max_items": self.max_items,
            "batch_size": self.batch_size,
            "groups": [group.to_dict() for group in self.groups],
            "summaries": [summary.to_dict() for summary in self.summaries],
        }


def generate_detail_summaries(
    documents: list[ExtractedDocument],
    llm_client: DetailSummaryLLMClient | None,
    *,
    enabled: bool,
    query: str = "",
    batch_size: int = DETAIL_SUMMARY_BATCH_SIZE,
    max_items: int = DETAIL_SUMMARY_MAX_ITEMS,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> DetailSummaryResult:
    if not enabled:
        return DetailSummaryResult(enabled=False)
    _check_cancelled(cancel_event)
    _emit(progress, "status", "Detail summaries enabled.\n")
    groups = build_detail_summary_groups(documents)
    truncated = len(groups) > max_items
    groups = groups[:max_items]
    _emit(progress, "status", f"Built {len(groups)} detail summary group(s).\n")
    if truncated:
        _emit(progress, "status", f"Detail summaries limited to first {max_items} item(s).\n")

    if not groups:
        return DetailSummaryResult(enabled=True, groups=[], summaries=[])

    total = len(groups)
    if llm_client is None:
        reason = "LLM client is not configured."
        summaries = []
        _emit(progress, "status", f"[detail] LLM unavailable, writing extractive fallback summaries for {total} item(s).\n")
        for index, group in enumerate(groups, start=1):
            _emit(progress, "status", f"[detail] Processing {_item_progress_label(group, index, total)} with extractive fallback...\n")
            summaries.append(_fallback_summary(group, reason))
            _emit(progress, "status", f"[detail] Completed {_item_progress_label(group, index, total)} with fallback.\n")
        _emit(progress, "status", f"[detail] Detail summaries complete: {len(summaries)}/{total} item(s), LLM used: no.\n")
        return DetailSummaryResult(
            enabled=True,
            groups=groups,
            summaries=summaries,
            used_llm=False,
            fallback_reason=reason,
            truncated=truncated,
            max_items=max_items,
            batch_size=batch_size,
        )

    summaries: list[DetailSummaryItem] = []
    used_llm = False
    fallback_reasons: list[str] = []
    _emit(progress, "status", f"Generating detail summaries with LLM batch size {batch_size}...\n")
    for start in range(0, total, batch_size):
        _check_cancelled(cancel_event)
        batch = groups[start : start + batch_size]
        for offset, group in enumerate(batch, start=start + 1):
            _emit(progress, "status", f"[detail] Processing {_item_progress_label(group, offset, total)}...\n")
        try:
            _emit(progress, "status", f"[detail] Sending {_batch_range_label(batch)} to LLM...\n")
            batch_summaries = _summarize_batch(llm_client, batch, query)
            used_llm = True
            _emit(progress, "status", f"[detail] Received LLM detail summaries for {_batch_range_label(batch)}.\n")
        except Exception as exc:  # noqa: BLE001 - artifact generation should degrade gracefully.
            reason = f"Detail summary LLM batch failed: {exc}"
            fallback_reasons.append(reason)
            _emit(progress, "status", f"[detail] LLM failed for {_batch_range_label(batch)}; using extractive fallback for that batch.\n")
            batch_summaries = [_fallback_summary(group, reason) for group in batch]
        summaries.extend(batch_summaries)
        summary_by_id = {summary.item_id: summary for summary in batch_summaries}
        for offset, group in enumerate(batch, start=start + 1):
            summary = summary_by_id.get(group.item_id)
            fallback_suffix = " with fallback" if summary is not None and summary.fallback_reason else ""
            _emit(progress, "status", f"[detail] Completed {_item_progress_label(group, offset, total)}{fallback_suffix}.\n")

    _emit(progress, "status", f"[detail] Detail summaries complete: {len(summaries)}/{total} item(s), LLM used: {'yes' if used_llm else 'no'}.\n")
    return DetailSummaryResult(
        enabled=True,
        groups=groups,
        summaries=summaries,
        used_llm=used_llm,
        fallback_reason="; ".join(fallback_reasons),
        truncated=truncated,
        max_items=max_items,
        batch_size=batch_size,
    )


def build_detail_summary_groups(documents: list[ExtractedDocument]) -> list[DetailSummaryGroup]:
    groups: dict[tuple[str, str, str], list[ExtractedBlock]] = {}
    order: dict[tuple[str, str, str], tuple[int, str]] = {}
    filenames: dict[str, str] = {document.document_id: document.source.filename for document in documents}
    for document in documents:
        for block in sorted(document.blocks, key=lambda item: item.order):
            location_type, location_label, sort_key = _block_location(block)
            key = (document.document_id, location_type, location_label)
            groups.setdefault(key, []).append(block)
            order.setdefault(key, sort_key)

    detail_groups: list[DetailSummaryGroup] = []
    for index, (key, blocks) in enumerate(groups.items(), start=1):
        document_id, location_type, location_label = key
        text = "\n".join(_format_block(block) for block in blocks if _block_text(block)).strip()
        if not text:
            continue
        item_id = f"{document_id}_{_slug(location_type)}_{_slug(location_label)}"
        detail_groups.append(
            DetailSummaryGroup(
                item_id=item_id,
                document_id=document_id,
                filename=filenames.get(document_id, document_id),
                location_type=location_type,
                location_label=location_label,
                sort_key=order.get(key, (index, location_label)),
                block_ids=[block.block_id for block in blocks],
                text=text,
            )
        )
    return sorted(detail_groups, key=lambda group: (group.filename, group.sort_key, group.item_id))


def detail_summaries_markdown(result: DetailSummaryResult) -> str:
    lines = ["# Detail Summaries", ""]
    if not result.enabled:
        lines.append("Detail summaries were not requested.")
        lines.append("")
        return "\n".join(lines)
    if result.truncated:
        lines.append(f"Limited to first {result.max_items} item(s).")
        lines.append("")
    for item in result.summaries:
        lines.append(f"## {item.filename} - {item.location_label}")
        lines.append("")
        lines.append(item.summary or "No summary text available.")
        lines.append("")
        if item.key_points:
            lines.append("Key points:")
            lines.extend(f"- {point}" for point in item.key_points)
            lines.append("")
        lines.append(f"Source blocks: {', '.join(item.block_ids) or 'none'}")
        if item.fallback_reason:
            lines.append(f"Fallback: {item.fallback_reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _summarize_batch(
    llm_client: DetailSummaryLLMClient,
    groups: list[DetailSummaryGroup],
    query: str,
) -> list[DetailSummaryItem]:
    payload = {
        "query": query,
        "items": [
            {
                "item_id": group.item_id,
                "filename": group.filename,
                "location": group.location_label,
                "block_ids": group.block_ids,
                "text": _truncate(group.text, DETAIL_SUMMARY_MAX_ITEM_CHARS),
            }
            for group in groups
        ],
    }
    prompt = (
        "Summarize each page, slide, sheet, or file item using only its provided text.\n"
        "Return JSON only with this shape: "
        '{"summaries":[{"item_id":"...","summary":"...","key_points":["..."]}]}.\n'
        "Keep each summary concise and evidence-grounded. Do not infer missing facts.\n\n"
        f"Items:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    content = llm_client.chat_completion(
        [
            {
                "role": "system",
                "content": "You write concise, evidence-grounded page and slide summaries as valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    parsed = _parse_json_object(content)
    raw_summaries = parsed.get("summaries")
    if not isinstance(raw_summaries, list):
        raise ValueError("detail summary response missing summaries list")
    by_id = {str(item.get("item_id", "")): item for item in raw_summaries if isinstance(item, dict)}
    summaries: list[DetailSummaryItem] = []
    for group in groups:
        raw = by_id.get(group.item_id, {})
        summary = str(raw.get("summary", "")).strip()
        key_points = raw.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = []
        clean_points = [str(point).strip() for point in key_points if str(point).strip()]
        if not summary:
            return [_fallback_summary(item, "LLM response omitted one or more detail summaries.") for item in groups]
        summaries.append(
            DetailSummaryItem(
                item_id=group.item_id,
                document_id=group.document_id,
                filename=group.filename,
                location_type=group.location_type,
                location_label=group.location_label,
                block_ids=group.block_ids,
                summary=summary,
                key_points=clean_points[:5],
                used_llm=True,
            )
        )
    return summaries


def _fallback_summary(group: DetailSummaryGroup, reason: str) -> DetailSummaryItem:
    text = " ".join(group.text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentence for sentence in sentences[:4] if sentence).strip()
    if not summary:
        summary = text
    return DetailSummaryItem(
        item_id=group.item_id,
        document_id=group.document_id,
        filename=group.filename,
        location_type=group.location_type,
        location_label=group.location_label,
        block_ids=group.block_ids,
        summary=_truncate(summary, DETAIL_SUMMARY_FALLBACK_CHARS),
        used_llm=False,
        fallback_reason=reason,
    )


def _item_progress_label(group: DetailSummaryGroup, index: int, total: int) -> str:
    return f"{group.location_label} {index}/{total}"


def _batch_range_label(groups: list[DetailSummaryGroup]) -> str:
    if not groups:
        return "0 item(s)"
    first = groups[0].location_label
    last = groups[-1].location_label
    if first == last:
        return first
    first_type = groups[0].location_type
    if all(group.location_type == first_type for group in groups):
        return f"{first} - {last}"
    return f"{first} - {last}"


def _block_location(block: ExtractedBlock) -> tuple[str, str, tuple[int, str]]:
    provenance = block.provenance
    if provenance.page is not None:
        return "page", f"Page {provenance.page}", (provenance.page, "")
    if provenance.slide is not None:
        return "slide", f"Slide {provenance.slide}", (provenance.slide, "")
    if provenance.sheet:
        return "sheet", f"Sheet {provenance.sheet}", (block.order, provenance.sheet)
    section = " > ".join(provenance.section_path).strip()
    if section:
        return "section", section, (block.order, section)
    return "file", "File", (block.order, "File")


def _format_block(block: ExtractedBlock) -> str:
    text = _block_text(block)
    return f"[{block.block_id} role={block.role or block.type}]\n{text}"


def _block_text(block: ExtractedBlock) -> str:
    if block.markdown.strip():
        return block.markdown.strip()
    if block.normalized_text.strip():
        return block.normalized_text.strip()
    if block.text.strip():
        return block.text.strip()
    if block.rows:
        return "\n".join(" | ".join(cell.strip() for cell in row) for row in block.rows)
    return ""


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("detail summary response must be a JSON object")
    return parsed


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "item"


def _emit(progress: ProgressCallback | None, kind: str, text: str) -> None:
    if progress is not None and text:
        progress(kind, text)


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Slash tool cancelled.")
