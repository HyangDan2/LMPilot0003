from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Protocol

from src.document_pipeline.schemas import ExtractedBlock, ExtractedDocument

from .select_evidence import NUMBER_RE, _block_text, _query_terms

ProgressCallback = Callable[[str, str], None]

LARGE_DOCUMENT_BLOCK_THRESHOLD = 80
GROUP_TARGET_CHARS = 24000
FINAL_GROUPED_CONTEXT_CHARS = 9000
MAX_RECURSIVE_EVIDENCE_GROUPS = 10
GROUP_PROGRESS_INTERVAL = 100
BUCKET_ORDER = ("overview", "methods", "data", "results", "limitations")

OVERVIEW_TERMS = (
    "abstract",
    "introduction",
    "overview",
    "background",
    "purpose",
    "objective",
    "summary",
    "초록",
    "개요",
    "배경",
    "목적",
)
METHOD_TERMS = (
    "method",
    "component",
    "architecture",
    "algorithm",
    "system",
    "workflow",
    "process",
    "design",
    "implementation",
    "model",
    "protocol",
    "interface",
    "module",
    "방법",
    "모듈",
    "시스템",
    "설계",
)
DATA_TERMS = (
    "figure",
    "fig.",
    "table",
    "caption",
    "data",
    "parameter",
    "specification",
    "표",
    "그림",
    "데이터",
)
RESULT_TERMS = (
    "result",
    "evaluation",
    "validation",
    "performance",
    "test",
    "finding",
    "experiment",
    "검증",
    "결과",
    "성능",
    "시험",
    "평가",
)
LIMITATION_TERMS = (
    "conclusion",
    "limitation",
    "constraint",
    "future",
    "risk",
    "issue",
    "discussion",
    "한계",
    "제약",
    "결론",
    "리스크",
    "문제",
)
BOILERPLATE_TERMS = ("copyright", "all rights reserved", "confidential", "footer", "header")


class RecursiveLLMClient(Protocol):
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    document_id: str
    block_ids: list[str]
    text: str
    char_count: int
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "document_id": self.document_id,
            "block_ids": list(self.block_ids),
            "text": self.text,
            "char_count": self.char_count,
            "index": self.index,
        }


@dataclass(frozen=True)
class GroupSummary:
    summary_id: str
    group_ids: list[str]
    block_ids: list[str]
    text: str
    fallback: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "group_ids": list(self.group_ids),
            "block_ids": list(self.block_ids),
            "text": self.text,
            "fallback": self.fallback,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class RankedEvidenceGroup:
    group: EvidenceGroup
    score: int
    bucket: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.group.to_dict()
        payload.update(
            {
                "score": self.score,
                "bucket": self.bucket,
                "reasons": list(self.reasons),
            }
        )
        return payload


@dataclass(frozen=True)
class RecursiveSummaryResult:
    mode: str
    groups: list[EvidenceGroup] = field(default_factory=list)
    group_summaries: list[GroupSummary] = field(default_factory=list)
    merge_levels: list[list[GroupSummary]] = field(default_factory=list)
    final_summary: str = ""
    ranked_groups: list[RankedEvidenceGroup] = field(default_factory=list)
    selected_groups: list[RankedEvidenceGroup] = field(default_factory=list)

    @property
    def merge_level_count(self) -> int:
        return len(self.merge_levels)

    @property
    def selected_group_count(self) -> int:
        return len(self.selected_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "groups": [group.to_dict() for group in self.groups],
            "group_summaries": [summary.to_dict() for summary in self.group_summaries],
            "merge_levels": [
                [summary.to_dict() for summary in level]
                for level in self.merge_levels
            ],
            "ranked_groups": [group.to_dict() for group in self.ranked_groups],
            "selected_groups": [group.to_dict() for group in self.selected_groups],
            "selected_group_count": self.selected_group_count,
            "final_summary": self.final_summary,
        }


def should_use_recursive_summary(documents: list[ExtractedDocument], llm_input_chars: int) -> bool:
    blocks = _candidate_blocks(documents)
    total_chars = sum(len(_block_text(block)) for block in blocks)
    return len(blocks) > LARGE_DOCUMENT_BLOCK_THRESHOLD or total_chars > llm_input_chars * 2


def build_evidence_groups(
    documents: list[ExtractedDocument],
    target_chars: int = GROUP_TARGET_CHARS,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> list[EvidenceGroup]:
    blocks = _candidate_blocks(documents)
    _emit(progress, "status", f"Building evidence groups from {len(blocks)} block(s)...\n")
    groups: list[EvidenceGroup] = []
    grouped_count = 0
    for document in documents:
        current: list[ExtractedBlock] = []
        current_chars = 0
        for block in _candidate_blocks([document]):
            _check_cancelled(cancel_event)
            grouped_count += 1
            if grouped_count % GROUP_PROGRESS_INTERVAL == 0:
                _emit(progress, "status", f"Grouping progress: processed {grouped_count}/{len(blocks)} block(s)...\n")
            text = _format_source_block(block)
            block_chars = len(text)
            if current and current_chars + block_chars > target_chars:
                groups.append(_make_group(document.document_id, len(groups) + 1, current))
                current = []
                current_chars = 0
            current.append(block)
            current_chars += block_chars
        if current:
            groups.append(_make_group(document.document_id, len(groups) + 1, current))
    return groups


def run_recursive_summary(
    documents: list[ExtractedDocument],
    llm_client: RecursiveLLMClient | None,
    query: str,
    llm_input_chars: int,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
    selected_block_ids: set[str] | None = None,
) -> RecursiveSummaryResult:
    del llm_client
    groups = build_evidence_groups(documents, progress=progress, cancel_event=cancel_event)
    if not groups:
        return RecursiveSummaryResult(mode="one-shot")

    _emit(progress, "status", f"Built {len(groups)} raw evidence group(s).\n")
    _emit(progress, "status", "Ranking evidence groups locally; no LLM calls before final generation.\n")
    ranked_groups = rank_evidence_groups(groups, query, selected_block_ids or set())
    selected_groups = select_top_evidence_groups(ranked_groups, MAX_RECURSIVE_EVIDENCE_GROUPS)
    selected_block_count = len({block_id for ranked in selected_groups for block_id in ranked.group.block_ids})
    selected_chars = sum(ranked.group.char_count for ranked in selected_groups)
    top_scores = ", ".join(f"{ranked.group.group_id}={ranked.score}" for ranked in selected_groups[:3])
    _emit(
        progress,
        "status",
        (
            f"Selected top {len(selected_groups)} evidence group(s), covering "
            f"{selected_block_count} block(s), {selected_chars} chars.\n"
        ),
    )
    if top_scores:
        _emit(progress, "status", f"Top group scores: {top_scores}.\n")

    final_summary = _format_grouped_evidence_context(selected_groups)
    return RecursiveSummaryResult(
        mode="ranked-groups",
        groups=groups,
        final_summary=final_summary,
        ranked_groups=ranked_groups,
        selected_groups=selected_groups,
    )


def rank_evidence_groups(
    groups: list[EvidenceGroup],
    query: str,
    selected_block_ids: set[str],
) -> list[RankedEvidenceGroup]:
    query_terms = _query_terms(query)
    ranked = [_score_group(group, query_terms, selected_block_ids) for group in groups]
    return sorted(ranked, key=lambda item: (item.score, -item.group.index), reverse=True)


def select_top_evidence_groups(
    ranked_groups: list[RankedEvidenceGroup],
    limit: int = MAX_RECURSIVE_EVIDENCE_GROUPS,
) -> list[RankedEvidenceGroup]:
    selected: list[RankedEvidenceGroup] = []
    selected_ids: set[str] = set()

    for bucket in BUCKET_ORDER:
        best = next((group for group in ranked_groups if group.bucket == bucket and group.group.group_id not in selected_ids), None)
        if best is not None:
            selected.append(best)
            selected_ids.add(best.group.group_id)
        if len(selected) >= limit:
            return selected[:limit]

    for group in ranked_groups:
        if group.group.group_id in selected_ids:
            continue
        if _too_close_to_selected(group, selected):
            continue
        selected.append(group)
        selected_ids.add(group.group.group_id)
        if len(selected) >= limit:
            return selected

    for group in ranked_groups:
        if group.group.group_id in selected_ids:
            continue
        selected.append(group)
        selected_ids.add(group.group.group_id)
        if len(selected) >= limit:
            break
    return selected


def _score_group(
    group: EvidenceGroup,
    query_terms: set[str],
    selected_block_ids: set[str],
) -> RankedEvidenceGroup:
    text = group.text.lower()
    score = 0
    reasons: list[str] = []

    query_matches = sum(1 for term in query_terms if term in text)
    if query_matches:
        score += query_matches * 8
        reasons.append(f"query term matches: {query_matches}")

    bucket = _group_bucket(text)
    bucket_scores = {
        "overview": 6,
        "methods": 8,
        "data": 8,
        "results": 10,
        "limitations": 8,
        "remaining": 0,
    }
    if bucket_scores[bucket]:
        score += bucket_scores[bucket]
        reasons.append(f"{bucket} signal")

    if NUMBER_RE.search(text):
        score += 10
        reasons.append("quantitative value")
    elif any(char.isdigit() for char in text):
        score += 5
        reasons.append("numeric text")

    overlaps = selected_block_ids.intersection(group.block_ids)
    if overlaps:
        score += 12 + max(0, len(overlaps) - 1) * 3
        reasons.append(f"selected evidence overlap: {len(overlaps)}")

    if group.char_count > 600:
        score += 4
        reasons.append("substantive length")
    if _looks_like_boilerplate(text):
        score -= 15
        reasons.append("boilerplate penalty")

    return RankedEvidenceGroup(group=group, score=score, bucket=bucket, reasons=reasons)


def _group_bucket(text: str) -> str:
    if _has_any(text, OVERVIEW_TERMS):
        return "overview"
    if _has_any(text, DATA_TERMS):
        return "data"
    if _has_any(text, METHOD_TERMS):
        return "methods"
    if _has_any(text, RESULT_TERMS):
        return "results"
    if _has_any(text, LIMITATION_TERMS):
        return "limitations"
    return "remaining"


def _too_close_to_selected(group: RankedEvidenceGroup, selected: list[RankedEvidenceGroup]) -> bool:
    return any(
        group.group.document_id == item.group.document_id
        and abs(group.group.index - item.group.index) <= 1
        for item in selected
    )


def _format_grouped_evidence_context(groups: list[RankedEvidenceGroup]) -> str:
    parts = []
    for group in groups:
        reasons = ", ".join(group.reasons) or "general relevance"
        parts.append(
            f"## {group.group.group_id} | score={group.score} | bucket={group.bucket}\n"
            f"Reasons: {reasons}\n"
            f"Block IDs: {', '.join(group.group.block_ids)}\n\n"
            f"{group.group.text}"
        )
    return _truncate("\n\n".join(parts), FINAL_GROUPED_CONTEXT_CHARS)


def _candidate_blocks(documents: list[ExtractedDocument]) -> list[ExtractedBlock]:
    result: list[ExtractedBlock] = []
    for document in documents:
        has_section_blocks = any(block.role != "document_text" and _block_text(block) for block in document.blocks)
        for block in sorted(document.blocks, key=lambda item: item.order):
            if has_section_blocks and block.role == "document_text":
                continue
            if _block_text(block):
                result.append(block)
    return result


def _make_group(document_id: str, index: int, blocks: list[ExtractedBlock]) -> EvidenceGroup:
    text = "\n\n".join(_format_source_block(block) for block in blocks)
    return EvidenceGroup(
        group_id=f"group_{index:04d}",
        document_id=document_id,
        block_ids=[block.block_id for block in blocks],
        text=text,
        char_count=len(text),
        index=index,
    )


def _format_source_block(block: ExtractedBlock) -> str:
    page = f" page={block.provenance.page}" if block.provenance.page is not None else ""
    slide = f" slide={block.provenance.slide}" if block.provenance.slide is not None else ""
    return (
        f"[{block.block_id}{page}{slide} role={block.role or block.type}]\n"
        f"{_block_text(block)}"
    )


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _looks_like_boilerplate(text: str) -> bool:
    if not any(term in text for term in BOILERPLATE_TERMS):
        return False
    return len(text) < 1200 or _repeated_line_ratio(text) > 0.4


def _repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _emit(progress: ProgressCallback | None, kind: str, text: str) -> None:
    if progress is not None and text:
        progress(kind, text)


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Slash tool cancelled.")
