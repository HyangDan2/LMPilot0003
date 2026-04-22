from __future__ import annotations

import re

from src.document_pipeline.schemas import ExtractedBlock, ExtractedDocument, OutputPlan, SelectedEvidence
from src.document_pipeline.schemas import SelectedEvidenceBlock

ENGINEERING_TERMS = {
    "analysis",
    "calculation",
    "constraint",
    "design",
    "failure",
    "issue",
    "performance",
    "requirement",
    "result",
    "risk",
    "spec",
    "test",
    "validation",
    "검증",
    "결과",
    "리스크",
    "성능",
    "요구",
}
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|mm|cm|m|kg|g|ms|s|sec|min|h|hr|v|a|w|kw|kwh|pa|kpa|mpa|hz|rpm|°c|c)?\b", re.IGNORECASE)
MAX_EVIDENCE_BLOCKS_PER_DOCUMENT = 12
BUCKET_QUOTA = 2
BUCKET_ORDER = ("overview", "methods", "data", "results", "limitations", "remaining")


def select_evidence_blocks(
    documents: list[ExtractedDocument],
    output_plan: OutputPlan,
    query: str,
    max_input_chars: int = 12000,
) -> SelectedEvidence:
    """Select prompt-sized evidence from extracted blocks without chunking."""

    blocks = _candidate_blocks(documents)
    if not blocks:
        return SelectedEvidence(query=query, max_input_chars=max_input_chars, blocks=[])

    docs_by_id = {document.document_id: document for document in documents}
    planned_block_ids = [block_id for section in output_plan.sections for block_id in section.source_block_ids]
    terms = _query_terms(query)
    ordered = _ordered_blocks(blocks, planned_block_ids, terms)
    diverse_ordered = _diverse_block_order(blocks, ordered)

    selected: list[SelectedEvidenceBlock] = []
    selected_blocks: list[ExtractedBlock] = []
    used_chars = 0
    skipped_oversized = 0
    per_document_counts: dict[str, int] = {}
    budget = max(800, max_input_chars)
    for block in diverse_ordered:
        if (
            per_document_counts.get(block.document_id, 0) >= MAX_EVIDENCE_BLOCKS_PER_DOCUMENT
            and len(per_document_counts) > 1
        ):
            continue
        if any(selected_block.block_id == block.block_id for selected_block in selected_blocks):
            continue
        document = docs_by_id.get(block.document_id)
        source_filename = document.source.filename if document is not None else ""
        text = _block_text(block)
        evidence = SelectedEvidenceBlock(
            document_id=block.document_id,
            block_id=block.block_id,
            source_filename=source_filename,
            role=block.role or block.type,
            text=text,
            provenance=block.provenance,
            score=_block_score(block, terms, planned_block_ids),
        )
        formatted_len = len(_format_evidence_block(evidence))
        if selected and used_chars + formatted_len > budget:
            skipped_oversized += 1
            continue
        selected.append(evidence)
        selected_blocks.append(block)
        per_document_counts[block.document_id] = per_document_counts.get(block.document_id, 0) + 1
        used_chars += formatted_len
    if not selected:
        block = blocks[0]
        document = docs_by_id.get(block.document_id)
        selected.append(
            SelectedEvidenceBlock(
                document_id=block.document_id,
                block_id=block.block_id,
                source_filename=document.source.filename if document is not None else "",
                role=block.role or block.type,
                text=_block_text(block),
                provenance=block.provenance,
                score=_block_score(block, terms, planned_block_ids),
            )
        )
        selected_blocks.append(block)
    selected_by_id = {block.block_id: evidence for block, evidence in zip(selected_blocks, selected, strict=True)}
    ordered_selected_blocks = sorted(selected_blocks, key=lambda block: (block.document_id, block.order))
    ordered_selected = [selected_by_id[block.block_id] for block in ordered_selected_blocks]
    return SelectedEvidence(query=query, max_input_chars=max_input_chars, blocks=ordered_selected)


def _candidate_blocks(documents: list[ExtractedDocument]) -> list[ExtractedBlock]:
    result: list[ExtractedBlock] = []
    for document in documents:
        has_specific_blocks = any(block.role != "document_text" and _block_text(block) for block in document.blocks)
        for block in document.blocks:
            if has_specific_blocks and block.role == "document_text":
                continue
            if _block_text(block):
                result.append(block)
    return result


def _diverse_block_order(blocks: list[ExtractedBlock], ordered: list[ExtractedBlock]) -> list[ExtractedBlock]:
    buckets: dict[str, list[ExtractedBlock]] = {bucket: [] for bucket in BUCKET_ORDER}
    for block in ordered:
        buckets[_block_bucket(block)].append(block)

    chosen: list[ExtractedBlock] = []
    chosen_ids: set[str] = set()
    for bucket in BUCKET_ORDER:
        for block in buckets[bucket][:BUCKET_QUOTA]:
            _append_with_context(block, blocks, chosen, chosen_ids)
    for block in ordered:
        _append_with_context(block, blocks, chosen, chosen_ids)
    return chosen


def _append_with_context(
    block: ExtractedBlock,
    blocks: list[ExtractedBlock],
    chosen: list[ExtractedBlock],
    chosen_ids: set[str],
) -> None:
    for context_block in _adjacent_context_blocks(block, blocks):
        if context_block.block_id not in chosen_ids:
            chosen.append(context_block)
            chosen_ids.add(context_block.block_id)
    if block.block_id not in chosen_ids:
        chosen.append(block)
        chosen_ids.add(block.block_id)


def _adjacent_context_blocks(block: ExtractedBlock, blocks: list[ExtractedBlock]) -> list[ExtractedBlock]:
    context: list[ExtractedBlock] = []
    prior_blocks = [
        candidate
        for candidate in blocks
        if candidate.document_id == block.document_id and 0 < block.order - candidate.order <= 3
    ]
    for candidate in sorted(prior_blocks, key=lambda item: item.order, reverse=True):
        if candidate.role in {"title", "heading", "section"} and _block_text(candidate):
            context.append(candidate)
            break
    return context


def _block_bucket(block: ExtractedBlock) -> str:
    text = _block_text(block).lower()
    role = (block.role or "").lower()
    block_type = (block.type or "").lower()
    if role in {"title", "heading"} or _has_any(text, ("abstract", "introduction", "overview", "background", "purpose", "objective", "summary", "초록", "개요", "배경", "목적")):
        return "overview"
    if block_type == "table" or block.rows or block.markdown or _has_any(text, ("figure", "fig.", "table", "caption", "data", "parameter", "specification", "표", "그림", "데이터")):
        return "data"
    if _has_any(text, ("method", "component", "architecture", "algorithm", "system", "workflow", "process", "design", "implementation", "model", "protocol", "interface", "module", "방법", "모듈", "시스템", "설계")):
        return "methods"
    if _has_any(text, ("result", "evaluation", "validation", "performance", "test", "finding", "experiment", "검증", "결과", "성능", "시험", "평가")):
        return "results"
    if _has_any(text, ("conclusion", "limitation", "constraint", "future", "risk", "issue", "discussion", "한계", "제약", "결론", "리스크", "문제")):
        return "limitations"
    return "remaining"


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def format_selected_evidence(evidence: SelectedEvidence, max_text_chars: int = 1200) -> str:
    return "\n\n".join(_format_evidence_block(block, max_text_chars=max_text_chars) for block in evidence.blocks)


def _ordered_blocks(
    blocks: list[ExtractedBlock],
    planned_block_ids: list[str],
    terms: set[str],
) -> list[ExtractedBlock]:
    planned_lookup = {block_id: index for index, block_id in enumerate(planned_block_ids)}
    return sorted(
        blocks,
        key=lambda block: (
            _block_score(block, terms, planned_block_ids),
            -planned_lookup.get(block.block_id, len(planned_lookup)),
            -block.order,
        ),
        reverse=True,
    )


def _block_score(block: ExtractedBlock, terms: set[str], planned_block_ids: list[str]) -> int:
    text = _block_text(block).lower()
    score = 0
    if block.block_id in planned_block_ids:
        score += 10
    for term in terms:
        if term in text:
            score += 3
    for term in ENGINEERING_TERMS:
        if term in text:
            score += 2
    if NUMBER_RE.search(text):
        score += 3
    if block.type in {"table"} or block.rows or block.markdown:
        score += 2
    if block.role in {"title", "heading", "section"}:
        score += 1
    return score


def _block_text(block: ExtractedBlock) -> str:
    return (block.normalized_text or block.markdown or block.text).strip()


def _format_evidence_block(block: SelectedEvidenceBlock, max_text_chars: int = 1200) -> str:
    return (
        f"source: {block.source_filename}\n"
        f"document_id: {block.document_id}\n"
        f"block_id: {block.block_id}\n"
        f"role: {block.role}\n"
        f"score: {block.score}\n"
        f"evidence:\n{_truncate(' '.join(block.text.split()), max_text_chars)}"
    )


def _query_terms(query: str) -> set[str]:
    terms = set(re.findall(r"[A-Za-z0-9가-힣_]{3,}", query.lower()))
    stopwords = {"generate", "report", "summary", "summarize", "about", "folder", "output", "this", "that"}
    return {term for term in terms if term not in stopwords}


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."
