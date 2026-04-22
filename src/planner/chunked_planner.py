from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.schemas import PresentationPlan
from src.planner.llm_client import LLMClientError, LLMSettings, OpenAICompatibleLLMClient
from src.planner.planner import PlannerError, parse_presentation_plan
from src.planner.prompts import (
    CHUNK_SUMMARY_SYSTEM_PROMPT,
    FINAL_FROM_SUMMARIES_SYSTEM_PROMPT,
    build_chunk_summary_prompt,
    build_final_summary_prompt,
)
from src.utils.io import ensure_dir, save_json


class ChunkedPlannerError(Exception):
    """Raised when adaptive chunked planning cannot produce a plan."""


@dataclass(frozen=True)
class ChunkedPlannerSettings:
    chunk_chars: int = 6000
    min_chunk_chars: int = 800
    max_retries: int = 3
    intermediate_max_tokens: int = 1024
    final_max_tokens: int = 2048
    allow_response_format_retry: bool = True
    enable_local_fallback: bool = True


@dataclass(frozen=True)
class ChunkedPlannerResult:
    plan: PresentationPlan
    summary_json: Path
    attempts_json: Path
    chunk_count: int
    fallback_count: int
    attempts: list[dict[str, Any]] = field(default_factory=list)


def create_chunked_presentation_plan(
    *,
    llm_settings: LLMSettings,
    planner_settings: ChunkedPlannerSettings,
    user_goal: str,
    knowledge_map_md: str,
    artifact_dir: Path,
) -> ChunkedPlannerResult:
    artifact_dir = ensure_dir(artifact_dir)
    chunk_dir = ensure_dir(artifact_dir / "planner_chunks")
    attempts: list[dict[str, Any]] = []
    chunks = split_text_into_chunks(knowledge_map_md, max(planner_settings.chunk_chars, 1))

    summaries: list[dict[str, Any]] = []
    fallback_count = 0
    for index, chunk in enumerate(chunks, start=1):
        summary_path = chunk_dir / f"chunk_{index:03d}_summary.json"
        cached = _read_cached_summary(summary_path)
        if cached is not None:
            summaries.append(cached)
            attempts.append({"stage": "chunk", "chunk": index, "status": "reused", "path": str(summary_path)})
            if cached.get("fallback"):
                fallback_count += 1
            continue

        summary = _summarize_chunk_adaptive(
            llm_settings=llm_settings,
            planner_settings=planner_settings,
            user_goal=user_goal,
            chunk_text=chunk,
            chunk_label=f"{index:03d}",
            attempt_dir=chunk_dir,
            attempts=attempts,
            depth=0,
        )
        save_json(summary_path, summary)
        summaries.append(summary)
        if summary.get("fallback"):
            fallback_count += 1

    summary_payload = {"chunks": summaries}
    summary_json = artifact_dir / "planner_chunk_summary.json"
    save_json(summary_json, summary_payload)

    plan = _create_final_plan_adaptive(
        llm_settings=llm_settings,
        planner_settings=planner_settings,
        user_goal=user_goal,
        summaries=summaries,
        attempts=attempts,
    )

    attempts_json = artifact_dir / "planner_attempts.json"
    save_json(attempts_json, {"attempts": attempts})
    return ChunkedPlannerResult(
        plan=plan,
        summary_json=summary_json,
        attempts_json=attempts_json,
        chunk_count=len(summaries),
        fallback_count=fallback_count,
        attempts=attempts,
    )


def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in normalized.splitlines():
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0
        if line_len > max_chars:
            chunks.extend(_split_long_line(line, max_chars))
            continue
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _summarize_chunk_adaptive(
    *,
    llm_settings: LLMSettings,
    planner_settings: ChunkedPlannerSettings,
    user_goal: str,
    chunk_text: str,
    chunk_label: str,
    attempt_dir: Path,
    attempts: list[dict[str, Any]],
    depth: int,
) -> dict[str, Any]:
    detail_level = _detail_level(depth)
    errors: list[str] = []
    use_response_format_options = [True]
    if planner_settings.allow_response_format_retry:
        use_response_format_options.append(False)

    for response_format in use_response_format_options:
        try:
            client = OpenAICompatibleLLMClient(
                _replace_max_tokens(llm_settings, planner_settings.intermediate_max_tokens)
            )
            content = client.chat_completion(
                [
                    {"role": "system", "content": CHUNK_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": build_chunk_summary_prompt(user_goal, chunk_text, detail_level)},
                ],
                response_format=response_format,
            )
            summary = _parse_chunk_summary(content)
            summary["chunk_label"] = chunk_label
            summary["fallback"] = False
            attempts.append(
                {
                    "stage": "chunk",
                    "chunk": chunk_label,
                    "depth": depth,
                    "chars": len(chunk_text),
                    "response_format": response_format,
                    "status": "succeeded",
                }
            )
            return summary
        except (LLMClientError, PlannerError) as exc:
            errors.append(str(exc))
            attempts.append(
                {
                    "stage": "chunk",
                    "chunk": chunk_label,
                    "depth": depth,
                    "chars": len(chunk_text),
                    "response_format": response_format,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    can_split = (
        depth < planner_settings.max_retries
        and len(chunk_text) > planner_settings.min_chunk_chars
    )
    if can_split:
        child_size = max(planner_settings.min_chunk_chars, len(chunk_text) // 2)
        child_chunks = split_text_into_chunks(chunk_text, child_size)
        if len(child_chunks) > 1:
            child_summaries = []
            for child_index, child_chunk in enumerate(child_chunks, start=1):
                child_label = f"{chunk_label}_{child_index}"
                child_summary = _summarize_chunk_adaptive(
                    llm_settings=llm_settings,
                    planner_settings=planner_settings,
                    user_goal=user_goal,
                    chunk_text=child_chunk,
                    chunk_label=child_label,
                    attempt_dir=attempt_dir,
                    attempts=attempts,
                    depth=depth + 1,
                )
                save_json(attempt_dir / f"chunk_{child_label}_summary.json", child_summary)
                child_summaries.append(child_summary)
            return _merge_child_summaries(chunk_label, child_summaries)

    if planner_settings.enable_local_fallback:
        fallback = _local_fallback_summary(chunk_label, chunk_text, errors)
        attempts.append(
            {
                "stage": "chunk",
                "chunk": chunk_label,
                "depth": depth,
                "chars": len(chunk_text),
                "status": "fallback",
                "error": errors[-1] if errors else "",
            }
        )
        return fallback
    raise ChunkedPlannerError(f"Chunk {chunk_label} failed after adaptive retries: {errors[-1] if errors else ''}")


def _create_final_plan_adaptive(
    *,
    llm_settings: LLMSettings,
    planner_settings: ChunkedPlannerSettings,
    user_goal: str,
    summaries: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> PresentationPlan:
    compact_levels = ["normal", "compact", "minimal"]
    last_error = ""
    use_response_format_options = [True]
    if planner_settings.allow_response_format_retry:
        use_response_format_options.append(False)

    for index in range(planner_settings.max_retries + 1):
        level = compact_levels[min(index, len(compact_levels) - 1)]
        grouped_summaries = _group_summaries_for_final(summaries, group_size=max(1, 2**index))
        summary_text = json.dumps(
            {"chunks": [_compact_summary(summary, level) for summary in grouped_summaries]},
            ensure_ascii=False,
            sort_keys=True,
        )
        for response_format in use_response_format_options:
            try:
                client = OpenAICompatibleLLMClient(
                    _replace_max_tokens(llm_settings, planner_settings.final_max_tokens)
                )
                content = client.chat_completion(
                    [
                        {"role": "system", "content": FINAL_FROM_SUMMARIES_SYSTEM_PROMPT},
                        {"role": "user", "content": build_final_summary_prompt(user_goal, summary_text, level)},
                    ],
                    response_format=response_format,
                )
                plan = parse_presentation_plan(content)
                attempts.append(
                    {
                        "stage": "final",
                        "attempt": index + 1,
                        "detail_level": level,
                        "summary_count": len(grouped_summaries),
                        "chars": len(summary_text),
                        "response_format": response_format,
                        "status": "succeeded",
                    }
                )
                return plan
            except (LLMClientError, PlannerError) as exc:
                last_error = str(exc)
                attempts.append(
                    {
                        "stage": "final",
                        "attempt": index + 1,
                        "detail_level": level,
                        "summary_count": len(grouped_summaries),
                        "chars": len(summary_text),
                        "response_format": response_format,
                        "status": "failed",
                        "error": last_error,
                    }
                )
    raise ChunkedPlannerError(f"Final planner failed after adaptive retries: {last_error}")


def _parse_chunk_summary(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"Chunk planner did not return valid JSON. Response preview: {content[:500]}") from exc
    if not isinstance(payload, dict):
        raise PlannerError("Chunk planner JSON must be an object.")
    return {
        "summary": _string_value(payload.get("summary")),
        "key_points": _string_list(payload.get("key_points")),
        "candidate_slides": _candidate_slides(payload.get("candidate_slides")),
        "source_refs": _string_list(payload.get("source_refs")),
        "image_refs": _string_list(payload.get("image_refs")),
    }


def _local_fallback_summary(chunk_label: str, chunk_text: str, errors: list[str]) -> dict[str, Any]:
    source_refs = sorted(set(re.findall(r"`([^`]+(?:section|slide|page)[^`]*)`", chunk_text)))
    image_refs = sorted(set(re.findall(r"`([^`]+asset[^`]*)`", chunk_text)))
    preview = " ".join(chunk_text.split())[:800]
    return {
        "chunk_label": chunk_label,
        "summary": "Local fallback summary created because backend planning failed for this chunk.",
        "key_points": [preview] if preview else [],
        "candidate_slides": [
            {
                "slide_title": "Source Summary",
                "purpose": preview[:300] or "Summarize available source material.",
                "source_refs": source_refs[:8],
                "image_refs": image_refs[:4],
            }
        ],
        "source_refs": source_refs[:20],
        "image_refs": image_refs[:20],
        "fallback": True,
        "errors": errors[-3:],
    }


def _merge_child_summaries(chunk_label: str, child_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    key_points: list[str] = []
    candidate_slides: list[dict[str, Any]] = []
    source_refs: list[str] = []
    image_refs: list[str] = []
    for summary in child_summaries:
        key_points.extend(_string_list(summary.get("key_points")))
        candidate_slides.extend(_candidate_slides(summary.get("candidate_slides")))
        source_refs.extend(_string_list(summary.get("source_refs")))
        image_refs.extend(_string_list(summary.get("image_refs")))
    return {
        "chunk_label": chunk_label,
        "summary": "Merged summaries from smaller adaptive chunks.",
        "key_points": key_points[:20],
        "candidate_slides": candidate_slides[:12],
        "source_refs": _dedupe(source_refs)[:30],
        "image_refs": _dedupe(image_refs)[:30],
        "fallback": any(bool(summary.get("fallback")) for summary in child_summaries),
        "children": [summary.get("chunk_label", "") for summary in child_summaries],
    }


def _compact_summary(summary: dict[str, Any], level: str) -> dict[str, Any]:
    key_point_limit = 8 if level == "normal" else 4 if level == "compact" else 2
    slide_limit = 6 if level == "normal" else 3 if level == "compact" else 1
    ref_limit = 20 if level == "normal" else 10 if level == "compact" else 5
    return {
        "chunk_label": summary.get("chunk_label", ""),
        "summary": _truncate(_string_value(summary.get("summary")), 700 if level == "normal" else 350),
        "key_points": [_truncate(point, 300) for point in _string_list(summary.get("key_points"))[:key_point_limit]],
        "candidate_slides": _candidate_slides(summary.get("candidate_slides"))[:slide_limit],
        "source_refs": _string_list(summary.get("source_refs"))[:ref_limit],
        "image_refs": _string_list(summary.get("image_refs"))[:ref_limit],
        "fallback": bool(summary.get("fallback")),
    }


def _group_summaries_for_final(summaries: list[dict[str, Any]], group_size: int) -> list[dict[str, Any]]:
    if group_size <= 1:
        return summaries
    grouped: list[dict[str, Any]] = []
    for index in range(0, len(summaries), group_size):
        group = summaries[index : index + group_size]
        if len(group) == 1:
            grouped.append(group[0])
        else:
            grouped.append(_merge_child_summaries(f"final_group_{(index // group_size) + 1}", group))
    return grouped


def _candidate_slides(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    slides: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        slides.append(
            {
                "slide_title": _string_value(item.get("slide_title")) or "Untitled",
                "purpose": _string_value(item.get("purpose")),
                "source_refs": _string_list(item.get("source_refs")),
                "image_refs": _string_list(item.get("image_refs")),
            }
        )
    return slides


def _replace_max_tokens(settings: LLMSettings, max_tokens: int) -> LLMSettings:
    return LLMSettings(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        timeout=settings.timeout,
        max_tokens=max_tokens,
    )


def _read_cached_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("summary") else None


def _split_long_line(line: str, max_chars: int) -> list[str]:
    return [line[index : index + max_chars] for index in range(0, len(line), max_chars)]


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _detail_level(depth: int) -> str:
    if depth <= 0:
        return "normal"
    if depth == 1:
        return "compact"
    return "minimal"
