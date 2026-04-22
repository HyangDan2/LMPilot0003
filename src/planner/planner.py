from __future__ import annotations

import json
import re
from typing import Any

from src.models.schemas import PresentationPlan, SlidePlan
from src.planner.llm_client import OpenAICompatibleLLMClient
from src.planner.prompts import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt


class PlannerError(Exception):
    """Raised when the presentation plan cannot be created or validated."""


def create_presentation_plan(
    client: OpenAICompatibleLLMClient,
    user_goal: str,
    knowledge_map_md: str,
) -> PresentationPlan:
    """Ask the LLM for a JSON-only presentation plan and validate it."""

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": build_planner_user_prompt(user_goal, knowledge_map_md)},
    ]
    content = client.chat_completion(messages)
    return parse_presentation_plan(content)


def parse_presentation_plan(content: str) -> PresentationPlan:
    """Parse and validate planner JSON."""

    json_text = _strip_json_fence(content)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        preview = content.replace("\n", " ")[:500]
        raise PlannerError(f"Planner did not return valid JSON. Response preview: {preview}") from exc

    if not isinstance(payload, dict):
        raise PlannerError("Planner JSON must be an object.")
    if payload.get("output_type") != "pptx":
        raise PlannerError("Planner JSON output_type must be 'pptx'.")

    title = _required_string(payload, "title")
    target_audience = _required_string(payload, "target_audience")
    raw_slides = payload.get("slides")
    if not isinstance(raw_slides, list) or not raw_slides:
        raise PlannerError("Planner JSON must include at least one slide.")

    slides: list[SlidePlan] = []
    for index, raw_slide in enumerate(raw_slides, start=1):
        if not isinstance(raw_slide, dict):
            raise PlannerError(f"Slide {index} must be an object.")
        slides.append(
            SlidePlan(
                slide_title=_required_string(raw_slide, "slide_title", label=f"Slide {index}"),
                purpose=_required_string(raw_slide, "purpose", label=f"Slide {index}"),
                source_refs=_string_list(raw_slide.get("source_refs")),
                image_refs=_string_list(raw_slide.get("image_refs")),
            )
        )

    return PresentationPlan(
        output_type="pptx",
        title=title,
        target_audience=target_audience,
        slides=slides,
    )


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _required_string(payload: dict[str, Any], key: str, label: str = "Plan") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlannerError(f"{label} is missing required string field '{key}'.")
    return value.strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]

