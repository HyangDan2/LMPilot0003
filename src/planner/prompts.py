from __future__ import annotations


PLANNER_SYSTEM_PROMPT = """You are a presentation planning engine.
Return strict JSON only. Do not include markdown, comments, or prose.
Create a concise PowerPoint plan grounded in the provided knowledge map.
Use only source_refs and image_refs that appear in the knowledge map.
The output must match this shape:
{
  "output_type": "pptx",
  "title": "string",
  "target_audience": "string",
  "slides": [
    {
      "slide_title": "string",
      "purpose": "string",
      "source_refs": ["string"],
      "image_refs": ["string"]
    }
  ]
}
"""


def build_planner_user_prompt(user_goal: str, knowledge_map_md: str) -> str:
    """Build the user message sent to the planner LLM."""

    return (
        f"User goal:\n{user_goal.strip()}\n\n"
        "Knowledge map:\n"
        f"{knowledge_map_md.strip()}\n\n"
        "Return JSON only."
    )


CHUNK_SUMMARY_SYSTEM_PROMPT = """You are a presentation planning assistant.
Summarize the provided knowledge-map chunk for later PowerPoint planning.
Return strict JSON only. Do not include markdown, comments, or prose.
The output must match this shape:
{
  "summary": "string",
  "key_points": ["string"],
  "candidate_slides": [
    {
      "slide_title": "string",
      "purpose": "string",
      "source_refs": ["string"],
      "image_refs": ["string"]
    }
  ],
  "source_refs": ["string"],
  "image_refs": ["string"]
}
"""


FINAL_FROM_SUMMARIES_SYSTEM_PROMPT = """You are a presentation planning engine.
Create a concise PowerPoint plan from compact chunk summaries.
Return strict JSON only. Do not include markdown, comments, or prose.
Use only source_refs and image_refs that appear in the summaries.
The output must match this shape:
{
  "output_type": "pptx",
  "title": "string",
  "target_audience": "string",
  "slides": [
    {
      "slide_title": "string",
      "purpose": "string",
      "source_refs": ["string"],
      "image_refs": ["string"]
    }
  ]
}
"""


def build_chunk_summary_prompt(user_goal: str, chunk_text: str, detail_level: str = "normal") -> str:
    return (
        f"User goal:\n{user_goal.strip()}\n\n"
        f"Detail level: {detail_level}\n\n"
        "Knowledge-map chunk:\n"
        f"{chunk_text.strip()}\n\n"
        "Return JSON only."
    )


def build_final_summary_prompt(user_goal: str, summaries_json: str, detail_level: str = "normal") -> str:
    return (
        f"User goal:\n{user_goal.strip()}\n\n"
        f"Detail level: {detail_level}\n\n"
        "Chunk summaries JSON:\n"
        f"{summaries_json.strip()}\n\n"
        "Return final presentation plan JSON only."
    )
