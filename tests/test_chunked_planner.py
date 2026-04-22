import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.planner.chunked_planner import (
    CHUNK_SUMMARY_SYSTEM_PROMPT,
    ChunkedPlannerSettings,
    create_chunked_presentation_plan,
    split_text_into_chunks,
)
from src.planner.llm_client import LLMClientError, LLMSettings


class FakeAdaptivePlannerClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def chat_completion(self, messages: list[dict[str, str]], *, response_format: bool = True) -> str:
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        if system_prompt == CHUNK_SUMMARY_SYSTEM_PROMPT:
            if len(user_prompt) > 1200:
                raise LLMClientError("context too large")
            return """
            {
              "summary": "small chunk summary",
              "key_points": ["point"],
              "candidate_slides": [
                {
                  "slide_title": "Finding",
                  "purpose": "Summarize the finding",
                  "source_refs": ["doc-1-section-1"],
                  "image_refs": []
                }
              ],
              "source_refs": ["doc-1-section-1"],
              "image_refs": []
            }
            """
        return """
        {
          "output_type": "pptx",
          "title": "Adaptive Plan",
          "target_audience": "Team",
          "slides": [
            {
              "slide_title": "Finding",
              "purpose": "Summarize the finding",
              "source_refs": ["doc-1-section-1"],
              "image_refs": []
            }
          ]
        }
        """


class ChunkedPlannerTests(unittest.TestCase):
    def test_split_text_into_chunks_keeps_chunks_under_limit(self) -> None:
        chunks = split_text_into_chunks("a\nbb\nccc\ndddd", 6)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 6 for chunk in chunks))

    def test_adaptive_planner_splits_failed_chunk_and_writes_artifacts(self) -> None:
        artifact_dir = Path(tempfile.mkdtemp())
        knowledge_map = "# Knowledge Map\n\n`doc-1-section-1` " + ("large context " * 120)

        with patch("src.planner.chunked_planner.OpenAICompatibleLLMClient", FakeAdaptivePlannerClient):
            result = create_chunked_presentation_plan(
                llm_settings=LLMSettings(base_url="http://localhost:8000/v1", model="local"),
                planner_settings=ChunkedPlannerSettings(
                    chunk_chars=2000,
                    min_chunk_chars=500,
                    max_retries=2,
                    intermediate_max_tokens=256,
                    final_max_tokens=512,
                    enable_local_fallback=False,
                ),
                user_goal="Create a short deck",
                knowledge_map_md=knowledge_map,
                artifact_dir=artifact_dir,
            )

        self.assertEqual(result.plan.title, "Adaptive Plan")
        self.assertTrue(result.summary_json.exists())
        self.assertTrue(result.attempts_json.exists())
        self.assertTrue(any(attempt["status"] == "failed" for attempt in result.attempts))
        self.assertTrue(any(str(attempt.get("chunk", "")).startswith("001_") for attempt in result.attempts))


if __name__ == "__main__":
    unittest.main()
