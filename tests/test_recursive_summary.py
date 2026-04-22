import unittest
from pathlib import Path

from src.document_pipeline.high_level.recursive_summary import (
    MAX_RECURSIVE_EVIDENCE_GROUPS,
    run_recursive_summary,
)
from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo


class RecursiveSummaryTests(unittest.TestCase):
    def test_ranked_groups_do_not_call_llm_before_final_generation(self) -> None:
        document = _large_document()
        client = _FailingClient()
        events: list[tuple[str, str]] = []

        result = run_recursive_summary(
            [document],
            client,
            "summarize method results",
            12000,
            progress=lambda kind, text: events.append((kind, text)),
            selected_block_ids={"blk_010"},
        )

        self.assertEqual(client.calls, 0)
        self.assertEqual(result.mode, "ranked-groups")
        self.assertLessEqual(result.selected_group_count, MAX_RECURSIVE_EVIDENCE_GROUPS)
        self.assertGreater(len(result.final_summary), 0)
        self.assertTrue(any("no LLM calls before final generation" in text for _, text in events))
        self.assertTrue(any("Selected top" in text for _, text in events))


class _FailingClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_completion(self, messages, response_format=None):
        self.calls += 1
        raise AssertionError("recursive grouping must not call the LLM")


def _large_document() -> ExtractedDocument:
    source_path = str(Path("work") / "large.pdf")
    return ExtractedDocument(
        schema_version="0.1",
        document_id="doc_large",
        source=SourceInfo(
            path=source_path,
            filename="large.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=6,
            sha256="large",
        ),
        metadata=DocumentMetadata(title="Large"),
        blocks=[
            ExtractedBlock(
                block_id=f"blk_{index:03d}",
                document_id="doc_large",
                type="text",
                role="section",
                order=index,
                text=f"Section {index} method result value {index} with explicit evidence.",
                normalized_text=f"Section {index} method result value {index} with explicit evidence.",
                provenance=Provenance(source_path=source_path, location_type="page", page=index + 1),
            )
            for index in range(90)
        ],
    )


if __name__ == "__main__":
    unittest.main()
