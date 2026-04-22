import json
import unittest
from pathlib import Path
from typing import Any

from src.document_pipeline.high_level.detail_summary import (
    build_detail_summary_groups,
    detail_summaries_markdown,
    generate_detail_summaries,
)
from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo


class FakeDetailLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self.calls += 1
        self.response_format = response_format
        prompt = str(messages[-1]["content"])
        item_ids = []
        for line in prompt.splitlines():
            if '"item_id":' in line:
                item_ids.append(line.split('"item_id":', 1)[1].strip().strip('",'))
        return json.dumps(
            {
                "summaries": [
                    {
                        "item_id": item_id,
                        "summary": f"Summary for {item_id}.",
                        "key_points": ["grounded point"],
                    }
                    for item_id in item_ids
                ]
            }
        )


class DetailSummaryTests(unittest.TestCase):
    def test_build_detail_summary_groups_uses_pages_and_slides(self) -> None:
        document = _document("doc_pdf", "sample.pdf", ".pdf", [(1, "Intro text."), (2, "Method text.")])

        groups = build_detail_summary_groups([document])

        self.assertEqual([group.location_label for group in groups], ["Page 1", "Page 2"])
        self.assertEqual(groups[0].block_ids, ["blk_001"])
        self.assertIn("Intro text.", groups[0].text)

    def test_generate_detail_summaries_batches_llm_calls(self) -> None:
        document = _document("doc_pdf", "sample.pdf", ".pdf", [(1, "Intro text."), (2, "Method text.")])
        client = FakeDetailLLMClient()
        events: list[tuple[str, str]] = []

        result = generate_detail_summaries(
            [document],
            client,
            enabled=True,
            batch_size=10,
            progress=lambda kind, text: events.append((kind, text)),
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(client.response_format, {"type": "json_object"})
        self.assertTrue(result.used_llm)
        self.assertEqual(result.summary_count, 2)
        self.assertIn("Summary for doc_pdf_page_page_1.", detail_summaries_markdown(result))
        self.assertTrue(any("[detail] Processing Page 1 1/2" in text for _, text in events))
        self.assertTrue(any("[detail] Completed Page 1 1/2." in text for _, text in events))
        self.assertTrue(any("[detail] Processing Page 2 2/2" in text for _, text in events))
        self.assertTrue(any("[detail] Completed Page 2 2/2." in text for _, text in events))

    def test_generate_detail_summaries_falls_back_without_llm(self) -> None:
        document = _document("doc_pdf", "sample.pdf", ".pdf", [(1, "Intro text. More text.")])
        events: list[tuple[str, str]] = []

        result = generate_detail_summaries(
            [document],
            None,
            enabled=True,
            progress=lambda kind, text: events.append((kind, text)),
        )

        self.assertFalse(result.used_llm)
        self.assertEqual(result.summary_count, 1)
        self.assertIn("LLM client is not configured", result.summaries[0].fallback_reason)
        self.assertTrue(any("[detail] Processing Page 1 1/1 with extractive fallback" in text for _, text in events))
        self.assertTrue(any("[detail] Completed Page 1 1/1 with fallback" in text for _, text in events))


def _document(document_id: str, filename: str, extension: str, pages: list[tuple[int, str]]) -> ExtractedDocument:
    source_path = str(Path("work") / filename)
    return ExtractedDocument(
        schema_version="0.1",
        document_id=document_id,
        source=SourceInfo(
            path=source_path,
            filename=filename,
            extension=extension,
            mime_type="application/pdf",
            size_bytes=10,
            sha256=document_id,
        ),
        metadata=DocumentMetadata(title=filename),
        blocks=[
            ExtractedBlock(
                block_id=f"blk_{index:03d}",
                document_id=document_id,
                type="text",
                role="section",
                order=index,
                text=text,
                normalized_text=text,
                provenance=Provenance(source_path=source_path, location_type="page", page=page),
            )
            for index, (page, text) in enumerate(pages, start=1)
        ],
    )


if __name__ == "__main__":
    unittest.main()
