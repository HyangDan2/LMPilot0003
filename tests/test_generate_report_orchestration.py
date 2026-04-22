import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.document_pipeline.high_level import generate_report, write_output_plan
from src.document_pipeline.high_level.select_evidence import select_evidence_blocks
from src.document_pipeline.mid_level import build_doc_map
from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo


class FakeReportLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        prompt = str(messages[-1]["content"])
        self.prompts.append(prompt)
        if response_format is not None:
            raise AssertionError("generate_report should not request JSON chunk summaries.")
        return (
            "# Engineering Report\n\n"
            "## Summary\n\n### What the Document Explicitly Describes\n\nRevenue grew by 10% according to `report.pptx / blk_001`. The calculation basis is missing.\n\n"
            "## Source Documents\n\n| Source | Type | Evidence Used |\n|---|---|---:|\n| report.pptx | .pptx | 1 |\n\n"
            "## Open Issues and Next Actions\n\n- Verify the source calculation.\n"
        )


@dataclass(frozen=True)
class FakeStreamChunk:
    kind: str
    text: str


class FakeStreamingReportLLMClient(FakeReportLLMClient):
    def stream_chat_completion(self, messages: list[dict[str, Any]]):
        self.prompts.append(str(messages[-1]["content"]))
        yield FakeStreamChunk("final", "# Final")
        yield FakeStreamChunk("final", " Report\n\nStreaming report body.")


class GenerateReportOrchestrationTests(unittest.TestCase):
    def test_generate_report_uses_query_and_returns_markdown(self) -> None:
        document = _sample_document()
        doc_map = build_doc_map([document])
        plan = write_output_plan([document], doc_map, goal="summarize about revenue")
        selected_evidence = select_evidence_blocks([document], plan, "summarize about revenue")
        client = FakeReportLLMClient()

        result = generate_report(plan, [document], doc_map, selected_evidence, client, "summarize about revenue", 1200)

        self.assertTrue(result.used_llm)
        self.assertIn("# Engineering Report", result.markdown)
        self.assertIn("Revenue grew by 10% according to `report.pptx / blk_001`.\nThe calculation basis is missing.", result.markdown)
        self.assertEqual(result.chunk_summaries, [])
        self.assertEqual(result.section_summaries, [])
        self.assertEqual(len(client.prompts), 1)
        self.assertTrue(any("summarize about revenue" in prompt for prompt in client.prompts))
        self.assertTrue(any("Selected evidence for citation checks" in prompt for prompt in client.prompts))
        self.assertTrue(any("Summary, Source Documents, Open Issues and Next Actions" in prompt for prompt in client.prompts))
        self.assertTrue(any("What the Document Explicitly Describes, Main Methods or Components Explicitly Mentioned" in prompt for prompt in client.prompts))
        self.assertTrue(any("Do not infer architecture, databases" in prompt for prompt in client.prompts))
        self.assertTrue(any("Write each sentence on its own line" in prompt for prompt in client.prompts))

    def test_generate_report_falls_back_without_llm_client(self) -> None:
        document = _sample_document()
        doc_map = build_doc_map([document])
        plan = write_output_plan([document], doc_map)
        selected_evidence = select_evidence_blocks([document], plan, plan.goal)

        result = generate_report(plan, [document], doc_map, selected_evidence, None)

        self.assertFalse(result.used_llm)
        self.assertIn("# Engineering Report for Report", result.markdown)
        self.assertIn("## Summary", result.markdown)
        self.assertIn("### What the Document Explicitly Describes", result.markdown)
        self.assertIn("### Unclear or Not Specified in Selected Evidence", result.markdown)
        self.assertIn("## Source Documents", result.markdown)
        self.assertIn("## Open Issues and Next Actions", result.markdown)
        self.assertIn("LLM client is not configured", result.fallback_reason)

    def test_generate_report_streams_final_markdown_when_client_supports_it(self) -> None:
        document = _sample_document()
        doc_map = build_doc_map([document])
        plan = write_output_plan([document], doc_map)
        selected_evidence = select_evidence_blocks([document], plan, plan.goal)
        events: list[tuple[str, str]] = []

        result = generate_report(
            plan,
            [document],
            doc_map,
            selected_evidence,
            FakeStreamingReportLLMClient(),
            progress=lambda kind, text: events.append((kind, text)),
        )

        self.assertTrue(result.used_llm)
        self.assertIn("# Final Report", result.markdown)
        markdown_events = [text for kind, text in events if kind == "markdown"]
        self.assertIn("# Final", markdown_events)
        self.assertIn(" Report\n\nStreaming report body.", markdown_events)


def _sample_document() -> ExtractedDocument:
    source_path = str(Path("work") / "report.pptx")
    provenance = Provenance(source_path=source_path, location_type="slide", slide=1)
    return ExtractedDocument(
        schema_version="0.1",
        document_id="doc_report",
        source=SourceInfo(
            path=source_path,
            filename="report.pptx",
            extension=".pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            size_bytes=6,
            sha256="abc",
        ),
        metadata=DocumentMetadata(title="Report"),
        blocks=[
            ExtractedBlock(
                block_id="blk_001",
                document_id="doc_report",
                type="text",
                role="section",
                order=0,
                text="Revenue grew by 10%.",
                normalized_text="Revenue grew by 10%.",
                provenance=provenance,
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
