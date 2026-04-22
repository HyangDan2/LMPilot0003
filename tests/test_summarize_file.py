import unittest
from pathlib import Path

from src.document_pipeline.high_level import write_output_plan
from src.document_pipeline.high_level.select_evidence import select_evidence_blocks
from src.document_pipeline.high_level.summarize_file import _fallback_file_summary, _file_summary_prompt
from src.document_pipeline.mid_level import build_doc_map
from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo


class SummarizeFileTests(unittest.TestCase):
    def test_file_summary_prompt_uses_file_specific_sections(self) -> None:
        document = _sample_document()
        doc_map = build_doc_map([document])
        plan = write_output_plan([document], doc_map, goal="summarize risks")
        selected_evidence = select_evidence_blocks([document], plan, "summarize risks")

        prompt = _file_summary_prompt(plan, document, selected_evidence, "summarize risks", 2000)

        self.assertIn("File Summary: <filename>", prompt)
        self.assertIn("Summary, Source Details, Open Issues and Next Actions", prompt)
        self.assertIn("What the Document Explicitly Describes, Main Methods or Components Explicitly Mentioned", prompt)
        self.assertIn("Do not infer architecture, databases", prompt)
        self.assertIn("Write each sentence on its own line", prompt)

    def test_fallback_file_summary_uses_source_details(self) -> None:
        document = _sample_document()
        doc_map = build_doc_map([document])
        plan = write_output_plan([document], doc_map)
        selected_evidence = select_evidence_blocks([document], plan, plan.goal)

        markdown = _fallback_file_summary(plan, document, selected_evidence)

        self.assertIn("# File Summary: report.pptx", markdown)
        self.assertIn("## Summary", markdown)
        self.assertIn("### What the Document Explicitly Describes", markdown)
        self.assertIn("## Source Details", markdown)
        self.assertIn("## Open Issues and Next Actions", markdown)
        self.assertIn("Revenue grew by 10%.", markdown)


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
