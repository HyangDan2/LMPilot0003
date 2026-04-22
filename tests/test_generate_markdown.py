import unittest
from pathlib import Path

from src.document_pipeline.high_level import generate_markdown_report
from src.document_pipeline.mid_level import build_doc_map, chunk_sections
from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo


class GenerateMarkdownTests(unittest.TestCase):
    def test_generate_markdown_report_includes_evidence_and_artifacts(self) -> None:
        document = _sample_document()
        doc_map = build_doc_map([document])
        chunks = chunk_sections([document])

        markdown = generate_markdown_report([document], doc_map, chunks)

        self.assertIn("# Generated Document Report", markdown)
        self.assertIn("report.pptx", markdown)
        self.assertIn("Revenue grew by 10%.", markdown)
        self.assertIn("extracted_documents.json", markdown)

    def test_generate_markdown_report_handles_missing_chunks(self) -> None:
        markdown = generate_markdown_report([_sample_document()], chunks=[])

        self.assertIn("No chunks are available yet", markdown)


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
