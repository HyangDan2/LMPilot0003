import json
import tempfile
import unittest
from pathlib import Path

from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo
from src.slash_tools import SlashToolContext, run_slash_command


class SlashDocumentToolsTests(unittest.TestCase):
    def test_help_lists_document_pipeline_tools_and_saved_outputs(self) -> None:
        result = run_slash_command("/help", None, SlashToolContext())

        assert result is not None
        self.assertIn("/extract_docs", result.text)
        self.assertIn("/workspace_status", result.text)
        self.assertIn("/generate_markdown", result.text)
        self.assertIn("llm_result/document_pipeline/extracted_documents.json", result.text)
        self.assertNotIn("llm_chunk_summaries.json", result.text)

    def test_non_slash_prompt_is_not_handled(self) -> None:
        self.assertIsNone(run_slash_command("hello", None, SlashToolContext()))

    def test_unknown_slash_command_returns_helpful_error(self) -> None:
        result = run_slash_command("/missing", None, SlashToolContext())

        assert result is not None
        self.assertIn("unknown slash command", result.text)
        self.assertIn("/help", result.text)

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = run_slash_command("/read_file_info ../secret.txt", root, SlashToolContext())

        assert result is not None
        self.assertIn("outside the attached working folder", result.text)

    def test_detect_file_type_and_read_file_info_use_attached_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_path = root / "report.pptx"
            file_path.write_bytes(b"sample")
            context = SlashToolContext()

            detected = run_slash_command("/detect_file_type report.pptx", root, context)
            info = run_slash_command("/read_file_info report.pptx", root, context)

        assert detected is not None
        assert info is not None
        self.assertIn("family: pptx", detected.text)
        self.assertIn("size_bytes: 6", info.text)

    def test_extract_docs_auto_saves_empty_folder_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = SlashToolContext()

            result = run_slash_command("/extract_docs", root, context)

            extracted_path = root / "llm_result" / "document_pipeline" / "extracted_documents.json"
            manifest_path = root / "llm_result" / "document_pipeline" / "extraction_manifest.json"
            self.assertTrue(extracted_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(json.loads(extracted_path.read_text(encoding="utf-8")), {"documents": []})

        assert result is not None
        self.assertIn("Extracted 0 document(s)", result.text)

    def test_build_doc_map_auto_saves_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = SlashToolContext(working_folder=root, documents=[_sample_document(root)])

            doc_map_result = run_slash_command("/build_doc_map", root, context)

            self.assertTrue((root / "llm_result" / "document_pipeline" / "document_map.json").exists())

        assert doc_map_result is not None
        self.assertIn("Built document map", doc_map_result.text)

    def test_workspace_status_reports_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = SlashToolContext()
            run_slash_command("/extract_docs", root, context)

            result = run_slash_command("/workspace_status", root, context)

        assert result is not None
        self.assertIn("Workspace status", result.text)
        self.assertIn("extracted_documents.json: found", result.text)

    def test_generate_markdown_auto_saves_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = SlashToolContext(working_folder=root, documents=[_sample_document(root)])

            result = run_slash_command("/generate_markdown", root, context)

            report_path = root / "llm_result" / "document_pipeline" / "generated_report.md"
            self.assertTrue(report_path.exists())
            self.assertIn("# Generated Document Report", report_path.read_text(encoding="utf-8"))

        assert result is not None
        self.assertIn("Generated markdown report", result.text)
        self.assertIn("generated_report.md", result.text)

def _sample_document(root: Path) -> ExtractedDocument:
    provenance = Provenance(source_path=str(root / "report.pptx"), location_type="slide", slide=1)
    return ExtractedDocument(
        schema_version="0.1",
        document_id="doc_report",
        source=SourceInfo(
            path=str(root / "report.pptx"),
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
