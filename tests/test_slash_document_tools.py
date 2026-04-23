import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo
from src.document_pipeline.storage import save_extracted_documents
from src.gui.llm_client import OpenAIConnectionSettings
from src.slash_tools import SlashToolContext, run_slash_command


class SlashDocumentToolsTests(unittest.TestCase):
    @staticmethod
    def _structured_workspace_summary(label: str) -> str:
        return (
            "Overall Summary\n"
            f"{label} overall summary.\n\n"
            "Features\n"
            f"1. {label} feature one.\n"
            f"2. {label} feature two.\n"
            f"3. {label} feature three.\n\n"
            "Next Action\n"
            f"{label} next action."
        )

    @staticmethod
    def _engineering_workspace_summary(label: str) -> str:
        return (
            "Features\n"
            f"1. {label} feature one.\n"
            f"2. {label} feature two.\n"
            f"3. {label} feature three.\n\n"
            "Quantitative Information\n"
            f"{label} uses 10 ms latency and 3 retries.\n\n"
            "Recommended Action\n"
            f"{label} validate engineering limits."
        )

    def test_help_lists_document_pipeline_tools_and_saved_outputs(self) -> None:
        result = run_slash_command("/help", None, SlashToolContext())

        assert result is not None
        self.assertIn("/extract_docs", result.text)
        self.assertIn("/workspace_status", result.text)
        self.assertIn("/generate_markdown", result.text)
        self.assertIn("/summarize_doc", result.text)
        self.assertIn("HD2docpipe/artifacts/extracted_documents.json", result.text)
        self.assertIn("document_summaries.json", result.text)
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

            extracted_path = root / "HD2docpipe" / "artifacts" / "extracted_documents.json"
            manifest_path = root / "HD2docpipe" / "artifacts" / "extraction_manifest.json"
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

            self.assertTrue((root / "HD2docpipe" / "artifacts" / "report_pptx" / "document_map.json").exists())

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

            report_path = root / "HD2docpipe" / "artifacts" / "report_pptx" / "generated_report.md"
            self.assertTrue(report_path.exists())
            self.assertIn("# Generated Document Report", report_path.read_text(encoding="utf-8"))

        assert result is not None
        self.assertIn("Generated markdown report", result.text)
        self.assertIn("generated_report.md", result.text)

    def test_summarize_doc_auto_extracts_documents_when_artifacts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "report.pptx").write_bytes(b"sample")
            context = SlashToolContext(
                llm_settings=OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
            )
            calls: list[dict[str, object]] = []

            def fake_chat_completion(self, messages):
                calls.append({"max_tokens": self.settings.max_tokens, "prompt": messages[-1]["content"]})
                return SlashDocumentToolsTests._structured_workspace_summary("Auto")

            with patch("src.slash_tools.document_pipeline.extract_docs_mid_level", return_value=[_sample_document(root)]), patch(
                "src.slash_tools.document_pipeline.OpenAICompatibleClient.chat_completion", new=fake_chat_completion
            ):
                result = run_slash_command("/summarize_doc", root, context)

        assert result is not None
        self.assertIn("Auto-ran /extract_docs before summarizing.", result.text)
        self.assertIn("Generated hierarchical document summaries", result.text)
        self.assertTrue(calls)

    def test_summarize_doc_requires_llm_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = SlashToolContext(working_folder=root, documents=[_sample_document(root)])

            result = run_slash_command("/summarize_doc", root, context)

        assert result is not None
        self.assertIn("LLM settings are not configured", result.text)

    def test_summarize_doc_loads_saved_documents_and_writes_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_extracted_documents(root, [_sample_document(root), _sample_document(root, name="notes.pdf", doc_id="doc_notes")])
            context = SlashToolContext(
                llm_settings=OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
            )
            calls: list[dict[str, object]] = []

            def fake_chat_completion(self, messages):
                prompt = messages[-1]["content"]
                calls.append({"max_tokens": self.settings.max_tokens, "prompt": prompt})
                return SlashDocumentToolsTests._structured_workspace_summary(f"Summary {len(calls)}")

            with patch("src.slash_tools.document_pipeline.OpenAICompatibleClient.chat_completion", new=fake_chat_completion):
                result = run_slash_command("/summarize_doc", root, context)

            summary_runs = list((root / "HD2docpipe" / "summaries").iterdir())
            self.assertEqual(len(summary_runs), 1)
            run_dir = summary_runs[0]
            summaries_path = run_dir / "document_summaries.json"
            workspace_path = run_dir / "workspace_summary.md"
            self.assertTrue(summaries_path.exists())
            self.assertTrue(workspace_path.exists())
            payload = json.loads(summaries_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_name"], run_dir.name)
            self.assertEqual(payload["document_count"], 2)
            self.assertEqual(len(payload["documents"]), 2)
            self.assertEqual(payload["workspace_summary"]["features"][0], "Summary 3 feature one.")
            workspace_text = workspace_path.read_text(encoding="utf-8")
            self.assertIn("## Overall Summary", workspace_text)
            self.assertIn("## Features", workspace_text)
            self.assertIn("## Next Action", workspace_text)
            self.assertIn("Summary 3 overall summary.", workspace_text)
            self.assertIn(b"\r\n", workspace_path.read_bytes())
            self.assertNotIn("## Document Summaries", workspace_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(calls), 3)

        assert result is not None
        self.assertIn("Generated hierarchical document summaries", result.text)
        self.assertIn("workspace_summary.md", result.text)
        self.assertRegex(result.text, r"output_run: [a-z0-9._-]+_\d{8}_\d{6}")
        self.assertIn("workspace_summary_sections:", result.text)
        self.assertIn("Ask a normal question about the generated summary", result.text)

    def test_summarize_doc_path_limits_to_selected_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_extracted_documents(root, [_sample_document(root), _sample_document(root, name="notes.pdf", doc_id="doc_notes")])
            context = SlashToolContext(
                llm_settings=OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
            )

            def fake_chat_completion(self, messages):
                return SlashDocumentToolsTests._structured_workspace_summary("Single")

            with patch("src.slash_tools.document_pipeline.OpenAICompatibleClient.chat_completion", new=fake_chat_completion):
                result = run_slash_command("/summarize_doc notes.pdf", root, context)

            summary_runs = list((root / "HD2docpipe" / "summaries").iterdir())
            self.assertEqual(len(summary_runs), 1)
            run_dir = summary_runs[0]
            payload = json.loads((run_dir / "document_summaries.json").read_text(encoding="utf-8"))

        assert result is not None
        self.assertEqual(payload["document_count"], 1)
        self.assertEqual(payload["target_path"], "notes.pdf")
        self.assertTrue(run_dir.name.startswith("notes_"))
        self.assertIn("target_path: notes.pdf", result.text)

    def test_summarize_doc_engineering_mode_changes_workspace_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_extracted_documents(root, [_sample_document(root, name="notes.pdf", doc_id="doc_notes")])
            context = SlashToolContext(
                llm_settings=OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
            )

            def fake_chat_completion(client, messages):
                self.assertIn("Quantitative Information", messages[-1]["content"])
                return SlashDocumentToolsTests._engineering_workspace_summary("Engineering")

            with patch("src.slash_tools.document_pipeline.OpenAICompatibleClient.chat_completion", new=fake_chat_completion):
                result = run_slash_command("/summarize_doc --engineering True notes.pdf", root, context)

            run_dir = next((root / "HD2docpipe" / "summaries").iterdir())
            payload = json.loads((run_dir / "document_summaries.json").read_text(encoding="utf-8"))
            workspace_text = (run_dir / "workspace_summary.md").read_text(encoding="utf-8")

        assert result is not None
        self.assertEqual(payload["summary_mode"], "engineering")
        self.assertEqual(payload["workspace_summary"]["mode"], "engineering")
        self.assertIn("## Features", workspace_text)
        self.assertIn("## Quantitative Information", workspace_text)
        self.assertIn("## Recommended Action", workspace_text)
        self.assertNotIn("## Overall Summary", workspace_text)
        self.assertIn("mode: engineering", result.text)

    def test_summarize_doc_path_auto_extracts_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "report.pptx").write_bytes(b"sample")
            context = SlashToolContext(
                llm_settings=OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
            )

            def fake_chat_completion(self, messages):
                return SlashDocumentToolsTests._structured_workspace_summary("Target")

            with patch(
                "src.slash_tools.document_pipeline.extract_single_doc_mid_level",
                return_value=_sample_document(root),
            ), patch("src.slash_tools.document_pipeline.OpenAICompatibleClient.chat_completion", new=fake_chat_completion):
                result = run_slash_command("/summarize_doc report.pptx", root, context)

        assert result is not None
        self.assertIn("Auto-extracted report.pptx before summarizing.", result.text)

    def test_extract_single_doc_saves_file_scoped_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "report.pptx").write_bytes(b"sample")

            with patch(
                "src.slash_tools.document_pipeline.extract_single_doc_mid_level",
                return_value=_sample_document(root),
            ):
                result = run_slash_command("/extract_single_doc report.pptx", root, SlashToolContext())

        assert result is not None
        self.assertIn("HD2docpipe/artifacts/report_pptx/documents/doc_report.json", result.text)
        self.assertIn("HD2docpipe/artifacts/report_pptx/extraction_manifest.json", result.text)

    def test_summarize_doc_progress_messages_are_line_delimited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_extracted_documents(root, [_sample_document(root)])
            context = SlashToolContext(
                llm_settings=OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
            )
            progress_events: list[tuple[str, str]] = []

            def fake_chat_completion(self, messages):
                return SlashDocumentToolsTests._structured_workspace_summary("Progress")

            def capture_progress(kind: str, text: str) -> None:
                progress_events.append((kind, text))

            with patch("src.slash_tools.document_pipeline.OpenAICompatibleClient.chat_completion", new=fake_chat_completion):
                result = run_slash_command("/summarize_doc", root, context, progress=capture_progress)

        assert result is not None
        self.assertTrue(progress_events)
        self.assertTrue(all(text.endswith("\n") for _, text in progress_events))
        self.assertTrue(any("Summarizing document 1/1" in text for _, text in progress_events))

    def test_workspace_status_reports_latest_summary_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_root = root / "HD2docpipe" / "summaries"
            old_run = summary_root / "workspace_20240101_000000"
            new_run = summary_root / "notes_20240101_000001"
            old_run.mkdir(parents=True)
            new_run.mkdir(parents=True)
            (old_run / "document_summaries.json").write_text("{}", encoding="utf-8")
            (old_run / "workspace_summary.md").write_text("old", encoding="utf-8")
            (new_run / "document_summaries.json").write_text("{}", encoding="utf-8")
            (new_run / "workspace_summary.md").write_text("new", encoding="utf-8")

            result = run_slash_command("/workspace_status", root, SlashToolContext())

        assert result is not None
        self.assertIn("HD2docpipe/summaries/notes_20240101_000001/document_summaries.json", result.text)
        self.assertIn("HD2docpipe/summaries/notes_20240101_000001/workspace_summary.md", result.text)


def _sample_document(root: Path, name: str = "report.pptx", doc_id: str = "doc_report") -> ExtractedDocument:
    provenance = Provenance(source_path=str(root / name), location_type="slide", slide=1)
    return ExtractedDocument(
        schema_version="0.1",
        document_id=doc_id,
        source=SourceInfo(
            path=str(root / name),
            filename=name,
            extension=Path(name).suffix,
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            size_bytes=6,
            sha256="abc",
        ),
        metadata=DocumentMetadata(title=Path(name).stem.title()),
        blocks=[
            ExtractedBlock(
                block_id="blk_001",
                document_id=doc_id,
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
