import json
import tempfile
import unittest
from pathlib import Path

from src.document_pipeline.high_level import generate_report_pipeline


class GenerateReportPipelineTests(unittest.TestCase):
    def test_generate_report_pipeline_saves_plan_and_report_for_empty_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events: list[tuple[str, str]] = []

            result = generate_report_pipeline(root, goal="Demo report", progress=lambda kind, text: events.append((kind, text)))

            output_dir = root / "llm_result" / "document_pipeline"
            plan_path = output_dir / "output_plan.json"
            report_path = output_dir / "generated_report.md"
            selected_evidence_path = output_dir / "selected_evidence.json"
            self.assertTrue(plan_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(selected_evidence_path.exists())
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan_payload["goal"], "Demo report")
            self.assertEqual(plan_payload["sections"][0]["max_chars"], 20480)
            self.assertEqual(plan_payload["sections"][1]["max_chars"], 200)
            self.assertEqual(plan_payload["sections"][2]["max_chars"], 200)
            self.assertIn("# Engineering Report", report_path.read_text(encoding="utf-8"))
            self.assertIn("### What the Document Explicitly Describes", report_path.read_text(encoding="utf-8"))
            self.assertIn(plan_path.resolve(), result.saved_files)
            self.assertIn(report_path.resolve(), result.saved_files)
            self.assertTrue(any("[1/8] Extracting documents" in text for _, text in events))
            self.assertTrue(any("[4/8] Selecting representative evidence" in text for _, text in events))
            self.assertTrue(any("[6/8] Preparing grouped evidence context" in text for _, text in events))
            self.assertTrue(any("Timings:" in text for _, text in events))
            self.assertTrue(any("Saved" in text for _, text in events))
            self.assertTrue(any(kind == "markdown" for kind, _ in events))
            self.assertIn("total", result.timings)
            self.assertEqual(result.mode, "one-shot")
            self.assertTrue((output_dir / "final_prompt_preview.txt").exists())

    def test_generate_report_pipeline_uses_ranked_groups_mode_for_large_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "llm_result" / "document_pipeline"
            output_dir.mkdir(parents=True)
            events: list[tuple[str, str]] = []

            from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo
            from src.document_pipeline.storage import save_extracted_documents, save_manifest

            source_path = root / "large.pdf"
            source_path.write_text("sample", encoding="utf-8")
            document = ExtractedDocument(
                schema_version="0.1",
                document_id="doc_large",
                source=SourceInfo(
                    path=str(source_path),
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
                        provenance=Provenance(source_path=str(source_path), location_type="page", page=index + 1),
                    )
                    for index in range(90)
                ],
            )
            save_extracted_documents(root, [document])
            save_manifest(root, [document])

            result = generate_report_pipeline(
                root,
                goal="Large report",
                generate_detail=True,
                progress=lambda kind, text: events.append((kind, text)),
            )

            self.assertEqual(result.mode, "ranked-groups")
            self.assertGreater(result.evidence_group_count, 0)
            self.assertGreater(result.selected_evidence_group_count, 0)
            self.assertLessEqual(result.selected_evidence_group_count, 10)
            self.assertEqual(result.detail_summary_count, 90)
            self.assertTrue((output_dir / "evidence_groups.json").exists())
            self.assertTrue((output_dir / "selected_evidence_groups.json").exists())
            self.assertTrue((output_dir / "detail_summaries.json").exists())
            self.assertTrue((output_dir / "detail_summaries.md").exists())
            self.assertTrue(any("Large document set detected" in text for _, text in events))
            self.assertTrue(any("Ranking evidence groups locally; no LLM calls before final generation" in text for _, text in events))
            self.assertTrue(any("Detail summaries enabled" in text for _, text in events))
            self.assertTrue(any("[detail] Processing Page 1 1/90" in text for _, text in events))
            self.assertTrue(any("[detail] Completed Page 90 90/90" in text for _, text in events))

    def test_generate_report_pipeline_reuses_empty_extraction_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            first = generate_report_pipeline(root, goal="Demo report")
            second = generate_report_pipeline(root, goal="Demo report")

            self.assertFalse(first.extraction_cache_used)
            self.assertTrue(second.extraction_cache_used)

    def test_generate_report_pipeline_fresh_bypasses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            generate_report_pipeline(root, goal="Demo report")
            second = generate_report_pipeline(root, goal="Demo report", force_refresh=True)

            self.assertFalse(second.extraction_cache_used)

    def test_generate_report_pipeline_excludes_generated_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "llm_result" / "document_pipeline"
            output_dir.mkdir(parents=True)
            (output_dir / "ignored.docx").write_bytes(b"not a real docx")

            result = generate_report_pipeline(root, goal="Demo report")

            self.assertEqual(result.documents, [])


if __name__ == "__main__":
    unittest.main()
