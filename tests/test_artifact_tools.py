import tempfile
import unittest
from pathlib import Path

from src.gui.artifact_tools import (
    execute_artifact_request,
    extract_artifact_requests,
    resolve_output_artifact_path,
)
from src.gui.artifact_tools import ArtifactRequest


class ArtifactToolsTests(unittest.TestCase):
    def test_extracts_qwen_style_read_file_request(self) -> None:
        requests = extract_artifact_requests("[read_file] llm/document_pipeline/generated_report.md [/read_file]")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].command, "read_file")
        self.assertEqual(requests[0].requested_path, "llm/document_pipeline/generated_report.md")

    def test_resolves_aliases_under_llm_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            path = resolve_output_artifact_path(root, "document_pipeline/generated_report.md")

            self.assertEqual(path, root / "llm_result" / "document_pipeline" / "generated_report.md")

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                resolve_output_artifact_path(Path(temp_dir), "../secret.txt")

    def test_reads_generated_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = root / "llm_result" / "document_pipeline" / "generated_report.md"
            report.parent.mkdir(parents=True)
            report.write_text("# Report\n\nSentence.", encoding="utf-8")

            result = execute_artifact_request(
                root,
                ArtifactRequest("read_file", "llm/document_pipeline/generated_report.md"),
            )

            self.assertTrue(result.ok)
            self.assertIn("# Report", result.text)

    def test_lists_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "llm_result" / "document_pipeline"
            output_dir.mkdir(parents=True)
            (output_dir / "generated_report.md").write_text("report", encoding="utf-8")

            result = execute_artifact_request(root, ArtifactRequest("list_outputs", "document_pipeline"))

            self.assertTrue(result.ok)
            self.assertIn("generated_report.md", result.text)


if __name__ == "__main__":
    unittest.main()
