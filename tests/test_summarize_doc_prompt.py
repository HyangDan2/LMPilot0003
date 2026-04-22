import unittest
from pathlib import Path

from src.document_pipeline.high_level.summarize_doc import (
    _document_chunk_prompt,
    _document_consolidation_prompt,
    _workspace_final_prompt,
    _workspace_group_prompt,
    SummaryBudget,
)
from src.document_pipeline.schemas import DocumentMetadata, ExtractedBlock, ExtractedDocument, Provenance, SourceInfo


class SummarizeDocPromptTests(unittest.TestCase):
    def test_document_chunk_prompt_uses_system_instruction_and_natural_language_user_prompt(self) -> None:
        document = _sample_document()

        messages = _document_chunk_prompt(document, "Block 0 (text, content, slide 1)\nRevenue grew by 10%.")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Do not explain your task", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn('Please read the following excerpts from the document "Report"', messages[1]["content"])
        self.assertNotIn("Input budget", messages[1]["content"])
        self.assertNotIn("Output budget", messages[1]["content"])
        self.assertNotIn("chunk 1/1", messages[1]["content"])

    def test_consolidation_prompt_keeps_instructions_in_system_message(self) -> None:
        document = _sample_document()

        messages = _document_consolidation_prompt(document, ["first note", "second note"], SummaryBudget())

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Do not expose reasoning", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("The notes below each summarize different excerpts", messages[1]["content"])
        self.assertNotIn("Output budget", messages[1]["content"])

    def test_workspace_prompts_are_written_as_natural_language_requests(self) -> None:
        group_messages = _workspace_group_prompt("Document name: notes.pdf\nSummary:\nA spectrometer document.")
        final_messages = _workspace_final_prompt(["A draft workspace summary."], SummaryBudget())

        self.assertIn("Please write a concise workspace-level summary in natural language", group_messages[1]["content"])
        self.assertNotIn("group 1/1", group_messages[1]["content"])
        self.assertIn("Please combine them into one final workspace summary", final_messages[1]["content"])
        self.assertNotIn("Input budget", final_messages[1]["content"])


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
