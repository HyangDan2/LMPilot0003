import unittest
from pathlib import Path

from src.document_pipeline.high_level.summarize_doc import (
    _document_chunk_prompt,
    _document_consolidation_prompt,
    _workspace_final_prompt,
    _workspace_group_prompt,
    render_workspace_summary_markdown,
    SummaryBudget,
    WorkspaceSummarySections,
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
        final_messages = _workspace_final_prompt(
            [
                WorkspaceSummarySections(
                    overall_summary="A draft workspace summary.",
                    features=["Feature one", "Feature two", "Feature three"],
                    next_action="Follow up with the product team.",
                )
            ],
            SummaryBudget(),
        )

        self.assertIn("Please write a detailed workspace-level summary in natural language", group_messages[1]["content"])
        self.assertIn('"Overall Summary", "Features", and "Next Action"', group_messages[1]["content"])
        self.assertNotIn("group 1/1", group_messages[1]["content"])
        self.assertIn("Please combine them into one final workspace summary", final_messages[1]["content"])
        self.assertIn("list exactly 3 numbered items", final_messages[1]["content"])
        self.assertNotIn("Input budget", final_messages[1]["content"])

    def test_summary_budget_doubles_output_token_allowances(self) -> None:
        budget = SummaryBudget()

        self.assertEqual(budget.per_doc_output_tokens, 480)
        self.assertEqual(budget.consolidate_output_tokens, 720)
        self.assertEqual(budget.workspace_output_tokens, 3000)

    def test_workspace_summary_markdown_splits_plain_paragraph_sentences(self) -> None:
        markdown = render_workspace_summary_markdown(
            [],
            WorkspaceSummarySections(
                overall_summary="First sentence. Second sentence.",
                features=["Feature one. Keep list intact.", "Feature two.", "Feature three."],
                next_action="Review this. Then act.",
            ),
        )

        self.assertIn("First sentence.\nSecond sentence.", markdown)
        self.assertIn("1. Feature one. Keep list intact.", markdown)
        self.assertIn("Review this.\nThen act.", markdown)


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
