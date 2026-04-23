import unittest
from pathlib import Path

from src.document_pipeline.high_level.summarize_doc import (
    _document_chunk_prompt,
    _document_consolidation_prompt,
    _parse_workspace_summary_sections,
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
        self.assertIn("Preserve specific numbers", messages[1]["content"])
        self.assertNotIn("Input budget", messages[1]["content"])
        self.assertNotIn("Output budget", messages[1]["content"])
        self.assertNotIn("chunk 1/1", messages[1]["content"])

    def test_consolidation_prompt_keeps_instructions_in_system_message(self) -> None:
        document = _sample_document()

        messages = _document_consolidation_prompt(document, ["first note", "second note"], SummaryBudget())

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Do not expose reasoning", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("The notes below summarize different excerpts", messages[1]["content"])
        self.assertIn("retain concrete quantitative or technical details", messages[1]["content"])
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

        self.assertIn("Write a detailed workspace-level summary", group_messages[1]["content"])
        self.assertIn('"Overall Summary", "Features", and "Next Action"', group_messages[1]["content"])
        self.assertIn("at least 6 paragraphs", group_messages[1]["content"])
        self.assertNotIn("group 1/1", group_messages[1]["content"])
        self.assertIn("Combine these draft workspace summaries", final_messages[1]["content"])
        self.assertIn("write at least 6 sentences for each item", final_messages[1]["content"])
        self.assertNotIn("Input budget", final_messages[1]["content"])

    def test_summary_budget_uses_bounded_output_token_allowances(self) -> None:
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
        self.assertIn("1. Feature one.\n   Keep list intact.", markdown)
        self.assertIn("Review this.\nThen act.", markdown)

    def test_workspace_summary_markdown_splits_features_and_nextaction_sentences(self) -> None:
        sections = _parse_workspace_summary_sections(
            "Overall Summary\nA concise summary.\n\n"
            "Features\n"
            "1. First feature starts here.\nIt continues with another sentence.\n"
            "2. Second feature has one sentence. It has another sentence.\n"
            "3. Third feature is short.\n\n"
            "NextAction\nReview the implementation. Validate the output formatting.",
        )
        markdown = render_workspace_summary_markdown([], sections)

        self.assertIn("1. First feature starts here.\n   It continues with another sentence.", markdown)
        self.assertIn("2. Second feature has one sentence.\n   It has another sentence.", markdown)
        self.assertIn("Review the implementation.\nValidate the output formatting.", markdown)

    def test_engineering_workspace_prompt_and_markdown_use_engineering_sections(self) -> None:
        group_messages = _workspace_group_prompt("Document name: notes.pdf\nSummary:\nLatency is 10 ms.", engineering=True)
        final_messages = _workspace_final_prompt(
            [
                WorkspaceSummarySections(
                    features=["Fast path", "Telemetry", "Failover"],
                    quantitative_information="Latency is 10 ms.",
                    recommended_action="Validate load limits.",
                    mode="engineering",
                )
            ],
            SummaryBudget(),
            engineering=True,
        )

        self.assertIn('"Features", "Quantitative Information", and "Recommended Action"', group_messages[1]["content"])
        self.assertIn("Do not add a report title", group_messages[1]["content"])
        self.assertIn("write at least 8 sentences for each feature", group_messages[1]["content"])
        self.assertIn("Quantitative Information should preserve concrete source values", final_messages[1]["content"])
        self.assertIn("include at least 10 actionable steps", final_messages[1]["content"])

        sections = _parse_workspace_summary_sections(
            "Features\n1. Fast path\n2. Telemetry\n3. Failover\n\n"
            "Quantitative Information\nLatency is 10 ms.\n\n"
            "Recommended Action\nValidate load limits.",
            engineering=True,
        )
        markdown = render_workspace_summary_markdown([], sections)

        self.assertIn("## Features", markdown)
        self.assertIn("## Quantitative Information", markdown)
        self.assertIn("## Recommended Action", markdown)
        self.assertNotIn("## Overall Summary", markdown)
        self.assertEqual(sections.to_dict()["mode"], "engineering")

    def test_engineering_parser_handles_bold_headings_and_extra_report_title(self) -> None:
        sections = _parse_workspace_summary_sections(
            "**ESRGAN Workspace Summary**\n\n"
            "**Features**\n\n"
            "1. ESRGAN utilizes a deep network architecture featuring a Recurrent Residual Dense Block (RRDB).\n"
            "2. The model employs perceptual loss, adversarial training, and network interpolation.\n"
            "3. Training leverages the DF2K dataset and varying patch sizes.\n\n"
            "**Quantitative Information**\n\n"
            "* ESRGAN is compared with EDSR, RCAN, and EnhanceNet on DIV2K, Flickr2K, and Set5.\n"
            "* PSNR scores are reported as higher than competing models.\n"
            "* The model was trained using the DF2K dataset.\n\n"
            "**Recommended Action**\n\n"
            "Investigate the impact of different patch sizes and model capacities.",
            engineering=True,
        )
        markdown = render_workspace_summary_markdown([], sections)

        self.assertEqual(len(sections.features), 3)
        self.assertIn("RRDB", sections.features[0])
        self.assertIn("DIV2K", sections.quantitative_information)
        self.assertIn("patch sizes", sections.recommended_action)
        self.assertNotIn("No distinct features were identified", markdown)
        self.assertNotIn("**ESRGAN Workspace Summary**", markdown)


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
