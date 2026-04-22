from __future__ import annotations

from .results import SlashToolResult


HELP_TEXT = """LLM Workspace Help

Local slash tools:

/help
Show this help text.

/detect_file_type PATH
Detect extension, MIME type, document family, and confidence for a file inside the attached folder.

/read_file_info PATH
Show file size and SHA-256 hash for a file inside the attached folder.

/normalize_text TEXT
Normalize whitespace, Unicode compatibility characters, and control characters.

/extract_single_doc PATH
Extract one supported document from the attached folder.

/extract_docs
Extract all supported documents from the attached folder.

/build_doc_map
Build a structural document map from the latest /extract_docs result. If needed, it loads saved extracted_documents.json.

/workspace_status
Show which document-pipeline artifacts are available in the attached folder.

/generate_markdown
Generate a deterministic markdown report from extracted evidence.

/summarize_file PATH [--no-llm] [--generate-detail true|false] [--llm-input-chars N] [query...]
Summarize one supported file from the attached folder.
Use --generate-detail true to save optional LLM page, slide, sheet, or file summaries as detail_summaries.json and detail_summaries.md. Detail progress prints every item as it is processed and completed.
The saved file summary uses Summary, Source Details, and Open Issues and Next Actions.
File summaries are saved under llm_result/document_pipeline/file_summaries/DOCUMENT_ID/.

/generate_report [--no-llm] [--fresh] [--generate-detail true|false] [--llm-input-chars N] [query...]
Run extraction, mapping, output planning, representative evidence selection, optional local ranked evidence grouping, and one final engineering Markdown call in one step.
No prerequisite slash command is required. It reuses unchanged extraction artifacts unless --fresh is provided.
Use --generate-detail true to save optional LLM page, slide, sheet, or file summaries without adding them to the final report prompt. Detail progress prints every item as it is processed and completed.
Progress, ranked-groups mode status, timings, and final Markdown stream into the chat while generated_report.md is saved.
The saved report uses Summary, Source Documents, and Open Issues and Next Actions as top-level sections.
Summary may include What the Document Explicitly Describes, Main Methods or Components Explicitly Mentioned, Quantitative Values Explicitly Present, Explicit Limitations or Constraints, and Unclear or Not Specified in Selected Evidence.
Saved report paragraphs place each sentence on a separate line.
Other sessions can run slash tools while /generate_report is active. Stop cancels the selected session's running tool.

Normal chat generated-artifact access:
When a model needs a previous generated output, it can request:
  [read_output] document_pipeline/generated_report.md [/read_output]
  [list_outputs] document_pipeline [/list_outputs]
Qwen-style aliases such as [read_file] llm/document_pipeline/generated_report.md [/read_file] are supported.
Only files under the attached folder's llm_result/ directory can be read or listed.

Examples:
  /summarize_file design_review.pptx
  /summarize_file test_results.xlsx summarize risks and quantitative results
  /generate_report summarize all output in this folder
  /generate_report summarize about project risks
  /generate_report --no-llm summarize briefly

Supported file types:
.pptx, .docx, .xlsx, .pdf

Suggested flow:
1. Attach a folder.
2. Run /summarize_file FILE or /generate_report.
3. Run /workspace_status.

Advanced evidence flow:
1. Run /extract_docs.
2. Run /build_doc_map.
3. Run /generate_markdown.

Automatic saved outputs:
- /extract_docs saves llm_result/document_pipeline/extracted_documents.json
- /extract_docs saves llm_result/document_pipeline/extraction_manifest.json
- /extract_single_doc saves llm_result/document_pipeline/documents/DOCUMENT_ID.json
- /build_doc_map saves llm_result/document_pipeline/document_map.json
- /generate_report saves llm_result/document_pipeline/output_plan.json
- /generate_report saves llm_result/document_pipeline/selected_evidence.json
- /generate_report saves llm_result/document_pipeline/evidence_groups.json
- /generate_report saves llm_result/document_pipeline/selected_evidence_groups.json
- /generate_report saves llm_result/document_pipeline/group_summaries.json
- /generate_report saves llm_result/document_pipeline/recursive_summary_levels.json
- /generate_report saves llm_result/document_pipeline/final_prompt_preview.txt
- /generate_report saves llm_result/document_pipeline/detail_summaries.json
- /generate_report saves llm_result/document_pipeline/detail_summaries.md
- /generate_report saves llm_result/document_pipeline/llm_report_attempts.json
- /summarize_file saves llm_result/document_pipeline/file_summaries/DOCUMENT_ID/detail_summaries.json
- /summarize_file saves llm_result/document_pipeline/file_summaries/DOCUMENT_ID/detail_summaries.md
- /summarize_file saves llm_result/document_pipeline/file_summaries/DOCUMENT_ID/generated_summary.md
- /generate_markdown saves llm_result/document_pipeline/generated_report.md
- /generate_report saves llm_result/document_pipeline/generated_report.md

Not included yet:
- summarize_map
- integrated_result
- active /generate_report cancellation
"""


def help_command(args, working_folder, context, progress=None) -> SlashToolResult:
    return SlashToolResult(
        text=HELP_TEXT,
        tool_name="/help",
        next_actions=["/extract_docs", "/workspace_status"],
    )
