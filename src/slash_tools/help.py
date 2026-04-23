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

/summarize_doc [--engineering True|False] [PATH]
Generate hierarchical LLM-backed summaries from extracted documents without sending the full raw corpus in one request.
If PATH is given, summarize only that file inside the attached folder.
The saved workspace summary is structured into Overall Summary, Features (3 items), and Next Action.
With --engineering True, the saved summary uses Features, Quantitative Information, and Recommended Action.
Summary output length is controlled by substantial minimum-detail prompt instructions; token budgets remain bounded safety caps.
The saved workspace_summary.md formats prose with one sentence per line, including multi-sentence list items.

/summarize_docs [--engineering True|False]
Run /summarize_doc-style single-file summaries sequentially for every supported document in the attached folder.
Each processable document gets its own summary run under HD2docpipe/summaries/.

Normal chat generated-artifact access:
When a model needs a previous generated output, it can request:
  [read_output] HD2docpipe/artifacts/generated_report.md [/read_output]
  [list_outputs] HD2docpipe/artifacts [/list_outputs]
  [read_file] HD2docpipe/summaries/workspace_20260422_231501/workspace_summary.md [/read_file]
Only files under the attached folder's HD2docpipe/ directory can be read or listed.

Examples:
  /extract_single_doc design_review.pptx
  /extract_docs
  /summarize_doc
  /summarize_doc --engineering True design_review.pptx
  /summarize_docs
  /summarize_doc design_review.pptx
  /generate_markdown

Supported file types:
.pptx, .docx, .xlsx, .pdf

Suggested flow:
1. Attach a folder.
2. Run /extract_docs.
3. Run /workspace_status.

Workspace folder memory:
The app remembers the last attached folder in config. If that folder no longer exists, it starts with no attached folder.

Summary flow:
1. Run /extract_docs.
2. Run /summarize_doc.
3. Run /workspace_status.

Advanced evidence flow:
1. Run /extract_docs.
2. Run /build_doc_map.
3. Run /generate_markdown.

Automatic saved outputs:
- /extract_docs saves HD2docpipe/artifacts/extracted_documents.json
- /extract_docs saves HD2docpipe/artifacts/extraction_manifest.json
- /extract_single_doc saves HD2docpipe/artifacts/FILE_SCOPE/documents/DOCUMENT_ID.json
- /extract_single_doc saves HD2docpipe/artifacts/FILE_SCOPE/extracted_documents.json
- /build_doc_map saves HD2docpipe/artifacts/document_map.json or HD2docpipe/artifacts/FILE_SCOPE/document_map.json
- /generate_markdown saves HD2docpipe/artifacts/generated_report.md or HD2docpipe/artifacts/FILE_SCOPE/generated_report.md
- /summarize_doc saves HD2docpipe/summaries/RUN_NAME/document_summaries.json
- /summarize_doc saves HD2docpipe/summaries/RUN_NAME/workspace_summary.md
- /summarize_doc stores structured workspace_summary fields in document_summaries.json.
  Standard mode: overall_summary, features, next_action
  Engineering mode: features, quantitative_information, recommended_action
- /summarize_docs creates one summary run per supported document.

Summary run naming:
- Folder-wide summaries use workspace_TIMESTAMP
- Single-file summaries use FILE_STEM_TIMESTAMP

Single-file pipeline scopes:
- Single-file extraction and follow-up artifacts use HD2docpipe/artifacts/FILE_SCOPE/

Not included yet:
- summarize_map
- integrated_result
"""


def help_command(args, working_folder, context, progress=None) -> SlashToolResult:
    return SlashToolResult(
        text=HELP_TEXT,
        tool_name="/help",
        next_actions=["/extract_docs", "/workspace_status"],
    )
