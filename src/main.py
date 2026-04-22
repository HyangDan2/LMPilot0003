from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.config import PipelineConfig, load_config
from src.ingestion.dispatcher import parse_document
from src.ingestion.parsers.base import ParserError
from src.ingestion.scanner import scan_supported_files
from src.models.schemas import ParsedDocument
from src.planner.chunked_planner import ChunkedPlannerSettings, create_chunked_presentation_plan
from src.planner.llm_client import LLMSettings
from src.renderer.pptx_renderer import PptxRenderer
from src.transform.knowledge_map import build_knowledge_map, render_knowledge_map_markdown
from src.utils.io import ensure_dir, save_json, save_text
from src.utils.logging import get_logger


@dataclass(frozen=True)
class RenderPptxResult:
    """Artifacts produced by a render_pptx pipeline run."""

    scanned_files: int
    parsed_documents: int
    normalized_files: list[Path]
    knowledge_map_md: Path
    knowledge_map_json: Path
    planner_json: Path
    output_pptx: Path
    parse_errors: list[str]
    planner_summary_json: Path | None = None
    planner_attempts_json: Path | None = None
    planner_chunk_count: int = 0
    planner_fallback_count: int = 0


def render_pptx_pipeline(user_goal: str, config: PipelineConfig) -> RenderPptxResult:
    """Run scan, parse, normalize, plan, and deterministic PPTX rendering."""

    goal = user_goal.strip()
    if not goal:
        raise ValueError("A user goal is required, for example: Create a 7-slide executive summary")

    logger = get_logger()
    working_dir = ensure_dir(config.working_dir)
    normalized_dir = ensure_dir(config.normalized_dir)
    output_dir = ensure_dir(config.output_dir)

    files = scan_supported_files(working_dir, excluded_dirs={normalized_dir, output_dir})
    if not files:
        raise FileNotFoundError(f"No supported files found in working directory: {working_dir}")

    documents: list[ParsedDocument] = []
    normalized_files: list[Path] = []
    parse_errors: list[str] = []
    for file_path in files:
        try:
            document = parse_document(file_path)
        except ParserError as exc:
            parse_errors.append(f"{file_path}: {exc}")
            logger.warning("Skipping %s: %s", file_path, exc)
            continue
        documents.append(document)
        normalized_path = normalized_dir / f"{document.doc_id}.json"
        save_json(normalized_path, document.to_dict())
        normalized_files.append(normalized_path)

    if not documents:
        joined_errors = "\n".join(parse_errors) if parse_errors else "No parser output."
        raise RuntimeError(f"No documents could be parsed.\n{joined_errors}")

    knowledge_map = build_knowledge_map(documents)
    knowledge_map_json = normalized_dir / "knowledge_map.json"
    knowledge_map_md = normalized_dir / "knowledge_map.md"
    save_json(knowledge_map_json, knowledge_map.to_dict())
    knowledge_map_markdown = render_knowledge_map_markdown(knowledge_map)
    save_text(knowledge_map_md, knowledge_map_markdown)

    planner_result = create_chunked_presentation_plan(
        llm_settings=LLMSettings(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            timeout=config.timeout,
            max_tokens=config.planner_intermediate_max_tokens,
        ),
        planner_settings=ChunkedPlannerSettings(
            chunk_chars=config.planner_chunk_chars,
            min_chunk_chars=config.planner_min_chunk_chars,
            max_retries=config.planner_max_retries,
            intermediate_max_tokens=config.planner_intermediate_max_tokens,
            final_max_tokens=config.planner_final_max_tokens,
            allow_response_format_retry=config.planner_allow_response_format_retry,
            enable_local_fallback=config.planner_enable_local_fallback,
        ),
        user_goal=goal,
        knowledge_map_md=knowledge_map_markdown,
        artifact_dir=normalized_dir,
    )
    plan = planner_result.plan
    planner_json = output_dir / "planner_output.json"
    save_json(planner_json, plan.to_dict())

    output_pptx = PptxRenderer().render(plan, output_dir, config.output_filename)
    return RenderPptxResult(
        scanned_files=len(files),
        parsed_documents=len(documents),
        normalized_files=normalized_files,
        knowledge_map_md=knowledge_map_md,
        knowledge_map_json=knowledge_map_json,
        planner_json=planner_json,
        output_pptx=output_pptx,
        parse_errors=parse_errors,
        planner_summary_json=planner_result.summary_json,
        planner_attempts_json=planner_result.attempts_json,
        planner_chunk_count=planner_result.chunk_count,
        planner_fallback_count=planner_result.fallback_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a PPTX from workspace documents.")
    parser.add_argument("goal", nargs="+", help="Presentation goal, for example: Create a 7-slide summary")
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--normalized-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config(
        working_dir=args.working_dir,
        normalized_dir=args.normalized_dir,
        output_dir=args.output_dir,
        llm_base_url=args.base_url,
        llm_api_key=args.api_key,
        llm_model=args.model,
    )
    result = render_pptx_pipeline(" ".join(args.goal), config)
    print(f"Created: {result.output_pptx}")
    print(f"Knowledge map: {result.knowledge_map_md}")
    print(f"Planner JSON: {result.planner_json}")
    if result.parse_errors:
        print(f"Skipped {len(result.parse_errors)} file(s); see console warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
