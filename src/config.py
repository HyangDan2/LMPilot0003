from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the render_pptx workspace pipeline."""

    working_dir: Path
    normalized_dir: Path
    output_dir: Path
    output_filename: str | None = None
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    timeout: float = 120.0
    planner_chunk_chars: int = 6000
    planner_min_chunk_chars: int = 800
    planner_max_retries: int = 3
    planner_intermediate_max_tokens: int = 1024
    planner_final_max_tokens: int = 2048
    planner_allow_response_format_retry: bool = True
    planner_enable_local_fallback: bool = True


def load_config(
    *,
    working_dir: str | None = None,
    normalized_dir: str | None = None,
    output_dir: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
) -> PipelineConfig:
    """Load configuration from explicit values first, then environment variables."""

    return PipelineConfig(
        working_dir=Path(working_dir or os.environ.get("WORKING_DIR", "data/working")),
        normalized_dir=Path(normalized_dir or os.environ.get("NORMALIZED_DIR", "data/normalized")),
        output_dir=Path(output_dir or os.environ.get("OUTPUT_DIR", "data/outputs")),
        output_filename=None,
        llm_base_url=llm_base_url or os.environ.get("LLM_BASE_URL", ""),
        llm_api_key=llm_api_key or os.environ.get("LLM_API_KEY", ""),
        llm_model=llm_model or os.environ.get("LLM_MODEL", ""),
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        planner_chunk_chars=int(os.environ.get("PLANNER_CHUNK_CHARS", "6000")),
        planner_min_chunk_chars=int(os.environ.get("PLANNER_MIN_CHUNK_CHARS", "800")),
        planner_max_retries=int(os.environ.get("PLANNER_MAX_RETRIES", "3")),
        planner_intermediate_max_tokens=int(os.environ.get("PLANNER_INTERMEDIATE_MAX_TOKENS", "1024")),
        planner_final_max_tokens=int(os.environ.get("PLANNER_FINAL_MAX_TOKENS", "2048")),
        planner_allow_response_format_retry=_parse_bool(
            os.environ.get("PLANNER_ALLOW_RESPONSE_FORMAT_RETRY", "true")
        ),
        planner_enable_local_fallback=_parse_bool(os.environ.get("PLANNER_ENABLE_LOCAL_FALLBACK", "true")),
    )


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() not in {"0", "false", "no", "off"}
