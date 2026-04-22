from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .llm_client import OpenAIConnectionSettings


DEFAULT_CONNECTION_SETTINGS_PATH = "openai_settings.json"


@dataclass
class AppConfig:
    llama_cli_path: str
    model_path: str
    config_path: str = ""
    backend: str = "openai"
    server_url: str = ""
    server_endpoint: str = "auto"
    n_predict: int = 512
    system_prompt: str = "You are a helpful assistant."
    threads: int = 4
    ctx_size: int = 2048
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: float = 180.0
    response_timeout: float = 180.0
    db_path: str = "./data/app.db"
    window_title: str = "Gemma Console GUI (PySide6)"
    window_width: int = 1000
    window_height: int = 720
    response_token_reserve: int = 256
    max_prompt_chars: int = 12000
    connection_settings_path: str = DEFAULT_CONNECTION_SETTINGS_PATH
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    openai_embedding_model: str = ""
    temperature: float = 0.7
    recent_message_limit: int = 40
    rag_top_k: int = 5
    rag_min_score: float = 0.2
    rag_chunk_chars: int = 1200
    rag_chunk_overlap: int = 150
    memory_context_char_limit: int = 4000
    last_working_folder: str = ""

    def connection_settings(self) -> OpenAIConnectionSettings:
        return OpenAIConnectionSettings(
            base_url=self.openai_base_url,
            api_key=self.openai_api_key,
            model=self.openai_model,
            embedding_model=self.openai_embedding_model,
            temperature=self.temperature,
            max_tokens=self.n_predict,
            timeout=self.response_timeout,
        )


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    connection_settings_path = raw.get("connection_settings_path", DEFAULT_CONNECTION_SETTINGS_PATH)
    saved_connection = load_connection_settings(connection_settings_path)

    base_url = raw.get("openai_base_url", raw.get("base_url", raw.get("server_url", "")))
    api_key = raw.get("openai_api_key", raw.get("api_key", ""))
    model = raw.get("openai_model", raw.get("model", ""))
    embedding_model = raw.get("openai_embedding_model", raw.get("embedding_model", ""))
    temperature = float(raw.get("temperature", saved_connection.temperature))
    n_predict = int(raw.get("n_predict", raw.get("max_tokens", saved_connection.max_tokens)))

    if saved_connection.base_url:
        base_url = saved_connection.base_url
    if saved_connection.api_key:
        api_key = saved_connection.api_key
    if saved_connection.model:
        model = saved_connection.model
    if saved_connection.embedding_model:
        embedding_model = saved_connection.embedding_model
    temperature = saved_connection.temperature if saved_connection.temperature != 0.7 else temperature
    n_predict = saved_connection.max_tokens if saved_connection.max_tokens != 512 else n_predict

    last_working_folder = str(raw.get("last_working_folder", "") or "").strip()
    if last_working_folder:
        folder_path = Path(last_working_folder).expanduser()
        last_working_folder = str(folder_path.resolve()) if folder_path.exists() and folder_path.is_dir() else ""

    return AppConfig(
        config_path=str(config_path),
        llama_cli_path=raw.get(
            "llama_cli_path",
            "/home/pi/Downloads/llama.cpp/build/bin/llama-cli",
        ),
        model_path=raw.get(
            "model_path",
            "/home/pi/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27/gemma-3-1b-it-Q4_K_M.gguf",
        ),
        backend=raw.get("backend", "openai"),
        server_url=raw.get("server_url", base_url),
        server_endpoint=raw.get("server_endpoint", "auto"),
        n_predict=n_predict,
        system_prompt=raw.get("system_prompt", "You are a helpful assistant."),
        threads=int(raw.get("threads", 4)),
        ctx_size=int(raw.get("ctx_size", 2048)),
        extra_args=list(raw.get("extra_args", [])),
        startup_timeout=float(raw.get("startup_timeout", 180.0)),
        response_timeout=float(raw.get("response_timeout", 180.0)),
        db_path=raw.get("db_path", "./data/app.db"),
        window_title=raw.get("window_title", "Gemma Console GUI (PySide6)"),
        window_width=int(raw.get("window_width", 1000)),
        window_height=int(raw.get("window_height", 720)),
        response_token_reserve=int(raw.get("response_token_reserve", 256)),
        max_prompt_chars=int(raw.get("max_prompt_chars", 12000)),
        connection_settings_path=connection_settings_path,
        openai_base_url=base_url,
        openai_api_key=api_key,
        openai_model=model,
        openai_embedding_model=embedding_model,
        temperature=temperature,
        recent_message_limit=int(raw.get("recent_message_limit", 40)),
        rag_top_k=int(raw.get("rag_top_k", 5)),
        rag_min_score=float(raw.get("rag_min_score", 0.2)),
        rag_chunk_chars=int(raw.get("rag_chunk_chars", 1200)),
        rag_chunk_overlap=int(raw.get("rag_chunk_overlap", 150)),
        memory_context_char_limit=int(raw.get("memory_context_char_limit", 4000)),
        last_working_folder=last_working_folder,
    )


def load_connection_settings(path: str) -> OpenAIConnectionSettings:
    settings_path = Path(path)
    if not settings_path.exists():
        return OpenAIConnectionSettings()

    try:
        with settings_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return OpenAIConnectionSettings()

    return OpenAIConnectionSettings(
        base_url=str(raw.get("base_url", "")),
        api_key=str(raw.get("api_key", "")),
        model=str(raw.get("model", "")),
        embedding_model=str(raw.get("embedding_model", "")),
        temperature=float(raw.get("temperature", 0.7)),
        max_tokens=int(raw.get("max_tokens", 512)),
        timeout=float(raw.get("timeout", 180.0)),
    )


def save_connection_settings(path: str, settings: OpenAIConnectionSettings) -> None:
    settings_path = Path(path)
    if settings_path.parent != Path("."):
        settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def save_config(config: AppConfig) -> None:
    if not config.config_path.strip():
        return
    config_path = Path(config.config_path)
    if config_path.parent != Path("."):
        config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload.pop("config_path", None)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
