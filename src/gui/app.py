from __future__ import annotations

import argparse

from .config import AppConfig, load_config
from .console_session import ConsoleConfig, LlamaConsoleSession, OpenAICompatibleSession
from .database import ChatRepository
from .gui import ChatGUI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    config: AppConfig = load_config(args.config)

    console_config = ConsoleConfig(
        llama_cli_path=config.llama_cli_path,
        model_path=config.model_path,
        backend=config.backend,
        server_url=config.server_url,
        server_endpoint=config.server_endpoint,
        n_predict=config.n_predict,
        system_prompt=config.system_prompt,
        threads=config.threads,
        ctx_size=config.ctx_size,
        extra_args=config.extra_args,
        startup_timeout=config.startup_timeout,
        response_timeout=config.response_timeout,
        openai_base_url=config.openai_base_url,
        openai_api_key=config.openai_api_key,
        openai_model=config.openai_model,
        openai_embedding_model=config.openai_embedding_model,
        temperature=config.temperature,
    )

    if config.backend == "cli":
        console = LlamaConsoleSession(console_config)
    else:
        console = OpenAICompatibleSession(console_config)
    repository = ChatRepository(config.db_path)
    app = ChatGUI(console=console, repository=repository, app_config=config)
    app.run()
    return 0
