from __future__ import annotations

from dataclasses import dataclass

from src.gui.llm_client import (
    LLMClientError as GuiLLMClientError,
    OpenAICompatibleClient,
    OpenAIConnectionSettings,
)


class LLMClientError(Exception):
    """Raised when the OpenAI-compatible planner request fails."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    max_tokens: int = 1024


class OpenAICompatibleLLMClient:
    """Planner LLM client that shares the GUI chat transport."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client = OpenAICompatibleClient(
            OpenAIConnectionSettings(
                base_url=settings.base_url,
                api_key=settings.api_key,
                model=settings.model,
                temperature=0,
                timeout=settings.timeout,
                max_tokens=settings.max_tokens,
            )
        )

    def chat_completion(self, messages: list[dict[str, str]], *, response_format: bool = True) -> str:
        try:
            return self._client.chat_completion(
                messages,
                response_format={"type": "json_object"} if response_format else None,
            )
        except GuiLLMClientError as exc:
            raise LLMClientError(f"Planner request failed: {exc}") from exc
