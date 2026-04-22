from __future__ import annotations

import http.client
import json
import threading
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlparse


class LLMClientError(Exception):
    pass


FINAL_TEXT_FIELD_FALLBACKS = ("text", "output_text")
CHOICE_FINAL_TEXT_FIELDS = ("content", "output_text", "response", "completion")
REASONING_TEXT_FIELDS = ("reasoning", "reasoning_content")


@dataclass(frozen=True)
class ChatStreamChunk:
    kind: str
    text: str = ""


@dataclass
class OpenAIConnectionSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    embedding_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 180.0


class OpenAICompatibleClient:
    def __init__(self, settings: OpenAIConnectionSettings) -> None:
        self.settings = settings
        self._active_connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self._validate_for_chat()
        payload: dict[str, Any] = {
            "model": self.settings.model.strip(),
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        data = self._request_json("POST", "/chat/completions", payload)
        return self._extract_chat_text(data)

    def stream_chat_completion(self, messages: list[dict[str, Any]]) -> Iterator[ChatStreamChunk]:
        self._validate_for_chat()
        payload: dict[str, Any] = {
            "model": self.settings.model.strip(),
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
        }
        yield from self._extract_stream_chat_text(
            self._request_stream_events("POST", "/chat/completions", payload)
        )

    def embeddings(self, inputs: list[str], model: str | None = None) -> list[list[float]]:
        self._validate_base_url()
        embedding_model = (model or self.settings.embedding_model or self.settings.model).strip()
        if not embedding_model:
            raise LLMClientError("Embedding Model is required before building vector memory.")
        if not inputs:
            return []

        payload: dict[str, Any] = {
            "model": embedding_model,
            "input": inputs,
        }
        data = self._request_json("POST", "/embeddings", payload)
        return self._extract_embeddings(data)

    def list_models(self) -> list[str]:
        self._validate_base_url()
        data = self._request_json("GET", "/models")
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise LLMClientError("Malformed /models response: expected a data list.")
        names: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                names.append(item["id"])
        return names

    def test_connection(self) -> str:
        try:
            models = self.list_models()
        except LLMClientError as models_error:
            if not self.settings.model.strip():
                raise LLMClientError(
                    f"/models failed and Model Name is empty, so a chat fallback cannot run. {models_error}"
                ) from models_error
            answer = self.chat_completion([{"role": "user", "content": "Reply with OK."}])
            if not answer.strip():
                raise LLMClientError("Connection test returned an empty response.")
            return "Connection test succeeded with chat completions."

        if models:
            return f"Connection test succeeded. {len(models)} model(s) available."
        return "Connection test succeeded. /models returned no model IDs."

    def close_active_request(self) -> None:
        with self._lock:
            self._stop_requested = True
            conn = self._active_connection
        if conn is not None:
            conn.close()

    def _validate_for_chat(self) -> None:
        self._validate_base_url()
        if not self.settings.model.strip():
            raise LLMClientError("Model Name is required before sending a prompt.")

    def _validate_base_url(self) -> None:
        parsed = urlparse(self._normalized_base_url())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise LLMClientError("Base URL must be a valid http:// or https:// URL.")

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        base_url = self._normalized_base_url()
        parsed = urlparse(base_url)
        path = self._join_paths(parsed.path, endpoint)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        conn = self._create_connection(parsed)
        headers = {"Content-Type": "application/json"}
        api_key = self.settings.api_key.strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with self._lock:
            self._stop_requested = False
            self._active_connection = conn

        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            response_body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise LLMClientError(
                    f"HTTP {response.status} from {path}: {self._safe_error_body(response_body)}"
                )
            try:
                data = json.loads(response_body)
            except json.JSONDecodeError as exc:
                raise LLMClientError(f"Malformed JSON response from {path}: {response_body[:500]}") from exc
            if not isinstance(data, dict):
                raise LLMClientError(f"Malformed response from {path}: expected a JSON object.")
            return data
        except LLMClientError:
            raise
        except (OSError, AttributeError, http.client.HTTPException) as exc:
            if self._stop_requested:
                raise LLMClientError("Generation stopped.") from exc
            raise LLMClientError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            with self._lock:
                if self._active_connection is conn:
                    self._active_connection = None
            conn.close()

    def _request_stream_events(
        self, method: str, endpoint: str, payload: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        base_url = self._normalized_base_url()
        parsed = urlparse(base_url)
        path = self._join_paths(parsed.path, endpoint)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        conn = self._create_connection(parsed)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        api_key = self.settings.api_key.strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with self._lock:
            self._stop_requested = False
            self._active_connection = conn

        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            if response.status >= 400:
                response_body = response.read().decode("utf-8", errors="replace")
                raise LLMClientError(
                    f"HTTP {response.status} from {path}: {self._safe_error_body(response_body)}"
                )

            while True:
                line_bytes = response.readline()
                if not line_bytes:
                    return
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue

                data_text = line[5:].strip()
                if data_text == "[DONE]":
                    return
                try:
                    event = json.loads(data_text)
                except json.JSONDecodeError as exc:
                    raise LLMClientError(f"Malformed streaming JSON response from {path}: {data_text[:500]}") from exc
                if not isinstance(event, dict):
                    raise LLMClientError(f"Malformed streaming response from {path}: expected a JSON object.")
                yield event
        except LLMClientError:
            raise
        except (OSError, AttributeError, http.client.HTTPException) as exc:
            if self._stop_requested:
                raise LLMClientError("Generation stopped.") from exc
            raise LLMClientError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            with self._lock:
                if self._active_connection is conn:
                    self._active_connection = None
            conn.close()

    def _create_connection(
        self, parsed_url
    ) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if parsed_url.scheme == "https":
            return http.client.HTTPSConnection(
                parsed_url.hostname,
                port=parsed_url.port,
                timeout=self.settings.timeout,
            )
        return http.client.HTTPConnection(
            parsed_url.hostname,
            port=parsed_url.port,
            timeout=self.settings.timeout,
        )

    def _normalized_base_url(self) -> str:
        return self.settings.base_url.strip().rstrip("/")

    @staticmethod
    def _join_paths(base_path: str, endpoint: str) -> str:
        base = base_path.rstrip("/")
        suffix = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return f"{base}{suffix}" if base else suffix

    @staticmethod
    def _extract_chat_text(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError("Malformed chat response: missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMClientError(
                "Malformed chat response: choice is not an object. "
                f"First choice: {OpenAICompatibleClient._preview_payload(first_choice)}"
            )
        text = OpenAICompatibleClient._extract_choice_text(
            first_choice,
            strip=True,
            include_reasoning=False,
        )
        if text:
            return text
        reasoning = OpenAICompatibleClient._extract_reasoning_text(first_choice, strip=True)
        if reasoning:
            raise LLMClientError("Backend returned reasoning only, but no final assistant answer.")
        raise LLMClientError(
            "Malformed chat response: missing assistant content."
            f"{OpenAICompatibleClient._finish_reason_text(first_choice)} "
            f"First choice: {OpenAICompatibleClient._preview_payload(first_choice)}"
        )

    @staticmethod
    def _extract_stream_chat_text(events: Iterator[dict[str, Any]]) -> Iterator[ChatStreamChunk]:
        emitted_final_text = False
        saw_reasoning = False
        last_choice: Any = None
        for event in events:
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            first_choice = choices[0]
            last_choice = first_choice
            if not isinstance(first_choice, dict):
                raise LLMClientError(
                    "Malformed streaming chat response: choice is not an object. "
                    f"First choice: {OpenAICompatibleClient._preview_payload(first_choice)}"
                )
            text = OpenAICompatibleClient._extract_choice_text(
                first_choice,
                strip=False,
                include_reasoning=False,
            )
            if text is not None and text != "":
                emitted_final_text = True
                yield ChatStreamChunk(kind="final", text=text)
                continue

            reasoning = OpenAICompatibleClient._extract_reasoning_text(first_choice, strip=False)
            if reasoning is not None and reasoning != "":
                saw_reasoning = True
                yield ChatStreamChunk(kind="reasoning")

        if not emitted_final_text and saw_reasoning:
            raise LLMClientError("Backend returned reasoning only, but no final assistant answer.")
        if not emitted_final_text:
            raise LLMClientError(
                "Malformed streaming chat response: missing assistant content. "
                f"First choice: {OpenAICompatibleClient._preview_payload(last_choice)}"
            )

    @staticmethod
    def _extract_choice_text(
        first_choice: dict[str, Any],
        strip: bool,
        include_reasoning: bool = False,
    ) -> str | None:
        message = first_choice.get("message")
        if isinstance(message, dict):
            block_text = OpenAICompatibleClient._extract_text_value(message.get("content"), strip=strip)
            if OpenAICompatibleClient._has_text(block_text, strip=strip):
                return block_text
        text = OpenAICompatibleClient._extract_text_value(first_choice.get("text"), strip=strip)
        if OpenAICompatibleClient._has_text(text, strip=strip):
            return text
        delta = first_choice.get("delta")
        if isinstance(delta, dict):
            block_text = OpenAICompatibleClient._extract_text_value(delta.get("content"), strip=strip)
            if OpenAICompatibleClient._has_text(block_text, strip=strip):
                return block_text
            for key in FINAL_TEXT_FIELD_FALLBACKS:
                block_text = OpenAICompatibleClient._extract_text_value(delta.get(key), strip=strip)
                if OpenAICompatibleClient._has_text(block_text, strip=strip):
                    return block_text
        if isinstance(message, dict):
            for key in FINAL_TEXT_FIELD_FALLBACKS:
                block_text = OpenAICompatibleClient._extract_text_value(message.get(key), strip=strip)
                if OpenAICompatibleClient._has_text(block_text, strip=strip):
                    return block_text
        for key in CHOICE_FINAL_TEXT_FIELDS:
            block_text = OpenAICompatibleClient._extract_text_value(first_choice.get(key), strip=strip)
            if OpenAICompatibleClient._has_text(block_text, strip=strip):
                return block_text
        if include_reasoning:
            return OpenAICompatibleClient._extract_reasoning_text(first_choice, strip=strip)
        return None

    @staticmethod
    def _extract_reasoning_text(first_choice: dict[str, Any], strip: bool) -> str | None:
        delta = first_choice.get("delta")
        if isinstance(delta, dict):
            for key in REASONING_TEXT_FIELDS:
                reasoning = OpenAICompatibleClient._extract_text_value(delta.get(key), strip=strip)
                if OpenAICompatibleClient._has_text(reasoning, strip=strip):
                    return reasoning

        message = first_choice.get("message")
        if isinstance(message, dict):
            for key in REASONING_TEXT_FIELDS:
                reasoning = OpenAICompatibleClient._extract_text_value(message.get(key), strip=strip)
                if OpenAICompatibleClient._has_text(reasoning, strip=strip):
                    return reasoning

        for key in REASONING_TEXT_FIELDS:
            reasoning = OpenAICompatibleClient._extract_text_value(first_choice.get(key), strip=strip)
            if OpenAICompatibleClient._has_text(reasoning, strip=strip):
                return reasoning
        return None

    @staticmethod
    def _finish_reason_text(first_choice: dict[str, Any]) -> str:
        finish_reason = first_choice.get("finish_reason")
        return f" Finish reason: {finish_reason}." if isinstance(finish_reason, str) else ""

    @staticmethod
    def _has_text(value: str | None, strip: bool) -> bool:
        if value is None:
            return False
        return bool(value.strip()) if strip else value != ""

    @staticmethod
    def _extract_text_value(value: Any, strip: bool = True) -> str | None:
        if isinstance(value, str):
            return value.strip() if strip else value
        return OpenAICompatibleClient._extract_text_blocks(value, strip=strip)

    @staticmethod
    def _extract_text_blocks(value: Any, strip: bool = True) -> str | None:
        if not isinstance(value, list):
            return None

        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in ("text", "content"):
                text = item.get(key)
                if isinstance(text, str):
                    parts.append(text)
                    break
        if not parts:
            return None
        text = "".join(parts)
        return text.strip() if strip else text

    @staticmethod
    def _preview_payload(value: Any) -> str:
        try:
            preview = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            preview = repr(value)
        sanitized = preview.replace("\n", " ").strip()
        return sanitized[:500] if sanitized else "<empty payload>"

    @staticmethod
    def _extract_embeddings(data: dict[str, Any]) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list):
            raise LLMClientError("Malformed embeddings response: missing data list.")

        vectors: list[list[float]] = []
        for item in sorted(items, key=lambda value: value.get("index", 0) if isinstance(value, dict) else 0):
            if not isinstance(item, dict):
                raise LLMClientError("Malformed embeddings response: item is not an object.")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise LLMClientError("Malformed embeddings response: missing embedding vector.")
            vector: list[float] = []
            for value in embedding:
                if not isinstance(value, int | float):
                    raise LLMClientError("Malformed embeddings response: vector contains a non-number.")
                vector.append(float(value))
            vectors.append(vector)
        return vectors

    @staticmethod
    def _safe_error_body(response_body: str) -> str:
        sanitized = response_body.replace("\n", " ").strip()
        return sanitized[:500] if sanitized else "<empty response>"
