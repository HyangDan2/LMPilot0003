from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import pexpect

from .artifact_tools import (
    ARTIFACT_ACCESS_INSTRUCTION,
    build_artifact_followup_messages,
    execute_artifact_requests,
    extract_artifact_requests,
)
from .llm_client import ChatStreamChunk, LLMClientError, OpenAICompatibleClient, OpenAIConnectionSettings
from .token_handler import ModelPrompt, message_content_to_text

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TIMING_LINE_RE = re.compile(r"^\[\s*Prompt:.*?\]\s*$", re.MULTILINE)
PROMPT_RE = re.compile(r"(?m)^\s*>\s*$")
ANSWER_CONTINUATION_RE = re.compile(
    r"(?m)^\s*(?:"
    r"\[You\]|"
    r"<start_of_turn>user|"
    r"User:|"
    r"Human:|"
    r"### User:|"
    r"---\s*$|"
    r"\*\*Key elements"
    r")"
)
LEADING_ASSISTANT_LABEL_RE = re.compile(
    r"^\s*(?:\[Gemma\]|Gemma:|Assistant:|<start_of_turn>model)\s*",
    re.IGNORECASE,
)

DEFAULT_SERVER_STOP_SEQUENCES = [
    "<end_of_turn>",
    "<start_of_turn>user",
    "\n[You]",
    "\nUser:",
    "\nHuman:",
    "\n### User:",
]

REASONING_ONLY_ERROR = "Backend returned reasoning only, but no final assistant answer."
FINAL_ANSWER_RETRY_INSTRUCTION = (
    "The previous attempt returned reasoning without a final answer. "
    "Reply again with only the final answer in the assistant content. "
    "Do not include reasoning, thinking process, analysis, or hidden chain-of-thought."
)

BANNER_SKIP_PATTERNS = [
    re.compile(r"^available commands:\s*$", re.IGNORECASE),
    re.compile(r"^\s*/exit.*$", re.IGNORECASE),
    re.compile(r"^\s*/regen.*$", re.IGNORECASE),
    re.compile(r"^\s*/clear.*$", re.IGNORECASE),
    re.compile(r"^\s*/read.*$", re.IGNORECASE),
    re.compile(r"^\s*/glob.*$", re.IGNORECASE),
    re.compile(r"^Loading model\.\.\.\s*$", re.IGNORECASE),
    re.compile(r"^build\s*:.*$", re.IGNORECASE),
    re.compile(r"^model\s*:.*$", re.IGNORECASE),
    re.compile(r"^modalities\s*:.*$", re.IGNORECASE),
    re.compile(r"^using custom system prompt\s*$", re.IGNORECASE),
    re.compile(r"^add a text file\s*$", re.IGNORECASE),
    re.compile(r"^add text files using globbing pattern\s*$", re.IGNORECASE),
]


class ConsoleSessionError(Exception):
    pass


@dataclass
class ConsoleConfig:
    llama_cli_path: str
    model_path: str
    backend: str = "server"
    server_url: str = "http://127.0.0.1:8080"
    server_endpoint: str = "auto"
    n_predict: int = 512
    system_prompt: Optional[str] = None
    threads: int = 4
    ctx_size: int = 2048
    extra_args: Optional[list[str]] = None
    startup_timeout: float = 180.0
    response_timeout: float = 180.0
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    openai_embedding_model: str = ""
    temperature: float = 0.7
    artifact_working_folder: str = ""
    max_artifact_tool_rounds: int = 2

    def openai_settings(self) -> OpenAIConnectionSettings:
        return OpenAIConnectionSettings(
            base_url=self.openai_base_url or self.server_url,
            api_key=self.openai_api_key,
            model=self.openai_model,
            embedding_model=self.openai_embedding_model,
            temperature=self.temperature,
            max_tokens=self.n_predict,
            timeout=self.response_timeout,
        )


class OpenAICompatibleSession:
    def __init__(self, config: ConsoleConfig) -> None:
        self.config = config
        self._client = OpenAICompatibleClient(config.openai_settings())
        self._started = False

    def start(self) -> None:
        # Startup should be safe even before users enter runtime credentials.
        self._started = True

    def is_alive(self) -> bool:
        return self._started

    def update_connection_settings(self, settings: OpenAIConnectionSettings) -> None:
        self.config.openai_base_url = settings.base_url
        self.config.openai_api_key = settings.api_key
        self.config.openai_model = settings.model
        self.config.openai_embedding_model = settings.embedding_model
        self.config.temperature = settings.temperature
        self.config.n_predict = settings.max_tokens
        self.config.response_timeout = settings.timeout
        self._client = OpenAICompatibleClient(settings)

    def ask(self, user_text: str | ModelPrompt) -> str:
        if not self.is_alive():
            self.start()

        messages = self._build_chat_messages(user_text)

        answer = self._chat_completion_with_retry(messages)
        answer = self._resolve_artifact_requests(messages, answer)

        if not answer.strip():
            raise ConsoleSessionError("Model returned an empty response.")
        return answer

    def ask_stream(self, user_text: str | ModelPrompt) -> Iterator[ChatStreamChunk]:
        if not self.is_alive():
            self.start()

        messages = self._build_chat_messages(user_text)
        if self.config.artifact_working_folder:
            answer = self.ask(user_text)
            yield ChatStreamChunk(kind="final", text=answer)
            return

        emitted_final_text = False

        try:
            for chunk in self._client.stream_chat_completion(messages):
                if chunk.kind == "final" and chunk.text:
                    emitted_final_text = True
                yield chunk
        except LLMClientError as exc:
            if str(exc) == "Generation stopped.":
                raise ConsoleSessionError(str(exc)) from exc
            if not emitted_final_text:
                answer = self.ask(user_text)
                yield ChatStreamChunk(kind="final", text=answer)
                return
            raise ConsoleSessionError(str(exc)) from exc

        if not emitted_final_text:
            raise ConsoleSessionError("Model returned an empty response.")

    def _build_chat_messages(self, user_text: str | ModelPrompt) -> list[dict[str, Any]]:
        if isinstance(user_text, ModelPrompt):
            messages = [dict(message) for message in user_text.messages]
        else:
            if not user_text.strip():
                raise ConsoleSessionError("Empty prompt is not allowed.")
            messages = [{"role": "user", "content": user_text}]

        if self.config.system_prompt and not any(message.get("role") == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        if self.config.artifact_working_folder:
            messages = self._with_artifact_access_instruction(messages)
        return messages

    def _chat_completion_with_retry(self, messages: list[dict[str, Any]]) -> str:
        try:
            return self._client.chat_completion(messages)
        except LLMClientError as exc:
            if self._is_reasoning_only_error(exc):
                try:
                    return self._client.chat_completion(self._with_final_answer_retry_instruction(messages))
                except LLMClientError as retry_exc:
                    raise ConsoleSessionError(str(retry_exc)) from retry_exc
            raise ConsoleSessionError(str(exc)) from exc

    def _resolve_artifact_requests(self, messages: list[dict[str, Any]], answer: str) -> str:
        if not self.config.artifact_working_folder:
            return answer
        current_answer = answer
        current_messages = messages
        for _ in range(max(0, self.config.max_artifact_tool_rounds)):
            requests = extract_artifact_requests(current_answer)
            if not requests:
                return current_answer
            results = execute_artifact_requests(self.config.artifact_working_folder, requests)
            current_messages = build_artifact_followup_messages(current_messages, current_answer, results)
            current_answer = self._chat_completion_with_retry(current_messages)
        return current_answer

    @staticmethod
    def _with_artifact_access_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated = [dict(message) for message in messages]
        for message in updated:
            if message.get("role") == "system":
                content = message_content_to_text(message.get("content", "")).strip()
                if "Generated artifact access:" not in content:
                    message["content"] = f"{content}\n\n{ARTIFACT_ACCESS_INSTRUCTION}".strip()
                return updated
        updated.insert(0, {"role": "system", "content": ARTIFACT_ACCESS_INSTRUCTION})
        return updated

    @staticmethod
    def _is_reasoning_only_error(exc: LLMClientError) -> bool:
        return REASONING_ONLY_ERROR in str(exc)

    @staticmethod
    def _with_final_answer_retry_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retry_messages = [dict(message) for message in messages]
        for message in retry_messages:
            if message.get("role") == "system":
                content = message_content_to_text(message.get("content", "")).strip()
                message["content"] = f"{content}\n\n{FINAL_ANSWER_RETRY_INSTRUCTION}".strip()
                return retry_messages
        retry_messages.insert(0, {"role": "system", "content": FINAL_ANSWER_RETRY_INSTRUCTION})
        return retry_messages

    def test_connection(self) -> str:
        try:
            return self._client.test_connection()
        except LLMClientError as exc:
            raise ConsoleSessionError(str(exc)) from exc

    def list_models(self) -> list[str]:
        try:
            return self._client.list_models()
        except LLMClientError as exc:
            raise ConsoleSessionError(str(exc)) from exc

    def stop(self, force: bool = False) -> None:
        self.stop_generation()
        self._started = False

    def stop_generation(self) -> None:
        self._client.close_active_request()


class LlamaServerSession:
    def __init__(self, config: ConsoleConfig) -> None:
        self.config = config
        self._active_connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
        self._lock = threading.Lock()
        self._started = False
        self._stop_requested = False

    def start(self) -> None:
        self._validate_server_url()
        self._started = True

    def is_alive(self) -> bool:
        return self._started

    def ask(self, user_text: str | ModelPrompt) -> str:
        if not self._prompt_has_text(user_text):
            raise ConsoleSessionError("Empty prompt is not allowed.")

        if not self.is_alive():
            self.start()

        last_response_body = ""
        for mode, endpoint in self._endpoint_candidates():
            if mode == "completion" and self._prompt_has_structured_content(user_text):
                raise ConsoleSessionError(
                    "Structured image content requires a chat-completions vision backend; "
                    "the completion endpoint cannot receive structured content."
                )
            payload = (
                self._build_chat_payload(user_text)
                if mode == "chat"
                else self._build_completion_payload(user_text)
            )
            status, response_body = self._post_json(endpoint, payload)
            if status >= 400:
                last_response_body = response_body
                if self._should_try_next_endpoint(mode, status):
                    continue
                raise ConsoleSessionError(
                    f"llama-server returned HTTP {status} from {endpoint}: {response_body}"
                )
            answer = self._clean_server_answer(self._extract_server_answer(response_body))
            if not answer.strip():
                raise ConsoleSessionError("Model returned an empty response.")
            return answer

        raise ConsoleSessionError(f"llama-server did not return a usable response: {last_response_body}")

    def stop(self, force: bool = False) -> None:
        self.stop_generation()
        self._started = False

    def stop_generation(self) -> None:
        with self._lock:
            self._stop_requested = True
            conn = self._active_connection
        if conn is not None:
            conn.close()

    def _validate_server_url(self) -> None:
        parsed = urlparse(self.config.server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConsoleSessionError(f"Invalid llama-server URL: {self.config.server_url}")

    def _create_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        parsed = urlparse(self.config.server_url)
        port = parsed.port
        timeout = self.config.response_timeout
        if parsed.scheme == "https":
            return http.client.HTTPSConnection(parsed.hostname, port=port, timeout=timeout)
        return http.client.HTTPConnection(parsed.hostname, port=port, timeout=timeout)

    def _endpoint_candidates(self) -> list[tuple[str, str]]:
        endpoint = self._normalized_server_endpoint()
        if endpoint == "auto":
            return [("chat", "/v1/chat/completions"), ("completion", "/completion")]
        if "chat/completions" in endpoint:
            return [("chat", endpoint)]
        return [("completion", endpoint)]

    def _should_try_next_endpoint(self, mode: str, status: int) -> bool:
        return (
            self._normalized_server_endpoint() == "auto"
            and mode == "chat"
            and status in {400, 404, 405}
        )

    def _normalized_server_endpoint(self) -> str:
        endpoint = (self.config.server_endpoint or "auto").strip()
        if endpoint.lower() in {"auto", "/auto"}:
            return "auto"
        if not endpoint.startswith("/"):
            return f"/{endpoint}"
        return endpoint

    def _post_json(self, endpoint: str, payload: dict[str, object]) -> tuple[int, str]:
        body = json.dumps(payload).encode("utf-8")
        conn = self._create_connection()

        with self._lock:
            self._stop_requested = False
            self._active_connection = conn

        try:
            conn.request(
                "POST",
                endpoint,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response_body = response.read().decode("utf-8", errors="replace")
            return response.status, response_body
        except (OSError, http.client.HTTPException) as exc:
            if self._stop_requested:
                raise ConsoleSessionError("Generation stopped.") from exc
            raise ConsoleSessionError(f"llama-server request failed: {exc}") from exc
        finally:
            with self._lock:
                if self._active_connection is conn:
                    self._active_connection = None
            conn.close()

    def _build_chat_payload(self, prompt: str | ModelPrompt) -> dict[str, object]:
        if isinstance(prompt, ModelPrompt):
            messages = [dict(message) for message in prompt.messages]
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        if self.config.system_prompt and not any(message.get("role") == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        payload: dict[str, object] = {
            "messages": messages,
            "max_tokens": self.config.n_predict,
            "stream": False,
        }
        payload.update(self._extra_args_as_payload())
        return payload

    def _build_completion_payload(self, prompt: str | ModelPrompt) -> dict[str, object]:
        if isinstance(prompt, ModelPrompt):
            prompt_text = prompt.completion_prompt
        else:
            prompt_text = str(prompt)

        payload: dict[str, object] = {
            "prompt": prompt_text,
            "n_predict": self.config.n_predict,
            "stream": False,
            "stop": DEFAULT_SERVER_STOP_SEQUENCES,
        }
        payload.update(self._extra_args_as_payload())
        return payload

    @staticmethod
    def _prompt_has_text(prompt: str | ModelPrompt) -> bool:
        if isinstance(prompt, ModelPrompt):
            return bool(
                prompt.completion_prompt.strip()
                or any(message_content_to_text(message.get("content", "")).strip() for message in prompt.messages)
            )
        return bool(prompt.strip())

    @staticmethod
    def _prompt_has_structured_content(prompt: str | ModelPrompt) -> bool:
        return isinstance(prompt, ModelPrompt) and any(
            isinstance(message.get("content"), list) for message in prompt.messages
        )

    def _extra_args_as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in self.config.extra_args or []:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            payload[key.strip()] = self._parse_payload_value(value.strip())
        return payload

    @staticmethod
    def _parse_payload_value(value: str) -> object:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def _extract_server_answer(response_body: str) -> str:
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ConsoleSessionError(f"Invalid llama-server JSON response: {response_body}") from exc

        if isinstance(data, dict):
            content = data.get("content")
            if isinstance(content, str | list):
                return message_content_to_text(content).strip()
            if isinstance(data.get("completion"), str):
                return data["completion"].strip()
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    text = choice.get("text")
                    if isinstance(text, str):
                        return text.strip()
                    message = choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str | list):
                            return message_content_to_text(content).strip()

        raise ConsoleSessionError(f"Unsupported llama-server response: {response_body}")

    @staticmethod
    def _clean_server_answer(answer: str) -> str:
        answer = answer.replace("<end_of_turn>", "").strip()
        answer = LEADING_ASSISTANT_LABEL_RE.sub("", answer).strip()
        match = ANSWER_CONTINUATION_RE.search(answer)
        if match and match.start() > 0:
            answer = answer[: match.start()]
        return answer.strip()


class LlamaConsoleSession:
    def __init__(self, config: ConsoleConfig) -> None:
        self.config = config
        self.child: Optional[pexpect.spawn] = None
        self._started = False

    def start(self) -> None:
        if self._started and self.is_alive():
            return

        self._validate_paths()
        cmd = self._build_command()
        env = self._build_env()

        self.child = pexpect.spawn(
            command=cmd[0],
            args=cmd[1:],
            env=env,
            encoding="utf-8",
            codec_errors="replace",
            timeout=self.config.startup_timeout,
        )

        try:
            self._wait_for_prompt(self.config.startup_timeout)
            self._started = True
        except Exception as exc:
            startup_dump = self._safe_before()
            self.stop(force=True)
            raise ConsoleSessionError(
                "Failed to start llama-cli.\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"Output:\n{startup_dump}"
            ) from exc

    def is_alive(self) -> bool:
        return self.child is not None and self.child.isalive()

    def ask(self, user_text: str | ModelPrompt) -> str:
        if isinstance(user_text, ModelPrompt):
            if LlamaServerSession._prompt_has_structured_content(user_text):
                raise ConsoleSessionError(
                    "Structured image content requires an OpenAI-compatible or chat-completions vision backend."
                )
            user_text = user_text.completion_prompt

        if not user_text.strip():
            raise ConsoleSessionError("Empty prompt is not allowed.")

        if not self.is_alive():
            self.start()

        assert self.child is not None
        self.child.sendline(user_text)

        raw_block = self._wait_for_prompt(self.config.response_timeout)
        answer = self._extract_answer(raw_block, user_text)

        if not answer.strip():
            raise ConsoleSessionError(
                "Model returned an empty response.\n"
                f"Raw output:\n{self._sanitize_text(raw_block)}"
            )

        return answer

    def stop(self, force: bool = False) -> None:
        if self.child is None:
            self._started = False
            return

        try:
            if self.child.isalive():
                if force:
                    self.child.terminate(force=True)
                else:
                    self.child.sendline("/exit")
                    try:
                        self.child.expect(pexpect.EOF, timeout=5)
                    except pexpect.TIMEOUT:
                        self.child.terminate(force=True)
        finally:
            self.child = None
            self._started = False

    def stop_generation(self) -> None:
        """Interrupt the current response and reset the console for the next prompt."""
        self.stop(force=True)

    def _validate_paths(self) -> None:
        if not os.path.isfile(self.config.llama_cli_path):
            raise ConsoleSessionError(f"llama-cli not found: {self.config.llama_cli_path}")
        if not os.path.isfile(self.config.model_path):
            raise ConsoleSessionError(f"model not found: {self.config.model_path}")

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        llama_bin_dir = os.path.dirname(self.config.llama_cli_path)
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{llama_bin_dir}:{current_ld}" if current_ld else llama_bin_dir
        env.setdefault("TERM", "xterm")
        env.setdefault("COLORTERM", "false")
        env.setdefault("CLICOLOR", "0")
        env.setdefault("NO_COLOR", "1")
        return env

    def _build_command(self) -> list[str]:
        cmd = [
            self.config.llama_cli_path,
            "-m",
            self.config.model_path,
            "--simple-io",
            "--threads",
            str(self.config.threads),
            "--ctx-size",
            str(self.config.ctx_size),
        ]

        if self.config.system_prompt:
            cmd.extend(["--system-prompt", self.config.system_prompt])
        if self.config.extra_args:
            cmd.extend(self.config.extra_args)
        return cmd

    def _wait_for_prompt(self, timeout: float) -> str:
        if self.child is None:
            raise ConsoleSessionError("Console session is not initialized.")

        collected: list[str] = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                idx = self.child.expect([PROMPT_RE, pexpect.EOF, pexpect.TIMEOUT], timeout=remaining)
                if idx == 0:
                    collected.append(self.child.before or "")
                    return "".join(collected)
                if idx == 1:
                    collected.append(self.child.before or "")
                    raise ConsoleSessionError(
                        "llama-cli terminated unexpectedly.\n"
                        f"Output:\n{self._sanitize_text(''.join(collected))}"
                    )
                if idx == 2:
                    continue
            except pexpect.TIMEOUT:
                continue

        raise ConsoleSessionError(
            "Timed out waiting for model response.\n"
            f"Partial output:\n{self._sanitize_text(''.join(collected))}"
        )

    def _extract_answer(self, raw_text: str, user_text: str) -> str:
        text = self._sanitize_text(raw_text)
        lines = text.splitlines()
        cleaned_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped == user_text.strip():
                continue
            if TIMING_LINE_RE.match(stripped):
                continue
            if self._should_skip_line(stripped):
                continue
            cleaned_lines.append(line)

        normalized = "\n".join(cleaned_lines)
        return self._collapse_blank_lines(normalized).strip()

    def _sanitize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = ANSI_ESCAPE_RE.sub("", text)
        text = text.replace("\x00", "")
        return text

    def _should_skip_line(self, line: str) -> bool:
        if line == ">":
            return True
        if re.fullmatch(r"[▄█▀ ]+", line):
            return True
        return any(pattern.match(line) for pattern in BANNER_SKIP_PATTERNS)

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        lines = text.split("\n")
        out: list[str] = []
        blank_count = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 1:
                    out.append("")
            else:
                blank_count = 0
                out.append(line.rstrip())
        return "\n".join(out).strip()

    def _safe_before(self) -> str:
        if self.child is None:
            return ""
        try:
            return self._sanitize_text(self.child.before or "")
        except Exception:
            return ""
