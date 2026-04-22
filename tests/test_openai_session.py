import unittest
import tempfile
from collections.abc import Iterator
from pathlib import Path

from src.gui.console_session import ConsoleConfig, ConsoleSessionError, OpenAICompatibleSession
from src.gui.llm_client import ChatStreamChunk, LLMClientError


class FakeOpenAIClient:
    def __init__(
        self,
        chunks: list[ChatStreamChunk],
        error: LLMClientError | None = None,
        chat_results: list[str | LLMClientError] | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.chat_results = chat_results or ["fallback answer"]
        self.chat_completion_calls = 0
        self.chat_messages: list[list[dict[str, str]]] = []

    def stream_chat_completion(self, messages: list[dict[str, str]]) -> Iterator[ChatStreamChunk]:
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        self.chat_completion_calls += 1
        self.chat_messages.append(messages)
        result = self.chat_results.pop(0)
        if isinstance(result, LLMClientError):
            raise result
        return result

    def close_active_request(self) -> None:
        pass


class OpenAICompatibleSessionTests(unittest.TestCase):
    def make_session(self, client: FakeOpenAIClient) -> OpenAICompatibleSession:
        session = OpenAICompatibleSession(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                openai_base_url="http://localhost:1234/v1",
                openai_model="local-model",
            )
        )
        session._client = client  # type: ignore[assignment]
        return session

    def test_stream_failure_before_final_text_falls_back_to_non_streaming(self) -> None:
        client = FakeOpenAIClient(
            [ChatStreamChunk(kind="reasoning")],
            LLMClientError("AttributeError: 'NoneType' object has no attribute 'peek'"),
        )
        session = self.make_session(client)

        chunks = list(session.ask_stream("hello"))

        self.assertEqual(
            [(chunk.kind, chunk.text) for chunk in chunks],
            [("reasoning", ""), ("final", "fallback answer")],
        )
        self.assertEqual(client.chat_completion_calls, 1)

    def test_generation_stopped_does_not_fall_back_to_non_streaming(self) -> None:
        client = FakeOpenAIClient([], LLMClientError("Generation stopped."))
        session = self.make_session(client)

        with self.assertRaises(ConsoleSessionError) as raised:
            list(session.ask_stream("hello"))

        self.assertEqual(str(raised.exception), "Generation stopped.")
        self.assertEqual(client.chat_completion_calls, 0)

    def test_reasoning_only_non_streaming_response_retries_for_final_answer(self) -> None:
        client = FakeOpenAIClient(
            [],
            chat_results=[
                LLMClientError("Backend returned reasoning only, but no final assistant answer."),
                "final answer",
            ],
        )
        session = self.make_session(client)

        self.assertEqual(session.ask("hello"), "final answer")
        self.assertEqual(client.chat_completion_calls, 2)
        self.assertEqual(client.chat_messages[1][0]["role"], "system")
        self.assertIn("only the final answer", client.chat_messages[1][0]["content"])

    def test_reasoning_only_stream_fallback_uses_final_answer_retry(self) -> None:
        client = FakeOpenAIClient(
            [ChatStreamChunk(kind="reasoning")],
            LLMClientError("Backend returned reasoning only, but no final assistant answer."),
            chat_results=[
                LLMClientError("Backend returned reasoning only, but no final assistant answer."),
                "final answer",
            ],
        )
        session = self.make_session(client)

        chunks = list(session.ask_stream("hello"))

        self.assertEqual(
            [(chunk.kind, chunk.text) for chunk in chunks],
            [("reasoning", ""), ("final", "final answer")],
        )
        self.assertEqual(client.chat_completion_calls, 2)

    def test_artifact_request_reads_generated_report_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = root / "llm_result" / "document_pipeline" / "generated_report.md"
            report.parent.mkdir(parents=True)
            report.write_text("# Saved Report\n\nGenerated earlier.", encoding="utf-8")
            client = FakeOpenAIClient(
                [],
                chat_results=[
                    "[read_file] llm/document_pipeline/generated_report.md [/read_file]",
                    "The saved report says: Generated earlier.",
                ],
            )
            session = self.make_session(client)
            session.config.artifact_working_folder = str(root)

            answer = session.ask("What did the generated report say?")

        self.assertEqual(answer, "The saved report says: Generated earlier.")
        self.assertEqual(client.chat_completion_calls, 2)
        self.assertIn("Generated artifact access:", client.chat_messages[0][0]["content"])
        self.assertIn("# Saved Report", client.chat_messages[1][-1]["content"])

    def test_artifact_request_stream_returns_final_followup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = root / "llm_result" / "document_pipeline" / "generated_report.md"
            report.parent.mkdir(parents=True)
            report.write_text("# Saved Report\n\nGenerated earlier.", encoding="utf-8")
            client = FakeOpenAIClient(
                [],
                chat_results=[
                    "[read_file] llm/document_pipeline/generated_report.md [/read_file]",
                    "final artifact answer",
                ],
            )
            session = self.make_session(client)
            session.config.artifact_working_folder = str(root)

            chunks = list(session.ask_stream("Read the generated report."))

        self.assertEqual([(chunk.kind, chunk.text) for chunk in chunks], [("final", "final artifact answer")])
        self.assertEqual(client.chat_completion_calls, 2)


if __name__ == "__main__":
    unittest.main()
