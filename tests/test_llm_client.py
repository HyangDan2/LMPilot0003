import json
import unittest

from src.gui.llm_client import LLMClientError, OpenAICompatibleClient, OpenAIConnectionSettings


class FakeResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


class FakeStreamingResponse:
    def __init__(
        self,
        status: int,
        lines: list[str],
        body: dict | None = None,
        readline_error: Exception | None = None,
    ) -> None:
        self.status = status
        self.lines = [line.encode("utf-8") for line in lines]
        self.body = body or {}
        self.readline_error = readline_error

    def read(self) -> bytes:
        if self.body:
            return json.dumps(self.body).encode("utf-8")
        return b"".join(self.lines)

    def readline(self) -> bytes:
        if self.readline_error is not None:
            raise self.readline_error
        if not self.lines:
            return b""
        return self.lines.pop(0)


class FakeConnection:
    requests: list[dict] = []
    responses: list[FakeResponse | FakeStreamingResponse] = []

    def __init__(self) -> None:
        self.closed = False

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None) -> None:
        self.requests.append(
            {
                "method": method,
                "path": path,
                "body": json.loads(body.decode("utf-8")) if body else None,
                "headers": headers or {},
            }
        )

    def getresponse(self) -> FakeResponse | FakeStreamingResponse:
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


class OpenAICompatibleClientTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeConnection.requests.clear()
        FakeConnection.responses.clear()

    def make_client(self, settings: OpenAIConnectionSettings) -> OpenAICompatibleClient:
        client = OpenAICompatibleClient(settings)
        client._create_connection = lambda parsed: FakeConnection()  # type: ignore[method-assign]
        return client

    def test_chat_completion_posts_openai_compatible_payload(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"choices": [{"message": {"content": "hello"}}]})
        )
        client = self.make_client(
            OpenAIConnectionSettings(
                base_url="http://localhost:1234/v1/",
                api_key="sk-secret",
                model="local-model",
                temperature=0.2,
                max_tokens=64,
            )
        )

        answer = client.chat_completion([{"role": "user", "content": "Hi"}])

        self.assertEqual(answer, "hello")
        request = FakeConnection.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer sk-secret")
        self.assertEqual(request["body"]["model"], "local-model")
        self.assertEqual(request["body"]["messages"], [{"role": "user", "content": "Hi"}])
        self.assertFalse(request["body"]["stream"])

    def test_chat_completion_can_request_json_response_format(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        answer = client.chat_completion(
            [{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        self.assertEqual(answer, "{}")
        self.assertEqual(FakeConnection.requests[0]["body"]["response_format"], {"type": "json_object"})

    def test_stream_chat_completion_posts_streaming_payload(self) -> None:
        FakeConnection.responses.append(
            FakeStreamingResponse(
                200,
                [
                    'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                    'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
                    'data: {"choices":[{"delta":{"content":[{"text":"world"}]}}]}\n',
                    "data: [DONE]\n",
                ],
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        chunks = list(client.stream_chat_completion([{"role": "user", "content": "Hi"}]))

        self.assertEqual([(chunk.kind, chunk.text) for chunk in chunks], [("final", "hello "), ("final", "world")])
        request = FakeConnection.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertTrue(request["body"]["stream"])

    def test_stream_chat_completion_reports_reasoning_without_text(self) -> None:
        FakeConnection.responses.append(
            FakeStreamingResponse(
                200,
                [
                    'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n',
                    'data: {"choices":[{"delta":{"content":"final answer"}}]}\n',
                    "data: [DONE]\n",
                ],
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(
            [
                (chunk.kind, chunk.text)
                for chunk in client.stream_chat_completion([{"role": "user", "content": "Hi"}])
            ],
            [("reasoning", ""), ("final", "final answer")],
        )

    def test_stream_chat_completion_errors_on_reasoning_only(self) -> None:
        FakeConnection.responses.append(
            FakeStreamingResponse(
                200,
                [
                    'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n',
                    "data: [DONE]\n",
                ],
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        with self.assertRaises(LLMClientError) as raised:
            list(client.stream_chat_completion([{"role": "user", "content": "Hi"}]))

        self.assertIn("reasoning only", str(raised.exception))

    def test_stream_chat_completion_wraps_attribute_error_from_backend(self) -> None:
        FakeConnection.responses.append(
            FakeStreamingResponse(
                200,
                [],
                readline_error=AttributeError("'NoneType' object has no attribute 'peek'"),
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        with self.assertRaises(LLMClientError) as raised:
            list(client.stream_chat_completion([{"role": "user", "content": "Hi"}]))

        self.assertIn("AttributeError", str(raised.exception))
        self.assertIn("peek", str(raised.exception))

    def test_chat_completion_accepts_message_content_blocks(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "hello "},
                                    {"content": "from "},
                                    "blocks",
                                ]
                            }
                        }
                    ]
                },
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(
            client.chat_completion([{"role": "user", "content": "Hi"}]),
            "hello from blocks",
        )

    def test_chat_completion_accepts_choice_text(self) -> None:
        FakeConnection.responses.append(FakeResponse(200, {"choices": [{"text": "legacy text"}]}))
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(client.chat_completion([{"role": "user", "content": "Hi"}]), "legacy text")

    def test_chat_completion_accepts_delta_content_blocks(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(
                200,
                {"choices": [{"delta": {"content": [{"text": "delta "}, {"content": "blocks"}]}}]},
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(client.chat_completion([{"role": "user", "content": "Hi"}]), "delta blocks")

    def test_chat_completion_accepts_delta_content_string(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"choices": [{"delta": {"content": "delta text"}}]})
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(client.chat_completion([{"role": "user", "content": "Hi"}]), "delta text")

    def test_chat_completion_accepts_message_text_fallbacks(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"choices": [{"message": {"output_text": "fallback text"}}]})
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(client.chat_completion([{"role": "user", "content": "Hi"}]), "fallback text")

    def test_chat_completion_errors_on_message_reasoning_only(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "index": 0,
                            "logprobs": None,
                            "message": {
                                "annotations": None,
                                "audio": None,
                                "content": None,
                                "context": None,
                                "function_call": None,
                                "reasoning": "Thinking Process:\n\n1. Use the available text.",
                            },
                        }
                    ]
                },
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        with self.assertRaises(LLMClientError) as raised:
            client.chat_completion([{"role": "user", "content": "Hi"}])

        self.assertIn("reasoning only", str(raised.exception))

    def test_chat_completion_tries_fallbacks_after_empty_content(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(
                200,
                {"choices": [{"message": {"content": "", "text": "message fallback"}}]},
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(client.chat_completion([{"role": "user", "content": "Hi"}]), "message fallback")

    def test_chat_completion_accepts_choice_level_content_fallback(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"choices": [{"content": [{"text": "choice content"}]}]})
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        self.assertEqual(client.chat_completion([{"role": "user", "content": "Hi"}]), "choice content")

    def test_malformed_chat_response_includes_choice_preview(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": [{"type": "image"}]},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", model="local-model")
        )

        with self.assertRaises(LLMClientError) as raised:
            client.chat_completion([{"role": "user", "content": "Hi"}])

        message = str(raised.exception)
        self.assertIn("Malformed chat response: missing assistant content.", message)
        self.assertIn("Finish reason: stop.", message)
        self.assertIn("finish_reason", message)
        self.assertIn("image", message)

    def test_list_models_uses_base_url_prefix(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"data": [{"id": "model-a"}, {"id": "model-b"}]})
        )
        client = self.make_client(OpenAIConnectionSettings(base_url="https://example.test/v1"))

        self.assertEqual(client.list_models(), ["model-a", "model-b"])
        self.assertEqual(FakeConnection.requests[0]["path"], "/v1/models")

    def test_embeddings_posts_openai_compatible_payload(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(
                200,
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )
        )
        client = self.make_client(
            OpenAIConnectionSettings(
                base_url="http://localhost:1234/v1",
                model="chat-model",
                embedding_model="embedding-model",
            )
        )

        vectors = client.embeddings(["first", "second"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        request = FakeConnection.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/v1/embeddings")
        self.assertEqual(request["body"]["model"], "embedding-model")
        self.assertEqual(request["body"]["input"], ["first", "second"])

    def test_http_error_is_readable_without_credentials(self) -> None:
        FakeConnection.responses.append(FakeResponse(401, {"error": "bad key"}))
        client = self.make_client(
            OpenAIConnectionSettings(base_url="http://localhost:1234/v1", api_key="sk-secret", model="model")
        )

        with self.assertRaises(LLMClientError) as raised:
            client.chat_completion([{"role": "user", "content": "Hi"}])

        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn("sk-secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
