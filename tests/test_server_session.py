import json
import unittest

from src.gui.console_session import ConsoleConfig, ConsoleSessionError, LlamaServerSession
from src.gui.token_handler import build_model_prompt_request


class FakeResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


class FakeConnection:
    requests: list[dict] = []
    responses: list[FakeResponse] = []

    def __init__(self) -> None:
        self.closed = False

    def request(self, method: str, endpoint: str, body: bytes, headers: dict) -> None:
        self.requests.append(
            {
                "method": method,
                "endpoint": endpoint,
                "body": json.loads(body.decode("utf-8")),
                "headers": headers,
            }
        )

    def getresponse(self) -> FakeResponse:
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


class ServerSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeConnection.requests.clear()
        FakeConnection.responses.clear()

    def make_session(self, config: ConsoleConfig) -> LlamaServerSession:
        session = LlamaServerSession(config)
        session._create_connection = FakeConnection  # type: ignore[method-assign]
        return session

    def test_server_session_prefers_chat_completions_payload(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"choices": [{"message": {"content": "server answer"}}]})
        )
        session = self.make_session(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                server_url="http://127.0.0.1:8080",
                n_predict=64,
                system_prompt="Be concise.",
                extra_args=["temperature=0.2"],
            )
        )
        prompt = build_model_prompt_request([], "hello", 100, system_prompt="Be concise.")

        answer = session.ask(prompt)

        self.assertEqual(answer, "server answer")
        request = FakeConnection.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["endpoint"], "/v1/chat/completions")
        self.assertEqual(request["body"]["messages"][0], {"role": "system", "content": "Be concise."})
        self.assertEqual(request["body"]["messages"][-1], {"role": "user", "content": "hello"})
        self.assertEqual(request["body"]["max_tokens"], 64)
        self.assertEqual(request["body"]["temperature"], 0.2)
        self.assertNotIn("stop", request["body"])

    def test_server_session_falls_back_to_completion_with_gemma_template(self) -> None:
        FakeConnection.responses.extend(
            [
                FakeResponse(404, {"error": "not found"}),
                FakeResponse(200, {"content": "fallback answer"}),
            ]
        )
        session = self.make_session(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                server_url="http://127.0.0.1:8080",
                n_predict=64,
            )
        )
        prompt = build_model_prompt_request([], "hello", 100)

        answer = session.ask(prompt)

        self.assertEqual(answer, "fallback answer")
        self.assertEqual(FakeConnection.requests[0]["endpoint"], "/v1/chat/completions")
        self.assertEqual(FakeConnection.requests[1]["endpoint"], "/completion")
        completion_body = FakeConnection.requests[1]["body"]
        self.assertIn("<start_of_turn>user\nhello<end_of_turn>", completion_body["prompt"])
        self.assertTrue(completion_body["prompt"].endswith("<start_of_turn>model"))
        self.assertIn("\n[You]", completion_body["stop"])

    def test_server_session_treats_slash_auto_as_auto_endpoint(self) -> None:
        FakeConnection.responses.extend(
            [
                FakeResponse(404, {"error": "not found"}),
                FakeResponse(200, {"content": "fallback answer"}),
            ]
        )
        session = self.make_session(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                server_url="http://127.0.0.1:8080",
                server_endpoint="/auto",
            )
        )
        prompt = build_model_prompt_request([], "hello", 100)

        answer = session.ask(prompt)

        self.assertEqual(answer, "fallback answer")
        self.assertEqual(FakeConnection.requests[0]["endpoint"], "/v1/chat/completions")
        self.assertEqual(FakeConnection.requests[1]["endpoint"], "/completion")

    def test_server_session_completion_endpoint_uses_gemma_template_directly(self) -> None:
        FakeConnection.responses.append(FakeResponse(200, {"content": "completion answer"}))
        session = self.make_session(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                server_url="http://127.0.0.1:8080",
                server_endpoint="/completion",
            )
        )
        prompt = build_model_prompt_request([], "hello", 100)

        answer = session.ask(prompt)

        self.assertEqual(answer, "completion answer")
        self.assertEqual(len(FakeConnection.requests), 1)
        self.assertEqual(FakeConnection.requests[0]["endpoint"], "/completion")
        self.assertIn("<start_of_turn>user\nhello<end_of_turn>", FakeConnection.requests[0]["body"]["prompt"])

    def test_server_session_sends_structured_image_content_to_chat_endpoint(self) -> None:
        FakeConnection.responses.append(FakeResponse(200, {"choices": [{"message": {"content": "vision answer"}}]}))
        session = self.make_session(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                server_url="http://127.0.0.1:8080",
            )
        )
        content = [
            {"type": "text", "text": "describe it"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        prompt = build_model_prompt_request([], content, 100)

        answer = session.ask(prompt)

        self.assertEqual(answer, "vision answer")
        self.assertEqual(FakeConnection.requests[0]["body"]["messages"][-1]["content"], content)
        self.assertIn("[image]", prompt.completion_prompt)

    def test_server_session_rejects_structured_image_content_for_completion_fallback(self) -> None:
        FakeConnection.responses.append(FakeResponse(404, {"error": "not found"}))
        session = self.make_session(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                server_url="http://127.0.0.1:8080",
            )
        )
        prompt = build_model_prompt_request(
            [],
            [
                {"type": "text", "text": "describe it"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
            100,
        )

        with self.assertRaises(ConsoleSessionError) as raised:
            session.ask(prompt)

        self.assertIn("vision backend", str(raised.exception))
        self.assertEqual(len(FakeConnection.requests), 1)

    def test_server_session_cleans_runaway_dialogue_continuation(self) -> None:
        FakeConnection.responses.append(
            FakeResponse(200, {"choices": [{"message": {"content": "Hey you!\n\n[You]\nWTF\n\n[Gemma]\nWhat?"}}]})
        )
        session = self.make_session(
            ConsoleConfig(
                llama_cli_path="/unused",
                model_path="/unused",
                server_url="http://127.0.0.1:8080",
            )
        )
        prompt = build_model_prompt_request([], "Hey!", 100)

        answer = session.ask(prompt)

        self.assertEqual(answer, "Hey you!")


if __name__ == "__main__":
    unittest.main()
