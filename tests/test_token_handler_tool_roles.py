import unittest

from src.gui.token_handler import build_model_prompt_request


class TokenHandlerToolRoleTests(unittest.TestCase):
    def test_local_tool_history_is_sent_as_assistant_context(self) -> None:
        prompt = build_model_prompt_request(
            [
                {"role": "user", "content": "/help"},
                {"role": "tool", "content": "Available tools: /extract_docs"},
            ],
            "What should I do next?",
            max_tokens=200,
        )

        self.assertEqual(
            [message["role"] for message in prompt.messages],
            ["user", "assistant", "user"],
        )
        self.assertIn("Local tool output:", prompt.messages[1]["content"])
        self.assertIn("Available tools: /extract_docs", prompt.messages[1]["content"])

    def test_adjacent_same_role_messages_are_coalesced(self) -> None:
        prompt = build_model_prompt_request(
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "answer"},
            ],
            "next",
            max_tokens=200,
        )

        self.assertEqual(
            [message["role"] for message in prompt.messages],
            ["user", "assistant", "user"],
        )
        self.assertEqual(prompt.messages[0]["content"], "first\n\nsecond")


if __name__ == "__main__":
    unittest.main()
