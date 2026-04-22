import unittest

from src.gui.gui import normalize_text_for_display
from src.gui.token_handler import (
    build_model_prompt,
    build_memory_context,
    build_model_prompt_request,
    handle_token_limits,
    normalize_prompt_text,
    prompt_token_budget,
    truncate_text_to_char_budget,
    truncate_text_to_token_budget,
)


class TextProcessingTests(unittest.TestCase):
    def test_unicode_escape_normalization_preserves_korean(self) -> None:
        text = "한글 " + chr(92) + "u003c ok"
        self.assertEqual(normalize_text_for_display(text), "한글 < ok")

    def test_slash_unicode_escape_is_normalized(self) -> None:
        self.assertEqual(normalize_text_for_display("/u003c"), "<")

    def test_token_limit_keeps_newest_turns(self) -> None:
        conversation = ["one two", "three four five", "six"]
        self.assertEqual(handle_token_limits(conversation, 4), ["three four five", "six"])

    def test_token_limit_truncates_overlong_turn(self) -> None:
        self.assertEqual(truncate_text_to_token_budget("one two three", 2), "one two")
        self.assertEqual(handle_token_limits(["one two three four"], 2), ["one two"])

    def test_char_limit_truncates_text_without_spaces(self) -> None:
        self.assertEqual(truncate_text_to_char_budget("abcdef", 3), "abc")
        limited_prompt = build_model_prompt_request([], "abcdef", max_tokens=10, max_chars=57)
        self.assertEqual(
            limited_prompt.completion_prompt,
            "<start_of_turn>user\nabc<end_of_turn>\n<start_of_turn>model",
        )
        self.assertTrue(limited_prompt.was_limited)
        self.assertEqual(
            build_model_prompt([], "abcdef", max_tokens=10, max_chars=60),
            "<start_of_turn>user\nabcdef<end_of_turn>\n<start_of_turn>model",
        )

    def test_prompt_normalization_preserves_multiline_paste(self) -> None:
        text = "Summarize this\n\n첫 문장입니다.\n￼\n둘째 문장입니다."
        self.assertEqual(normalize_prompt_text(text), "Summarize this\n\n첫 문장입니다.\n\n둘째 문장입니다.")

    def test_prompt_normalization_preserves_code_structure(self) -> None:
        text = "Traceback:\r\n  File \"app.py\", line 1\r\n    raise ValueError()\r\nValueError"
        self.assertEqual(
            normalize_prompt_text(text),
            "Traceback:\n  File \"app.py\", line 1\n    raise ValueError()\nValueError",
        )

    def test_build_model_prompt_includes_history(self) -> None:
        messages = [
            {"role": "user", "content": "Here is some code:\n    print('hi')"},
            {"role": "assistant", "content": "It prints hi."},
        ]
        prompt = build_model_prompt(messages, "Explain the previous code again.", 100)
        self.assertIn("<start_of_turn>user\nHere is some code:\n    print('hi')<end_of_turn>", prompt)
        self.assertIn("<start_of_turn>model\nIt prints hi.<end_of_turn>", prompt)
        self.assertTrue(
            prompt.endswith(
                "<start_of_turn>user\nExplain the previous code again.<end_of_turn>\n<start_of_turn>model"
            )
        )

    def test_build_model_prompt_request_includes_structured_chat_messages(self) -> None:
        prompt = build_model_prompt_request(
            [{"role": "assistant", "content": "Prior answer."}],
            "Follow up.",
            max_tokens=100,
            system_prompt="Be concise.",
        )
        self.assertEqual(
            prompt.messages,
            [
                {"role": "system", "content": "Be concise."},
                {"role": "assistant", "content": "Prior answer."},
                {"role": "user", "content": "Follow up."},
            ],
        )
        self.assertIn("<start_of_turn>model\nPrior answer.<end_of_turn>", prompt.completion_prompt)

    def test_build_model_prompt_request_includes_memory_context(self) -> None:
        prompt = build_model_prompt_request(
            [{"role": "assistant", "content": "Prior answer."}],
            "Follow up.",
            max_tokens=100,
            system_prompt="Be concise.",
            memory_context=build_memory_context(
                summary="The user prefers local vector memory.",
                retrieved_context="[1] design note\nUse top-k chunks.",
            ),
        )

        self.assertEqual(prompt.messages[0], {"role": "system", "content": "Be concise."})
        self.assertEqual(prompt.messages[1]["role"], "system")
        self.assertIn("Conversation summary:", prompt.messages[1]["content"])
        self.assertIn("Relevant retrieved context:", prompt.messages[1]["content"])
        self.assertEqual(prompt.messages[-1], {"role": "user", "content": "Follow up."})

    def test_build_model_prompt_trims_oldest_turns_first(self) -> None:
        messages = [
            {"role": "user", "content": "old user context " * 10},
            {"role": "assistant", "content": "old assistant context " * 10},
            {"role": "user", "content": "newer user"},
            {"role": "assistant", "content": "newer assistant"},
        ]
        prompt = build_model_prompt(messages, "current question", max_tokens=12)
        self.assertNotIn("old user context", prompt)
        self.assertNotIn("old assistant context", prompt)
        self.assertIn("newer assistant", prompt)
        self.assertIn("current question", prompt)
        self.assertTrue(prompt.endswith("<start_of_turn>model"))

    def test_prompt_token_budget_reserves_response_space(self) -> None:
        self.assertEqual(prompt_token_budget(128, 32), 96)
        self.assertEqual(prompt_token_budget(4, 32), 1)


if __name__ == "__main__":
    unittest.main()
