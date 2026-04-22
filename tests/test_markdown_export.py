import unittest

from src.gui.markdown_export import format_chat_markdown, safe_markdown_filename


class MarkdownExportTests(unittest.TestCase):
    def test_format_chat_markdown(self) -> None:
        markdown = format_chat_markdown(
            "Session Title",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "**hi**"},
            ],
        )

        self.assertEqual(
            markdown,
            "# Session Title\n\n## You\n\nhello\n\n## Assistant\n\n**hi**\n",
        )

    def test_safe_markdown_filename(self) -> None:
        self.assertEqual(safe_markdown_filename("Chat: one/two?"), "Chat_ one_two_")
        self.assertEqual(safe_markdown_filename(""), "chat-export")


if __name__ == "__main__":
    unittest.main()
