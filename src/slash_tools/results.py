from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SlashToolResult:
    """Structured local tool result with display and compact history text."""

    text: str
    tool_name: str
    saved_files: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    @property
    def history_text(self) -> str:
        return self.text.strip()


def error_result(message: str, tool_name: str = "/unknown") -> SlashToolResult:
    return SlashToolResult(text=f"Tool error: {message}", tool_name=tool_name)
