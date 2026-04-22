from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from src.document_pipeline.schemas import DocumentMap, ExtractedDocument


@dataclass
class SlashToolContext:
    """Runtime state shared by local slash tools in one GUI window."""

    working_folder: Path | None = None
    documents: list[ExtractedDocument] = field(default_factory=list)
    doc_map: DocumentMap | None = None
    saved_files: list[str] = field(default_factory=list)
    llm_settings: Any | None = None
    cancel_event: Event | None = None
    active_llm_client: Any | None = None
    last_tool_name: str = ""
    last_tool_summary: str = ""

    def reset_for_folder(self, working_folder: Path | None) -> None:
        self.working_folder = working_folder
        self.documents.clear()
        self.doc_map = None
        self.saved_files.clear()
        self.active_llm_client = None
        self.last_tool_name = ""
        self.last_tool_summary = ""

    def copy_for_worker(self) -> "SlashToolContext":
        return SlashToolContext(
            working_folder=self.working_folder,
            documents=list(self.documents),
            doc_map=self.doc_map,
            saved_files=list(self.saved_files),
            llm_settings=self.llm_settings,
            cancel_event=self.cancel_event,
            active_llm_client=self.active_llm_client,
            last_tool_name=self.last_tool_name,
            last_tool_summary=self.last_tool_summary,
        )

    def replace_from(self, other: "SlashToolContext") -> None:
        self.working_folder = other.working_folder
        self.documents = list(other.documents)
        self.doc_map = other.doc_map
        self.saved_files = list(other.saved_files)
        self.llm_settings = other.llm_settings
        self.cancel_event = other.cancel_event
        self.active_llm_client = other.active_llm_client
        self.last_tool_name = other.last_tool_name
        self.last_tool_summary = other.last_tool_summary

    def cancel_requested(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()
