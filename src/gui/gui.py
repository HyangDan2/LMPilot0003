from __future__ import annotations

import re
import shlex
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor, QKeySequence, QShortcut, QTextDocumentFragment
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.slash_tools import SlashToolContext, SlashToolResult, run_slash_command
from src.slash_tools.results import error_result

from .attachment_handler import (
    AttachmentError,
    extract_text_from_file,
    list_supported_files_in_folder,
    validate_attachment_path,
)
from .config import AppConfig, save_config, save_connection_settings
from .console_session import (
    ConsoleConfig,
    ConsoleSessionError,
    LlamaConsoleSession,
    LlamaServerSession,
    OpenAICompatibleSession,
)
from .database import ChatRepository
from .llm_client import ChatStreamChunk, OpenAIConnectionSettings
from .markdown_export import format_chat_markdown, safe_markdown_filename
from .rag_store import RagStore, build_rag_context, chunk_text
from .session_title import DEFAULT_SESSION_TITLE, derive_session_title_from_input
from .token_handler import (
    ModelPrompt,
    build_memory_context,
    build_model_prompt_request,
    normalize_prompt_text,
    prompt_token_budget,
)

UNICODE_ESCAPE_RE = re.compile(r'\\u[0-9a-fA-F]{4}|/u[0-9a-fA-F]{4}')
MAX_INPUT_HISTORY = 10


def normalize_text_for_display(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    def replace_escape(match: re.Match[str]) -> str:
        fixed = match.group(0).replace('/u', '\\u')
        try:
            return chr(int(fixed[2:], 16))
        except ValueError:
            return match.group(0)

    return UNICODE_ESCAPE_RE.sub(replace_escape, text)


def strip_unsupported_chars(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return ''.join(ch for ch in text if ord(ch) <= 0xFFFF)


class ChatSession(Protocol):
    config: ConsoleConfig

    def start(self) -> None: ...
    def stop(self, force: bool = False) -> None: ...
    def stop_generation(self) -> None: ...
    def ask(self, prompt_text: str | ModelPrompt) -> str: ...
    def ask_stream(self, prompt_text: str | ModelPrompt) -> Iterator[ChatStreamChunk]: ...
    def update_connection_settings(self, settings: OpenAIConnectionSettings) -> None: ...
    def test_connection(self) -> str: ...
    def list_models(self) -> list[str]: ...


class ChatWorker(QObject):
    chunk = Signal(int, str, str)
    finished = Signal(int, str, bool)
    error = Signal(int, str)
    done = Signal(int)

    def __init__(self, session_id: int, console: ChatSession, prompt_text: str | ModelPrompt) -> None:
        super().__init__()
        self.session_id = session_id
        self.console = console
        self.prompt_text = prompt_text

    @Slot()
    def run(self) -> None:
        try:
            stream_answer = getattr(self.console, 'ask_stream', None)
            if callable(stream_answer):
                answer_parts: list[str] = []
                for chunk in stream_answer(self.prompt_text):
                    if chunk.kind == 'final':
                        answer_parts.append(chunk.text)
                    self.chunk.emit(self.session_id, chunk.kind, chunk.text)
                self.finished.emit(self.session_id, ''.join(answer_parts), True)
                return

            answer = self.console.ask(self.prompt_text)
            self.finished.emit(self.session_id, answer, False)
        except ConsoleSessionError as exc:
            self.error.emit(self.session_id, str(exc))
        except Exception:
            traceback.print_exc()
            self.error.emit(self.session_id, 'Unexpected internal error while generating a response.')
        finally:
            self.done.emit(self.session_id)


class SlashToolWorker(QObject):
    progress = Signal(int, str, str)
    finished = Signal(int, object, object)
    error = Signal(int, str)
    done = Signal(int)

    def __init__(
        self,
        session_id: int,
        command_text: str,
        working_folder: str | None,
        context: SlashToolContext,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.command_text = command_text
        self.working_folder = working_folder
        self.context = context
        self.cancel_event = cancel_event
        self.context.cancel_event = cancel_event

    def request_stop(self) -> None:
        self.cancel_event.set()
        client = self.context.active_llm_client
        close_active_request = getattr(client, "close_active_request", None)
        if callable(close_active_request):
            close_active_request()

    @Slot()
    def run(self) -> None:
        try:
            def emit_progress(kind: str, text: str) -> None:
                if self.cancel_event.is_set():
                    raise RuntimeError("Slash tool cancelled.")
                self.progress.emit(self.session_id, kind, text)

            if self.cancel_event.is_set():
                raise RuntimeError("Slash tool cancelled.")
            result = run_slash_command(
                self.command_text,
                self.working_folder,
                self.context,
                progress=emit_progress,
            )
            if result is None:
                result = error_result("empty slash command", "/unknown")
            self.finished.emit(self.session_id, result, self.context)
        except Exception as exc:
            traceback.print_exc()
            message = "Slash tool cancelled." if self.cancel_event.is_set() else str(exc) or "Unexpected internal error while running slash tool."
            self.error.emit(self.session_id, message)
        finally:
            self.done.emit(self.session_id)


@dataclass
class ActiveGeneration:
    session_id: int
    thread: QThread
    worker: ChatWorker
    console: ChatSession
    stop_requested: bool = False
    was_streamed: bool = False
    partial_answer: list[str] = field(default_factory=list)


@dataclass
class ActiveSlashTool:
    session_id: int
    thread: QThread
    worker: SlashToolWorker
    command_text: str
    stream_parts: list[str] = field(default_factory=list)
    stop_requested: bool = False


class ConnectionTestWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, console: ChatSession) -> None:
        super().__init__()
        self.console = console

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(self.console.test_connection())
        except ConsoleSessionError as exc:
            self.error.emit(str(exc))
        except Exception:
            traceback.print_exc()
            self.error.emit('Unexpected internal error while testing the connection.')


class MainWindow(QMainWindow):
    def __init__(self, console: ChatSession, repository: ChatRepository, app_config: AppConfig) -> None:
        super().__init__()
        self.console = console
        self.repository = repository
        self.app_config = app_config
        self.current_session_id: int | None = None
        self._active_generations: dict[int, ActiveGeneration] = {}
        self._thread_session_ids: dict[QThread, int] = {}
        self._active_slash_tools: dict[int, ActiveSlashTool] = {}
        self._slash_tool_thread_session_ids: dict[QThread, int] = {}
        self._connection_test_thread: QThread | None = None
        self._connection_test_worker: ConnectionTestWorker | None = None
        self._streaming_assistant_started = False
        self._streaming_block_start: int | None = None
        self._reasoning_placeholder_start: int | None = None
        self._tool_stream_block_start: int | None = None
        self._attached_file_paths: list[str] = []
        self._attachment_folder_roots: dict[str, str] = {}
        self._slash_tool_context = SlashToolContext()
        self._rag_store = RagStore(app_config.db_path)
        self._rag_index_signature: tuple[tuple[str, int, int], ...] | None = None
        self._rag_settings_signature: tuple[str, str, str] | None = None
        self._input_history: list[str] = []
        self._input_history_index: int | None = None

        self.setWindowTitle(app_config.window_title)
        self.resize(app_config.window_width, app_config.window_height)

        self.session_list = QListWidget()
        self.session_list.itemClicked.connect(self._on_session_selected)

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText('Type your message here...')
        self.input_edit.setFixedHeight(120)

        self.attachment_list = QListWidget()
        self.attachment_list.setMaximumHeight(120)
        self.attachment_list.itemDoubleClicked.connect(self._on_attachment_double_clicked)

        self.attach_file_btn = QPushButton('Attach Folder')
        self.attach_file_btn.clicked.connect(self.on_attach_files)

        self.clear_attachments_btn = QPushButton('Clear Attachments')
        self.clear_attachments_btn.clicked.connect(self._clear_attached_files)
        self.clear_attachments_btn.setDisabled(True)

        self.send_btn = QPushButton('Send')
        self.send_btn.clicked.connect(self.on_send)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setDisabled(True)
        self.stop_btn.clicked.connect(self.on_stop_generation)

        self.new_chat_btn = QPushButton('New Chat')
        self.new_chat_btn.clicked.connect(self.on_new_chat)

        self.delete_chat_btn = QPushButton('Delete Chat')
        self.delete_chat_btn.clicked.connect(self._delete_current_chat)

        self.clear_view_btn = QPushButton('Clear View')
        self.clear_view_btn.clicked.connect(self._clear_view_only)

        self.copy_last_output_btn = QPushButton('Copy Last Output')
        self.copy_last_output_btn.clicked.connect(self.on_copy_last_output)

        self.save_chat_btn = QPushButton('Save Chat')
        self.save_chat_btn.clicked.connect(self.on_save_chat_markdown)

        settings_group = QGroupBox('OpenAI-Compatible Settings')
        settings_layout = QFormLayout(settings_group)
        self.base_url_edit = QLineEdit(app_config.openai_base_url)
        self.base_url_edit.setPlaceholderText('http://127.0.0.1:8000/v1')
        self.api_key_edit = QLineEdit(app_config.openai_api_key)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText('Optional for local servers')
        self.model_edit = QLineEdit(app_config.openai_model)
        self.model_edit.setPlaceholderText('Model name from your backend')
        self.embedding_model_edit = QLineEdit(app_config.openai_embedding_model)
        self.embedding_model_edit.setPlaceholderText('Optional embedding model for RAG')
        self.save_settings_btn = QPushButton('Save Settings')
        self.save_settings_btn.clicked.connect(self.on_save_connection_settings)
        self.test_connection_btn = QPushButton('Test Connection')
        self.test_connection_btn.clicked.connect(self.on_test_connection)
        settings_button_row = QHBoxLayout()
        settings_button_row.addStretch(1)
        settings_button_row.addWidget(self.save_settings_btn)
        settings_button_row.addWidget(self.test_connection_btn)
        settings_layout.addRow('Base URL', self.base_url_edit)
        settings_layout.addRow('API Key', self.api_key_edit)
        settings_layout.addRow('Model Name', self.model_edit)
        settings_layout.addRow('Embedding Model', self.embedding_model_edit)
        settings_layout.addRow(settings_button_row)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel('Sessions'))
        left_layout.addWidget(self.session_list)
        left_layout.addWidget(QLabel('Attachments'))
        left_layout.addWidget(self.attachment_list)
        attachment_button_row = QHBoxLayout()
        attachment_button_row.addWidget(self.attach_file_btn)
        left_layout.addLayout(attachment_button_row)
        left_layout.addWidget(self.clear_attachments_btn)
        left_layout.addWidget(self.new_chat_btn)
        left_layout.addWidget(self.delete_chat_btn)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(settings_group)
        right_layout.addWidget(QLabel('Chat'))
        right_layout.addWidget(self.chat_view, stretch=1)
        right_layout.addWidget(QLabel('Input'))
        right_layout.addWidget(self.input_edit)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.copy_last_output_btn)
        button_row.addWidget(self.save_chat_btn)
        button_row.addWidget(self.clear_view_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(self.send_btn)
        right_layout.addLayout(button_row)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 780])

        container = QWidget()
        root_layout = QVBoxLayout(container)
        root_layout.addWidget(splitter)
        self.setCentralWidget(container)

        status = QStatusBar()
        self.setStatusBar(status)
        self._set_status('Starting console session...')

        self._init_console()
        self._reload_sessions()
        self._restore_last_working_folder()
        self._set_status('Idle')
        self._append_block('System', 'Select a session or click New Chat.')

        # Ctrl+Enter → Send
        self.send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.input_edit)
        self.send_shortcut.activated.connect(self.on_send)
        self.input_history_shortcut = QShortcut(QKeySequence("Ctrl+Up"), self.input_edit)
        self.input_history_shortcut.activated.connect(self._recall_previous_input)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        for generation in list(self._active_generations.values()):
            generation.stop_requested = True
            generation.console.stop_generation()

        for generation in list(self._active_generations.values()):
            generation.thread.quit()
            generation.thread.wait(3000)
            if generation.thread.isRunning():
                self._set_status('Please wait for active responses to finish before closing.')
                event.ignore()
                return

        if self._connection_test_thread is not None and self._connection_test_thread.isRunning():
            self.console.stop_generation()
            self._connection_test_thread.quit()
            self._connection_test_thread.wait(3000)
            if self._connection_test_thread.isRunning():
                self._set_status('Please wait for the connection test to finish before closing.')
                event.ignore()
                return

        for active in list(self._active_slash_tools.values()):
            active.stop_requested = True
            active.worker.request_stop()

        for active in list(self._active_slash_tools.values()):
            active.thread.quit()
            active.thread.wait(3000)
            if active.thread.isRunning():
                self._set_status('Please wait for active slash tools to finish before closing.')
                event.ignore()
                return

        self.console.stop()
        super().closeEvent(event)

    def _init_console(self) -> None:
        try:
            self.console.start()
            self._set_status('Idle')
        except ConsoleSessionError as exc:
            QMessageBox.critical(self, 'Startup Error', str(exc))
            self._set_status('Startup error')
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, 'Startup Error', f'Unexpected startup error: {exc}')
            self._set_status('Startup error')

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _current_connection_settings(self) -> OpenAIConnectionSettings:
        return OpenAIConnectionSettings(
            base_url=self.base_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            model=self.model_edit.text().strip(),
            embedding_model=self.embedding_model_edit.text().strip(),
            temperature=self.app_config.temperature,
            max_tokens=self.app_config.n_predict,
            timeout=self.app_config.response_timeout,
        )

    def _apply_connection_settings(self) -> OpenAIConnectionSettings:
        settings = self._current_connection_settings()
        self.app_config.openai_base_url = settings.base_url
        self.app_config.openai_api_key = settings.api_key
        self.app_config.openai_model = settings.model
        self.app_config.openai_embedding_model = settings.embedding_model
        self.app_config.temperature = settings.temperature
        self.app_config.n_predict = settings.max_tokens
        if hasattr(self.console, 'update_connection_settings'):
            self.console.update_connection_settings(settings)
        return settings

    def _console_config_from_settings(self, settings: OpenAIConnectionSettings) -> ConsoleConfig:
        return ConsoleConfig(
            llama_cli_path=self.app_config.llama_cli_path,
            model_path=self.app_config.model_path,
            backend=self.app_config.backend,
            server_url=self.app_config.server_url or settings.base_url,
            server_endpoint=getattr(self.app_config, "server_endpoint", "auto"),
            n_predict=settings.max_tokens,
            system_prompt=self.app_config.system_prompt,
            threads=self.app_config.threads,
            ctx_size=self.app_config.ctx_size,
            extra_args=list(self.app_config.extra_args),
            startup_timeout=self.app_config.startup_timeout,
            response_timeout=settings.timeout,
            openai_base_url=settings.base_url,
            openai_api_key=settings.api_key,
            openai_model=settings.model,
            openai_embedding_model=settings.embedding_model,
            temperature=settings.temperature,
            artifact_working_folder=self._active_attachment_folder() or "",
        )

    def _create_generation_console(self, settings: OpenAIConnectionSettings) -> ChatSession:
        config = self._console_config_from_settings(settings)
        if isinstance(self.console, OpenAICompatibleSession) or self.app_config.backend == "openai":
            return OpenAICompatibleSession(config)
        if isinstance(self.console, LlamaServerSession) or self.app_config.backend == "server":
            return LlamaServerSession(config)
        if isinstance(self.console, LlamaConsoleSession):
            return LlamaConsoleSession(config)
        return self.console

    @Slot()
    def on_save_connection_settings(self) -> None:
        settings = self._apply_connection_settings()
        try:
            save_connection_settings(self.app_config.connection_settings_path, settings)
        except OSError as exc:
            QMessageBox.critical(self, 'Save Settings', f'Could not save settings: {exc}')
            self._set_status('Settings save failed')
            return
        self._set_status('Settings saved')
        self._append_block('System', 'Connection settings saved.')

    @Slot()
    def on_test_connection(self) -> None:
        if self._active_generations:
            QMessageBox.information(self, 'Test Connection', 'Please wait for active responses to finish first.')
            return
        if self._connection_test_thread is not None and self._connection_test_thread.isRunning():
            return
        self._apply_connection_settings()
        self.test_connection_btn.setDisabled(True)
        self.save_settings_btn.setDisabled(True)
        self._set_status('Testing connection...')

        self._connection_test_thread = QThread()
        self._connection_test_worker = ConnectionTestWorker(self.console)
        self._connection_test_worker.moveToThread(self._connection_test_thread)
        self._connection_test_thread.started.connect(self._connection_test_worker.run)
        self._connection_test_worker.finished.connect(self._on_connection_test_success)
        self._connection_test_worker.error.connect(self._on_connection_test_error)
        self._connection_test_worker.finished.connect(self._connection_test_thread.quit)
        self._connection_test_worker.error.connect(self._connection_test_thread.quit)
        self._connection_test_thread.finished.connect(self._cleanup_connection_test_worker)
        self._connection_test_thread.start()

    @Slot(str)
    def _on_connection_test_success(self, message: str) -> None:
        self._set_status('Connection OK')
        self._append_block('System', message)

    @Slot(str)
    def _on_connection_test_error(self, error_text: str) -> None:
        self._set_status('Connection failed')
        self._append_block('System', f'Connection test failed: {error_text}')

    @Slot()
    def _cleanup_connection_test_worker(self) -> None:
        if self._connection_test_worker is not None:
            self._connection_test_worker.deleteLater()
        if self._connection_test_thread is not None:
            self._connection_test_thread.deleteLater()
        self._connection_test_worker = None
        self._connection_test_thread = None
        self._refresh_controls()

    def _reload_sessions(self) -> None:
        selected_session_id = self.current_session_id
        self.session_list.clear()
        sessions = self.repository.list_sessions()
        for session in sessions:
            title = session['title']
            if session['id'] in self._active_generations:
                title = f"{title} [running]"
            elif session['id'] in self._active_slash_tools:
                title = f"{title} [tool]"
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, session['id'])
            self.session_list.addItem(item)
        if selected_session_id is not None:
            self._select_session_in_list(selected_session_id)

    def on_new_chat(self) -> None:
        self.current_session_id = self.repository.create_session(DEFAULT_SESSION_TITLE)
        self._reload_sessions()
        self._select_session_in_list(self.current_session_id)
        self.chat_view.clear()
        self._append_block('System', 'New chat started.')
        self._set_status('Idle')
        self._refresh_controls()

    def _select_session_in_list(self, session_id: int) -> None:
        for i in range(self.session_list.count()):
            item = self.session_list.item(i)
            if item.data(Qt.UserRole) == session_id:
                self.session_list.setCurrentItem(item)
                break

    @Slot()
    def on_attach_files(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            'Attach Workspace Folder',
            '',
        )
        if not folder:
            return
        try:
            self._attach_folder_path(folder, persist=True)
        except AttachmentError as exc:
            QMessageBox.warning(self, 'Attach Folder', str(exc))

    @Slot()
    def on_save_chat_markdown(self) -> None:
        if self.current_session_id is None:
            QMessageBox.information(self, 'Save Chat', 'Please select or create a session first.')
            return

        messages = self.repository.get_messages(self.current_session_id)
        if not messages:
            QMessageBox.information(self, 'Save Chat', 'This session has no chat messages to save.')
            return

        title = self.repository.get_session_title(self.current_session_id)
        default_path = f'{safe_markdown_filename(title)}.md'
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Chat Markdown',
            default_path,
            'Markdown files (*.md);;All files (*)',
        )
        if not output_path:
            return

        file_path = Path(output_path)
        if not file_path.suffix:
            file_path = file_path.with_suffix('.md')

        try:
            file_path.write_text(format_chat_markdown(title, messages), encoding='utf-8')
        except OSError as exc:
            QMessageBox.critical(self, 'Save Chat', f'Could not save chat: {exc}')
            self._set_status('Chat save failed')
            return
        self._set_status(f'Chat saved: {file_path}')

    @Slot()
    def on_copy_last_output(self) -> None:
        if self.current_session_id is None:
            QMessageBox.information(self, 'Copy Last Output', 'Please select or create a session first.')
            return

        last_output = self._last_assistant_output()
        if not last_output:
            QMessageBox.information(self, 'Copy Last Output', 'This session has no assistant output to copy.')
            return

        QApplication.clipboard().setText(last_output)
        self._set_status('Last output copied to clipboard')

    @Slot()
    def on_send(self) -> None:
        user_text = self.input_edit.toPlainText().strip()
        if not user_text:
            return
        normalized_user_text = normalize_prompt_text(user_text) if user_text else ''
        display_user_text = normalized_user_text
        if not display_user_text:
            return
        if self.current_session_id is not None and self.current_session_id in self._active_generations:
            QMessageBox.information(self, 'Send', 'This session is already generating a response.')
            return
        self._remember_input_history(display_user_text)
        if display_user_text.startswith("/") and self._handle_slash_tool_command(display_user_text):
            self.input_edit.clear()
            return

        settings = self._apply_connection_settings()
        if self.app_config.backend != "cli" and not settings.base_url:
            QMessageBox.warning(self, 'Missing Base URL', 'Enter a Base URL before sending a prompt.')
            self._set_status('Missing Base URL')
            return
        if self.app_config.backend != "cli" and not settings.model:
            QMessageBox.warning(self, 'Missing Model Name', 'Enter a Model Name before sending a prompt.')
            self._set_status('Missing Model Name')
            return
        if self.current_session_id is None:
            self.on_new_chat()

        assert self.current_session_id is not None
        target_session_id = self.current_session_id
        if target_session_id in self._active_generations:
            QMessageBox.information(self, 'Send', 'This session is already generating a response.')
            return

        prior_message_count = self.repository.count_messages(target_session_id)
        prior_messages = self.repository.get_recent_messages(
            target_session_id,
            self.app_config.recent_message_limit,
        )
        retrieved_context = self._build_retrieved_context(display_user_text, settings)
        memory_context = build_memory_context(
            summary=self.repository.get_session_summary(target_session_id),
            retrieved_context=retrieved_context,
            max_chars=self.app_config.memory_context_char_limit,
        )
        max_prompt_tokens = prompt_token_budget(
            self.console.config.ctx_size,
            self.app_config.response_token_reserve,
        )
        model_prompt = build_model_prompt_request(
            prior_messages,
            display_user_text,
            max_prompt_tokens,
            self.app_config.max_prompt_chars,
            self.app_config.system_prompt,
            memory_context,
        )
        was_limited = (
            (bool(user_text) and display_user_text != user_text)
            or model_prompt.was_limited
            or prior_message_count > len(prior_messages)
        )
        self.input_edit.clear()
        self._append_block('You', display_user_text)
        if was_limited:
            self._append_block('System', 'Prompt context was shortened to fit the configured context limit.')
        self.repository.add_message(target_session_id, 'user', display_user_text)
        self._set_new_session_title(target_session_id, display_user_text)
        self._reload_sessions()
        self._select_session_in_list(target_session_id)
        self._set_status('Generating response...')
        self._start_worker(target_session_id, model_prompt, settings)
        self._refresh_controls()

    def _remember_input_history(self, text: str) -> None:
        normalized = normalize_prompt_text(text)
        if not normalized:
            return
        self._input_history = [item for item in self._input_history if item != normalized]
        self._input_history.insert(0, normalized)
        del self._input_history[MAX_INPUT_HISTORY:]
        self._input_history_index = None

    @Slot()
    def _recall_previous_input(self) -> None:
        if not self._input_history:
            return
        if self._input_history_index is None:
            self._input_history_index = 0
        else:
            self._input_history_index = min(self._input_history_index + 1, len(self._input_history) - 1)
        self.input_edit.setPlainText(self._input_history[self._input_history_index])
        cursor = self.input_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.input_edit.setTextCursor(cursor)
        self.input_edit.setFocus()

    def _handle_slash_tool_command(self, display_user_text: str) -> bool:
        if self.current_session_id is None:
            self.on_new_chat()
        assert self.current_session_id is not None
        target_session_id = self.current_session_id
        if target_session_id in self._active_slash_tools:
            QMessageBox.information(self, 'Send', 'This session already has a slash tool running.')
            return True
        self._slash_tool_context.llm_settings = self._current_connection_settings()
        context_snapshot = self._slash_tool_context.copy_for_worker()
        self._append_block('You', display_user_text)
        self.repository.add_message(target_session_id, 'user', display_user_text)
        if self.repository.get_session_title(target_session_id) == DEFAULT_SESSION_TITLE:
            self._set_new_session_title(target_session_id, display_user_text)
        self._reload_sessions()
        self._select_session_in_list(target_session_id)
        self._start_slash_tool_worker(
            target_session_id,
            display_user_text,
            self._active_attachment_folder(),
            context_snapshot,
        )
        self._refresh_controls()
        return True

    def _start_slash_tool_worker(
        self,
        session_id: int,
        command_text: str,
        working_folder: str | None,
        context: SlashToolContext,
    ) -> None:
        thread = QThread()
        cancel_event = threading.Event()
        worker = SlashToolWorker(session_id, command_text, working_folder, context, cancel_event)
        active = ActiveSlashTool(
            session_id=session_id,
            thread=thread,
            worker=worker,
            command_text=command_text,
        )
        self._active_slash_tools[session_id] = active
        self._slash_tool_thread_session_ids[thread] = session_id
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_slash_tool_progress)
        worker.finished.connect(self._on_slash_tool_success)
        worker.error.connect(self._on_slash_tool_error)
        worker.done.connect(self._quit_slash_tool_thread)
        thread.finished.connect(self._cleanup_slash_tool_worker)
        thread.start()
        self._set_status('Running slash tool...')
        if self.current_session_id == session_id:
            self._start_tool_stream_block()
            self._append_tool_stream_text(f"Running {command_text}\n\n")
        active.stream_parts.append(f"Running {command_text}\n\n")
        self._reload_sessions()

    @Slot(int, str, str)
    def _on_slash_tool_progress(self, session_id: int, kind: str, text: str) -> None:
        active = self._active_slash_tools.get(session_id)
        if active is None or not text:
            return
        formatted = text if kind == "markdown" else text
        if not formatted.endswith("\n"):
            formatted = f"{formatted}\n"
        active.stream_parts.append(formatted)
        if self.current_session_id == session_id:
            self._append_tool_stream_text(formatted)
            if kind == "status":
                self._set_status(text.strip()[:120] or "Running slash tool...")

    @Slot(int, object, object)
    def _on_slash_tool_success(
        self,
        session_id: int,
        result: SlashToolResult,
        updated_context: SlashToolContext,
    ) -> None:
        self._slash_tool_context.replace_from(updated_context)
        self.repository.add_message(session_id, 'tool', result.history_text)
        if self.current_session_id == session_id:
            self._append_tool_stream_text("\n" + result.text + "\n\n")
            self._finish_tool_stream_block()
            self._set_status('Tool complete')
        else:
            self._set_status('Slash tool complete')
        self._reload_sessions()
        self._refresh_controls()

    @Slot(int, str)
    def _on_slash_tool_error(self, session_id: int, error_text: str) -> None:
        self.repository.add_message(session_id, 'tool', f'Tool error: {error_text}')
        if self.current_session_id == session_id:
            self._append_tool_stream_text(f"\nTool error: {error_text}\n\n")
            self._finish_tool_stream_block()
        self._set_status('Tool error')
        self._reload_sessions()
        self._refresh_controls()

    @Slot(int)
    def _quit_slash_tool_thread(self, session_id: int) -> None:
        active = self._active_slash_tools.get(session_id)
        if active is not None:
            active.thread.quit()

    @Slot()
    def _cleanup_slash_tool_worker(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QThread):
            return
        session_id = self._slash_tool_thread_session_ids.pop(sender, None)
        if session_id is None:
            return
        active = self._active_slash_tools.pop(session_id, None)
        if active is None:
            return
        active.worker.deleteLater()
        active.thread.deleteLater()
        if self.current_session_id == session_id:
            self._tool_stream_block_start = None
        self._reload_sessions()
        self._refresh_controls()

    @Slot()
    def on_stop_generation(self) -> None:
        if self.current_session_id is None:
            return
        generation = self._active_generations.get(self.current_session_id)
        if generation is not None:
            generation.stop_requested = True
            self.stop_btn.setDisabled(True)
            self._set_status('Stopping response...')
            generation.console.stop_generation()
            return

        slash_tool = self._active_slash_tools.get(self.current_session_id)
        if slash_tool is not None:
            slash_tool.stop_requested = True
            self.stop_btn.setDisabled(True)
            self._set_status('Stopping slash tool...')
            slash_tool.worker.request_stop()

    def _start_worker(
        self,
        session_id: int,
        prompt_text: str | ModelPrompt,
        settings: OpenAIConnectionSettings,
    ) -> None:
        self._streaming_assistant_started = False
        self._streaming_block_start = None
        self._reasoning_placeholder_start = None
        self._tool_stream_block_start = None
        self._tool_stream_block_start = None
        thread = QThread()
        console = self._create_generation_console(settings)
        worker = ChatWorker(session_id, console, prompt_text)
        generation = ActiveGeneration(
            session_id=session_id,
            thread=thread,
            worker=worker,
            console=console,
        )
        self._active_generations[session_id] = generation
        self._thread_session_ids[thread] = session_id
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunk.connect(self._on_generation_chunk)
        worker.finished.connect(self._on_generation_success)
        worker.error.connect(self._on_generation_error)
        worker.done.connect(self._quit_worker_thread)
        thread.finished.connect(self._cleanup_finished_worker)
        thread.start()
        self._reload_sessions()

    @Slot(int)
    def _quit_worker_thread(self, session_id: int) -> None:
        generation = self._active_generations.get(session_id)
        if generation is not None:
            generation.thread.quit()

    @Slot(int, str, str)
    def _on_generation_chunk(self, session_id: int, kind: str, chunk: str) -> None:
        generation = self._active_generations.get(session_id)
        if generation is None:
            return
        if kind == 'reasoning':
            if self.current_session_id == session_id:
                self._show_reasoning_placeholder()
                self._set_status('Reasoning...')
            return
        if kind != 'final' or not chunk:
            return
        generation.was_streamed = True
        generation.partial_answer.append(chunk)
        if self.current_session_id == session_id:
            self._clear_reasoning_placeholder()
            if not self._streaming_assistant_started:
                self._append_block_start('Assistant')
                self._streaming_assistant_started = True
            self._append_stream_text(chunk)
            self._set_status('Generating response...')

    @Slot(int, str, bool)
    def _on_generation_success(self, session_id: int, answer: str, was_streamed: bool) -> None:
        generation = self._active_generations.get(session_id)
        if generation is None:
            return

        is_visible = self.current_session_id == session_id
        if generation.stop_requested:
            if is_visible:
                self._clear_reasoning_placeholder()
                self._finish_stream_block()
                self._append_block('System', 'Generation stopped.')
            self._set_status('Idle')
            self._refresh_controls()
            return

        if is_visible:
            self._clear_reasoning_placeholder()
        if not answer.strip():
            if is_visible:
                self._append_block('System', 'Backend returned no final assistant answer.')
            self._set_status('Error')
            self._refresh_controls()
            return

        self.repository.add_message(session_id, 'assistant', answer)
        if is_visible:
            if was_streamed or self._streaming_assistant_started:
                self._replace_stream_block_with_markdown('Assistant', answer)
            else:
                self._append_block('Assistant', answer)
            self._set_status('Idle')
        else:
            self._set_status('Response finished')
        self._reload_sessions()
        self._refresh_controls()

    @Slot(int, str)
    def _on_generation_error(self, session_id: int, error_text: str) -> None:
        generation = self._active_generations.get(session_id)
        if generation is None:
            return

        is_visible = self.current_session_id == session_id
        if generation.stop_requested:
            if is_visible:
                self._clear_reasoning_placeholder()
                self._finish_stream_block()
                self._append_block('System', 'Generation stopped.')
            self._set_status('Idle')
            self._refresh_controls()
            return

        if is_visible:
            self._clear_reasoning_placeholder()
            self._finish_stream_block()
            self._append_block('System', f'Error: {error_text}')
        self._set_status('Error')
        self._refresh_controls()

    @Slot()
    def _cleanup_finished_worker(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QThread):
            return
        session_id = self._thread_session_ids.pop(sender, None)
        if session_id is None:
            return
        self._cleanup_worker(session_id)

    def _cleanup_worker(self, session_id: int) -> None:
        generation = self._active_generations.pop(session_id, None)
        if generation is not None:
            generation.worker.deleteLater()
            generation.thread.deleteLater()
            if generation.console is not self.console:
                generation.console.stop()
        if self.current_session_id == session_id:
            self._streaming_assistant_started = False
            self._streaming_block_start = None
            self._reasoning_placeholder_start = None
        self._reload_sessions()
        self._refresh_controls()

    def _set_new_session_title(
        self,
        session_id: int,
        user_text: str,
        attachment_filenames: list[str] | None = None,
    ) -> None:
        current_title = self.repository.get_session_title(session_id)
        if current_title != DEFAULT_SESSION_TITLE:
            return
        title = derive_session_title_from_input(user_text, attachment_filenames)
        if title != DEFAULT_SESSION_TITLE:
            self.repository.update_session_title(session_id, title)

    def _refresh_controls(self) -> None:
        current_generating = (
            self.current_session_id is not None
            and self.current_session_id in self._active_generations
        )
        has_active_generation = bool(self._active_generations)
        testing_connection = (
            self._connection_test_thread is not None
            and self._connection_test_thread.isRunning()
        )
        has_running_slash_tool = bool(self._active_slash_tools)
        current_slash_tool_running = (
            self.current_session_id is not None
            and self.current_session_id in self._active_slash_tools
        )
        self.send_btn.setDisabled(current_generating or testing_connection or current_slash_tool_running)
        self.stop_btn.setDisabled(not (current_generating or current_slash_tool_running))
        self.new_chat_btn.setDisabled(testing_connection)
        self.session_list.setDisabled(False)
        self.test_connection_btn.setDisabled(has_active_generation or testing_connection or has_running_slash_tool)
        self.save_settings_btn.setDisabled(testing_connection or has_running_slash_tool)
        self.attach_file_btn.setDisabled(testing_connection or has_running_slash_tool)
        self.copy_last_output_btn.setDisabled(testing_connection)
        self.save_chat_btn.setDisabled(testing_connection)
        self.clear_attachments_btn.setDisabled(testing_connection or has_running_slash_tool or not self._attached_file_paths)

    @Slot(QListWidgetItem)
    def _on_attachment_double_clicked(self, item: QListWidgetItem) -> None:
        row = self.attachment_list.row(item)
        if row < 0 or row >= len(self._attached_file_paths):
            return
        display_name = self._attachment_display_name(self._attached_file_paths[row])
        insertion = shlex.quote(display_name)
        if self.input_edit.toPlainText() and not self.input_edit.toPlainText().endswith((" ", "\n", "\t")):
            insertion = f" {insertion}"
        self.input_edit.insertPlainText(insertion)
        self.input_edit.setFocus()
        self._set_status(f'Inserted attachment filename: {display_name}')

    @Slot(QListWidgetItem)
    def _on_session_selected(self, item: QListWidgetItem) -> None:
        session_id = item.data(Qt.UserRole)
        if session_id is None:
            return
        self.current_session_id = int(session_id)
        self._load_session_messages(self.current_session_id)
        self._refresh_controls()

    def _load_session_messages(self, session_id: int) -> None:
        self.chat_view.clear()
        self._streaming_assistant_started = False
        self._streaming_block_start = None
        self._reasoning_placeholder_start = None
        messages = self.repository.get_messages(session_id)
        if not messages:
            self._append_block('System', 'New chat started.')
        else:
            for message in messages:
                self._append_block(self._display_label_for_role(message['role']), message['content'])
        generation = self._active_generations.get(session_id)
        if generation is not None:
            partial_answer = ''.join(generation.partial_answer)
            if partial_answer:
                self._append_block_start('Assistant')
                self._streaming_assistant_started = True
                self._append_stream_text(partial_answer)
            else:
                self._show_reasoning_placeholder()
            self._set_status('Generating response...')
        elif session_id in self._active_slash_tools:
            active_slash_tool = self._active_slash_tools[session_id]
            self._start_tool_stream_block()
            stream_text = ''.join(active_slash_tool.stream_parts)
            self._append_tool_stream_text(stream_text or f"Running {active_slash_tool.command_text}\n\n")
            self._set_status('Running slash tool...')
        else:
            self._set_status('Idle')
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _clear_view_only(self) -> None:
        self.chat_view.clear()

    def _last_assistant_output(self) -> str:
        if self.current_session_id is None:
            return ""
        for message in reversed(self.repository.get_messages(self.current_session_id)):
            if message.get('role') == 'assistant':
                return str(message.get('content', '')).strip()
        return ""

    def _attached_filenames(self, paths: list[str] | None = None) -> list[str]:
        source_paths = self._attached_file_paths if paths is None else paths
        return [Path(path).name for path in source_paths]

    def _attached_file_summary(self, paths: list[str] | None = None) -> str:
        source_paths = self._attached_file_paths if paths is None else paths
        lines: list[str] = []
        for path in source_paths:
            file_path = Path(path)
            file_type = file_path.suffix.lower().lstrip(".") or "file"
            lines.append(f'- {self._attachment_display_name(path)} ({file_type})')
        return '\n'.join(lines)

    def _active_attachment_folder(self) -> str | None:
        roots = [root for root in self._attachment_folder_roots.values() if root]
        if not roots:
            return None
        return roots[-1]

    def _attachment_display_name(self, path: str) -> str:
        root = self._attachment_folder_roots.get(path)
        if root:
            try:
                return str(Path(path).relative_to(root))
            except ValueError:
                pass
        return Path(path).name

    def _refresh_attachment_list(self) -> None:
        self.attachment_list.clear()
        for path in self._attached_file_paths:
            file_path = Path(path)
            file_type = file_path.suffix.lower().lstrip(".") or "file"
            item = QListWidgetItem(f'{self._attachment_display_name(path)} ({file_type})')
            item.setToolTip(path)
            self.attachment_list.addItem(item)
        self._refresh_controls()

    def _clear_attached_files(self) -> None:
        self._attached_file_paths.clear()
        self._attachment_folder_roots.clear()
        self._rag_store.clear_source_type("attachment")
        self._rag_index_signature = None
        self._rag_settings_signature = None
        self._slash_tool_context.reset_for_folder(None)
        self.app_config.last_working_folder = ""
        save_config(self.app_config)
        self._refresh_attachment_list()

    def _attach_folder_path(self, folder: str, persist: bool) -> None:
        files = list_supported_files_in_folder(folder)
        self._attached_file_paths.clear()
        self._attachment_folder_roots.clear()
        self.attachment_list.clear()

        root = str(Path(folder).expanduser().resolve())
        existing_paths = set(self._attached_file_paths)
        failures: list[str] = []
        for path in files:
            try:
                attachment_path = str(validate_attachment_path(str(path)))
            except AttachmentError as exc:
                failures.append(str(exc))
                continue
            self._attachment_folder_roots[attachment_path] = root
            if attachment_path in existing_paths:
                continue
            self._attached_file_paths.append(attachment_path)
            existing_paths.add(attachment_path)

        self._refresh_attachment_list()
        self._rag_index_signature = None
        self._rag_settings_signature = None
        self._slash_tool_context.reset_for_folder(Path(root))
        self.app_config.last_working_folder = root
        if persist:
            save_config(self.app_config)
        if failures:
            raise AttachmentError('\n'.join(failures))

    def _restore_last_working_folder(self) -> None:
        folder = self.app_config.last_working_folder.strip()
        if not folder:
            return
        folder_path = Path(folder).expanduser()
        if not folder_path.exists() or not folder_path.is_dir():
            self.app_config.last_working_folder = ""
            save_config(self.app_config)
            return
        try:
            self._attach_folder_path(str(folder_path), persist=True)
            self._set_status(f'Restored workspace folder: {folder_path}')
        except AttachmentError:
            self.app_config.last_working_folder = ""
            save_config(self.app_config)

    def _build_retrieved_context(
        self,
        query_text: str,
        settings: OpenAIConnectionSettings,
    ) -> str:
        if self.app_config.rag_top_k <= 0:
            return ""
        if not self._attached_file_paths:
            return ""
        normalized_query = normalize_prompt_text(query_text)
        if not normalized_query:
            return ""
        if not settings.base_url.strip():
            return ""
        if not (settings.embedding_model.strip() or settings.model.strip()):
            return ""

        try:
            client = self._ensure_rag_index(settings)
            vectors = client.embeddings([normalized_query])
            if not vectors:
                return ""
            results = self._rag_store.search(
                vectors[0],
                top_k=self.app_config.rag_top_k,
                min_score=self.app_config.rag_min_score,
            )
        except (AttachmentError, ValueError, OSError):
            return ""
        except Exception:
            return ""
        return build_rag_context(results, max_chars=self.app_config.memory_context_char_limit)

    def _ensure_rag_index(self, settings: OpenAIConnectionSettings):
        attachment_signature = self._attachment_signature()
        settings_signature = (
            settings.base_url.strip(),
            settings.embedding_model.strip(),
            settings.model.strip(),
        )
        client = getattr(self.console, "_client", None)
        if (
            attachment_signature == self._rag_index_signature
            and settings_signature == self._rag_settings_signature
            and attachment_signature
        ):
            return client or OpenAICompatibleSession(self._console_config_from_settings(settings))._client

        rag_client = client
        if rag_client is None or getattr(rag_client, "settings", None) != settings:
            rag_client = OpenAICompatibleSession(self._console_config_from_settings(settings))._client

        self._rag_store.clear_source_type("attachment")
        for path in self._attached_file_paths:
            attachment = extract_text_from_file(path)
            chunks = chunk_text(
                attachment.extracted_text,
                max_chars=self.app_config.rag_chunk_chars,
                overlap=self.app_config.rag_chunk_overlap,
            )
            if not chunks:
                continue
            embeddings = rag_client.embeddings(chunks)
            self._rag_store.replace_source_chunks(
                source_type="attachment",
                source_id=str(Path(path).resolve()),
                source_label=self._attachment_display_name(path),
                chunks=chunks,
                embeddings=embeddings,
            )
        self._rag_index_signature = attachment_signature
        self._rag_settings_signature = settings_signature
        return rag_client

    def _attachment_signature(self) -> tuple[tuple[str, int, int], ...]:
        signature: list[tuple[str, int, int]] = []
        for path in self._attached_file_paths:
            file_path = Path(path)
            try:
                stat = file_path.stat()
            except OSError:
                continue
            signature.append((str(file_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)))
        signature.sort()
        return tuple(signature)

    def _append_block(self, role: str, text: str) -> None:
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        if role == 'Tool':
            self._insert_plain_block(cursor, role, text)
        else:
            self._insert_markdown_block(cursor, role, text)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _append_block_start(self, role: str) -> None:
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._streaming_block_start = cursor.position()
        cursor.insertText(f'[{role}]\n')
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _start_tool_stream_block(self) -> None:
        if self._tool_stream_block_start is not None:
            return
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._tool_stream_block_start = cursor.position()
        cursor.insertText('[Tool]\n')
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _append_tool_stream_text(self, text: str) -> None:
        text = normalize_text_for_display(text)
        text = strip_unsupported_chars(text)
        if self._tool_stream_block_start is None:
            self._start_tool_stream_block()
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _finish_tool_stream_block(self) -> None:
        if self._tool_stream_block_start is None:
            return
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText('\n')
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()
        self._tool_stream_block_start = None

    def _show_reasoning_placeholder(self) -> None:
        if self._streaming_assistant_started or self._reasoning_placeholder_start is not None:
            return
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._reasoning_placeholder_start = cursor.position()
        cursor.insertText('[Assistant]\nReasoning...\n\n')
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _clear_reasoning_placeholder(self) -> None:
        if self._reasoning_placeholder_start is None:
            return
        cursor = self.chat_view.textCursor()
        cursor.setPosition(self._reasoning_placeholder_start)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()
        self._reasoning_placeholder_start = None

    def _append_stream_text(self, text: str) -> None:
        text = normalize_text_for_display(text)
        text = strip_unsupported_chars(text)
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _finish_stream_block(self) -> None:
        if not self._streaming_assistant_started:
            return
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText('\n\n')
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()
        self._streaming_assistant_started = False
        self._streaming_block_start = None

    def _replace_stream_block_with_markdown(self, role: str, text: str) -> None:
        cursor = self.chat_view.textCursor()
        if self._streaming_block_start is not None:
            cursor.setPosition(self._streaming_block_start)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
        else:
            cursor.movePosition(QTextCursor.End)
        self._insert_markdown_block(cursor, role, text)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()
        self._streaming_assistant_started = False
        self._streaming_block_start = None

    def _format_display_block(self, role: str, text: str) -> str:
        text = normalize_text_for_display(text)
        text = strip_unsupported_chars(text)
        return f'**{role}**\n\n{text}\n\n'

    def _insert_markdown_block(self, cursor: QTextCursor, role: str, text: str) -> None:
        block = self._format_display_block(role, text)
        try:
            cursor.insertFragment(QTextDocumentFragment.fromMarkdown(block))
            cursor.insertBlock()
        except Exception:
            cursor.insertText(block)

    def _insert_plain_block(self, cursor: QTextCursor, role: str, text: str) -> None:
        text = normalize_text_for_display(text)
        text = strip_unsupported_chars(text)
        cursor.insertText(f'[{role}]\n{text}\n\n')

    @staticmethod
    def _display_label_for_role(role: str) -> str:
        if role == 'user':
            return 'You'
        if role == 'assistant':
            return 'Assistant'
        if role == 'tool':
            return 'Tool'
        return 'System'

    @Slot()
    def _delete_current_chat(self) -> None:
        current_item = self.session_list.currentItem()
        if current_item is None:
            QMessageBox.information(self, 'Delete Session', 'Please select a session first.')
            return

        session_id = current_item.data(Qt.UserRole)
        if session_id is None:
            return
        session_id = int(session_id)
        if session_id in self._active_generations:
            QMessageBox.information(self, 'Delete Session', 'Stop this session before deleting it.')
            return
        if session_id in self._active_slash_tools:
            QMessageBox.information(self, 'Delete Session', 'Wait for the running slash tool before deleting it.')
            return

        reply = QMessageBox.question(
            self,
            'Delete Session',
            'Are you sure you want to delete this session?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.repository.delete_session(session_id)

        if self.current_session_id == session_id:
            self.current_session_id = None
            self.chat_view.clear()
            self._append_block('System', 'Select a session or click New Chat.')

        self._reload_sessions()
        self._refresh_controls()
        self._set_status('Session deleted')

class ChatGUI:
    def __init__(self, console: ChatSession, repository: ChatRepository, app_config: AppConfig) -> None:
        self.console = console
        self.repository = repository
        self.app_config = app_config


    def run(self) -> None:
        app = QApplication.instance() or QApplication([])
        app.setFont(QFont("Arial"))
        window = MainWindow(self.console, self.repository, self.app_config)
        window.show()
        app.exec()
