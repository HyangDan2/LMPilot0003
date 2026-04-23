# Generic LLM Communicator(for OpenAI Compatible model)

A lightweight desktop GUI for interacting with a local `llama-server` or
for connecting to frontier models such as ChatGPT, Gemini, and Claude by using an API key.

Designed for every environment with stability-focused output handling.
(Please change pexpect to wexpect when you use this application under the windows environment.)

---

## ✨ Features

* 🖥️ PySide6 desktop GUI (clean chat interface)
* 🔁 Local `llama-server` HTTP backend
  * Tries `/v1/chat/completions` first
  * Falls back to `/completion` with Gemma turn markers when chat completions are unavailable
* 💾 SQLite-based local chat history
* 🧹 ANSI / help banner / control sequence cleanup
* 🔤 Unicode normalization (`/uXXXX`, `\\uXXXX`) support
* 🚫 Emoji & unsupported glyph filtering (prevents rendering issues on Pi)
* ⚡ Multi-threaded inference (UI never freezes)
* 🧵 Per-session generation workers

  * One session can generate while you switch to another session and send another prompt.
  * The selected session is the only one whose Send button is disabled while that session is running.
  * The backend/server decides whether parallel requests run concurrently, queue, or fail.
* ⌨️ **Keyboard shortcut**

  * `Ctrl + Enter` → Send message
  * `Ctrl + Up` → Recall the most recent input. Up to 10 recent inputs are remembered.
* 🗂️ Session management

  * Create new session
  * Delete selected chat
* 📎 Folder attachments

  * Double-click an attached file to insert its filename into the input box
* 💾 Markdown export and quick copy

  * Use **Save Chat** to save the current session as a `.md` file
  * Use **Copy Last Output** to copy the latest assistant response to the clipboard

---

## 🧠 Behavior Changes (Important)

* ❌ **No auto session creation on startup**

  * User must manually click **"New Chat"**
* 🧭 Startup message:

  ```
  Select a session or click New Chat.
  ```
* 🧩 Prompt handling is history-aware:

  * Recent session messages are included in the model request.
  * Older context can be carried through a stored session summary.
  * Long context is trimmed from oldest turns first.
  * Multiline pasted text, code, logs, JSON, and YAML are preserved as much as practical.
* 🛑 Stop behavior:

  * The Stop button interrupts the current request and returns the UI to an idle state.
  * The next send should work without restarting the app.
  * If multiple sessions are generating or running slash tools, Stop applies only to the currently selected session.

---

## 🗂️ Session Management

* **New Chat**

  * Creates a new session in SQLite
* **Delete Chat**

  * Deletes:

    * chat session
    * all associated messages
* UI auto-refresh after deletion

---

## 🚀 Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
# edit config.yaml for your local server/model settings
# keep server_endpoint: "auto" unless you intentionally force an endpoint
llama-server -m /path/to/your/model.gguf --host 127.0.0.1 --port 8080
python run.py --config config.yaml
```

`config.yaml` is intentionally ignored by git. Keep machine-specific paths and local server settings there. The checked-in `config.example.yaml` is the template.

---

## ⚙️ Configuration

* **llama-server URL**

  ```
  http://127.0.0.1:8080
  ```

* **llama-server endpoint**

  ```
  auto
  ```

  `auto` is recommended. The app tries `/v1/chat/completions` first, then falls back to `/completion`.

  The app also accepts `/auto` and treats it the same as `auto`, but `auto` is clearer in config files.

* **Model (GGUF)**

  ```
  /path/to/your/gemma-3-1b-it-Q4_K_M.gguf
  ```

* **System prompt**

  ```yaml
  system_prompt: "You are a helpful assistant."
  ```

  In server mode, this is sent as the first structured system message when `/v1/chat/completions` is used. In `/completion` fallback mode, it is included in the Gemma-template prompt.

* **Response length**

  ```yaml
  n_predict: 512
  response_token_reserve: 256
  max_prompt_chars: 12000
  recent_message_limit: 40
  memory_context_char_limit: 4000
  ```

  If answers are too short, increase `n_predict`. If long pasted prompts are being trimmed too aggressively, increase `ctx_size` and `max_prompt_chars` in line with your model/server capacity.

* **Conversation memory**

  ```yaml
  recent_message_limit: 40
  rag_top_k: 0
  rag_min_score: 0.2
  openai_embedding_model: ""
  ```

  The prompt builder uses a recent-message window instead of reprocessing the entire session on every send. The database also includes a session summary table. RAG-style attachment retrieval is disabled by default; set `rag_top_k` to a positive value to enable it.

* **Attachments**

  Use **Attach Folder** in the left sidebar under **Sessions** to select a workspace folder. The attachment list shows supported files directly inside that selected folder, including `.pptx`; subfolders are not scanned. Selecting a new folder replaces the previous attachment list. Files are not automatically included in prompts. Double-click an attached file to insert its filename into the input box. Use **Clear Attachments** to clear the full list.

  Supported file types:

  ```text
  .txt .md .json .yaml .yml .csv .py .log .pdf .docx .png .jpg .jpeg .bmp .webp
  ```

  Plain text files are read as UTF-8 first, with fallback decoding for common local encodings. PDF extraction uses `pypdf`; DOCX extraction uses `python-docx`. Images use `Pillow` for metadata and a local heuristic caption, with OCR attempted through `pytesseract` or the native `tesseract` command when available. Unsupported files in the selected folder are skipped; unreadable supported files show a GUI warning.

* **Document pipeline slash tools**

  The prompt box supports local slash tools for the attached folder. Slash tools run before normal chat generation and save artifacts under:

  ```text
  <attached-folder>/HD2docpipe/artifacts/
  ```

  Summary outputs generated by `/summarize_doc` are saved under:

  ```text
  <attached-folder>/HD2docpipe/summaries/<run-name>/
  ```

  Run names include the source and timestamp so repeated summaries stay separated.
  Examples:

  ```text
  workspace_20260422_231501
  design_review_20260422_231530
  ```

  Examples:

  ```text
  /extract_single_doc design_review.pptx
  /extract_docs
  /build_doc_map
  /summarize_doc
  /summarize_doc --engineering True design_review.pptx
  /summarize_docs
  /generate_markdown
  ```

  Available document-pipeline tools:

  ```text
  /extract_single_doc <path>
  /extract_docs
  /build_doc_map
  /summarize_doc [--engineering True|False] [path]
  /summarize_docs [--engineering True|False]
  /generate_markdown
  /workspace_status
  ```

  `/extract_single_doc` extracts one supported file and saves a per-document JSON artifact.
  `/extract_docs` scans the attached folder and saves extracted document JSON plus a manifest.
  `/build_doc_map` builds a structural map from the latest extracted documents.
  `/summarize_doc` creates hierarchical LLM-backed summaries from extracted documents with bounded chunking and workspace-level synthesis. The default final workspace summary is organized into `Overall Summary`, `Features` (exactly 3 items), and `Next Action`. The prompt asks for a substantial minimum level of detail rather than relying on very large output-token caps. Use `--engineering True` to output `Features`, `Quantitative Information`, and `Recommended Action`. If a path is provided, it summarizes only that file from the attached folder.
  `/summarize_docs` runs the same single-file summary flow sequentially for every supported document in the attached folder. Each processable document gets its own summary run under `HD2docpipe/summaries/`.
  Reasoning-model backends that return `reasoning` or `reasoning_content` without final assistant content are handled automatically: the UI reports `Assistant: Reasoning...`, keeps reasoning visible, and continues follow-up requests (up to the configured limit) until final answer content is returned.
  `/generate_markdown` writes a deterministic markdown report from extracted evidence without any final LLM report-writing stage.

  Slash tools run in background workers so the GUI stays responsive. A session can run only one slash tool at a time, and Stop cancels the currently selected session's running slash tool or normal generation.

  Generated document-pipeline artifacts:

  ```text
  extracted_documents.json
  extraction_manifest.json
  document_map.json
  generated_report.md
  documents/<document_id>.json
  ```

  When a command is scoped to a single file, related pipeline artifacts are stored under a file-specific folder:

  ```text
  <attached-folder>/HD2docpipe/artifacts/<file-scope>/
  ```

  Examples:

  ```text
  HD2docpipe/artifacts/design_review_pptx/extracted_documents.json
  HD2docpipe/artifacts/design_review_pptx/document_map.json
  HD2docpipe/artifacts/design_review_pptx/generated_report.md
  ```

  Summary output artifacts:

  ```text
  <run-name>/document_summaries.json
  <run-name>/workspace_summary.md
  ```

  `workspace_summary.md` is rendered with these sections:

  ```text
  ## Overall Summary
  ## Features
  ## Next Action
  ```

  `document_summaries.json` also stores the final workspace summary in structured form:

  ```json
  {
    "workspace_summary": {
      "mode": "standard",
      "overall_summary": "...",
      "features": ["...", "...", "..."],
      "next_action": "..."
    }
  }
  ```

  Engineering mode stores this shape:

  ```json
  {
    "summary_mode": "engineering",
    "workspace_summary": {
      "mode": "engineering",
      "features": ["...", "...", "..."],
      "quantitative_information": "...",
      "recommended_action": "..."
    }
  }
  ```

  During normal chat, the model can request previously generated artifacts from the attached folder. The app safely executes these requests only under `<attached-folder>/HD2docpipe/`, then sends the artifact content back to the model for a follow-up answer. Supported tags:

  ```text
  [read_output] HD2docpipe/artifacts/generated_report.md [/read_output]
  [list_outputs] HD2docpipe/artifacts [/list_outputs]
  ```

  Generated-output reads can also be requested directly:

  ```text
  [read_file] HD2docpipe/artifacts/generated_report.md [/read_file]
  [read_file] HD2docpipe/summaries/workspace_20260422_231501/workspace_summary.md [/read_file]
  ```

  Paths are resolved inside the attached folder and cannot escape `HD2docpipe/`.

  Other useful tools:

  ```text
  /help
  /workspace_status
  /extract_single_doc <path>
  /extract_docs
  /build_doc_map
  /summarize_doc
  /summarize_docs
  /generate_markdown
  ```

* **Markdown export**

  Use **Save Chat** next to **Clear View** to export the current session messages as Markdown. Use **Copy Last Output** to copy only the latest assistant response to the clipboard.

Create and modify local settings in:

```bash
cp config.example.yaml config.yaml
```

The app stores the most recently used attached folder in `last_working_folder`. If that folder still exists the next time the app starts, it is restored automatically. If it no longer exists, the setting is treated as empty.

---

## 🧩 Architecture Overview

The code is consolidated under `src/`; the old top-level `app/` package has been removed. The main folders are:

```text
src/ingestion          deterministic file scanning and parsers
src/models             shared extraction dataclasses
src/utils              path, IO, and logging helpers
src/document_pipeline  extracted-document schema, deterministic markdown, and storage helpers
src/slash_tools        prompt-box local commands
src/gui                PySide6 GUI, sessions, database, and LLM client
```

```
GUI (PySide6)
   │
   ├── QThread (ChatWorker)
   │       │
   │       └── LlamaServerSession (HTTP)
   │              ├── /v1/chat/completions (preferred)
   │              └── /completion (Gemma-template fallback)
   │
   ├── SQLite (ChatRepository)
   │      ├── recent message window
   │      └── session summary memory
   │
   ├── RAG Foundation
   │      ├── OpenAI-compatible /v1/embeddings
   │      └── cosine similarity search
   │
  ├── Attachments
  │      └── folder file listing and filename insertion
  │
   └── Text Processing
          ├── ANSI strip
          ├── Unicode normalize
          └── Glyph filtering
```

---

## ⚠️ Notes (Raspberry Pi)

* Emoji may break rendering → filtered by design
* Ensure `llama-server` runs independently before GUI
* If output looks corrupted:

  * Check locale (`UTF-8`)
  * Verify terminal encoding

---

## 🔧 Troubleshooting

### 1. Model not loading

* Check your local `config.yaml` paths.
* Verify the `.gguf` file exists.
* `config.example.yaml` is only a template; copy it to `config.yaml` and edit it before running.

---

### 2. llama-server error

```bash
curl http://127.0.0.1:8080/health
```

→ confirm the server is reachable

If you see an error like:

```text
llama-server returned HTTP 404 from /auto
```

Update to the latest branch and set:

```yaml
server_endpoint: "auto"
```

The current code treats both `auto` and `/auto` as automatic endpoint selection, but `auto` is the recommended spelling.

---

### 3. No response / stuck

* Check:

  * `llama-server` is running and reachable
  * The GUI is using the latest code on the intended branch
  * The local `config.yaml` points to the same host/port where `llama-server` is listening

---

### 4. Output continues as fake dialogue

If a simple prompt such as `Hey!` produces a long scripted conversation with extra `[You]` / `[Gemma]` turns, the app is likely using raw completion behavior without chat formatting.

Current server behavior:

* Prefer `/v1/chat/completions` with structured messages.
* If unavailable, fall back to `/completion` using Gemma turn markers:

  ```text
  <start_of_turn>user
  ...
  <end_of_turn>
  <start_of_turn>model
  ```

* In fallback mode, stop sequences are used to prevent the model from generating the next user turn.

Start a new chat after updating if an older session already contains runaway generated dialogue; old history can influence the next answer.

---

### 5. Output is too short

Short output can be normal if the model gives a concise answer, but check these settings:

* `n_predict`: maximum generated tokens. Increase it for longer answers.
* `server_endpoint`: keep it as `auto` so chat completions are preferred.

`/summarize_doc` uses its own internal summary budget and final output structure. The token budget is a safety cap, not a guaranteed output length. Summary length is primarily controlled by the workspace prompt instructions in `src/document_pipeline/high_level/summarize_doc.py`, which specify substantial minimum paragraph, sentence, bullet, and action counts for standard and engineering summaries. The rendered `workspace_summary.md` also formats prose so each sentence starts on its own line, including multi-sentence list items.

For reasoning-model backends such as Qwen variants, the OpenAI-compatible client automatically detects `reasoning` / `reasoning_content` responses that contain no final assistant content. It shows `Assistant: Reasoning...` with streamed reasoning text and continues follow-up requests until final answer content appears or the follow-up limit is reached (`reasoning_followup_max_attempts`, default `10`). Follow-up reasoning text is display-oriented and bounded by `reasoning_display_max_chars`.

The app does not send fallback stop sequences to `/v1/chat/completions`; those stop sequences are only used for raw `/completion` fallback to prevent runaway dialogue.

---

### 6. Broken characters (e.g. `\ufffd`)

Handled internally by:

* unicode normalization
* unsupported char filtering

---

## 📌 Future Improvements (Optional)

* Streaming token output (real-time typing)
* Multi-model selection
* GPU offload tuning (Metal / Vulkan / OpenCL)
* Chat export (markdown / txt)

---

## OpenAI-Compatible Backend Settings

The app now uses an OpenAI-compatible Chat Completions backend by default. Start the GUI, then fill in the settings panel at the top of the main window:

* **Base URL**: your OpenAI-compatible `/v1` base URL, for example `http://127.0.0.1:8000/v1` or `http://localhost:1234/v1`
* **API Key**: optional for local servers; masked in the GUI and saved only in the local ignored settings file
* **Model Name**: the exact model ID exposed by your backend
* **Embedding Model**: optional model ID for `/v1/embeddings`; if blank, code can fall back to the chat model for embedding calls

Use **Save Settings** to persist those values to `openai_settings.json`. That file is ignored by git because it may contain secrets. Use **Test Connection** to check `/models` first; if model listing is unavailable, the app falls back to a minimal chat completion test when a model name is set.

The checked-in `config.example.yaml` contains non-secret defaults. Runtime connection values should be entered in the GUI or kept in your local, ignored `openai_settings.json`.

## Memory and RAG Foundation

Long chats no longer require every saved turn to be normalized and trimmed on each send. The GUI reads only `recent_message_limit` prior messages for the live prompt and can include a stored session summary as a compact memory block.

The RAG groundwork is present in the codebase and can be enabled by setting `rag_top_k` to a positive value:

* `OpenAICompatibleClient.embeddings(...)` calls `/v1/embeddings`
* `RagStore` saves chunk text, source metadata, and embedding vectors in SQLite
* cosine similarity search returns the top-k relevant chunks
* retrieved chunks can be formatted and passed into the prompt as memory context
* the live GUI prompt path currently uses recent messages plus an optional stored session summary
