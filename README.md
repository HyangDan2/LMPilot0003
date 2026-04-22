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
  rag_top_k: 5
  rag_min_score: 0.2
  openai_embedding_model: ""
  ```

  The prompt builder uses a recent-message window instead of reprocessing the entire session on every send. The database also includes a session summary table, and the codebase includes a vector-store foundation for future RAG-style retrieval.

* **Terminal PPTX rendering pipeline**

  The terminal pipeline scans a working folder recursively, parses supported documents, writes normalized JSON, creates a knowledge map, asks an OpenAI-compatible LLM for a strict JSON slide plan, and renders a deterministic PowerPoint file.

  Supported source files:

  ```text
  .pptx .docx .xlsx .pdf
  ```

  Install the document pipeline dependencies with:

  ```bash
  pip install -r requirements.txt
  ```

  Run the pipeline from a terminal:

  ```bash
  python -m src.main --working-dir data/working --output-dir data/outputs "Create a 7-slide executive summary"
  ```

  Generated files:

  ```text
  <normalized-dir>/<document-id>.json
  <normalized-dir>/knowledge_map.md
  <normalized-dir>/knowledge_map.json
  <normalized-dir>/planner_chunk_summary.json
  <normalized-dir>/planner_attempts.json
  <output-dir>/planner_output.json
  <output-dir>/<plan-title>.pptx
  ```

  For terminal usage, settings come from command-line overrides or environment variables such as `WORKING_DIR`, `NORMALIZED_DIR`, `OUTPUT_DIR`, `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. Planning is adaptive: the app summarizes the knowledge map in chunks, recursively splits failed chunks into smaller work, retries without JSON response-format enforcement when needed, and can write local fallback summaries for chunks that still fail. Tune `planner_chunk_chars`, `planner_min_chunk_chars`, `planner_max_retries`, `planner_intermediate_max_tokens`, `planner_final_max_tokens`, `planner_allow_response_format_retry`, and `planner_enable_local_fallback` for smaller local backends.

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
  <attached-folder>/llm_result/document_pipeline/
  ```

  Main report command:

  ```text
  /generate_report [--no-llm] [--fresh] [--generate-detail true|false] [--llm-input-chars N] [query...]
  ```

  Examples:

  ```text
  /summarize_file design_review.pptx
  /summarize_file test_results.xlsx summarize risks and quantitative results
  /generate_report summarize all output in this folder
  /generate_report summarize about project risks
  /generate_report --no-llm summarize briefly
  /summarize_file --generate-detail true design_review.pptx
  ```

  For quick inspection of one file, use:

  ```text
  /summarize_file <path> [--no-llm] [--generate-detail true|false] [--llm-input-chars N] [query...]
  ```

  `/summarize_file` extracts only the selected file and saves isolated artifacts under:

  ```text
  <attached-folder>/llm_result/document_pipeline/file_summaries/<document_id>/
  ```

  The generated single-file summary uses three top-level sections:

  ```text
  Summary
  Source Details
  Open Issues and Next Actions
  ```

  The `Summary` section may use the same detailed engineering subsections as `/generate_report` when evidence supports them.
  Saved `generated_summary.md` files also place each normal paragraph sentence on its own line.

  Use `--generate-detail true` with `/summarize_file` or `/generate_report` to generate optional LLM summaries for every page, slide, sheet, or file-level item. Detail summaries are batched for speed, but chat progress prints every current detail item as it is processed and completed. The summaries are saved as separate `detail_summaries.json` and `detail_summaries.md` artifacts and are not automatically added to the final report prompt. If no LLM client is configured, the app saves local extractive detail summaries with a fallback reason.

  `/generate_report` runs the complete engineering-report pipeline. You do not need to run `/extract_docs` or `/build_doc_map` first. The command scans the attached folder, reuses unchanged extraction artifacts when possible, then performs document mapping, output planning, representative evidence selection, optional local ranked evidence grouping for large documents, and one final LLM Markdown call. Use `--fresh` to force full re-extraction. Progress updates and per-stage timings stream into the chat while the tool runs; when the configured backend supports streaming, the final Markdown report streams into the Tool block while also being accumulated and saved as `generated_report.md`. The free-form text after the command becomes the report query/focus. If the configured LLM is unavailable, the command saves a deterministic fallback Markdown report instead of failing the whole pipeline.

  The generated engineering report uses three top-level sections:

  ```text
  Summary
  Source Documents
  Open Issues and Next Actions
  ```

  The `Summary` section is intentionally grounded in the selected evidence. When evidence supports it, `Summary` may include these subsections:

  ```text
  What the Document Explicitly Describes
  Main Methods or Components Explicitly Mentioned
  Quantitative Values Explicitly Present
  Explicit Limitations or Constraints
  Unclear or Not Specified in Selected Evidence
  ```

  Unsupported categories should be stated as not explicitly present in the selected evidence rather than filled with inferred architecture, risks, or recommendations.

  Large documents automatically use local ranked evidence grouping. Chat progress shows whether the run uses `one-shot` or `ranked-groups` mode, raw group count, selected top group count, representative evidence count, and final prompt size. The app does not call the LLM during grouping; the LLM is used only for the final report.

  Saved `generated_report.md` files place each normal paragraph sentence on its own line. Headings, tables, bullets, code fences, and blank lines are preserved.

  Slash tools run in background workers so the GUI stays responsive. A session can run only one slash tool at a time, but other sessions can run their own slash tools while `/generate_report` is active. Stop cancels the currently selected session's running slash tool or normal generation. Document-pipeline artifacts are still saved to shared paths under the attached folder, so simultaneous report commands against the same folder use last-writer-wins files.

  Generated document-pipeline artifacts:

  ```text
  extracted_documents.json
  extraction_manifest.json
  document_map.json
  output_plan.json
  selected_evidence.json
  evidence_groups.json
  selected_evidence_groups.json
  group_summaries.json
  recursive_summary_levels.json
  final_prompt_preview.txt
  detail_summaries.json
  detail_summaries.md
  llm_report_attempts.json
  generated_report.md
  file_summaries/<document_id>/detail_summaries.json
  file_summaries/<document_id>/detail_summaries.md
  file_summaries/<document_id>/generated_summary.md
  ```

  The LLM orchestration sends representative selected evidence and, for large documents, top-ranked evidence groups within `--llm-input-chars`. Lower `--llm-input-chars` for smaller local models.

  During normal chat, the model can request previously generated artifacts from the attached folder. The app safely executes these requests only under `<attached-folder>/llm_result/`, then sends the artifact content back to the model for a follow-up answer. Supported tags:

  ```text
  [read_output] document_pipeline/generated_report.md [/read_output]
  [list_outputs] document_pipeline [/list_outputs]
  ```

  Qwen-style generated-output aliases are also supported:

  ```text
  [read_file] llm/document_pipeline/generated_report.md [/read_file]
  [read_file] llm_result/document_pipeline/generated_report.md [/read_file]
  ```

  Paths are resolved inside the attached folder and cannot escape `llm_result/`.

  Other useful tools:

  ```text
  /help
  /workspace_status
  /summarize_file <path>
  /extract_docs
  /build_doc_map
  /generate_markdown
  ```

* **Markdown export**

  Use **Save Chat** next to **Clear View** to export the current session messages as Markdown. Use **Copy Last Output** to copy only the latest assistant response to the clipboard.

Create and modify local settings in:

```bash
cp config.example.yaml config.yaml
```

---

## 🧩 Architecture Overview

The code is consolidated under `src/`; the old top-level `app/` package has been removed. The main folders are:

```text
src/ingestion          deterministic file scanning and parsers
src/models             shared parser/planner dataclasses
src/utils              path, IO, and logging helpers
src/planner            adaptive PPTX planning pipeline
src/renderer           deterministic PPTX rendering
src/transform          knowledge-map construction
src/document_pipeline  extracted-document schema, evidence selection, and report generation
src/slash_tools        prompt-box local commands
src/gui                PySide6 GUI, sessions, database, and LLM client
src/tools              legacy local tools
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

The RAG groundwork is present in the codebase, but it is not yet fully wired into the normal chat flow:

* `OpenAICompatibleClient.embeddings(...)` calls `/v1/embeddings`
* `RagStore` saves chunk text, source metadata, and embedding vectors in SQLite
* cosine similarity search returns the top-k relevant chunks
* retrieved chunks can be formatted and passed into the prompt as memory context
* the live GUI prompt path currently uses recent messages plus an optional stored session summary
