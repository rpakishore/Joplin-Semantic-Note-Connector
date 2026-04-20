## Project Overview

This repository contains **Semantic Note Connector**, a Python-based tool that builds a semantic graph of your Joplin notes. It:

- Connects to Joplin via the **Joplin Data API** using the **`joppy`** library.
- Generates **semantic embeddings** for notes using an **OpenAI-compatible embeddings API** (local or cloud) via the `openai` Python client.
- Computes **cosine similarity** between notes to identify related content.
- Writes **internal Joplin links** back into each note under a managed section so that link-graph plugins can visualize the “web of connections”.

The long-term goal is:

- Zero manual export of Markdown: everything comes directly from Joplin.
- Deterministic, automation-friendly runs (cron, CLI agents).
- A clean separation between:
  - Joplin I/O (`joppy` adapter).
  - Semantic engine (embeddings + similarity).
  - CLI / orchestration.

This file explains how to work on this project as a developer: tech stack, structure, coding principles, and project “constitution”.

---

## Tech Stack

- **Language:** Python 3.13+ (project targets modern Python; avoid legacy compatibility hacks).
- **Package / Project Manager:** [`uv`](https://docs.astral.sh/uv/), used for:
  - Managing the project via `pyproject.toml`.
  - Creating virtual environments (`uv venv`, `uv sync`, `uv run`).
  - Running scripts and tests.
- **Dependency Specification:** `pyproject.toml` only.
  - **Do NOT introduce** `requirements.txt`, `Pipfile`, `poetry.lock`, etc.
- **Core Python Dependencies (target state):**
  - `joppy` – Joplin Data API / Joplin Server API client.
  - `openai` – embeddings client for OpenAI-compatible APIs.
  - `tiktoken` – tokenization and context-window enforcement.
  - `numpy` – numerical arrays for embeddings.
  - `scikit-learn` – `cosine_similarity` implementation.
  - `pytest` – testing (once test suite is introduced).

---

## Project Constitution

These are non-negotiable conventions for this repository:

1. **`uv` is the only Python tooling entrypoint.**
   - All docs and automation must use `uv` commands, e.g.:
     - `uv run semantic-note-connector`
     - `uv run pytest`
     - `uv sync`
   - No direct `pip install ...` or `python -m venv ...` in docs or scripts.

2. **`pyproject.toml` is the single source of truth.**
   - Dependencies, scripts, and project metadata live there.
   - Do not add `requirements.txt`, `setup.py`, or `setup.cfg`.

3. **Joplin-centric design.**
   - The primary use case is Joplin notes via the Data API (`joppy`).
   - The legacy “scan a directory of Markdown files” mode is considered deprecated and will be removed or isolated.

4. **Deterministic, automation-friendly behavior.**
   - No interactive prompts in the core CLI.
   - All behavior is driven by configuration (`config.toml`) and optional CLI flags.
   - Runs should be idempotent: re-running with the same config and note set should produce the same results (modulo timestamps and external changes).

5. **Privacy-first defaults.**
   - Sample configs and docs should favor local embedding endpoints (e.g., Ollama) over cloud APIs.
   - Never log note bodies or large snippets.
   - API keys and tokens must never be hard-coded or committed.

6. **Single managed semantic section per note.**
   - Auto-generated content inside notes is fully owned by the tool and lives within special markers:
     - `<!-- semantic-note-connector:begin -->`
     - `<!-- semantic-note-connector:end -->`
   - Outside of these markers, the tool must not alter note content.

---

## File and Module Layout

- `pyproject.toml` – Project metadata, dependencies, and CLI entrypoints.
- `config.toml` – User configuration (LLM, Joplin, semantic behavior).
- `process.py` (or `semantic_note_connector/process.py`) – CLI entry module that:
  - Parses config and CLI args.
  - Orchestrates Joplin adapter and semantic engine.
  - Handles modes (`local-embed`, `adapter-only`).
- `semantic_note_connector/` (new package directory):
  - `models.py` – Dataclasses for notes, connections, and config.
  - `config.py` – Loading and validation for `config.toml` and env overrides.
  - `joplin_adapter.py` – All interaction with Joplin via `joppy`.
  - `semantic_engine.py` – Embedding + similarity logic.
  - `logging_utils.py` (optional) – Shared logger configuration.
- `AGENTS.md` – (this file) Developer onboarding and conventions.
- `Deployment.md` – How to run, deploy, or automate the tool using `uv`.
- `README.md` – User-facing documentation (installation, usage, motivation).
- `Implementation.md` – Temporary, implementation-specific design doc used during the migration; remove it once work is complete (current code no longer relies on it).

---

## Coding Principles

We explicitly follow **KISS** and **DRY**:

- **KISS (Keep It Simple, Stupid):**
  - Prefer straightforward, readable code over clever abstractions.
  - Avoid unnecessary inheritance and metaprogramming.
  - favor simple functions and dataclasses over deep object hierarchies.

- **DRY (Don’t Repeat Yourself):**
  - Centralize shared logic in helper functions or modules (e.g., note body cleaning, cache handling).
  - Don’t duplicate logic between `process.py` and `semantic_engine.py`.

Additional guidelines:

- **Type hints everywhere:**
  - All new functions and methods must include type hints for parameters and return types.
  - Favor `dataclasses` for structured data (`Note`, `Connection`, config types).

- **Small, focused functions:**
  - Each function should do “one thing well”.
  - Extraction is preferred when a function grows too large or gains multiple responsibilities.

- **No side effects in core computation modules:**
  - `semantic_engine` should be as pure as possible: given notes and config, return connections.
  - Side-effectful operations (network calls, file I/O, Joplin writes) belong in adapters or the CLI layer.

- **Logging over printing:**
  - Use Python’s `logging` module with levels (`INFO`, `DEBUG`, `WARNING`, `ERROR`).
  - Reserve `print` only for exceptional cases in scripts; ideally, not at all.

---

## Style Guidelines

- **Formatting:**
  - Follow PEP 8 conventions (indentation, naming, etc.).
  - Function and variable names: `snake_case`.
  - Class names and dataclasses: `PascalCase`.
  - Max line length: 88–100 characters (black-compatible).

- **Imports:**
  - Standard library first, then third-party, then local imports.
  - Avoid wildcard imports (`from x import *`).

- **Error Handling:**
  - Prefer explicit, contextual error messages:
    - Example: “Joplin API unreachable at {base_url}. Ensure Joplin is running and Data API is enabled.”
  - Fail fast on configuration errors (missing tokens, invalid URLs, etc.).
  - Treat remote failures (embedding errors, transient HTTP issues) as recoverable per-note; warn and skip rather than aborting the entire run.

---

## Semantic Graph Behavior (Conceptual)

High-level semantics the code must respect:

- Each eligible Joplin note is represented by a `Note` dataclass:
  - `id`, `title`, `body`, `clean_body`, `notebook_id`, `tags`, `updated_time`.
- Only **eligible** notes are considered:
  - Not deleted.
  - Not conflict notes.
  - Not encrypted.
  - To-dos are included only if not completed.
  - Notes with any tag listed in `joplin.exclude_tags` are excluded.
- Embedding input:
  - `clean_body` is the note’s body with the semantic block removed.
  - Embedding text: `f"{title}\n\n{clean_body}"`.
  - The text is normalized (newlines, trailing whitespace) before hashing/embedding.
- Embeddings:
  - Generated via the configured OpenAI-compatible API.
  - Caching is keyed by content hash of the embedding text.
  - Cache file includes `model_name`, `model_dim`, and an `entries` map.
  - Incompatible cache (missing metadata or mismatched model/dim) is discarded.
- Similarity:
  - Cosine similarity between all note embeddings (O(N²)).
  - For each note, pick up to `top_n` neighbors above `min_similarity`.
  - Neighbor list ordered by descending score, then title, then ID.

---

## Semantic Section in Notes

The tool maintains a **single, tool-owned** markdown block in each note:

```markdown
<!-- semantic-note-connector:begin -->
## Semantic Connections

1. [Some Related Note](:/NOTE_ID_1)
2. [Another Related Note](:/NOTE_ID_2)
<!-- semantic-note-connector:end -->
```

Rules:

- The block is **fully owned** by the tool:
  - On each run, it is either fully regenerated or removed.
  - Any manual edits inside the block will be overwritten.
- Notes with no neighbors above `min_similarity`:
  - Have no semantic block at all (existing blocks are removed).
- Only notes with at least one qualifying neighbor get a semantic block.
- Internal links must be standard Joplin internal links: `[Title](:/NOTE_ID)`.

---

## Configuration Overview

Configuration is read from `config.toml` (see `Deployment.md` / `README.md` for user details). Internally:

- Sections:
  - `[llm]` – Embedding endpoint, key, model, context, tokenizer fallback.
  - `[joplin]` – Data API base URL, token, exclusion tags.
  - `[semantic]` – Graph behavior (thresholds, modes, limits, dry-run).
- Environment variable overrides:
  - `LLM_API_KEY` overrides `[llm].key`.
  - `JOPLIN_TOKEN` overrides `[joplin].token`.

The order of precedence for config file path:

1. `--config PATH` CLI flag.
2. `SEMANTIC_CONNECTOR_CONFIG` environment variable.
3. `./config.toml` (project root).

If the chosen config file is missing or invalid, the program must exit with a clear error and non-zero status.

---

## CLI & Modes (Developer View)

The primary CLI entrypoint (once `pyproject.toml` is in place) will be:

- `semantic-note-connector` (exposed via `uv`):
  - `uv run semantic-note-connector` – standard run.
  - `uv run semantic-note-connector --mode adapter-only --dry-run` – adapter-only flow.

Modes:

- `local-embed` (default):
  - Fetch notes from Joplin.
  - Generate embeddings (using cache).
  - Compute similarities and update notes’ semantic blocks.
  - Optionally write `relevance_results.md`.

- `adapter-only`:
  - Fetch notes and write `notes_dump.json`.
  - If `connections_update.json` exists, apply those connections to notes.
  - No embedding or similarity computation.

Exit codes:

- `0` – success (even with some per-note warnings).
- `1` – configuration/startup errors.
- `2` – unexpected internal errors.

---

## Testing Guidelines

We aim for a minimal but meaningful test suite:

- Unit tests for:
  - `semantic_engine`: similarity and neighbor selection logic.
  - Functions that strip and regenerate the semantic block in note bodies.
- Joplin adapter:
  - Write against a fake `ClientApi` interface to avoid requiring a live Joplin instance.
- Tests should be runnable via:
  - `uv run pytest`

Tests and CI details will be further documented in `Deployment.md` once they exist.

---

## Pitfalls & Gotchas (For Contributors)

- **Never hash or embed text that includes the semantic block.**
  - Always strip the block before computing the content hash or sending text to the embedding API.
- **Do not mix embeddings across models or dimensions.**
  - If `model_name` or `model_dim` in the cache does not match the current model, discard the cache.
- **Be precise when editing note bodies.**
  - Only modify the content between `<!-- semantic-note-connector:begin -->` and `<!-- semantic-note-connector:end -->`.
- **Respect concurrency.**
  - When writing back a note, confirm its `updated_time` has not changed since initial fetch; skip updates otherwise.
- **Tokenizer fallback is required.**
  - `tiktoken.encoding_for_model` can fail for non-OpenAI models; always have a fallback encoding.
- **Max note limits must be deterministic.**
  - When applying `max_notes`, always select by `updated_time` (or the agreed rule), not by arbitrary order.
- **Adapter JSON schemas are a contract.**
  - Changes to `notes_dump.json` or `connections_update.json` formats must be coordinated and documented; do not break them silently.

---

## Implementation Docs Lifecycle

For major features (like the `joppy` integration and semantic-section rewrite), we use a temporary `Implementation.md`:

- Serves as the detailed blueprint for the feature.
- Contains step-by-step implementation instructions.
- Once the feature is implemented, tested, and merged:
  - Move user-facing behavior and options into `README.md`.
  - Move roadmap items into `README.md` or an issue tracker.
  - Delete `Implementation.md`.

Always consult `AGENTS.md` before editing code in this repo; update it if the project architecture or expectations change.
