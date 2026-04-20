## Deployment & Operations Guide

This document explains how to **run**, **automate**, and **operate** Semantic Note Connector using the `uv` toolchain.

It is oriented toward:

- Local runs on a developer’s machine.
- Scripted or scheduled runs (cron, Task Scheduler, systemd).
- Integration with other tools or agents.

User-facing usage details are in `README.md`; this file focuses on governance and operational patterns.

---

## Prerequisites

### 1. Install `uv`

`uv` is the mandatory package and project manager for this repository.

Install on macOS / Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, ensure `uv` is on your `PATH`:

```sh
uv --version
```

### 2. Python

`uv` can manage Python versions itself. You can either:

- Use an existing system Python 3.13+, or
- Let `uv` manage one for you:

```sh
uv python install 3.13
uv python pin 3.13
```

---

## Project Environment Management

### Initialize / Sync the Environment

With `pyproject.toml` in place, the standard workflow is:

```sh
# From the project root
uv sync
```

This will:

- Create or update the project’s `.venv`.
- Install all dependencies declared in `pyproject.toml`.

You do **not** need to manually activate `.venv` to run project commands; `uv run` handles that.

---

## Configuration Management

Semantic Note Connector is configured via `config.toml` plus a few environment-variable overrides.

### Config File Location

Resolution order:

1. `--config PATH` flag (if provided).
2. `SEMANTIC_CONNECTOR_CONFIG` environment variable (if set).
3. `./config.toml` in the current working directory.

If the chosen file is missing or invalid, the CLI exits with a clear error and non-zero status.

### Config Sections (Overview)

The expected structure:

```toml
[llm]
url = "http://localhost:11434/v1"
key = "ollama"              # overridable by LLM_API_KEY
model = "nomic-embed-text"
model_context = 8192
fallback_encoding = "cl100k_base"

[joplin]
base_url = "http://127.0.0.1:41184"
token = "YOUR_JOPLIN_TOKEN" # overridable by JOPLIN_TOKEN
exclude_tags = ["no-semantic-links", "Private"]

[semantic]
top_n = 5
min_similarity = 0.6
generate_global_report = true
mode = "local-embed"        # or "adapter-only"
dry_run = false
max_notes = 5000
notes_dump_path = "notes_dump.json"
connections_update_path = "connections_update.json"
verbose = false
```

### Secret Handling

To avoid committing secrets:

- **Preferred:** Use environment variables:
  - `export LLM_API_KEY="..."` – overrides `[llm].key`.
  - `export JOPLIN_TOKEN="..."` – overrides `[joplin].token`.
- Keep `config.toml` **out** of version control or ensure it contains only placeholders.

---

## Running the Tool

The canonical commands are:

### Local-Embed Mode (Default)

Compute embeddings, similarities, and update Joplin notes:

```sh
# From project root, using ./config.toml
uv run semantic-note-connector

# With explicit config file
uv run semantic-note-connector --config /path/to/config.toml
```

Typical flow in `local-embed` mode:

1. Load `config.toml` and environment overrides.
2. Initialize logging (`INFO` by default, `DEBUG` when verbose).
3. Create a `joppy.client_api.ClientApi` using `[joplin]` section.
4. Perform a quick Joplin health check; exit with a clear error if unreachable.
5. Fetch and filter eligible notes from Joplin.
6. Load (or initialize) `embeddings_cache.json`.
7. Generate embeddings where needed; update cache.
8. Compute cosine similarity and determine top neighbors.
9. For each note:
   - Check `updated_time` for concurrent edits.
   - Regenerate or remove `Semantic Connections` block.
10. Optionally generate `relevance_results.md`.

### Adapter-Only Mode

Use the Joplin adapter without local embeddings:

```sh
uv run semantic-note-connector --mode adapter-only
```

Behavior:

- Writes `notes_dump.json` (location controlled by `[semantic].notes_dump_path`).
- If `connections_update.json` exists (matching the agreed schema), applies those connections to Joplin notes.
- Does **not** call any embedding or similarity APIs.

This mode is ideal for:

- External LLM agents that want to:
  - Read notes,
  - Compute connections themselves,
  - Hand the tool a `connections_update.json` file to apply.

### Dry-Run Mode

To preview changes without modifying Joplin:

```sh
uv run semantic-note-connector --dry-run
```

Effects in `local-embed` mode:

- Fetch, embed, and compute similarities as usual.
- Do **not** write any note updates back to Joplin.
- Still write `relevance_results.md` and optionally `planned_connections.json` (implementation detail).
- Exit with code `0` on success; logs must clearly state that no notes were modified.

---

## Logging and Exit Codes

### Logging

Default:

- Level: `INFO`.
- Single-line, human-readable log messages.

Verbose mode:

```sh
uv run semantic-note-connector --verbose
```

- Enables `DEBUG` logs:
  - Detailed cache stats (hit/miss).
  - Notes skipped (embedding failures, concurrency, filters).
  - Truncation details for long notes.

### Exit Codes

- `0` – Successful run:
  - Even if individual notes were skipped due to recoverable issues.
  - Errors are logged at `WARNING`/`ERROR` level but do not abort.
- `1` – Configuration or startup problem:
  - Missing/invalid config file.
  - Joplin unreachable or unauthorized.
  - LLM config invalid (e.g., missing model or URL).
- `2` – Unexpected internal error:
  - Unhandled exceptions; ideally rare.

Automations (cron, systemd, CI) should treat non-zero exit codes as failures.

---

## Automation Patterns

### Cron (Linux/macOS)

Example daily run at 03:00:

```cron
0 3 * * * cd /path/to/Joplin-Semantic-Note-Connector && \
  SEMANTIC_CONNECTOR_CONFIG=/path/to/config.toml \
  LLM_API_KEY=... \
  JOPLIN_TOKEN=... \
  uv run semantic-note-connector >> /var/log/semantic-note-connector.log 2>&1
```

Notes:

- Ensure Joplin Desktop (with Data API enabled) is running at that time.
- Store secrets securely (e.g., in a cron-specific environment file) rather than embedding them directly in crontab where possible.

### Windows Task Scheduler

Use a scheduled task that:

- Runs `powershell` with a command similar to:

```powershell
cd 'C:\path\to\Joplin-Semantic-Note-Connector'
$env:SEMANTIC_CONNECTOR_CONFIG = 'C:\path\to\config.toml'
$env:LLM_API_KEY = '...'
$env:JOPLIN_TOKEN = '...'
uv run semantic-note-connector
```

---

## Operational Safety & Privacy

- **Never log note bodies.**
  - Logs should reference note IDs and titles only.
- **Embedding provider choice matters.**
  - Defaults should favor local providers (e.g., Ollama).
  - If using cloud providers:
    - Document that note content leaves the machine.
    - Recommend reviewing provider’s privacy and data-retention policies.
- **Joplin token and LLM API key are secrets.**
  - Use environment variables or secret management systems.
  - Keep `config.toml` out of version control or filled with placeholders only.

---

## Testing and Verification

Once tests are in place, run them with:

```sh
uv run pytest
```

Operationally, after deploying a new version or changing config:

1. Run `uv sync` to ensure dependencies are in sync.
2. Run `uv run pytest` (if tests exist) to catch regressions.
3. Run `uv run semantic-note-connector --dry-run --verbose` to:
   - Check that Joplin connectivity and LLM config are valid.
   - Inspect logs for unexpected warnings or skips.

Only then enable or re-enable automation (cron, Task Scheduler, etc.).

---

## Maintenance & Upgrades

- **Changing embedding models:**
  - Update `[llm].model` and `[llm].model_context`.
  - On next run, the cache will detect mismatch and start fresh.
- **Changing note scope:**
  - Adjust `[joplin].exclude_tags`.
  - Optionally tune `[semantic].max_notes` as the collection size grows.
- **Scaling up:**
  - For larger collections, consider:
    - Increasing machine memory.
    - Reducing `max_notes`.
    - Lowering `top_n`.
- Future versions may introduce approximate nearest neighbor search; see README “Future Enhancements”.
