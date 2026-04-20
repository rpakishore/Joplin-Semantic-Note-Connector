import os
from pathlib import Path

from semantic_note_connector.config import load_config


CONFIG_BODY = """
[llm]
url = "http://localhost:11434/v1"
key = "placeholder"
model = "test-model"
model_context = 1024
fallback_encoding = "cl100k_base"

[joplin]
base_url = "http://127.0.0.1:41184"
token = "token"
exclude_tags = ["Private", "no-semantic-links"]

[semantic]
top_n = 3
min_similarity = 0.2
generate_global_report = false
mode = "local-embed"
dry_run = true
max_notes = 100
notes_dump_path = "notes_dump.json"
connections_update_path = "connections_update.json"
verbose = false
"""


def test_load_config_applies_env_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONFIG_BODY, encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("JOPLIN_TOKEN", "env-joplin-token")

    cfg = load_config(config_path)

    assert cfg.llm.key == "env-llm-key"
    assert cfg.joplin.token == "env-joplin-token"
    assert cfg.semantic.mode == "local-embed"
    assert cfg.joplin.exclude_tags == ["private", "no-semantic-links"]
