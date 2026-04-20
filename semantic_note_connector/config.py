import logging
import os
import sys
from pathlib import Path
from typing import Optional

import tomllib

from .models import AppConfig, JoplinConfig, LlmConfig, SemanticConfig


class ConfigError(Exception):
    """Raised when configuration is invalid."""


def resolve_config_path(argv_override: Optional[str]) -> Path:
    """Resolve configuration path following CLI/env precedence."""
    if argv_override:
        path = Path(argv_override).expanduser()
    elif os.environ.get("SEMANTIC_CONNECTOR_CONFIG"):
        path = Path(os.environ["SEMANTIC_CONNECTOR_CONFIG"]).expanduser()
    else:
        path = Path("config.toml")

    if not path.exists():
        logging.error("Config file not found at %s", path)
        sys.exit(1)

    return path


def _require_keys(section: str, data: dict, keys: list[str]) -> None:
    missing = [key for key in keys if data.get(key) in (None, "")]
    if missing:
        raise ConfigError(
            f"Missing required keys in [{section}]: {', '.join(sorted(missing))}"
        )


def load_config(path: Path) -> AppConfig:
    """Load and validate configuration from TOML."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logging.error("Config file %s not found", path)
        sys.exit(1)
    except tomllib.TOMLDecodeError as exc:
        logging.error("Config file %s is not valid TOML: %s", path, exc)
        sys.exit(1)

    try:
        llm_data = dict(raw.get("llm", {}))
        joplin_data = dict(raw.get("joplin", {}))
        semantic_data = dict(raw.get("semantic", {}))
    except AttributeError:
        logging.error("Config file %s must contain [llm], [joplin], and [semantic]", path)
        sys.exit(1)

    llm_data["key"] = os.environ.get("LLM_API_KEY", llm_data.get("key"))
    joplin_data["token"] = os.environ.get("JOPLIN_TOKEN", joplin_data.get("token"))

    try:
        _require_keys("llm", llm_data, ["url", "model", "model_context", "key"])
        _require_keys("joplin", joplin_data, ["base_url", "token"])
        _require_keys(
            "semantic",
            semantic_data,
            ["top_n", "min_similarity", "mode", "max_notes"],
        )
    except ConfigError as exc:
        logging.error(str(exc))
        sys.exit(1)

    mode = str(semantic_data.get("mode", "local-embed")).lower()
    if mode not in {"local-embed", "adapter-only"}:
        logging.error("Invalid semantic.mode '%s'. Use 'local-embed' or 'adapter-only'.", mode)
        sys.exit(1)

    exclude_tags = [str(tag).lower() for tag in joplin_data.get("exclude_tags", [])]

    llm_cfg = LlmConfig(
        url=str(llm_data["url"]),
        key=str(llm_data["key"]),
        model=str(llm_data["model"]),
        model_context=int(llm_data["model_context"]),
        fallback_encoding=str(llm_data.get("fallback_encoding", "cl100k_base")),
    )

    joplin_cfg = JoplinConfig(
        base_url=str(joplin_data["base_url"]),
        token=str(joplin_data["token"]),
        exclude_tags=exclude_tags,
    )

    semantic_cfg = SemanticConfig(
        top_n=int(semantic_data["top_n"]),
        min_similarity=float(semantic_data["min_similarity"]),
        generate_global_report=bool(semantic_data.get("generate_global_report", True)),
        mode=mode,
        dry_run=bool(semantic_data.get("dry_run", False)),
        max_notes=int(semantic_data.get("max_notes", 5000)),
        notes_dump_path=str(semantic_data.get("notes_dump_path", "notes_dump.json")),
        connections_update_path=str(
            semantic_data.get("connections_update_path", "connections_update.json")
        ),
        verbose=bool(semantic_data.get("verbose", False)),
    )

    return AppConfig(llm=llm_cfg, joplin=joplin_cfg, semantic=semantic_cfg)
