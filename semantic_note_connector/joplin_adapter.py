import logging
import re
from datetime import datetime
from typing import Dict, Iterable, List, Protocol, Tuple

from .models import AppConfig, Connection, JoplinConfig, Note

SEMANTIC_BEGIN = "<!-- semantic-note-connector:begin -->"
SEMANTIC_END = "<!-- semantic-note-connector:end -->"
SEMANTIC_BLOCK_PATTERN = re.compile(
    rf"{re.escape(SEMANTIC_BEGIN)}.*?{re.escape(SEMANTIC_END)}",
    flags=re.DOTALL,
)


class JoplinClientProtocol(Protocol):
    """Subset of ClientApi methods used by the connector."""

    def get_all_notes(self, **kwargs):
        ...

    def get_all_tags(self, **kwargs):
        ...

    def get_note(self, id_: str, **kwargs):
        ...

    def modify_note(self, id_: str, **kwargs):
        ...


def _field(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def strip_semantic_block(body: str) -> str:
    if not body:
        return ""
    cleaned = SEMANTIC_BLOCK_PATTERN.sub("", body)
    return cleaned.rstrip()


def insert_semantic_block(clean_body: str, entries: List[str]) -> str:
    if not entries:
        return clean_body

    trimmed = clean_body.rstrip()
    if trimmed:
        trimmed += "\n\n"
    block = "\n".join(
        [
            SEMANTIC_BEGIN,
            "## Semantic Connections",
            "",
            *entries,
            SEMANTIC_END,
        ]
    )
    return f"{trimmed}{block}"


def create_client(joplin_cfg: JoplinConfig) -> JoplinClientProtocol:
    from joppy.client_api import ClientApi

    return ClientApi(token=joplin_cfg.token, url=joplin_cfg.base_url)


def check_health(api: JoplinClientProtocol, base_url: str) -> None:
    try:
        notebooks = []
        if hasattr(api, "get_all_notebooks"):
            notebooks = api.get_all_notebooks()  # type: ignore[attr-defined]
        else:
            notebooks = api.get_all_notes(fields="id")
        logging.debug(
            "Health check succeeded; notebooks/notes result count: %s",
            len(notebooks) if notebooks else 0,
        )
    except Exception as exc:  # pragma: no cover - network specific
        logging.error(
            "Joplin API unreachable at %s. Ensure Joplin is running and Data API is enabled. (%s)",
            base_url,
            exc,
        )
        raise SystemExit(1)


def _build_tag_map(api: JoplinClientProtocol) -> Dict[str, List[str]]:
    tag_map: Dict[str, List[str]] = {}
    try:
        tags = api.get_all_tags()
    except Exception as exc:  # pragma: no cover - remote fetch
        logging.warning("Could not fetch tags from Joplin: %s", exc)
        return tag_map

    for tag in tags or []:
        tag_id = _field(tag, "id")
        tag_title = str(_field(tag, "title", "")).lower()
        if not tag_id or not tag_title:
            continue
        try:
            notes_for_tag = api.get_all_notes(tag_id=tag_id)
        except Exception as exc:  # pragma: no cover - remote fetch
            logging.warning("Failed to list notes for tag %s: %s", tag_id, exc)
            continue
        for note in notes_for_tag or []:
            note_id = _field(note, "id")
            if not note_id:
                continue
            tag_map.setdefault(note_id, []).append(tag_title)
    return tag_map


def _eligible(note) -> bool:
    if _field(note, "is_conflict", False):
        return False
    if _field(note, "deleted_time", 0):
        return False
    if _field(note, "encryption_applied", 0):  # best-effort check
        return False
    if _field(note, "is_todo", False) and _field(note, "todo_completed", 0):
        return False
    return True


def _normalize_updated_time(value) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fetch_notes(api: JoplinClientProtocol, cfg: AppConfig) -> List[Note]:
    try:
        raw_notes = api.get_all_notes(
            fields="id,title,body,parent_id,is_todo,todo_completed,is_conflict,updated_time,deleted_time",
        )
    except Exception as exc:  # pragma: no cover - remote fetch
        logging.error("Failed to fetch notes from Joplin: %s", exc)
        raise SystemExit(1)

    tag_map = _build_tag_map(api)
    exclude = set(cfg.joplin.exclude_tags)

    eligible: List[Note] = []
    for raw in raw_notes or []:
        if not _eligible(raw):
            continue
        note_tags = [tag for tag in tag_map.get(_field(raw, "id"), [])]
        if any(tag in exclude for tag in note_tags):
            continue
        body = _field(raw, "body", "") or ""
        clean_body = strip_semantic_block(body)
        note = Note(
            id=str(_field(raw, "id")),
            title=str(_field(raw, "title", "")),
            body=body,
            clean_body=clean_body,
            notebook_id=str(_field(raw, "parent_id", "")),
            tags=note_tags,
            updated_time=_normalize_updated_time(_field(raw, "updated_time", 0)),
        )
        eligible.append(note)

    eligible.sort(key=lambda n: n.updated_time, reverse=True)
    if len(eligible) > cfg.semantic.max_notes:
        logging.warning(
            "Eligible note count %s exceeds max_notes %s; truncating to most recently updated notes.",
            len(eligible),
            cfg.semantic.max_notes,
        )
        eligible = eligible[: cfg.semantic.max_notes]

    logging.info(
        "Fetched %s notes (%s after filters, max_notes=%s).",
        len(raw_notes or []),
        len(eligible),
        cfg.semantic.max_notes,
    )
    return eligible


def update_note_connections(
    api: JoplinClientProtocol,
    note: Note,
    connections: List[Tuple[str, float]],
    id_to_title: Dict[str, str],
    cfg: AppConfig,
) -> None:
    if cfg.semantic.dry_run:
        logging.debug("Dry-run: skipping update for note %s", note.id)
        return

    try:
        latest = api.get_note(note.id, fields="id,updated_time")
        latest_updated_time = _normalize_updated_time(_field(latest, "updated_time", note.updated_time))
    except Exception as exc:  # pragma: no cover - remote fetch
        logging.warning("Could not re-fetch note %s for concurrency check: %s", note.id, exc)
        latest_updated_time = note.updated_time

    if latest_updated_time != note.updated_time:
        logging.warning("Note %s changed since fetch; skipping update.", note.id)
        return

    if not connections:
        desired_body = strip_semantic_block(note.body)
        if desired_body == note.body:
            logging.debug("No semantic block to remove for note %s", note.id)
            return
        try:
            api.modify_note(id_=note.id, body=desired_body)
            logging.info("Removed semantic block from note %s", note.id)
        except Exception as exc:  # pragma: no cover - remote write
            logging.error("Failed to update note %s: %s", note.id, exc)
        return

    entries = [
        f"{idx + 1}. [{id_to_title.get(target_id, target_id)}](:/{target_id})"
        for idx, (target_id, _score) in enumerate(connections)
    ]
    desired_body = insert_semantic_block(note.clean_body, entries)

    if desired_body == note.body:
        logging.debug("Note %s already has up-to-date semantic block", note.id)
        return

    try:
        api.modify_note(id_=note.id, body=desired_body)
        logging.info("Updated semantic block for note %s", note.id)
    except Exception as exc:  # pragma: no cover - remote write
        logging.error("Failed to update note %s: %s", note.id, exc)
