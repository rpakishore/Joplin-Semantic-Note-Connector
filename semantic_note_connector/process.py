import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config, resolve_config_path
from .joplin_adapter import (
    create_client,
    check_health,
    fetch_notes,
    update_note_connections,
)
from .logging_utils import setup_logging
from .models import AppConfig, Connection, Note
from .semantic_engine import compute_connections, generate_embeddings


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic Note Connector")
    parser.add_argument("--config", dest="config_path", help="Path to config file")
    parser.add_argument("--mode", choices=["local-embed", "adapter-only"], help="Override semantic mode")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes to Joplin")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--top-n", type=int, dest="top_n", help="Override top_n neighbors")
    parser.add_argument("--min-similarity", type=float, dest="min_similarity", help="Override similarity threshold")
    return parser.parse_args(argv)


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.mode:
        cfg.semantic.mode = args.mode
    if args.dry_run:
        cfg.semantic.dry_run = True
    if args.verbose:
        cfg.semantic.verbose = True
    if args.top_n is not None:
        cfg.semantic.top_n = args.top_n
    if args.min_similarity is not None:
        cfg.semantic.min_similarity = args.min_similarity
    return cfg


def write_connections_file(path: Path, connections: List[Connection]) -> None:
    payload = {
        "connections": [
            {
                "source_id": conn.source_id,
                "targets": [
                    {"id": target_id, "score": score} for target_id, score in conn.targets
                ],
            }
            for conn in connections
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote planned connections to %s", path)


def write_relevance_report(path: Path, notes: List[Note], connections: List[Connection], cfg: AppConfig) -> None:
    id_to_note: Dict[str, Note] = {note.id: note for note in notes}
    lines: List[str] = []
    lines.append("# Semantic Connections Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(f"Model: {cfg.llm.model}\n")
    lines.append(f"Top N: {cfg.semantic.top_n}, Min Similarity: {cfg.semantic.min_similarity}\n")
    lines.append("---\n")

    connection_map: Dict[str, List[tuple[str, float]]] = {}
    for conn in connections:
        connection_map[conn.source_id] = conn.targets

    for note in notes:
        targets = connection_map.get(note.id, [])
        lines.append(f"## {note.title}\n")
        lines.append(f"- Note ID: {note.id}\n")
        if not targets:
            lines.append("- No related notes above threshold.\n\n")
            continue
        lines.append("- Related notes:\n")
        for target_id, score in targets:
            target = id_to_note.get(target_id)
            title = target.title if target else target_id
            lines.append(f"  - [{title}](:/{target_id}) — {score:.4f}\n")
        lines.append("\n")

    path.write_text("".join(lines), encoding="utf-8")
    logging.info("Wrote relevance report to %s", path)


def write_notes_dump(path: Path, notes: List[Note]) -> None:
    payload = {
        "notes": [
            {
                "id": note.id,
                "title": note.title,
                "body": note.body,
                "clean_body": note.clean_body,
                "notebook_id": note.notebook_id,
                "tags": note.tags,
            }
            for note in notes
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote notes dump to %s", path)


def load_connections_update(path: Path) -> List[Connection]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.error("connections_update.json is not valid JSON: %s", exc)
        raise SystemExit(1)

    connections: List[Connection] = []
    for item in data.get("connections", []):
        source_id = item.get("source_id") or item.get("id") or item.get("note_id")
        targets_raw = item.get("targets", [])
        if not source_id:
            logging.warning("Skipping connection with missing source_id: %s", item)
            continue
        targets: List[tuple[str, float]] = []
        for target in targets_raw:
            target_id = target.get("id") or target.get("note_id")
            score = target.get("score")
            if not target_id or score is None:
                continue
            targets.append((target_id, float(score)))
        connections.append(Connection(source_id=source_id, targets=targets))
    return connections


def run_local_embed(cfg: AppConfig) -> int:
    setup_logging(cfg.semantic.verbose)
    client = create_client(cfg.joplin)
    check_health(client, cfg.joplin.base_url)

    notes = fetch_notes(client, cfg)
    if len(notes) < 2:
        logging.info("Not enough notes to compute similarities (found %s).", len(notes))
        return 0

    embedded_notes, embeddings, _cache = generate_embeddings(notes, cfg)
    if len(embedded_notes) < 2:
        logging.info("Not enough embeddings to proceed (generated %s).", len(embedded_notes))
        return 0

    connections = compute_connections(embedded_notes, embeddings, cfg.semantic)

    id_to_title = {note.id: note.title for note in embedded_notes}
    connection_map = {conn.source_id: conn.targets for conn in connections}

    if cfg.semantic.dry_run:
        write_connections_file(Path("planned_connections.json"), connections)
        if cfg.semantic.generate_global_report:
            write_relevance_report(Path("relevance_results.md"), embedded_notes, connections, cfg)
        return 0

    for note in embedded_notes:
        targets = connection_map.get(note.id, [])
        update_note_connections(client, note, targets, id_to_title, cfg)

    if cfg.semantic.generate_global_report:
        write_relevance_report(Path("relevance_results.md"), embedded_notes, connections, cfg)

    return 0


def run_adapter_only(cfg: AppConfig) -> int:
    setup_logging(cfg.semantic.verbose)
    client = create_client(cfg.joplin)
    check_health(client, cfg.joplin.base_url)

    notes = fetch_notes(client, cfg)
    write_notes_dump(Path(cfg.semantic.notes_dump_path), notes)

    update_path = Path(cfg.semantic.connections_update_path)
    if not update_path.exists():
        logging.info("No connections_update file found at %s; nothing to apply.", update_path)
        return 0

    connections = load_connections_update(update_path)
    id_to_title = {note.id: note.title for note in notes}
    note_map = {note.id: note for note in notes}

    for connection in connections:
        note = note_map.get(connection.source_id)
        if not note:
            logging.warning("Skipping connection for unknown note %s", connection.source_id)
            continue
        update_note_connections(client, note, connection.targets, id_to_title, cfg)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config_path(args.config_path)
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, args)

    if cfg.semantic.mode == "adapter-only":
        return run_adapter_only(cfg)

    return run_local_embed(cfg)


if __name__ == "__main__":
    sys.exit(main())
