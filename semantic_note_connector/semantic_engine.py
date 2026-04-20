import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from .models import AppConfig, Connection, Note, SemanticConfig

CACHE_PATH = Path("embeddings_cache.json")


def prepare_embedding_text(note: Note) -> str:
    text = f"{note.title}\n\n{note.clean_body}"
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines())
    return normalized.strip()


def get_tokenizer(llm_cfg) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(llm_cfg.model)
    except Exception:
        try:
            return tiktoken.get_encoding(llm_cfg.fallback_encoding)
        except Exception as exc:  # pragma: no cover - tokenizer failure
            logging.error(
                "Failed to initialize tokenizer for model %s (fallback %s): %s",
                llm_cfg.model,
                llm_cfg.fallback_encoding,
                exc,
            )
            raise SystemExit(1)


def truncate_to_context(text: str, tokenizer: tiktoken.Encoding, max_tokens: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logging.debug("Truncating text from %s to %s tokens", len(tokens), max_tokens)
    return tokenizer.decode(tokens[:max_tokens])


def load_cache(path: Path = CACHE_PATH) -> Tuple[Dict[str, List[float]], Optional[str], Optional[int]]:
    if not path.exists():
        return {}, None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Embeddings cache is invalid JSON; starting fresh.")
        return {}, None, None
    except Exception as exc:  # pragma: no cover - IO error
        logging.warning("Could not read embeddings cache: %s", exc)
        return {}, None, None

    if not isinstance(data, dict) or "entries" not in data:
        logging.warning("Old cache format detected; starting with an empty cache.")
        return {}, None, None

    return (
        dict(data.get("entries", {})),
        data.get("model_name"),
        data.get("model_dim"),
    )


def save_cache(
    path: Path, model_name: str, model_dim: int, entries: Dict[str, List[float]]
) -> None:
    payload = {
        "model_name": model_name,
        "model_dim": model_dim,
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _embedding_client(cfg: AppConfig) -> OpenAI:
    return OpenAI(base_url=cfg.llm.url, api_key=cfg.llm.key)


def generate_embeddings(
    notes: List[Note], cfg: AppConfig
) -> Tuple[List[Note], np.ndarray, Dict[str, List[float]]]:
    tokenizer = get_tokenizer(cfg.llm)
    cache_entries, cache_model, cache_dim = load_cache(CACHE_PATH)

    if cache_model and cache_model != cfg.llm.model:
        logging.warning(
            "Cache model %s does not match requested model %s; discarding cache.",
            cache_model,
            cfg.llm.model,
        )
        cache_entries = {}
        cache_dim = None
    if cache_model is None or cache_dim is None:
        if cache_entries:
            logging.warning("Cache missing metadata; discarding cached entries.")
        cache_entries = {}
        cache_dim = None

    client = _embedding_client(cfg)
    embeddings: List[List[float]] = []
    successful_notes: List[Note] = []
    current_model_dim: Optional[int] = cache_dim

    for note in notes:
        text = prepare_embedding_text(note)
        truncated = truncate_to_context(text, tokenizer, cfg.llm.model_context)
        content_hash = hashlib.sha256(truncated.encode("utf-8")).hexdigest()

        embedding = cache_entries.get(content_hash)
        if embedding is not None:
            embeddings.append(embedding)
            successful_notes.append(note)
            current_model_dim = current_model_dim or len(embedding)
            continue

        retries = 3
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                response = client.embeddings.create(
                    model=cfg.llm.model,
                    input=truncated,
                    encoding_format="float",
                )
                embedding = response.data[0].embedding
                break
            except Exception as exc:  # pragma: no cover - network
                last_exc = exc
                logging.warning(
                    "Embedding attempt %s/%s failed for note %s: %s",
                    attempt + 1,
                    retries,
                    note.id,
                    exc,
                )
        else:
            logging.warning("Skipping note %s due to embedding errors: %s", note.id, last_exc)
            continue

        cache_entries[content_hash] = embedding
        embeddings.append(embedding)
        successful_notes.append(note)
        current_model_dim = current_model_dim or len(embedding)

    if not embeddings:
        logging.error("No embeddings generated; aborting run.")
        raise SystemExit(1)

    if current_model_dim is None:
        logging.error("Could not determine embedding dimension; aborting.")
        raise SystemExit(1)

    save_cache(CACHE_PATH, cfg.llm.model, current_model_dim, cache_entries)

    return successful_notes, np.array(embeddings, dtype=np.float32), cache_entries


def compute_connections(
    notes: List[Note], embeddings: np.ndarray, cfg: SemanticConfig
) -> List[Connection]:
    if len(notes) < 2 or embeddings.shape[0] < 2:
        return []

    similarity_matrix = cosine_similarity(embeddings)
    connections: List[Connection] = []

    for idx, note in enumerate(notes):
        row = similarity_matrix[idx]
        candidates: List[Tuple[int, float]] = []
        for j, score in enumerate(row):
            if j == idx:
                continue
            candidates.append((j, float(score)))

        filtered = [c for c in candidates if c[1] >= cfg.min_similarity]
        filtered.sort(
            key=lambda pair: (
                -pair[1],
                notes[pair[0]].title.lower(),
                notes[pair[0]].id,
            )
        )
        filtered = filtered[: cfg.top_n]
        if not filtered:
            continue

        connection = Connection(
            source_id=note.id,
            targets=[(notes[j].id, score) for j, score in filtered],
        )
        connections.append(connection)

    return connections
