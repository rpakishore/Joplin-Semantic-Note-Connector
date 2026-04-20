import numpy as np

from semantic_note_connector.models import Note, SemanticConfig
from semantic_note_connector.semantic_engine import compute_connections, prepare_embedding_text


def _note(idx: str, title: str, body: str = "body") -> Note:
    return Note(
        id=idx,
        title=title,
        body=body,
        clean_body=body,
        notebook_id="n",
        tags=[],
        updated_time=0,
    )


def test_prepare_embedding_text_normalizes_newlines():
    note = _note("1", "Title", "line1\r\nline2\rline3  ")
    text = prepare_embedding_text(note)
    assert text == "Title\n\nline1\nline2\nline3"


def test_compute_connections_orders_by_score_then_title():
    notes = [
        _note("1", "Alpha"),
        _note("2", "Beta"),
        _note("3", "Gamma"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.1],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )
    cfg = SemanticConfig(
        top_n=2,
        min_similarity=0.1,
        generate_global_report=False,
        mode="local-embed",
        dry_run=True,
        max_notes=10,
        notes_dump_path="",
        connections_update_path="",
        verbose=False,
    )
    connections = compute_connections(notes, embeddings, cfg)
    first = next(c for c in connections if c.source_id == "1")
    assert first.targets[0][0] == "2"
    assert len(first.targets) == 2
