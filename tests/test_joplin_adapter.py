from semantic_note_connector.joplin_adapter import fetch_notes
from semantic_note_connector.models import AppConfig, JoplinConfig, LlmConfig, SemanticConfig


class FakeClient:
    def __init__(self, notes, tag_map):
        self._notes = notes
        self._tag_map = tag_map

    def get_all_notes(self, **kwargs):
        tag_id = kwargs.get("tag_id")
        if tag_id:
            note_ids = self._tag_map.get(tag_id, [])
            return [n for n in self._notes if n["id"] in note_ids]
        return self._notes

    def get_all_tags(self, **kwargs):
        return [
            {"id": tag_id, "title": title}
            for tag_id, title in {
                "t1": "Private",
                "t2": "Public",
            }.items()
        ]


def make_cfg(max_notes: int = 10) -> AppConfig:
    llm = LlmConfig(
        url="http://localhost",
        key="k",
        model="m",
        model_context=10,
        fallback_encoding="cl100k_base",
    )
    joplin = JoplinConfig(base_url="http://localhost", token="t", exclude_tags=["private"])
    semantic = SemanticConfig(
        top_n=1,
        min_similarity=0.1,
        generate_global_report=False,
        mode="local-embed",
        dry_run=True,
        max_notes=max_notes,
        notes_dump_path="notes.json",
        connections_update_path="conn.json",
        verbose=False,
    )
    return AppConfig(llm=llm, joplin=joplin, semantic=semantic)


def test_fetch_notes_filters_conflicts_and_tags():
    notes = [
        {"id": "1", "title": "Keep", "body": "b", "parent_id": "p", "updated_time": 5},
        {"id": "2", "title": "Conflict", "body": "b", "parent_id": "p", "is_conflict": True, "updated_time": 4},
        {"id": "3", "title": "Todo", "body": "b", "parent_id": "p", "is_todo": True, "todo_completed": 1, "updated_time": 3},
        {"id": "4", "title": "Tagged", "body": "b", "parent_id": "p", "updated_time": 2},
    ]
    tag_map = {"t1": ["4"]}
    client = FakeClient(notes, tag_map)
    cfg = make_cfg()

    result = fetch_notes(client, cfg)
    ids = [n.id for n in result]
    assert ids == ["1"]


def test_fetch_notes_respects_max_notes():
    notes = [
        {"id": "1", "title": "A", "body": "b", "parent_id": "p", "updated_time": 1},
        {"id": "2", "title": "B", "body": "b", "parent_id": "p", "updated_time": 3},
        {"id": "3", "title": "C", "body": "b", "parent_id": "p", "updated_time": 2},
    ]
    client = FakeClient(notes, {})
    cfg = make_cfg(max_notes=2)

    result = fetch_notes(client, cfg)
    ids = [n.id for n in result]
    assert ids == ["2", "3"]
