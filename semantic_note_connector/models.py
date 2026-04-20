from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Note:
    id: str
    title: str
    body: str
    clean_body: str
    notebook_id: str
    tags: List[str]
    updated_time: int


@dataclass
class Connection:
    source_id: str
    targets: List[Tuple[str, float]]


@dataclass
class LlmConfig:
    url: str
    key: str
    model: str
    model_context: int
    fallback_encoding: str


@dataclass
class JoplinConfig:
    base_url: str
    token: str
    exclude_tags: List[str]


@dataclass
class SemanticConfig:
    top_n: int
    min_similarity: float
    generate_global_report: bool
    mode: str
    dry_run: bool
    max_notes: int
    notes_dump_path: str
    connections_update_path: str
    verbose: bool


@dataclass
class AppConfig:
    llm: LlmConfig
    joplin: JoplinConfig
    semantic: SemanticConfig
