from semantic_note_connector.joplin_adapter import (
    SEMANTIC_BEGIN,
    SEMANTIC_END,
    insert_semantic_block,
    strip_semantic_block,
)


def test_strip_no_block():
    body = "Hello world"\
        "\nSecond line"
    assert strip_semantic_block(body) == body


def test_strip_single_block():
    body = (
        "Intro text\n"
        f"{SEMANTIC_BEGIN}\n## Semantic Connections\n\n1. [A](:/a)\n{SEMANTIC_END}\n"
        "Tail text\n"
    )
    expected = "Intro text\nTail text"
    assert strip_semantic_block(body) == expected


def test_strip_multiple_blocks():
    body = (
        f"{SEMANTIC_BEGIN}ignore{SEMANTIC_END}\nContent\n"
        f"{SEMANTIC_BEGIN}more{SEMANTIC_END}"
    )
    assert strip_semantic_block(body) == "Content"


def test_insert_semantic_block():
    clean = "Body"
    entries = ["1. [Note](:/id)", "2. [Second](:/id2)"]
    result = insert_semantic_block(clean, entries)
    assert SEMANTIC_BEGIN in result and SEMANTIC_END in result
    assert result.startswith("Body\n\n")
    assert "## Semantic Connections" in result
