"""Backward-compatible entrypoint for Semantic Note Connector."""
from semantic_note_connector.process import main

if __name__ == "__main__":
    raise SystemExit(main())
