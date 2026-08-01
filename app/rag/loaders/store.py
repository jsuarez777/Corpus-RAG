"""Reading and writing ``data/extracted/`` — the cache between load and chunk.

Extraction runs once per corpus; chunking runs once per chunk config, and the
12-config grid multiplies from there. Persisting Documents keeps the slow,
config-independent stage out of that multiplication (and pdfplumber extraction
is slow enough that re-running it per config would dominate the grid).
"""

import json
from pathlib import Path

from app.rag.models import Document


def document_path(out_dir: Path, document: Document) -> Path:
    """Where ``document`` is cached: ``<out_dir>/<paper_id>.json``."""
    stem = getattr(document.metadata, "paper_id", None) or Path(document.metadata.source).stem
    return Path(out_dir) / f"{stem}.json"


def save_document(document: Document, out_dir: Path) -> Path:
    """Write ``document`` as JSON under ``out_dir``, returning the file path."""
    target = document_path(out_dir, document)
    target.parent.mkdir(parents=True, exist_ok=True)
    # mode="json" so the UUID serializes as a string rather than as an object
    # repr; the file has to round-trip back through Document.model_validate.
    target.write_text(json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return target


def load_document(path: Path) -> Document:
    """Read back a Document written by :func:`save_document`."""
    return Document.model_validate(json.loads(Path(path).read_text()))


def load_documents(directory: Path) -> list[Document]:
    """Read every cached Document under ``directory``, sorted by filename."""
    return [load_document(path) for path in sorted(Path(directory).glob("*.json"))]
