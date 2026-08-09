"""Reading and writing ``data/chunks/`` — the cache between chunk and embed.

One file per chunker config, holding every chunk for the corpus. The config is
the natural unit because that is what the experiment grid varies: embedding a
config's chunks is the next stage, and it should never have to re-chunk to find
out what it is embedding.

``semantic`` is the exception: its boundaries come out of an embedding model,
so the same spec yields different chunks per embedder. Those files carry the
embedder in the name (``semantic_512_90__minilm``) so the two cannot overwrite
each other. Every other strategy stays embedder-independent and keeps one file
shared by the whole grid.

Each write also stamps a fresh :func:`new_chunk_set_id`. ``Chunk.id`` is a
``uuid4`` minted at construction, so re-chunking renames every chunk in the
corpus while the filename stays put — and an index or a saved ranking built
from the previous generation goes on referring to ids that no longer exist.
The stamp travels into those artifacts so the mismatch is one string comparison
at startup rather than a silent join across two id spaces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from app.rag.models import Chunk

#: Key the stamp is stored under, in the chunk file and in everything derived
#: from it.
CHUNK_SET_KEY = "chunk_set_id"


def new_chunk_set_id() -> str:
    """A fresh id for one chunking run.

    Random rather than a hash of the content: two runs over an unchanged corpus
    still produce different ``Chunk.id`` values, so identical content is not
    the same chunk set for any purpose that stores ids.
    """
    return str(uuid4())


def chunk_set_id(path: Path) -> str:
    """The stamp on a chunk file, or "" for one written before stamping.

    An unstamped file cannot be shown to mismatch, so callers treat "" as
    "unknown" and let it pass rather than failing on every pre-existing file.
    """
    try:
        return str(json.loads(Path(path).read_text()).get(CHUNK_SET_KEY, ""))
    except (OSError, json.JSONDecodeError):
        return ""


def config_slug(spec: str, embedder: str | None = None) -> str:
    """Filename-safe id for a chunker spec: ``fixed_size:512:25%`` -> ``fixed_size_512_25pct``.

    ``embedder`` is appended only for chunkers that depend on one; passing it
    for the others would fragment a cache they are meant to share.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.replace("%", "pct")).strip("_")
    return f"{slug}__{embedder}" if embedder else slug


def save_chunks(
    chunks: list[Chunk],
    out_dir: Path,
    spec: str,
    embedder: str | None = None,
    chunk_set: str | None = None,
) -> Path:
    """Write ``chunks`` to ``<out_dir>/<config_slug(spec, embedder)>.json``.

    ``chunk_set`` defaults to a fresh id. Pass one only to re-write a file
    without declaring its chunks new — nothing in the pipeline does that today.
    """
    target = Path(out_dir) / f"{config_slug(spec, embedder)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Identifies this chunking run, so an index or ranking built from it can
        # say whether it is looking at the same ids.
        CHUNK_SET_KEY: chunk_set or new_chunk_set_id(),
        # The spec is kept verbatim alongside the slug: the slug is lossy
        # (25% and 25pct collide), and results tables want the real thing.
        "chunker": spec,
        # None for the embedder-independent strategies, which is the honest
        # answer: their chunks were not produced by any embedding model.
        "embedder": embedder,
        "num_chunks": len(chunks),
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def load_chunks(path: Path) -> list[Chunk]:
    """Read back the chunks written by :func:`save_chunks`."""
    payload = json.loads(Path(path).read_text())
    return [Chunk.model_validate(record) for record in payload["chunks"]]
