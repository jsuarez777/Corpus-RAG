"""Reading and writing ``data/chunks/`` — the cache between chunk and embed.

One file per chunker config, holding every chunk for the corpus. The config is
the natural unit because that is what the experiment grid varies: embedding a
config's chunks is the next stage, and it should never have to re-chunk to find
out what it is embedding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.rag.models import Chunk


def config_slug(spec: str) -> str:
    """Filename-safe id for a chunker spec: ``fixed_size:512:25%`` -> ``fixed_size_512_25pct``."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.replace("%", "pct")).strip("_")


def save_chunks(chunks: list[Chunk], out_dir: Path, spec: str) -> Path:
    """Write ``chunks`` to ``<out_dir>/<config_slug(spec)>.json``."""
    target = Path(out_dir) / f"{config_slug(spec)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # The spec is kept verbatim alongside the slug: the slug is lossy
        # (25% and 25pct collide), and results tables want the real thing.
        "chunker": spec,
        "num_chunks": len(chunks),
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def load_chunks(path: Path) -> list[Chunk]:
    """Read back the chunks written by :func:`save_chunks`."""
    payload = json.loads(Path(path).read_text())
    return [Chunk.model_validate(record) for record in payload["chunks"]]
