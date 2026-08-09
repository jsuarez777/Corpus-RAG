"""Naming and locating ``data/indices/`` — the cache between embed and retrieve.

An index is identified by the pair that produced it, chunker *and* embedder,
because that pair is what the experiment grid varies. Two indices built from
the same chunks by different models are different indices, and a filename that
said only ``fixed_size_512_128`` would let one silently overwrite the other.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.rag.chunking.store import config_slug

META_FILE = "meta.json"


def config_id(chunker_spec: str, embedder: str) -> str:
    """Directory name for one (chunker, embedder) pair.

    ``("fixed_size:512:128", "minilm")`` -> ``fixed_size_512_128__minilm``.
    """
    return f"{config_slug(chunker_spec)}__{config_slug(embedder)}"


def index_dir(base: Path, chunker_spec: str, embedder: str) -> Path:
    """Where the index for this pair lives under ``base``."""
    return Path(base) / config_id(chunker_spec, embedder)


def require_same_chunk_set(chunk_path: Path, index_path: Path) -> None:
    """Stop unless an index and a chunk file describe the same chunk ids.

    Hybrid retrieval fuses dense results, whose ids come from the index, with
    BM25 results, whose ids come from the chunk file. Re-chunking renames every
    chunk, so a stale index makes those two id spaces disjoint — the fusion
    still runs, the metrics are still produced, and every one of them is wrong.
    Nothing else in the pipeline would notice.

    An unstamped artifact fails too. It cannot be shown to mismatch, but it
    cannot be shown to match either, and "unknown" is the state every artifact
    was in when the bug this prevents was possible. Rebuilding is cheap next to
    a grid of metrics that are quietly wrong.
    """
    from app.rag.chunking.store import chunk_set_id

    chunks, index = chunk_set_id(chunk_path), index_chunk_set_id(index_path)
    if not chunks:
        raise SystemExit(
            f"{Path(chunk_path).name} carries no chunk_set_id — it predates the check. "
            f"Re-run `python app/chunk.py`, then `python app/align.py` and "
            f"`python app/index.py --force`."
        )
    if not index:
        raise SystemExit(
            f"{Path(index_path).name} carries no chunk_set_id — it predates the check "
            f"and cannot be shown to hold the chunks now on disk. "
            f"Re-run `python app/index.py --force`."
        )
    if chunks != index:
        raise SystemExit(
            f"{Path(index_path).name} was built from chunk set {index[:8]}, but "
            f"{Path(chunk_path).name} is now {chunks[:8]} — the chunks were rebuilt "
            f"after the index. Re-run `python app/index.py --force`."
        )


def index_chunk_set_id(path: Path) -> str:
    """Which chunking run an index was built from, read from its manifest.

    Reads the manifest alone rather than loading the index, so a caller can
    check a 46MB directory for free. "" means the index predates the stamp,
    which cannot be shown to mismatch and so is allowed through.
    """
    try:
        return str(json.loads((Path(path) / META_FILE).read_text()).get("chunk_set_id", ""))
    except (OSError, json.JSONDecodeError):
        return ""
