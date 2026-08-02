"""Naming and locating ``data/indices/`` — the cache between embed and retrieve.

An index is identified by the pair that produced it, chunker *and* embedder,
because that pair is what the experiment grid varies. Two indices built from
the same chunks by different models are different indices, and a filename that
said only ``fixed_size_512_128`` would let one silently overwrite the other.
"""

from __future__ import annotations

from pathlib import Path

from app.rag.chunking.store import config_slug


def config_id(chunker_spec: str, embedder: str) -> str:
    """Directory name for one (chunker, embedder) pair.

    ``("fixed_size:512:128", "minilm")`` -> ``fixed_size_512_128__minilm``.
    """
    return f"{config_slug(chunker_spec)}__{config_slug(embedder)}"


def index_dir(base: Path, chunker_spec: str, embedder: str) -> Path:
    """Where the index for this pair lives under ``base``."""
    return Path(base) / config_id(chunker_spec, embedder)
