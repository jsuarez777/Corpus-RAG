"""Embedding models, and the registry a config names one by.

``get_embedder("minilm")`` is the swap point. The two grid models are
registered under their own aliases rather than one entry taking a model
argument, so a config line reads ``embedder: mpnet`` and result files carry a
name that means something in a table.
"""

from app.rag.base import BaseEmbedder
from app.rag.embedding.sentence_transformer import (
    DEFAULT_MODEL,
    MODELS,
    SentenceTransformerEmbedder,
)

EMBEDDERS: dict[str, type[BaseEmbedder]] = {alias: SentenceTransformerEmbedder for alias in MODELS}

DEFAULT_EMBEDDER = DEFAULT_MODEL


def get_embedder(name: str = DEFAULT_EMBEDDER, **kwargs) -> BaseEmbedder:
    """Build the embedder registered under ``name``.

    Constructing one does not load the model — the weights arrive on first use.
    """
    try:
        embedder_cls = EMBEDDERS[name]
    except KeyError:
        available = ", ".join(sorted(EMBEDDERS))
        raise ValueError(f"Unknown embedder {name!r}. Available: {available}") from None
    return embedder_cls(name, **kwargs)


__all__ = [
    "DEFAULT_EMBEDDER",
    "EMBEDDERS",
    "MODELS",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
]
