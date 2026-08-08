"""Rerankers, and the registry a config names one by.

A second stage exists because the two ranking problems are not the same one.
The first stage searches the whole corpus and must be cheap enough to do that;
the second sees ten candidates and can afford a model that reads the query and
the passage together. The grid measures whether that trade pays here.

Both implementations put their scores on ``[0, 1]`` — Cohere's arrive that way,
the cross-encoder's are sigmoid-squashed — so the two are comparable with each
other. Neither is comparable with the retriever score it replaces, which is why
:class:`~app.rag.models.RetrievalResult` carries ``original_rank``.
"""

from __future__ import annotations

from app.rag.base import BaseReranker
from app.rag.reranking.cohere import CohereReranker
from app.rag.reranking.cross_encoder import CrossEncoderReranker

RERANKERS: dict[str, type[BaseReranker]] = {
    CrossEncoderReranker.name: CrossEncoderReranker,
    CohereReranker.name: CohereReranker,
}

#: The local one: it needs no API key, so a clone can reproduce the numbers.
DEFAULT_RERANKER = CrossEncoderReranker.name


def get_reranker(name: str = DEFAULT_RERANKER, **kwargs) -> BaseReranker:
    """Build the reranker registered under ``name``.

    Construction never loads weights or checks credentials — both are deferred
    to first use, so naming a reranker in a config is free until it runs.
    """
    try:
        reranker_cls = RERANKERS[name]
    except KeyError:
        available = ", ".join(sorted(RERANKERS))
        raise ValueError(f"Unknown reranker {name!r}. Available: {available}") from None
    return reranker_cls(**kwargs)


def reranker_from_spec(spec: str) -> BaseReranker:
    """Build from ``name[:model]``, the form that survives a filename.

    ``cross_encoder:ms-marco-l12`` and ``cohere:rerank-v3.5`` — one positional
    argument, because a reranker's only real parameter is which model it is.
    """
    name, _, model = spec.partition(":")
    return get_reranker(name, model=model) if model else get_reranker(name)


__all__ = [
    "DEFAULT_RERANKER",
    "RERANKERS",
    "BaseReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    "get_reranker",
    "reranker_from_spec",
]
