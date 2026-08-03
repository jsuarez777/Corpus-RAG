"""Retrievers, and the registry a config names one by.

Unlike the loader and chunker registries, these do not share a constructor
signature — dense needs a store and an embedder, BM25 needs the chunks, hybrid
needs two retrievers. So ``RETRIEVERS`` maps names to classes for lookup and
validation, and callers construct with the arguments their retriever needs.
"""

from app.rag.base import BaseRetriever
from app.rag.retrieval._tokenize import STOPWORDS, tokenize
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.hybrid import DEFAULT_ALPHA, FUSIONS, HybridRetriever, normalize_scores

RETRIEVERS: dict[str, type[BaseRetriever]] = {
    DenseRetriever.name: DenseRetriever,
    BM25Retriever.name: BM25Retriever,
    HybridRetriever.name: HybridRetriever,
}

DEFAULT_RETRIEVER = HybridRetriever.name


def get_retriever(name: str = DEFAULT_RETRIEVER, *args, **kwargs) -> BaseRetriever:
    """Build the retriever registered under ``name``.

    Positional arguments are passed through, since what a retriever needs
    differs by kind: ``get_retriever("dense", store, embedder)``,
    ``get_retriever("bm25", chunks)``, ``get_retriever("hybrid", dense, sparse)``.
    """
    try:
        retriever_cls = RETRIEVERS[name]
    except KeyError:
        available = ", ".join(sorted(RETRIEVERS))
        raise ValueError(f"Unknown retriever {name!r}. Available: {available}") from None
    return retriever_cls(*args, **kwargs)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RETRIEVER",
    "FUSIONS",
    "RETRIEVERS",
    "STOPWORDS",
    "BM25Retriever",
    "BaseRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "get_retriever",
    "normalize_scores",
    "tokenize",
]
