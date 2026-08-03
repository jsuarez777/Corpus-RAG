"""Dense retrieval — embed the query, ask the vector store for its neighbours.

Deliberately thin. All the substance lives in the embedder (what the vectors
mean) and the store (how they are compared); this class exists so the pipeline
can name a retriever in a config and swap dense for BM25 or hybrid without any
caller changing.
"""

from __future__ import annotations

import logging

from app.rag.base import BaseEmbedder, BaseRetriever, BaseVectorStore
from app.rag.models import RetrievalResult, RetrieverType

log = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Nearest neighbours of the query vector, by cosine similarity."""

    name = "dense"

    def __init__(self, store: BaseVectorStore, embedder: BaseEmbedder) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.DENSE

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the ``top_k`` chunks nearest the query, best first."""
        if not query.strip():
            return []
        # embed_query rather than embed_texts: models with asymmetric
        # query/passage prefixes encode the two differently, and this is the
        # hook where that happens.
        return self._store.search(self._embedder.embed_query(query), top_k=top_k)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._embedder!r}, {len(self._store)} chunks)"
