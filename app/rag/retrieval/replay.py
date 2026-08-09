"""A retriever that replays a search someone already ran.

The other three retrievers compute a ranking. This one reads it back: given the
rankings `app/retrieve.py` saved and the chunk set they were retrieved from, it
answers `retrieve()` from a dict.

That sounds like a cache, and the reason it is not is what it removes rather
than what it saves. A real retriever needs a faiss index and a torch embedder in
the process; `app/__init__.py` keeps those two from killing each other by
pinning `OMP_NUM_THREADS=1`, which only holds while nothing runs in parallel.
Replaying touches neither library, so a stage built on this one is free to use a
worker pool — which is what makes batch answer generation parallelizable at all.

Keyed by query text, not query id, because that is the only thing
:class:`BaseRetriever` is handed. Queries in this benchmark are unique; a
duplicate would take the ranking of whichever came last, and the stage that
builds the map logs when it collapses one.
"""

from __future__ import annotations

import logging

from app.rag.base import BaseRetriever
from app.rag.models import RetrievalResult, RetrieverType

log = logging.getLogger(__name__)


class ReplayRetriever(BaseRetriever):
    """Serves saved rankings, one list of results per query string."""

    name = "replay"

    def __init__(
        self,
        by_query: dict[str, list[RetrievalResult]],
        *,
        retriever_type: RetrieverType | str = RetrieverType.DENSE,
    ) -> None:
        self.by_query = by_query
        self._retriever_type = RetrieverType(retriever_type)

    @property
    def retriever_type(self) -> RetrieverType:
        """What produced the saved ranking, not what replayed it.

        A replayed hybrid result has to keep calling itself hybrid or every
        record downstream misattributes where its passages came from.
        """
        return self._retriever_type

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """The saved ranking for ``query``, truncated to ``top_k``.

        A query with no saved ranking returns nothing rather than raising: the
        caller asked a question this file has no answer to, which is the same
        situation as a retriever finding no matches, and the stages above
        already handle an empty result.
        """
        results = self.by_query.get(query.strip())
        if results is None:
            log.warning(f"No saved ranking for {query[:60]!r}")
            return []
        return results[:top_k]

    def __len__(self) -> int:
        return len(self.by_query)

    def __repr__(self) -> str:
        return f"ReplayRetriever({len(self.by_query)} queries, {self._retriever_type.value})"
