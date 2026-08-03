"""Hybrid retrieval — fusing the dense and sparse rankings into one.

The two retrievers fail on different queries, so the union of their candidates
is a better pool than either. The difficulty is combining the scores: cosine
similarity lives in [-1, 1] and clusters tightly around 0.3 on this corpus,
while BM25 is unbounded positive and depends on corpus statistics. Adding them
directly would let BM25 dominate every result purely through scale.

Two fusion rules are provided, because they fail differently:

**weighted** normalizes each retriever's scores to [0, 1] across the candidate
pool, then takes ``alpha * dense + (1 - alpha) * sparse``. This is what the
alpha sweep varies. Its weakness is that min-max normalization is relative to
whatever came back: if every candidate is mediocre, the least mediocre still
scores 1.0, so the fused score says nothing about absolute quality.

**rrf** (reciprocal rank fusion) ignores scores entirely and combines
``1 / (RRF_K + rank)``. Immune to scale, robust when one retriever's scores are
badly distributed, and it cannot express "dense was much better here" — only
"dense ranked it higher".

Candidates are pooled well beyond ``top_k`` before fusing. A chunk ranked 30th
by dense and 2nd by BM25 should surface, and it cannot if each list was already
truncated to 5.
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.rag.base import BaseRetriever
from app.rag.models import RetrievalResult, RetrieverType

log = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.5

# How many candidates to pull from each retriever, as a multiple of top_k.
# Four is enough for a chunk to climb from either list's tail; larger pools cost
# only the fusion arithmetic, since both retrievers score the whole corpus anyway.
DEFAULT_POOL = 4
MIN_POOL = 20

# The constant in reciprocal rank fusion. 60 is the value from the original
# paper and is not tuned here — it flattens the difference between the top few
# ranks, so a chunk ranked 1st and one ranked 3rd contribute comparably.
RRF_K = 60

FUSIONS = ("weighted", "rrf")


def normalize_scores(results: list[RetrievalResult]) -> dict[UUID, float]:
    """Min-max each result set onto [0, 1], keyed by chunk id.

    When every score is identical — one result, or a perfect tie — the spread
    is zero and there is no information to scale. Everything gets 1.0 rather
    than a division by zero.
    """
    if not results:
        return {}
    scores = [result.score for result in results]
    low, high = min(scores), max(scores)
    if high == low:
        return {result.chunk.id: 1.0 for result in results}
    return {result.chunk.id: (result.score - low) / (high - low) for result in results}


class HybridRetriever(BaseRetriever):
    """Weighted or rank-based fusion of a dense and a sparse retriever.

    ``alpha`` is the dense share: 1.0 is dense only, 0.0 is sparse only, and the
    sweep in the experiment grid walks the space between. It applies to
    ``weighted`` fusion only — ``rrf`` weights the two equally by construction.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: BaseRetriever,
        sparse: BaseRetriever,
        *,
        alpha: float = DEFAULT_ALPHA,
        fusion: str = "weighted",
        pool: int = DEFAULT_POOL,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {', '.join(FUSIONS)}, got {fusion!r}")
        self._dense = dense
        self._sparse = sparse
        self.alpha = alpha
        self.fusion = fusion
        self.pool = pool

    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.HYBRID

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Fuse both retrievers' candidates and return the best ``top_k``."""
        if top_k <= 0:
            return []
        depth = max(top_k * self.pool, MIN_POOL)
        dense = self._dense.retrieve(query, top_k=depth)
        sparse = self._sparse.retrieve(query, top_k=depth)
        if not dense and not sparse:
            return []

        fused = (
            self._fuse_by_rank(dense, sparse)
            if self.fusion == "rrf"
            else self._fuse_by_score(dense, sparse)
        )

        # Keep the chunk object from whichever list held it; both are the same
        # chunk, but only one may carry an embedding.
        by_id = {result.chunk.id: result.chunk for result in sparse + dense}
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(
                chunk=by_id[chunk_id],
                score=float(score),
                retriever_type=RetrieverType.HYBRID,
            )
            # A fused score of zero means neither retriever gave the chunk any
            # weight — at alpha=0 that is every dense-only candidate. Keeping
            # them would pad the list with chunks BM25 explicitly rejected, and
            # would stop alpha=0 degenerating to pure sparse retrieval.
            for chunk_id, score in ranked[:top_k]
            if score > 0
        ]

    def _fuse_by_score(
        self, dense: list[RetrievalResult], sparse: list[RetrievalResult]
    ) -> dict[UUID, float]:
        """alpha-weighted sum of separately normalized scores."""
        dense_scores = normalize_scores(dense)
        sparse_scores = normalize_scores(sparse)
        # A chunk missing from one list scores 0 there, not "unknown". It was
        # offered the chance to rank and did not take it.
        return {
            chunk_id: self.alpha * dense_scores.get(chunk_id, 0.0)
            + (1 - self.alpha) * sparse_scores.get(chunk_id, 0.0)
            for chunk_id in dense_scores.keys() | sparse_scores.keys()
        }

    def _fuse_by_rank(
        self, dense: list[RetrievalResult], sparse: list[RetrievalResult]
    ) -> dict[UUID, float]:
        """Reciprocal rank fusion — scale-free, uses only the ordering."""
        fused: dict[UUID, float] = {}
        for results in (dense, sparse):
            for rank, result in enumerate(results, start=1):
                fused[result.chunk.id] = fused.get(result.chunk.id, 0.0) + 1.0 / (RRF_K + rank)
        return fused

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self.alpha}, fusion={self.fusion!r})"
