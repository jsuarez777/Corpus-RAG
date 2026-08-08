"""Local cross-encoder reranking.

A bi-encoder embeds the query and the passage separately and compares two
vectors, which is what makes an index possible: passages are encoded once,
ahead of time, and a query is one dot product away from all of them. The price
is that the passage is encoded without ever having seen the query.

A cross-encoder concatenates the pair and runs them through the model together,
so every layer attends across the boundary. That is strictly more informative
and strictly more expensive — it cannot be precomputed, and the cost is one
forward pass per candidate rather than one per query. Which is exactly why it
belongs in a second stage over a handful of first-stage results.

The model emits a single logit per pair, on no fixed scale. It is squashed
through a sigmoid to ``[0, 1]``: the ordering is unchanged, but the numbers
become comparable with the Cohere reranker's, which is what lets the two be
plotted on one axis.
"""

from __future__ import annotations

import logging
import math
from functools import cached_property

from app.rag.base import BaseReranker
from app.rag.models import RetrievalResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Short names, so a config carries `ms-marco` rather than a HuggingFace path.
MODELS = {
    "ms-marco": DEFAULT_MODEL,
    "ms-marco-l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}


def sigmoid(value: float) -> float:
    """Squash a logit to ``[0, 1]``, saturating instead of overflowing.

    ``math.exp`` raises OverflowError below about -710, which a confident
    negative logit can reach. The clamp is far outside the range where the
    result is distinguishable from 0.0 or 1.0 anyway.
    """
    if value < -60.0:
        return 0.0
    if value > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


class CrossEncoderReranker(BaseReranker):
    """Rerank with a local sentence-transformers cross-encoder.

    ``model`` is an alias from :data:`MODELS` or any model id the library
    accepts. Weights load on first :meth:`rerank`, not on construction, so
    building one in a registry sweep or a ``--help`` path costs nothing.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.alias = model
        self.model_id = MODELS.get(model, model)
        self.batch_size = batch_size
        self.device = device

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        log.info(f"Loading {self.model_id} (first use downloads it)")
        return CrossEncoder(self.model_id, device=self.device)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Rescore every result against ``query`` and reorder by the new score.

        Results are copied rather than mutated: the caller's list is usually
        the first-stage output, and a baseline that silently became the
        reranked run would make the comparison meaningless.
        """
        if not results:
            return []

        pairs = [(query, result.chunk.content) for result in results]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        reranked = [
            result.model_copy(
                update={
                    "score": sigmoid(float(score)),
                    # Recorded on the way past, since `score` is about to stop
                    # being comparable with what produced this ordering.
                    "original_rank": result.original_rank or rank,
                }
            )
            for rank, (result, score) in enumerate(zip(results, scores, strict=True), start=1)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k] if top_k else reranked

    def __repr__(self) -> str:
        return f"CrossEncoderReranker({self.alias!r})"
