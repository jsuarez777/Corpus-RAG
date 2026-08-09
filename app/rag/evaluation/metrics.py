"""Retrieval metrics, computed over chunks against section-level ground truth.

**The ground truth is section-level, and the retrieval unit is the chunk.**
Every query has exactly one gold section; that section is covered by several
chunks. What the benchmark asserts is "the answer to this query is in section
23" — *not* "all four chunks of section 23 are relevant". The other chunks are
unlabelled, not judged relevant.

That distinction decides which numbers mean anything:

* :func:`hit_rate_at_k` is **the** retrieval metric here. Did any retrieved
  chunk come from the gold section? With one gold section per query, that *is*
  recall at the granularity the labels have. It is also the only metric that
  does not move merely because a chunker produces smaller pieces.

* :func:`section_coverage_at_k` is what a naive chunk-level "recall" computes,
  and it is deliberately not called recall. If the answer lives in one chunk
  and the retriever returns exactly that chunk at rank 1, coverage scores 0.25
  for a four-chunk section — penalizing a perfect result for not also fetching
  three chunks nobody said were relevant. It is reported because it answers a
  real question for the generation stage — how much of the gold section reached
  the context window — but it is not a measure of retrieval quality.

* :func:`precision_at_k` and :func:`ndcg_at_k` inherit the same caveat in
  milder form: a chunk from the gold section counts as a hit whether or not it
  contains the answer, so both are upper bounds on answer-bearing relevance.
  They are reported because the spec asks for them and they rank configs
  consistently, not because their absolute values are trustworthy.

* :func:`reciprocal_rank` is well behaved: the rank of the first gold-section
  chunk is exactly what MRR is meant to capture.

Gains are binary throughout: the benchmark labels a section relevant or not,
with no graded judgments to draw on.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import log2
from uuid import UUID

from app.rag.base import BaseRetriever
from app.rag.evaluation.qrels import QueryRelevance

log = logging.getLogger(__name__)

# Reported at several depths because they answer different questions: @1 is
# "would a single-chunk prompt work", @10 is "is it in the candidate pool a
# reranker could fix".
DEFAULT_KS = (1, 3, 5, 10)


def hit_rate_at_k(retrieved: Sequence[UUID], relevant: Iterable[UUID], k: int) -> float:
    """1.0 if any of the top ``k`` came from the gold section.

    With one gold section per query this is recall at the granularity the
    benchmark actually labels, which is why it is the headline metric rather
    than :func:`section_coverage_at_k`.
    """
    return float(bool(set(retrieved[:k]) & set(relevant)))


def precision_at_k(retrieved: Sequence[UUID], relevant: Iterable[UUID], k: int) -> float:
    """Fraction of the top ``k`` that are relevant.

    Divided by ``k``, not by the number returned: a retriever that returns two
    results and gets both right has not achieved precision 1.0 at k=5, it has
    left three slots empty.
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def section_coverage_at_k(retrieved: Sequence[UUID], relevant: Iterable[UUID], k: int) -> float:
    """Fraction of the gold section's chunks that reached the top ``k``.

    **Not recall.** The benchmark says the answer is in this section; it does
    not say every chunk of the section is relevant. A retriever that returns the
    one answer-bearing chunk at rank 1 scores 1/N here for an N-chunk section,
    which is a perfect result scored as a partial one.

    What it does measure is how much of the right section reached the context
    window — a genuine question for generation, where more of the section means
    a better chance the answer is present. Also bounded above by
    ``k / len(relevant)`` when the section has more chunks than there are slots.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[UUID], relevant: Iterable[UUID]) -> float:
    """1 / rank of the first relevant result, or 0 if none was retrieved."""
    relevant = set(relevant)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[UUID], relevant: Iterable[UUID], k: int) -> float:
    """Normalized discounted cumulative gain with binary gains.

    The ideal ranking places ``min(len(relevant), k)`` relevant items first, so
    a query with more relevant chunks than slots is not penalized for the slots
    it never had.
    """
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0
    gains = sum(
        1.0 / log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return gains / ideal if ideal else 0.0


def evaluate_query(
    retrieved: Sequence[UUID], relevant: Iterable[UUID], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    """Every metric for one query, keyed as ``metric@k``."""
    relevant = set(relevant)
    scores: dict[str, float] = {"mrr": reciprocal_rank(retrieved, relevant)}
    for k in ks:
        scores[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        scores[f"coverage@{k}"] = section_coverage_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return scores


@dataclass
class EvaluationResult:
    """Mean metrics over a query set, with the detail behind them."""

    num_queries: int = 0
    means: dict[str, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)
    mean_latency_ms: float = 0.0
    mean_relevant_chunks: float = 0.0

    def summary(self, ks: Sequence[int] = DEFAULT_KS) -> str:
        headline = " ".join(f"hit@{k} {self.means.get(f'hit_rate@{k}', 0):.3f}" for k in ks)
        return (
            f"{self.num_queries} queries | {headline} | "
            f"mrr {self.means.get('mrr', 0):.3f} | "
            f"ndcg@5 {self.means.get('ndcg@5', 0):.3f} | "
            f"{self.mean_latency_ms:.0f}ms/query | "
            f"{self.mean_relevant_chunks:.0f} relevant chunks/query"
        )


@dataclass
class Ranking:
    """What one query retrieved, and what it cost to retrieve it.

    The entire output of the retrieval half, and everything the scoring half
    needs from it. Ids rather than chunk text: the text is already in
    data/chunks/, and inlining it would multiply the file by ``top_k`` for
    nothing.
    """

    query_id: str
    retrieved: list[UUID] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    retriever_type: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "retrieved": [str(chunk_id) for chunk_id in self.retrieved],
            # Parallel to `retrieved` rather than paired with it, so the ids
            # stay a plain list — the charts and the qrels only ever want those.
            "scores": [round(score, 6) for score in self.scores],
            "retriever_type": self.retriever_type,
            "latency_ms": round(self.latency_ms, 3),
        }

    @classmethod
    def from_dict(cls, row: dict) -> Ranking:
        return cls(
            query_id=str(row["query_id"]),
            retrieved=[UUID(str(chunk_id)) for chunk_id in row.get("retrieved", ())],
            scores=[float(score) for score in row.get("scores", ())],
            retriever_type=row.get("retriever_type", ""),
            latency_ms=float(row.get("latency_ms", 0.0)),
        )


def retrieve_all(
    retriever: BaseRetriever,
    relevance: list[QueryRelevance],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    top_k: int | None = None,
) -> list[Ranking]:
    """Run every query through ``retriever``, keeping the ranked ids only.

    ``top_k`` defaults to the largest k being measured — retrieving fewer than
    that would cap the deepest metric at whatever was fetched.

    This is the only part of evaluation that touches an index, an embedder, or
    a model. Everything downstream — scoring, answering, judging — works from
    the ids this returns, which is what lets those stages run without loading
    faiss or torch and without re-running a search that has already happened.
    """
    depth = top_k or max(ks)
    rankings: list[Ranking] = []

    for item in relevance:
        started = time.perf_counter()
        results = retriever.retrieve(item.query, top_k=depth)
        rankings.append(
            Ranking(
                query_id=item.query_id,
                retrieved=[result.chunk.id for result in results],
                scores=[result.score for result in results],
                retriever_type=str(results[0].retriever_type.value) if results else "",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
    return rankings


def score_rankings(
    rankings: Sequence[Ranking],
    relevance: list[QueryRelevance],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    keep_per_query: bool = True,
) -> EvaluationResult:
    """Average every metric over rankings already produced, joining on query id.

    Pure arithmetic over ids: no retriever, no index, no model. A query present
    in ``relevance`` but missing from ``rankings`` is left out rather than
    scored as a miss — an absent ranking means it was never run, which is not
    the same as running and finding nothing.
    """
    by_id = {ranking.query_id: ranking for ranking in rankings}
    matched = [(item, by_id[item.query_id]) for item in relevance if item.query_id in by_id]
    if not matched:
        return EvaluationResult()

    missing = len(relevance) - len(matched)
    if missing:
        log.warning(f"{missing} of {len(relevance)} queries have no ranking — scoring the rest")

    totals: dict[str, list[float]] = {}
    per_query: list[dict] = []

    for item, ranking in matched:
        scores = evaluate_query(ranking.retrieved, item.relevant, ks)
        for name, value in scores.items():
            totals.setdefault(name, []).append(value)

        if keep_per_query:
            per_query.append(
                {
                    "query_id": item.query_id,
                    "query": item.query,
                    "doc_id": item.doc_id,
                    "section_id": item.section_id,
                    "num_relevant": len(item.relevant),
                    "num_retrieved": len(ranking.retrieved),
                    # Kept per query, not just averaged: the mean hides the
                    # shape, and the shape is the interesting part. Dense
                    # latency is tight around its median while hybrid carries a
                    # tail from BM25 scoring the whole corpus, and two configs
                    # can share a mean while one of them is occasionally slow.
                    "latency_ms": round(ranking.latency_ms, 3),
                    # The ranked ids, so a reranker — or the generation stage —
                    # can work from this run without repeating the retrieval.
                    "retrieved": [str(chunk_id) for chunk_id in ranking.retrieved],
                    "relevant": [str(chunk_id) for chunk_id in item.relevant],
                    **scores,
                }
            )

    return EvaluationResult(
        num_queries=len(matched),
        means={name: statistics.mean(values) for name, values in totals.items()},
        per_query=per_query,
        mean_latency_ms=statistics.mean(ranking.latency_ms for _, ranking in matched),
        mean_relevant_chunks=statistics.mean(len(item.relevant) for item, _ in matched),
    )


def evaluate(
    retriever: BaseRetriever,
    relevance: list[QueryRelevance],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    top_k: int | None = None,
    keep_per_query: bool = True,
) -> EvaluationResult:
    """Retrieve, then score — the two halves run back to back in one process.

    Kept as the convenient path for a terminal question. A grid run splits them
    so the expensive half writes its rankings once and every later stage reads
    them.
    """
    if not relevance:
        return EvaluationResult()

    rankings = retrieve_all(retriever, relevance, ks=ks, top_k=top_k)
    return score_rankings(rankings, relevance, ks=ks, keep_per_query=keep_per_query)
