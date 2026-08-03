"""Tests for retrieval metrics and qrels resolution.

Two things here are easy to get wrong and expensive to get wrong quietly:

* **Unscoreable queries must be excluded, not scored zero.** A query whose gold
  section produced no chunks is a gap in the corpus, not a retriever failure,
  and counting it would make every metric move when the working set changed.
* **NDCG's ideal ranking is capped at k.** Without that, a query with 200
  relevant chunks can never reach 1.0 at k=5 however perfect the ranking, and
  the chunker that makes smaller pieces looks worse for arithmetic reasons.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.rag.evaluation import (
    QueryRelevance,
    build_relevance,
    evaluate,
    evaluate_query,
    hit_rate_at_k,
    index_chunks_by_section,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    section_coverage_at_k,
)
from app.rag.models import Chunk, ChunkMetadata, RetrievalResult, RetrieverType

A, B, C, D, E = (uuid4() for _ in range(5))


def make_chunk(paper: str, sections: list[int], index: int = 0) -> Chunk:
    return Chunk(
        content=f"chunk {index} of {paper}",
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source=f"{paper}.pdf",
            start_char=index * 100,
            end_char=index * 100 + 50,
            chunk_index=index,
            section_indices=sections,
        ),
    )


class TestHitRate:
    def test_one_when_any_hit(self) -> None:
        assert hit_rate_at_k([A, B, C], {C}, 3) == 1.0

    def test_zero_when_none(self) -> None:
        assert hit_rate_at_k([A, B], {C}, 2) == 0.0

    def test_respects_the_cutoff(self) -> None:
        assert hit_rate_at_k([A, B, C], {C}, 2) == 0.0

    def test_is_insensitive_to_how_many_are_relevant(self) -> None:
        """Why it is the headline: a chunker producing 200 relevant pieces
        rather than 3 does not get a different number for the same behaviour."""
        assert hit_rate_at_k([A], {A, B, C, D, E}, 5) == hit_rate_at_k([A], {A}, 5)


class TestPrecision:
    def test_counts_against_k_not_against_what_was_returned(self) -> None:
        """Two results, both right, is not precision 1.0 at k=5 — three slots
        were left empty."""
        assert precision_at_k([A, B], {A, B}, 5) == pytest.approx(0.4)

    def test_all_relevant(self) -> None:
        assert precision_at_k([A, B], {A, B}, 2) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k([A, B], {C}, 2) == 0.0

    def test_non_positive_k(self) -> None:
        assert precision_at_k([A], {A}, 0) == 0.0


class TestSectionCoverage:
    def test_fraction_of_the_section_retrieved(self) -> None:
        assert section_coverage_at_k([A, B], {A, B, C, D}, 5) == 0.5

    def test_a_perfect_answer_scores_partial(self) -> None:
        """Why this is not called recall. The benchmark says the answer is in
        this section, not that all four of its chunks are relevant — so
        retrieving the answer-bearing chunk at rank 1 is a perfect result, and
        coverage still reports 0.25. hit_rate is the metric that gets it right.
        """
        assert section_coverage_at_k([A], {A, B, C, D}, 5) == 0.25
        assert hit_rate_at_k([A], {A, B, C, D}, 5) == 1.0

    def test_is_capped_by_k_over_section_size(self) -> None:
        """With more chunks in the section than there are slots, coverage
        cannot reach 1.0 however good the ranking is."""
        assert section_coverage_at_k([A, B, C, D, E], {A, B, C, D, E, uuid4()}, 5) < 1.0

    def test_no_relevant_chunks(self) -> None:
        assert section_coverage_at_k([A], set(), 5) == 0.0


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank([A, B], {A}) == 1.0

    def test_third_position(self) -> None:
        assert reciprocal_rank([A, B, C], {C}) == pytest.approx(1 / 3)

    def test_uses_the_first_hit_only(self) -> None:
        assert reciprocal_rank([A, B, C], {B, C}) == 0.5

    def test_nothing_relevant(self) -> None:
        assert reciprocal_rank([A, B], {C}) == 0.0


class TestNDCG:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k([A, B], {A, B}, 5) == pytest.approx(1.0)

    def test_ideal_is_capped_at_k(self) -> None:
        """Five perfect hits out of 200 relevant is still a perfect top-5."""
        many = {A, B, C, D, E} | {uuid4() for _ in range(195)}
        assert ndcg_at_k([A, B, C, D, E], many, 5) == pytest.approx(1.0)

    def test_a_later_hit_scores_less(self) -> None:
        assert ndcg_at_k([B, A], {A}, 5) < ndcg_at_k([A, B], {A}, 5)

    def test_nothing_relevant(self) -> None:
        assert ndcg_at_k([A, B], {C}, 5) == 0.0

    def test_no_relevant_set(self) -> None:
        assert ndcg_at_k([A], set(), 5) == 0.0


class TestEvaluateQuery:
    def test_every_metric_is_present_at_every_k(self) -> None:
        scores = evaluate_query([A], {A}, ks=(1, 5))
        assert set(scores) == {
            "mrr",
            "hit_rate@1",
            "precision@1",
            "coverage@1",
            "ndcg@1",
            "hit_rate@5",
            "precision@5",
            "coverage@5",
            "ndcg@5",
        }

    def test_an_empty_result_list_scores_zero_throughout(self) -> None:
        assert set(evaluate_query([], {A}, ks=(5,)).values()) == {0.0}


class TestIndexChunksBySection:
    def test_groups_by_paper_and_section(self) -> None:
        chunks = [make_chunk("p1", [0], 0), make_chunk("p1", [1], 1), make_chunk("p2", [0], 0)]
        by_section = index_chunks_by_section(chunks)
        assert set(by_section) == {("p1", 0), ("p1", 1), ("p2", 0)}

    def test_a_straddling_chunk_appears_under_both(self) -> None:
        """What makes both neighbours of a boundary scoreable."""
        chunk = make_chunk("p1", [3, 4], 0)
        by_section = index_chunks_by_section([chunk])
        assert by_section[("p1", 3)] == by_section[("p1", 4)] == {chunk.id}

    def test_unlabelled_chunks_are_skipped(self) -> None:
        assert index_chunks_by_section([make_chunk("p1", [], 0)]) == {}


class TestBuildRelevance:
    QUERIES = {
        "q1": {"query": "what is a cell?"},
        "q2": {"query": "what is a treaty?"},
        "q3": {"query": "unindexed paper"},
    }
    QRELS = {
        "q1": {"doc_id": "p1", "section_id": 0},
        "q2": {"doc_id": "p1", "section_id": 9},
        "q3": {"doc_id": "p9", "section_id": 0},
    }

    @property
    def chunks(self) -> list[Chunk]:
        return [make_chunk("p1", [0], 0), make_chunk("p1", [1], 1)]

    def test_resolves_a_query_to_its_chunks(self) -> None:
        relevance, _ = build_relevance(self.chunks, self.QUERIES, self.QRELS)
        assert [item.query_id for item in relevance] == ["q1"]
        assert len(relevance[0].relevant) == 1
        assert relevance[0].query == "what is a cell?"

    def test_a_query_on_an_unindexed_paper_is_not_counted_at_all(self) -> None:
        """q3 targets a paper outside the working set — it is not a failure of
        the retriever and must not appear in the denominator."""
        _, report = build_relevance(self.chunks, self.QUERIES, self.QRELS)
        assert report.queries_total == 3
        assert report.queries_in_corpus == 2

    def test_a_gold_section_with_no_chunks_is_excluded_and_reported(self) -> None:
        """q2's section 9 was never aligned, so no chunk can answer it.
        Scoring it zero would blame the retriever for a labelling gap."""
        relevance, report = build_relevance(self.chunks, self.QUERIES, self.QRELS)
        assert "q2" not in {item.query_id for item in relevance}
        assert report.unscoreable_sections == [("p1", 9)]
        assert report.queries_scoreable == 1

    def test_answers_are_carried_through_for_the_generation_stage(self) -> None:
        relevance, _ = build_relevance(
            self.chunks, self.QUERIES, self.QRELS, {"q1": "A cell is..."}
        )
        assert relevance[0].answer == "A cell is..."

    def test_summary_names_the_gap(self) -> None:
        _, report = build_relevance(self.chunks, self.QUERIES, self.QRELS)
        assert "1 scoreable" in report.summary()

    def test_no_chunks_means_nothing_is_scoreable(self) -> None:
        relevance, report = build_relevance([], self.QUERIES, self.QRELS)
        assert relevance == []
        assert report.queries_scoreable == 0


class StubRetriever:
    """Returns a fixed ranking, so evaluate() is tested without a real index."""

    def __init__(self, ranking: list[UUID]) -> None:
        self._ranking = ranking

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk=Chunk(
                    id=chunk_id,
                    content="x",
                    metadata=ChunkMetadata(
                        document_id=uuid4(), source="p.pdf", start_char=0, end_char=1, chunk_index=0
                    ),
                ),
                score=1.0 / (rank + 1),
                retriever_type=RetrieverType.DENSE,
            )
            for rank, chunk_id in enumerate(self._ranking[:top_k])
        ]


class TestEvaluate:
    def _relevance(self, relevant: set[UUID], count: int = 2) -> list[QueryRelevance]:
        return [
            QueryRelevance(f"q{i}", f"query {i}", "p1", 0, frozenset(relevant))
            for i in range(count)
        ]

    def test_averages_across_queries(self) -> None:
        result = evaluate(StubRetriever([A, B, C]), self._relevance({A}), ks=(1, 5))
        assert result.num_queries == 2
        assert result.means["hit_rate@1"] == 1.0
        assert result.means["mrr"] == 1.0

    def test_a_miss_lowers_the_mean(self) -> None:
        result = evaluate(StubRetriever([C, D]), self._relevance({A}), ks=(5,))
        assert result.means["hit_rate@5"] == 0.0

    def test_records_per_query_detail(self) -> None:
        result = evaluate(StubRetriever([A]), self._relevance({A}), ks=(1,))
        assert len(result.per_query) == 2
        assert result.per_query[0]["num_relevant"] == 1
        assert result.per_query[0]["doc_id"] == "p1"

    def test_per_query_detail_can_be_turned_off(self) -> None:
        result = evaluate(StubRetriever([A]), self._relevance({A}), ks=(1,), keep_per_query=False)
        assert result.per_query == []

    def test_retrieval_depth_covers_the_largest_k(self) -> None:
        """Fetching only 5 would cap ndcg@10 at whatever the top 5 held."""
        ranking = [uuid4() for _ in range(9)] + [A]
        result = evaluate(StubRetriever(ranking), self._relevance({A}), ks=(1, 10))
        assert result.means["hit_rate@10"] == 1.0
        assert result.means["hit_rate@1"] == 0.0

    def test_no_queries(self) -> None:
        assert evaluate(StubRetriever([A]), []).num_queries == 0

    def test_summary_reports_the_headline(self) -> None:
        result = evaluate(StubRetriever([A]), self._relevance({A}), ks=(1, 5))
        assert "2 queries" in result.summary(ks=(1, 5))
