"""Tests for the rankings artifact and the retrieve/score split.

The point of the split is that no stage after retrieval needs an index, so
these tests use a stub retriever and hand-built chunks — if any of this reached
for faiss or an embedder, that would be the bug.

What has to hold:

* scoring from saved ids gives the same numbers as scoring live, or the split
  silently changed every metric in the results directory;
* a rankings file round-trips, including the scores and retriever kind a
  replayed result needs to be indistinguishable from a live one;
* stale ids — the chunk set rebuilt after retrieval — fail loudly, because
  quietly dropping them would write answers from fewer passages than asked for.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from app.rag.evaluation.metrics import (
    Ranking,
    evaluate,
    retrieve_all,
    score_rankings,
)
from app.rag.evaluation.qrels import QueryRelevance
from app.rag.evaluation.rankings import (
    StaleRankings,
    as_results,
    load_rankings,
    load_rankings_dir,
    rankings_file,
    write_rankings,
)
from app.rag.models import Chunk, ChunkMetadata, RetrievalResult, RetrieverType
from app.rag.retrieval import ReplayRetriever


def make_chunk(index: int) -> Chunk:
    return Chunk(
        content=f"passage {index}",
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source=f"paper{index}.pdf",
            page_number=index + 1,
            start_char=index * 100,
            end_char=index * 100 + 10,
            chunk_index=index,
        ),
    )


CHUNKS = [make_chunk(i) for i in range(6)]
BY_ID = {chunk.id: chunk for chunk in CHUNKS}


class StubRetriever:
    """Returns a fixed slice of CHUNKS, so a ranking is known in advance."""

    def __init__(self, order: list[int]) -> None:
        self.order = order
        self.calls: list[str] = []

    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.HYBRID

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append(query)
        return [
            RetrievalResult(
                chunk=CHUNKS[i], score=1.0 - position / 10, retriever_type=RetrieverType.HYBRID
            )
            for position, i in enumerate(self.order[:top_k])
        ]


def relevance(query_id: str = "q1", relevant: set | None = None) -> QueryRelevance:
    return QueryRelevance(
        query_id=query_id,
        query=f"question {query_id}",
        doc_id="2411.00000v1",
        section_id=1,
        answer="the reference answer",
        relevant=relevant if relevant is not None else {CHUNKS[0].id},
    )


class TestSplit:
    def test_scoring_saved_ids_matches_scoring_live(self) -> None:
        """The whole refactor rests on this: rankings are not an approximation."""
        retriever = StubRetriever([2, 0, 3, 1])
        items = [relevance("q1"), relevance("q2", {CHUNKS[3].id})]

        live = evaluate(retriever, items, ks=(1, 3))
        replayed = score_rankings(
            retrieve_all(StubRetriever([2, 0, 3, 1]), items, ks=(1, 3)), items, ks=(1, 3)
        )

        assert live.means == replayed.means
        assert live.num_queries == replayed.num_queries

    def test_retrieval_keeps_scores_and_the_retriever_kind(self) -> None:
        """A replayed result has to be indistinguishable from a live one, and
        `RetrievalResult` requires both fields to be built at all."""
        (ranking,) = retrieve_all(StubRetriever([1, 0]), [relevance()], top_k=2)

        assert ranking.retrieved == [CHUNKS[1].id, CHUNKS[0].id]
        assert ranking.scores == [1.0, 0.9]
        assert ranking.retriever_type == "hybrid"

    def test_a_query_with_no_ranking_is_skipped_not_scored_as_a_miss(self) -> None:
        """Never retrieved is not the same as retrieved and found nothing —
        counting it as a miss would drag the mean of a partial run downward."""
        items = [relevance("q1"), relevance("q2")]
        rankings = [Ranking(query_id="q1", retrieved=[CHUNKS[0].id], scores=[1.0])]

        result = score_rankings(rankings, items, ks=(1,))

        assert result.num_queries == 1
        assert result.means["hit_rate@1"] == 1.0

    def test_nothing_to_score_reports_nothing(self) -> None:
        assert score_rankings([], [relevance()], ks=(1,)).num_queries == 0

    def test_latency_survives_the_round_trip(self) -> None:
        """The latency charts are drawn from this, and retrieval is the only
        stage that can measure it."""
        rankings = [Ranking(query_id="q1", retrieved=[CHUNKS[0].id], latency_ms=12.5)]
        result = score_rankings(rankings, [relevance()], ks=(1,))

        assert result.mean_latency_ms == 12.5
        assert result.per_query[0]["latency_ms"] == 12.5


class TestArtifact:
    def test_a_rankings_file_round_trips(self, tmp_path) -> None:
        rankings = retrieve_all(StubRetriever([2, 0]), [relevance()], ks=(1,))
        path = write_rankings(
            tmp_path,
            rankings,
            config_id="fixed_size_512_128__minilm__dense",
            label="fixed_size:512:128 | minilm | dense",
            experiment="grid_12",
            top_k=10,
            config={"chunker": {"name": "fixed_size"}},
        )

        loaded = load_rankings(path)

        assert loaded.config_id == "fixed_size_512_128__minilm__dense"
        assert loaded.label == "fixed_size:512:128 | minilm | dense"
        assert loaded.top_k == 10
        assert loaded.config["chunker"]["name"] == "fixed_size"
        assert loaded.rankings[0].retrieved == rankings[0].retrieved
        assert loaded.rankings[0].scores == rankings[0].scores
        assert loaded.rankings[0].retriever_type == "hybrid"

    def test_the_file_is_named_for_the_config_not_the_run(self, tmp_path) -> None:
        """A re-run of the same cell overwrites: a ranking is a function of the
        config and the corpus, and every consumer names it without a timestamp."""
        first = write_rankings(tmp_path, [Ranking("q1")], config_id="cell")
        second = write_rankings(tmp_path, [Ranking("q1"), Ranking("q2")], config_id="cell")

        assert first == second == rankings_file(tmp_path, "cell")
        assert len(load_rankings(first)) == 2

    def test_a_file_that_is_not_rankings_is_refused(self, tmp_path) -> None:
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"config_id": "cell", "means": {"mrr": 0.5}}))

        with pytest.raises(ValueError, match="no rankings"):
            load_rankings(path)

    def test_a_directory_load_skips_what_does_not_parse(self, tmp_path) -> None:
        write_rankings(tmp_path, [Ranking("q1")], config_id="good")
        (tmp_path / "bad.json").write_text("{not json")

        assert [s.config_id for s in load_rankings_dir(tmp_path)] == ["good"]


class TestAsResults:
    def test_ids_become_results_the_generator_can_answer_from(self) -> None:
        ranking = Ranking(
            query_id="q1",
            retrieved=[CHUNKS[2].id, CHUNKS[0].id],
            scores=[0.9, 0.4],
            retriever_type="bm25",
        )

        results = as_results(ranking, BY_ID)

        assert [r.chunk.id for r in results] == [CHUNKS[2].id, CHUNKS[0].id]
        assert [r.score for r in results] == [0.9, 0.4]
        assert results[0].retriever_type is RetrieverType.BM25

    def test_top_k_truncates_the_saved_ranking(self) -> None:
        """The grid retrieves 10; an answer prompt wants 5 of them."""
        ranking = Ranking(query_id="q1", retrieved=[c.id for c in CHUNKS], scores=[1.0] * 6)
        assert len(as_results(ranking, BY_ID, top_k=2)) == 2

    def test_a_rebuilt_chunk_set_raises_rather_than_dropping_passages(self) -> None:
        """Chunk ids are generated per chunking run, so re-chunking invalidates
        every saved ranking. Skipping the misses would answer from fewer
        passages than asked for and look like a slightly worse config."""
        ranking = Ranking(query_id="q1", retrieved=[uuid4()], scores=[1.0])

        with pytest.raises(StaleRankings, match="retrieve.py"):
            as_results(ranking, BY_ID)

    def test_a_file_written_before_scores_existed_still_loads(self) -> None:
        """Older rankings carry ids only; a zero score is wrong but harmless,
        since nothing downstream ranks by it — the order is the ranking."""
        ranking = Ranking(query_id="q1", retrieved=[CHUNKS[1].id])
        results = as_results(ranking, BY_ID)

        assert results[0].chunk.id == CHUNKS[1].id
        assert results[0].score == 0.0


class TestReplayRetriever:
    def test_it_serves_the_saved_ranking(self) -> None:
        ranking = Ranking(query_id="q1", retrieved=[CHUNKS[1].id], scores=[0.7])
        retriever = ReplayRetriever(
            {"question q1": as_results(ranking, BY_ID)}, retriever_type="hybrid"
        )

        (result,) = retriever.retrieve("question q1")

        assert result.chunk.id == CHUNKS[1].id
        assert retriever.retriever_type is RetrieverType.HYBRID

    def test_it_reports_the_kind_that_produced_the_ranking(self) -> None:
        """Not "replay" — every downstream record would misattribute its
        passages if replaying changed what the retriever claims to be."""
        assert ReplayRetriever({}, retriever_type="bm25").retriever_type is RetrieverType.BM25

    def test_an_unseen_query_returns_nothing_rather_than_raising(self) -> None:
        """Same situation as a retriever finding no matches, which the answer
        stage already handles."""
        assert ReplayRetriever({}).retrieve("never asked") == []

    def test_top_k_is_honoured(self) -> None:
        ranking = Ranking(query_id="q1", retrieved=[c.id for c in CHUNKS], scores=[1.0] * 6)
        retriever = ReplayRetriever({"q": as_results(ranking, BY_ID)})

        assert len(retriever.retrieve("q", top_k=3)) == 3

    def test_whitespace_around_a_query_does_not_lose_its_ranking(self) -> None:
        """`AnswerGenerator.answer` strips before retrieving; the map is built
        from the benchmark's raw text."""
        retriever = ReplayRetriever({"question q1": []})
        assert retriever.retrieve("  question q1  ") == []
        assert "question q1" in retriever.by_query


class TestUUIDHandling:
    def test_ids_survive_json_as_strings(self, tmp_path) -> None:
        """The file is JSON, the metrics compare UUID objects. A str/UUID
        mismatch here would score every query as a miss."""
        chunk_id = CHUNKS[0].id
        path = write_rankings(
            tmp_path, [Ranking("q1", retrieved=[chunk_id], scores=[1.0])], config_id="cell"
        )

        loaded = load_rankings(path).rankings[0]

        assert isinstance(loaded.retrieved[0], UUID)
        assert loaded.retrieved[0] == chunk_id
        assert score_rankings([loaded], [relevance()], ks=(1,)).means["hit_rate@1"] == 1.0
