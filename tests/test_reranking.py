"""Tests for the reranking stage.

The property that matters: reranking reorders a set it was handed, and never
changes which chunks are in it. Every metric at the retrieved depth is
therefore invariant — reranking the top 10 cannot move hit@10 — and a change
there means results are being dropped or invented rather than reordered.

The models are stubbed throughout. A cross-encoder is a few hundred MB and the
Cohere client needs a key; neither belongs in a test that runs on every commit,
and neither is what these tests are about.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.rag.base import BaseLLM, BaseRetriever
from app.rag.generation import AnswerGenerator
from app.rag.models import Chunk, ChunkMetadata, RetrievalResult, RetrieverType
from app.rag.reranking import (
    DEFAULT_RERANKER,
    RERANKERS,
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    get_reranker,
    reranker_from_spec,
)
from app.rag.reranking.cohere import KEY_VARIABLES, find_api_key, is_rate_limit
from app.rag.reranking.cross_encoder import sigmoid

DOCUMENT_ID = uuid4()


def make_results(count: int = 5) -> list[RetrievalResult]:
    """First-stage output: descending scores, so rank order is unambiguous."""
    return [
        RetrievalResult(
            chunk=Chunk(
                content=f"passage {index}",
                metadata=ChunkMetadata(
                    document_id=DOCUMENT_ID,
                    source="p.pdf",
                    chunk_index=index,
                    start_char=index * 10,
                    end_char=index * 10 + 9,
                ),
            ),
            score=1.0 - index * 0.1,
            retriever_type=RetrieverType.HYBRID,
        )
        for index in range(count)
    ]


class StubCrossEncoder:
    """Returns the logits it was given, in the order the pairs arrive."""

    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=32):
        self.seen = list(pairs)
        return self.logits[: len(self.seen)]


def stubbed(logits: list[float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker()
    # cached_property, so assigning into __dict__ pre-empts the real load.
    reranker.__dict__["_model"] = StubCrossEncoder(logits)
    return reranker


class TestSigmoid:
    def test_maps_zero_to_a_half(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_is_monotonic(self) -> None:
        values = [sigmoid(x) for x in (-5.0, -1.0, 0.0, 1.0, 5.0)]
        assert values == sorted(values)

    def test_stays_inside_the_unit_interval(self) -> None:
        assert all(0.0 <= sigmoid(x) <= 1.0 for x in (-1e4, -60.0, 0.0, 60.0, 1e4))

    @pytest.mark.parametrize("extreme", [-1e9, -800.0, 800.0, 1e9])
    def test_saturates_instead_of_overflowing(self, extreme: float) -> None:
        """math.exp raises OverflowError below about -710, and a confident
        negative logit gets there."""
        assert sigmoid(extreme) in (0.0, 1.0)


class TestCrossEncoderReranker:
    def test_is_a_reranker(self) -> None:
        assert isinstance(CrossEncoderReranker(), BaseReranker)

    def test_construction_loads_no_weights(self) -> None:
        """Registry sweeps and --help paths build one; neither should pay for
        several hundred MB of model."""
        assert "_model" not in CrossEncoderReranker().__dict__

    def test_reorders_by_the_new_score(self) -> None:
        results = make_results(3)
        # Reverses the first-stage order.
        reranked = stubbed([-2.0, 0.0, 2.0]).rerank("q", results)
        assert [r.chunk.content for r in reranked] == ["passage 2", "passage 1", "passage 0"]

    def test_records_where_each_result_came_from(self) -> None:
        reranked = stubbed([-2.0, 0.0, 2.0]).rerank("q", make_results(3))
        assert [r.original_rank for r in reranked] == [3, 2, 1]

    def test_scores_are_squashed_to_the_unit_interval(self) -> None:
        reranked = stubbed([-9.0, 0.0, 9.0]).rerank("q", make_results(3))
        assert all(0.0 <= r.score <= 1.0 for r in reranked)

    def test_the_query_is_paired_with_every_passage(self) -> None:
        reranker = stubbed([1.0, 1.0, 1.0])
        reranker.rerank("what is smoothness?", make_results(3))
        assert reranker._model.seen == [
            ("what is smoothness?", "passage 0"),
            ("what is smoothness?", "passage 1"),
            ("what is smoothness?", "passage 2"),
        ]

    def test_the_first_stage_results_are_left_alone(self) -> None:
        """The caller's list is usually the baseline being compared against."""
        results = make_results(3)
        before = [(r.score, r.original_rank) for r in results]
        stubbed([-2.0, 0.0, 2.0]).rerank("q", results)
        assert [(r.score, r.original_rank) for r in results] == before

    def test_top_k_truncates_after_reordering(self) -> None:
        """Truncating first would discard the result the reranker was meant to
        promote, which is the entire point of the stage."""
        reranked = stubbed([-2.0, -1.0, 0.0, 1.0, 2.0]).rerank("q", make_results(5), top_k=2)
        assert [r.chunk.content for r in reranked] == ["passage 4", "passage 3"]

    def test_the_retrieved_set_is_unchanged(self) -> None:
        """Reranking reorders; it never adds or drops. Every metric at the
        retrieved depth is invariant because of this."""
        results = make_results(5)
        reranked = stubbed([3.0, -1.0, 0.5, -4.0, 2.0]).rerank("q", results)
        assert {r.chunk.id for r in reranked} == {r.chunk.id for r in results}
        assert len(reranked) == len(results)

    def test_an_empty_candidate_list(self) -> None:
        assert CrossEncoderReranker().rerank("q", []) == []

    def test_an_existing_original_rank_is_not_overwritten(self) -> None:
        """Reranking twice must still point at the first-stage position."""
        once = stubbed([-2.0, 0.0, 2.0]).rerank("q", make_results(3))
        twice = stubbed([2.0, 0.0, -2.0]).rerank("q", once)
        assert sorted(r.original_rank for r in twice) == [1, 2, 3]

    def test_an_alias_resolves_to_a_model_id(self) -> None:
        assert CrossEncoderReranker("ms-marco").model_id.startswith("cross-encoder/")

    def test_an_unknown_model_passes_straight_through(self) -> None:
        assert CrossEncoderReranker("some/other-model").model_id == "some/other-model"


class TestCohereKeyDiscovery:
    def test_prefers_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COHERE_API_KEY", "from-env")
        assert find_api_key() == "from-env"

    def test_accepts_the_older_variable_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in KEY_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("CO_API_KEY", "from-legacy")
        assert find_api_key() == "from-legacy"

    def test_reads_a_shell_profile_when_the_environment_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Exporting a key in ~/.profile and finding a launched process cannot
        see it is a common half-hour."""
        for name in KEY_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        profile = tmp_path / ".profile"
        profile.write_text('# comment\nexport COHERE_API_KEY="from-profile"  # trailing\n')
        monkeypatch.setattr("app.rag.reranking.cohere.PROFILE_FILES", (str(profile),))
        assert find_api_key() == "from-profile"

    def test_missing_everywhere_is_none_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in KEY_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr("app.rag.reranking.cohere.PROFILE_FILES", ())
        assert find_api_key() is None


class TestRateLimitDetection:
    """Matched on the status code rather than Cohere's exception classes,
    which get renamed between SDK majors."""

    def test_detects_a_status_code_attribute(self) -> None:
        error = Exception("boom")
        error.status_code = 429
        assert is_rate_limit(error)

    def test_detects_it_in_the_message(self) -> None:
        assert is_rate_limit(Exception("429 Too Many Requests"))
        assert is_rate_limit(Exception("You are being rate limited"))

    def test_leaves_other_failures_alone(self) -> None:
        """A 401 must surface immediately — retrying it just wastes minutes."""
        error = Exception("unauthorized")
        error.status_code = 401
        assert not is_rate_limit(error)


class TestCohereReranker:
    def test_is_a_reranker(self) -> None:
        assert isinstance(CohereReranker(), BaseReranker)

    def test_construction_needs_no_credentials(self) -> None:
        """So a registry sweep works on a machine with no Cohere account."""
        assert CohereReranker().model_id == "rerank-v3.5"

    def test_an_empty_candidate_list_makes_no_call(self) -> None:
        assert CohereReranker().rerank("q", []) == []


class TestRegistry:
    def test_both_providers_are_registered(self) -> None:
        assert sorted(RERANKERS) == ["cohere", "cross_encoder"]

    def test_the_default_needs_no_api_key(self) -> None:
        """A clone with no Cohere account must still reproduce the numbers."""
        assert DEFAULT_RERANKER == "cross_encoder"

    def test_builds_by_name(self) -> None:
        assert isinstance(get_reranker("cohere"), CohereReranker)

    def test_a_spec_string_picks_the_model(self) -> None:
        reranker = reranker_from_spec("cross_encoder:ms-marco-l12")
        assert reranker.model_id.endswith("L-12-v2")

    def test_a_bare_spec_takes_the_default_model(self) -> None:
        assert reranker_from_spec("cohere").model_id == "rerank-v3.5"

    def test_an_unknown_name_lists_the_real_ones(self) -> None:
        with pytest.raises(ValueError, match="Available: cohere, cross_encoder"):
            get_reranker("nope")


class RecordingRetriever(BaseRetriever):
    """Returns fixed results and records the depth it was asked for."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.asked_for: list[int] = []

    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.HYBRID

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.asked_for.append(top_k)
        return self._results[:top_k]


class SilentLLM(BaseLLM):
    def generate(self, prompt: str, *, temperature: float = 0.0, **kwargs) -> str:
        return "an answer [1]"


def generator_with(reranker, **kwargs) -> tuple[AnswerGenerator, RecordingRetriever]:
    retriever = RecordingRetriever(make_results(40))
    return AnswerGenerator(retriever, SilentLLM(), reranker=reranker, **kwargs), retriever


class TestRerankDepth:
    """A reranker pays off by narrowing, so the retriever has to be asked for
    more than will be kept.

    Fetching exactly ``k`` and reranking to ``k`` hands the model the same
    passages in a different order. Measured on the benchmark, that is not a
    small effect — reranking 10 candidates down to 10 moved hit@10 by exactly
    zero, because no passage entered or left the set.
    """

    def test_without_a_reranker_the_depth_is_just_top_k(self) -> None:
        """Nobody pays for candidates that will not be chosen between."""
        generator, retriever = generator_with(None, top_k=5)
        generator.retrieve("q")
        assert retriever.asked_for == [5]

    def test_a_reranker_fetches_deeper_than_it_keeps(self) -> None:
        generator, retriever = generator_with(stubbed([0.0] * 40), top_k=5)
        kept = generator.retrieve("q")
        assert retriever.asked_for == [20]
        assert len(kept) == 5

    def test_an_explicit_depth_wins(self) -> None:
        generator, retriever = generator_with(stubbed([0.0] * 40), top_k=5, rerank_depth=30)
        generator.retrieve("q")
        assert retriever.asked_for == [30]

    def test_a_depth_below_top_k_is_raised_to_it(self) -> None:
        """Otherwise the reranker would silently return fewer passages than
        the caller asked the generator for."""
        generator, retriever = generator_with(stubbed([0.0] * 40), top_k=10, rerank_depth=3)
        assert retriever.asked_for == [] and generator.retrieve("q") is not None
        assert retriever.asked_for == [10]

    def test_the_per_call_top_k_still_drives_the_depth(self) -> None:
        generator, retriever = generator_with(stubbed([0.0] * 40), top_k=5)
        generator.retrieve("q", top_k=2)
        assert retriever.asked_for == [8]

    def test_the_reranker_chooses_which_passages_survive(self) -> None:
        """The whole point: a candidate outside the kept window is promoted
        into it, which reordering a fetched-to-size list cannot do."""
        logits = [0.0] * 40
        logits[7] = 9.0  # rank 8 of 20 fetched, outside a top-5 retrieval
        generator, _ = generator_with(stubbed(logits), top_k=5)
        assert generator.retrieve("q")[0].chunk.content == "passage 7"
