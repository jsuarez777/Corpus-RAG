"""Tests for context building, citation parsing and answer assembly.

No network: the LLM is a stub returning a fixed string, which is the point —
everything that can be wrong about a citation is wrong before or after the API
call, never inside it. What matters here:

* the ``[N]`` -> chunk mapping is positional, and survives whatever order the
  model cites in;
* a marker with no passage behind it is reported, not resolved into a citation
  for whichever chunk happens to sit at that index;
* an empty result set never reaches the model, because a model asked to answer
  with no context answers from its training data and the evaluation cannot
  tell that apart from a real retrieval.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.rag.base import BaseLLM, BaseRetriever
from app.rag.generation import (
    AnswerGenerator,
    build_context,
    latest_version,
    load_prompt,
    parse_citations,
    parse_markers,
    unresolved_markers,
)
from app.rag.generation.generator import ABSTAINED_ANSWER, NO_CONTEXT_ANSWER
from app.rag.generation.prompts import INSUFFICIENT, passage_label
from app.rag.models import Chunk, ChunkMetadata, RetrievalResult, RetrieverType

PASSAGES = [
    "Cells were tracked with live imaging microscopy.",
    "The XGBoost model predicts active cells per frame.",
    "Ripley's K measures spatial clustering.",
]


def make_result(text: str, index: int, *, page: int | None = 1, score: float = 0.9):
    chunk = Chunk(
        content=text,
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source=f"paper{index}.pdf",
            page_number=page,
            start_char=index * 100,
            end_char=index * 100 + len(text),
            chunk_index=index,
        ),
    )
    return RetrievalResult(chunk=chunk, score=score, retriever_type=RetrieverType.HYBRID)


@pytest.fixture
def results() -> list[RetrievalResult]:
    return [make_result(text, i, score=1.0 - i / 10) for i, text in enumerate(PASSAGES)]


class StubRetriever(BaseRetriever):
    """Returns fixed results, and records the top_k it was asked for."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.asked_for: list[int] = []

    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.HYBRID

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.asked_for.append(top_k)
        return self._results[:top_k]


class StubLLM(BaseLLM):
    """Returns a canned answer and remembers what it was prompted with."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def generate(self, prompt: str, *, temperature: float = 0.0, **kwargs) -> str:
        self.prompts.append(prompt)
        self.systems.append(kwargs.get("system"))
        return self.answer


class TestBuildContext:
    def test_passages_are_numbered_from_one_in_rank_order(self, results) -> None:
        context = build_context(results)
        assert context.startswith("[1] paper0.pdf p1\n")
        assert "[2] paper1.pdf p1" in context
        assert context.index("[1]") < context.index("[2]") < context.index("[3]")

    def test_whitespace_inside_a_passage_is_collapsed(self) -> None:
        result = make_result("line one\nbroken   mid\tsentence", 0)
        assert "line one broken mid sentence" in build_context([result])

    def test_long_passages_are_truncated_not_dropped(self, results) -> None:
        context = build_context(results, passage_chars=10)
        # Truncation must not renumber: all three markers still present.
        assert [f"[{n}]" in context for n in (1, 2, 3)] == [True, True, True]
        assert "..." in context

    def test_missing_page_is_omitted_from_the_label(self) -> None:
        assert passage_label(make_result("text", 0, page=None), 1) == "[1] paper0.pdf"

    def test_no_results_is_empty_context(self) -> None:
        assert build_context([]) == ""


class TestPrompts:
    def test_latest_version_picks_the_highest_number(self, tmp_path) -> None:
        for name in ("v1", "v2", "v10", "notes"):
            (tmp_path / name).mkdir()
        assert latest_version(tmp_path) == "v10"

    def test_missing_version_dir_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            latest_version(tmp_path)

    def test_shipped_prompt_loads_and_formats(self) -> None:
        prompt = load_prompt()
        assert INSUFFICIENT in prompt.system
        filled = prompt.format("what is X?", "[1] a.pdf\ntext")
        assert "what is X?" in filled and "[1] a.pdf" in filled
        assert "{context}" not in filled and "{query}" not in filled


class TestParseMarkers:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("a claim [1] and another [2].", [1, 2]),
            ("stacked [1][3]", [1, 3]),
            ("comma separated [1, 3]", [1, 3]),
            ("a range [1-3]", [1, 2, 3]),
            ("repeated [2] then [2] again", [2]),
            ("cited out of order [3] then [1]", [3, 1]),
            ("no markers at all", []),
            ("not a marker [sic] nor [Smith et al.]", []),
        ],
    )
    def test_marker_forms(self, answer: str, expected: list[int]) -> None:
        assert parse_markers(answer) == expected

    def test_an_implausible_range_is_not_expanded(self) -> None:
        """ "[2020-2024]" is a date; expanding it would invent 5 citations."""
        assert parse_markers("published [2020-2024]") == [2020, 2024]


class TestParseCitations:
    def test_citations_follow_the_answer_not_the_ranking(self, results) -> None:
        citations = parse_citations("first [3] then [1]", results)
        assert [c.source for c in citations] == ["paper2.pdf", "paper0.pdf"]

    def test_citation_carries_page_and_retrieval_score(self, results) -> None:
        (citation,) = parse_citations("claim [2]", results)
        assert citation.chunk_id == results[1].chunk.id
        assert citation.page_number == 1
        assert citation.relevance_score == pytest.approx(results[1].score)

    def test_uncited_passages_produce_no_citation(self, results) -> None:
        assert len(parse_citations("only one [1]", results)) == 1

    def test_out_of_range_marker_is_skipped_not_misresolved(self, results) -> None:
        assert parse_citations("claim [9]", results) == []
        assert unresolved_markers("claim [9] and [1]", results) == [9]

    def test_zero_is_out_of_range(self, results) -> None:
        """Numbering starts at 1; [0] must not resolve to the last passage."""
        assert parse_citations("claim [0]", results) == []
        assert unresolved_markers("claim [0]", results) == [0]

    def test_snippet_is_truncated(self, results) -> None:
        (citation,) = parse_citations("claim [1]", results, snippet_chars=10)
        assert citation.text_snippet == "Cells were..."


class TestAnswerGenerator:
    def test_answer_carries_citations_and_the_chunks_behind_them(self, results) -> None:
        llm = StubLLM("Cells are tracked by microscopy [1], and counted by a model [2].")
        generator = AnswerGenerator(StubRetriever(results), llm, top_k=3)

        response = generator.answer("how are cells tracked?")

        assert response.query == "how are cells tracked?"
        assert [c.source for c in response.citations] == ["paper0.pdf", "paper1.pdf"]
        assert [c.id for c in response.chunks_used] == [r.chunk.id for r in results]
        assert response.confidence is None  # not measured here; the judge scores it

    def test_the_prompt_carries_the_numbered_context_and_the_system_text(self, results) -> None:
        llm = StubLLM("answer [1]")
        AnswerGenerator(StubRetriever(results), llm).answer("q")

        assert "[1] paper0.pdf p1" in llm.prompts[0]
        assert PASSAGES[0] in llm.prompts[0]
        assert llm.systems[0] and INSUFFICIENT in llm.systems[0]

    def test_top_k_reaches_the_retriever(self, results) -> None:
        retriever = StubRetriever(results)
        generator = AnswerGenerator(retriever, StubLLM("a [1]"), top_k=2)

        generator.answer("q")
        generator.answer("q", top_k=1)

        assert retriever.asked_for == [2, 1]

    def test_empty_retrieval_never_reaches_the_model(self) -> None:
        llm = StubLLM("this must not be returned")
        response = AnswerGenerator(StubRetriever([]), llm).answer("q")

        assert llm.prompts == []
        assert response.answer == NO_CONTEXT_ANSWER
        assert response.confidence == 0.0
        assert response.citations == [] and response.chunks_used == []

    def test_abstention_is_recorded_as_such(self, results) -> None:
        generator = AnswerGenerator(StubRetriever(results), StubLLM(INSUFFICIENT))
        response = generator.answer("what colour is the sky?")

        assert response.answer == ABSTAINED_ANSWER
        assert response.confidence == 0.0
        assert response.citations == []
        assert len(response.chunks_used) == len(results)  # what it declined on

    def test_an_uncited_answer_is_still_returned(self, results) -> None:
        response = AnswerGenerator(StubRetriever(results), StubLLM("no markers")).answer("q")
        assert response.answer == "no markers" and response.citations == []

    def test_answer_from_reuses_the_given_results(self, results) -> None:
        """The grid scores metrics and generates from one result set."""
        retriever = StubRetriever(results)
        generator = AnswerGenerator(retriever, StubLLM("a [2]"))

        response = generator.answer_from("q", results[:2])

        assert retriever.asked_for == []  # no second retrieval
        assert [c.id for c in response.chunks_used] == [r.chunk.id for r in results[:2]]

    def test_empty_query_raises(self, results) -> None:
        with pytest.raises(ValueError, match="empty"):
            AnswerGenerator(StubRetriever(results), StubLLM("a")).answer("   ")


class TestOpenAILLM:
    """Usage accounting, with no API call — the SDK is needed to construct one."""

    @pytest.fixture
    def llm(self):
        pytest.importorskip("openai")
        from app.rag.generation import OpenAILLM

        return OpenAILLM(model="gpt-4.1-mini", api_key="test-key-not-used")

    def test_default_model(self, llm) -> None:
        assert llm.model == "gpt-4.1-mini"
        assert llm.temperature == 0.0

    def test_usage_accumulates_across_calls(self, llm) -> None:
        class Usage:
            input_tokens, output_tokens, input_tokens_details = 1000, 200, None

        llm._record_usage(type("R", (), {"usage": Usage})())
        llm._record_usage(type("R", (), {"usage": Usage})())

        assert (llm.calls, llm.input_tokens, llm.output_tokens) == (2, 2000, 400)
        assert llm.cost_usd == pytest.approx(2000 * 0.40 / 1e6 + 400 * 1.60 / 1e6)

    def test_cached_tokens_are_billed_at_the_cached_rate(self, llm) -> None:
        class Details:
            cached_tokens = 800

        class Usage:
            input_tokens, output_tokens, input_tokens_details = 1000, 0, Details

        llm._record_usage(type("R", (), {"usage": Usage})())

        assert (llm.input_tokens, llm.cached_input_tokens) == (200, 800)
        assert llm.cost_usd == pytest.approx(200 * 0.40 / 1e6 + 800 * 0.10 / 1e6)

    def test_a_response_without_usage_is_not_an_error(self, llm) -> None:
        llm._record_usage(object())
        assert llm.calls == 1 and llm.cost_usd == 0.0

    def test_unknown_model_has_no_cost(self, llm) -> None:
        llm.model = "gpt-imaginary"
        assert llm.cost_usd is None
        assert "cost unknown" in llm.usage_summary()
