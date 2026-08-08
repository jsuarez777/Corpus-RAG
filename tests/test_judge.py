"""Tests for LLM-as-judge scoring.

No network: the judge's LLM is a stub returning a fixed JSON string. What that
leaves testable is everything the judge is actually responsible for —

* the passages the judge sees are numbered the same way the answer's markers
  were, or every citation-quality score is graded against the wrong text;
* a score outside 1-5, or text that is not JSON, is refused rather than
  averaged into the report;
* one unusable score does not lose the other nineteen in a grid run.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.rag.base import BaseLLM
from app.rag.evaluation.judge import (
    CITATION_QUALITY_FLOOR,
    DIMENSIONS,
    JudgeError,
    JudgeScore,
    LLMJudge,
)
from app.rag.models import Chunk, ChunkMetadata, QAResponse

PASSAGES = [
    "Cells were tracked with live imaging microscopy.",
    "The XGBoost model predicts active cells per frame.",
]


def make_chunk(text: str, index: int) -> Chunk:
    return Chunk(
        content=text,
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source=f"paper{index}.pdf",
            page_number=index + 1,
            start_char=index * 100,
            end_char=index * 100 + len(text),
            chunk_index=index,
        ),
    )


def make_response(answer: str = "Cells are tracked by microscopy [1].") -> QAResponse:
    return QAResponse(
        query="how are cells tracked?",
        answer=answer,
        chunks_used=[make_chunk(text, i) for i, text in enumerate(PASSAGES)],
    )


def score_json(relevance=5, accuracy=5, completeness=4, citation_quality=5, rationale="ok") -> str:
    return json.dumps(
        {
            "relevance": relevance,
            "accuracy": accuracy,
            "completeness": completeness,
            "citation_quality": citation_quality,
            "rationale": rationale,
        }
    )


class StubLLM(BaseLLM):
    """Returns canned responses in order, and records how it was called."""

    model = "stub-judge"

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.kwargs: list[dict] = []

    def generate(self, prompt: str, *, temperature: float = 0.0, **kwargs) -> str:
        self.prompts.append(prompt)
        self.systems.append(kwargs.get("system"))
        self.kwargs.append({"temperature": temperature, **kwargs})
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def usage_summary(self) -> str:
        return "stub"


class TestJudgeScore:
    def test_average_is_the_mean_of_four_dimensions(self) -> None:
        score = JudgeScore(relevance=5, accuracy=4, completeness=3, citation_quality=4)
        assert score.average == 4.0

    def test_confidence_maps_the_scale_floor_to_zero(self) -> None:
        """1/5 is the bottom of the scale, not 20% quality."""
        worst = JudgeScore(relevance=1, accuracy=1, completeness=1, citation_quality=1)
        best = JudgeScore(relevance=5, accuracy=5, completeness=5, citation_quality=5)
        assert (worst.confidence, best.confidence) == (0.0, 1.0)

    def test_confidence_is_a_valid_qa_response_confidence(self) -> None:
        score = JudgeScore(relevance=4, accuracy=4, completeness=3, citation_quality=5)
        response = make_response()
        response.confidence = score.confidence  # would raise if outside 0-1
        assert 0.0 < response.confidence < 1.0

    @pytest.mark.parametrize("value", [0, 6, -1])
    def test_scores_outside_the_scale_are_rejected(self, value: int) -> None:
        with pytest.raises(Exception):
            JudgeScore(relevance=value, accuracy=3, completeness=3, citation_quality=3)


class TestScoring:
    def test_the_judge_sees_question_answer_and_numbered_passages(self) -> None:
        llm = StubLLM(score_json())
        LLMJudge(llm).score(make_response())

        prompt = llm.prompts[0]
        assert "how are cells tracked?" in prompt
        assert "Cells are tracked by microscopy [1]." in prompt
        # Numbering must match the markers in the answer being graded.
        assert "[1] paper0.pdf p1" in prompt and "[2] paper1.pdf p2" in prompt
        assert prompt.index("[1] paper0.pdf") < prompt.index("[2] paper1.pdf")

    def test_the_rubric_is_sent_as_the_system_prompt(self) -> None:
        llm = StubLLM(score_json())
        LLMJudge(llm).score(make_response())
        assert llm.systems[0] and "Citation Quality" in llm.systems[0]

    def test_scoring_is_deterministic_and_schema_constrained(self) -> None:
        llm = StubLLM(score_json())
        LLMJudge(llm).score(make_response())

        call = llm.kwargs[0]
        assert call["temperature"] == 0.0
        assert call["text"]["format"]["strict"] is True

    def test_a_reference_answer_is_included_when_given(self) -> None:
        llm = StubLLM(score_json())
        judge = LLMJudge(llm)

        judge.score(make_response(), reference="Tracked by live imaging.")
        judge.score(make_response())

        assert "Reference answer" in llm.prompts[0]
        assert "Tracked by live imaging." in llm.prompts[0]
        assert "Reference answer" not in llm.prompts[1]

    def test_an_answer_with_no_passages_still_scores(self) -> None:
        """Retrieval returning nothing is a result the judge has to grade."""
        empty = QAResponse(query="q", answer="No relevant passages were retrieved.")
        score = LLMJudge(StubLLM(score_json(relevance=1))).score(empty)
        assert score.relevance == 1

    @pytest.mark.parametrize(
        "raw",
        [
            "not json at all",
            '{"relevance": 5}',  # missing dimensions
            '{"relevance": 9, "accuracy": 5, "completeness": 5, "citation_quality": 5}',
        ],
    )
    def test_an_unusable_score_raises_rather_than_defaulting(self, raw: str) -> None:
        with pytest.raises(JudgeError):
            LLMJudge(StubLLM(raw)).score(make_response())


class TestScoreAll:
    def test_report_averages_every_dimension(self) -> None:
        llm = StubLLM(score_json(relevance=5, accuracy=3, completeness=3, citation_quality=5))
        report = LLMJudge(llm).score_all([make_response(), make_response()])

        assert report.num_scored == 2
        assert report.means["relevance"] == 5.0
        assert report.means["accuracy"] == 3.0
        assert report.means["average"] == 4.0
        assert set(DIMENSIONS) <= set(report.means)

    def test_per_query_detail_carries_the_rationale(self) -> None:
        llm = StubLLM(score_json(rationale="cites [1] for a claim [1] does not make"))
        report = LLMJudge(llm).score_all([make_response()])

        (row,) = report.per_query
        assert row["query"] == "how are cells tracked?"
        assert "does not make" in row["rationale"]

    def test_one_bad_score_does_not_lose_the_others(self) -> None:
        llm = StubLLM("garbage", score_json(), score_json())
        report = LLMJudge(llm).score_all([make_response(), make_response(), make_response()])

        assert report.num_scored == 2
        assert len(report.per_query) == 2

    def test_low_citation_quality_is_flagged(self) -> None:
        low = LLMJudge(StubLLM(score_json(citation_quality=3))).score_all([make_response()])
        high = LLMJudge(StubLLM(score_json(citation_quality=5))).score_all([make_response()])

        assert low.citation_quality_is_low
        assert str(CITATION_QUALITY_FLOOR) in low.summary()
        assert not high.citation_quality_is_low

    def test_an_empty_run_reports_nothing_rather_than_dividing_by_zero(self) -> None:
        report = LLMJudge(StubLLM(score_json())).score_all([])
        assert report.num_scored == 0 and report.means == {}
        assert not report.citation_quality_is_low

    def test_the_report_records_what_produced_it(self) -> None:
        report = LLMJudge(StubLLM(score_json())).score_all([make_response()])
        assert report.model == "stub-judge"
        assert report.prompt_version.startswith("v")


class TestJudgePrompt:
    def test_the_shipped_prompt_loads_and_fills(self) -> None:
        judge = LLMJudge(StubLLM(score_json()))
        assert all(name.replace("_", " ") in judge.prompt.system.lower() for name in ("relevance",))
        filled = judge.prompt.render(question="q", context="[1] a.pdf", answer="a", reference="")
        assert "{question}" not in filled and "{reference}" not in filled
