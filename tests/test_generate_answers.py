"""Tests for batch answer generation and the judging pass over it.

No network: both stages take a stubbed LLM. What that leaves testable is the
part that costs money to get wrong —

* a grid with twelve cells does not silently answer whichever one came first;
* the answers file is appended per query and re-readable mid-run, so a stop at
  400 of 488 keeps 400 paid-for answers and ``--resume`` does not re-buy them;
* one failed query does not end the run;
* the judged report is the exact shape ``app/visualize.py`` reads, on the axes
  ``judge.DIMENSIONS`` names — the radar is the only consumer, and a mismatch
  there is four zeros on a chart rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.generate_answers import (
    as_record,
    generate,
    header,
    label_for,
    select_config,
)
from app.judge_answers import judge_file, read_answers, report_from, write_report
from app.rag.base import BaseLLM, BaseRetriever
from app.rag.config import PipelineConfig
from app.rag.evaluation.judge import DIMENSIONS
from app.rag.evaluation.plots import GENERATION_AXES
from app.rag.evaluation.qrels import QueryRelevance
from app.rag.generation import AnswerGenerator
from app.rag.models import Chunk, ChunkMetadata, RetrievalResult
from app.visualize import load_judged

PASSAGE = "Cells were tracked with live imaging microscopy."


def make_chunk(index: int = 0) -> Chunk:
    return Chunk(
        content=PASSAGE,
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source=f"paper{index}.pdf",
            page_number=index + 1,
            start_char=0,
            end_char=len(PASSAGE),
            chunk_index=index,
        ),
    )


def make_item(number: int) -> QueryRelevance:
    return QueryRelevance(
        query_id=f"q{number}",
        query=f"question {number}?",
        doc_id="2412.06611v2",
        section_id=number,
        relevant=frozenset({uuid4()}),
        answer=f"reference {number}",
    )


class StubRetriever(BaseRetriever):
    """Returns one passage, always."""

    @property
    def retriever_type(self):
        return "dense"

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [RetrievalResult(chunk=make_chunk(), score=0.9, retriever_type="dense")]


class StubLLM(BaseLLM):
    """Returns canned text in order; a canned ``Exception`` instance is raised."""

    model = "stub"
    temperature = 0.0

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def usage_summary(self) -> str:
        return "stub: no cost"


def make_generator(*replies) -> AnswerGenerator:
    return AnswerGenerator(StubRetriever(), StubLLM(*replies), top_k=1)


def score_json(relevance=5, accuracy=4, completeness=4, citation_quality=5) -> str:
    return json.dumps(
        {
            "relevance": relevance,
            "accuracy": accuracy,
            "completeness": completeness,
            "citation_quality": citation_quality,
            "rationale": "grounded and cited",
        }
    )


GRID = [
    PipelineConfig(name="a", chunker="fixed_size:512:128", embedder="mpnet", retriever="dense"),
    PipelineConfig(name="b", chunker="sentence:5:1", embedder="minilm", retriever="hybrid:0.5"),
]


class TestSelectConfig:
    def test_picks_the_named_cell(self) -> None:
        assert select_config(GRID, GRID[1].id) is GRID[1]

    def test_a_multi_cell_grid_needs_an_explicit_choice(self) -> None:
        """Defaulting to the first cell would spend money on an unasked config."""
        with pytest.raises(SystemExit) as error:
            select_config(GRID, None)
        assert GRID[0].id in str(error.value) and GRID[1].id in str(error.value)

    def test_a_single_cell_experiment_needs_no_choice(self) -> None:
        assert select_config(GRID[:1], None) is GRID[0]

    def test_an_unknown_id_lists_the_real_ones(self) -> None:
        with pytest.raises(SystemExit) as error:
            select_config(GRID, "no_such_config")
        assert GRID[0].id in str(error.value)

    def test_label_matches_the_retrieval_charts(self) -> None:
        """RunResult.label is chunker | embedder | retriever; the radar has to
        name its polygons the same or no figure can be read against another."""
        assert label_for(GRID[1]) == "sentence:5:1 | minilm | hybrid:0.5"


class TestGenerate:
    def test_writes_a_header_then_one_line_per_query(self, tmp_path: Path) -> None:
        out = tmp_path / "answers.jsonl"
        written = generate(
            make_generator("An answer [1]."),
            [make_item(1), make_item(2)],
            out,
            config=GRID[0],
        )
        head, records = read_answers(out)

        assert written == 2
        assert head["config_id"] == GRID[0].id
        assert head["num_queries"] == 2
        assert [record["query_id"] for record in records] == ["q1", "q2"]

    def test_the_reference_answer_travels_with_the_response(self, tmp_path: Path) -> None:
        """The judge grades against it; re-reading the benchmark to score a file
        would make the file not stand on its own."""
        out = tmp_path / "answers.jsonl"
        generate(make_generator("An answer [1]."), [make_item(1)], out, config=GRID[0])
        _, records = read_answers(out)

        assert records[0]["reference"] == "reference 1"
        assert records[0]["response"]["chunks_used"][0]["content"] == PASSAGE

    def test_a_failed_query_does_not_end_the_run(self, tmp_path: Path) -> None:
        out = tmp_path / "answers.jsonl"
        generator = make_generator(RuntimeError("rate limited"), "Recovered [1].")
        written = generate(generator, [make_item(1), make_item(2)], out, config=GRID[0])
        _, records = read_answers(out)

        assert written == 1
        assert [record["query_id"] for record in records] == ["q2"]

    def test_resume_skips_what_is_already_answered(self, tmp_path: Path) -> None:
        out = tmp_path / "answers.jsonl"
        items = [make_item(1), make_item(2), make_item(3)]
        generate(make_generator("First pass [1]."), items[:1], out, config=GRID[0])

        generator = make_generator("Second pass [1].")
        written = generate(generator, items, out, config=GRID[0], resume=True)
        _, records = read_answers(out)

        assert (written, generator.llm.calls) == (2, 2)
        assert [record["query_id"] for record in records] == ["q1", "q2", "q3"]
        assert records[0]["response"]["answer"] == "First pass [1]."

    def test_without_resume_the_file_is_rewritten(self, tmp_path: Path) -> None:
        out = tmp_path / "answers.jsonl"
        generate(make_generator("First [1]."), [make_item(1), make_item(2)], out, config=GRID[0])
        generate(make_generator("Second [1]."), [make_item(1)], out, config=GRID[0])
        _, records = read_answers(out)

        assert len(records) == 1
        assert records[0]["response"]["answer"] == "Second [1]."

    def test_the_header_records_what_would_change_the_answers(self) -> None:
        generator = make_generator("x")
        head = header(GRID[0], generator, 488)
        assert head["model"] == "stub"
        assert head["prompt_version"] == generator.prompt.version
        assert head["top_k"] == 1

    def test_a_record_carries_its_own_latency(self) -> None:
        response = make_generator("An answer [1].").answer_from("q?", StubRetriever().retrieve("q"))
        record = as_record(make_item(1), response, 123.456)
        assert record["latency_ms"] == 123.5


class TestReport:
    def test_means_average_each_dimension_and_then_the_four(self) -> None:
        rows = [
            {"relevance": 5, "accuracy": 3, "completeness": 4, "citation_quality": 4},
            {"relevance": 3, "accuracy": 5, "completeness": 4, "citation_quality": 4},
        ]
        report = report_from(rows, model="stub", prompt_version="v1")

        assert report.num_scored == 2
        assert report.means["relevance"] == 4.0
        assert report.means["average"] == 4.0

    def test_no_scores_is_not_a_failing_report(self) -> None:
        """Nothing was measured, so the citation floor has nothing to be below."""
        report = report_from([], model="stub", prompt_version="v1")
        assert report.means == {} and not report.citation_quality_is_low


class TestJudgeFile:
    @pytest.fixture
    def answers(self, tmp_path: Path) -> Path:
        out = tmp_path / "answers.jsonl"
        generate(
            make_generator("An answer [1]."),
            [make_item(1), make_item(2)],
            out,
            config=GRID[0],
        )
        return out

    def test_scores_every_answer_into_a_report(
        self, answers: Path, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr("app.judge_answers.OpenAILLM", lambda model: StubLLM(score_json()))
        target = judge_file(answers, judged_dir=tmp_path / "judged")
        data = json.loads(target.read_text())

        assert data["config_id"] == GRID[0].id
        assert data["num_scored"] == 2
        assert data["means"]["relevance"] == 5.0
        assert data["means"]["average"] == 4.5

    def test_the_report_is_what_the_radar_chart_reads(
        self, answers: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """load_judged lowercases the axis names to reach into ``means``; if the
        two ever drift apart the radar plots four zeros and says nothing."""
        monkeypatch.setattr("app.judge_answers.OpenAILLM", lambda model: StubLLM(score_json()))
        judged_dir = tmp_path / "judged"
        judge_file(answers, judged_dir=judged_dir)

        scores = load_judged(judged_dir)
        assert list(scores) == [label_for(GRID[0])]
        assert scores[label_for(GRID[0])] == {
            "Relevance": 5.0,
            "Accuracy": 4.0,
            "Completeness": 4.0,
            "Citation Quality": 5.0,
        }

    def test_the_radar_axes_are_the_judge_dimensions(self) -> None:
        assert tuple(axis.lower().replace(" ", "_") for axis in GENERATION_AXES) == DIMENSIONS

    def test_resume_skips_what_is_already_scored(
        self, answers: Path, tmp_path: Path, monkeypatch
    ) -> None:
        judged_dir = tmp_path / "judged"
        monkeypatch.setattr("app.judge_answers.OpenAILLM", lambda model: StubLLM(score_json()))
        judge_file(answers, judged_dir=judged_dir, limit=1)

        second = StubLLM(score_json(relevance=1, accuracy=1, completeness=1, citation_quality=1))
        monkeypatch.setattr("app.judge_answers.OpenAILLM", lambda model: second)
        target = judge_file(answers, judged_dir=judged_dir, resume=True)
        data = json.loads(target.read_text())

        assert second.calls == 1  # only the answer that had not been scored
        assert data["num_scored"] == 2
        assert data["means"]["relevance"] == 3.0  # the 5 kept, the 1 added

    def test_an_unusable_score_drops_that_query_only(
        self, answers: Path, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "app.judge_answers.OpenAILLM",
            lambda model: StubLLM("not json at all", score_json()),
        )
        target = judge_file(answers, judged_dir=tmp_path / "judged")

        assert json.loads(target.read_text())["num_scored"] == 1

    def test_a_file_without_a_run_header_is_refused(self, tmp_path: Path) -> None:
        """Reading answers off a headerless file would produce a report that
        cannot say which config, model or prompt produced it."""
        out = tmp_path / "headerless.jsonl"
        out.write_text(json.dumps({"query_id": "q1"}) + "\n")
        with pytest.raises(ValueError, match="run header"):
            read_answers(out)


class TestWriteReport:
    def test_the_label_falls_back_to_the_config_id(self, tmp_path: Path) -> None:
        report = report_from([], model="stub", prompt_version="v1")
        target = write_report(tmp_path, {"config_id": "some_config"}, report)
        assert json.loads(target.read_text())["label"] == "some_config"
