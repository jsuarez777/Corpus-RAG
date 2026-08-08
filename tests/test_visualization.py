"""Tests for reading results back and turning them into figures.

Charts are hard to assert on and easy to get subtly wrong, so these tests draw
the line at the two places a mistake would be silent and consequential:

* **Reading a result file back.** A config round-trips through JSON on the way
  into a chart, and a component that comes back as ``hybrid`` where the run was
  ``hybrid:0.5`` produces a chart that is wrong rather than one that fails.
* **Which runs a chart is given.** The same cell can belong to two experiments
  and a re-run leaves the old file in place, so the selection rules decide what
  ends up on the axes.

The figures themselves are checked for structure — the number of rows, the
values annotated, the axis order — not for appearance. No file is written and
no window is opened; the backend is Agg.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.rag.config import PipelineConfig
from app.rag.evaluation import plots
from app.rag.evaluation.results import (
    RunResult,
    available_metrics,
    latest_per_config,
    load_result,
    load_results,
    sweep_experiments,
    varying_axes,
)

MEANS = {
    "hit_rate@1": 0.5,
    "hit_rate@3": 0.7,
    "hit_rate@5": 0.8,
    "hit_rate@10": 0.9,
    "mrr": 0.62,
    "ndcg@5": 0.43,
    "precision@5": 0.25,
    "coverage@5": 0.42,
}


def make_run(
    chunker: str = "fixed_size:512:128",
    embedder: str = "minilm",
    retriever: str = "dense",
    *,
    experiment: str = "grid_12",
    stamp: str = "20260808_120000",
    latencies: list[float] | None = None,
    **overrides: float,
) -> RunResult:
    config = PipelineConfig(chunker=chunker, embedder=embedder, retriever=retriever)
    per_query = [{"latency_ms": value} for value in (latencies or [])]
    return RunResult(
        path=Path(f"{stamp}_{config.id}.json"),
        config=config,
        config_id=config.id,
        means={**MEANS, **overrides},
        num_queries=488,
        mean_latency_ms=8.0,
        per_query=per_query,
        experiment=experiment,
    )


def write_run(directory: Path, run: RunResult) -> Path:
    path = directory / run.path.name
    path.write_text(
        json.dumps(
            {
                "config": run.config.model_dump(mode="json"),
                "config_id": run.config_id,
                "experiment": run.experiment,
                "num_queries": run.num_queries,
                "mean_latency_ms": run.mean_latency_ms,
                "means": run.means,
                "per_query": run.per_query,
            }
        )
    )
    return path


class TestConfigRoundTrip:
    """A result file is written from ``model_dump`` and read back through the
    same validator. If that trip is lossy every chart silently loses its
    parameters."""

    @pytest.mark.parametrize(
        "chunker,retriever",
        [
            ("fixed_size:512:128", "dense"),
            ("semantic:512:90", "hybrid:0.5"),
            ("sentence:5:1", "hybrid:0.3:rrf"),
        ],
    )
    def test_the_spec_survives_a_dump_and_reload(self, chunker: str, retriever: str) -> None:
        config = PipelineConfig(chunker=chunker, retriever=retriever, embedder="mpnet")
        back = PipelineConfig(**config.model_dump(mode="json"))
        assert (back.chunker.spec, back.retriever.spec) == (chunker, retriever)

    def test_the_identifier_survives_too(self) -> None:
        """It names the index directory, so a changed id is a chart pointing at
        a run that does not exist."""
        config = PipelineConfig(chunker="semantic:512:90", embedder="mpnet", retriever="hybrid:0.7")
        assert PipelineConfig(**config.model_dump(mode="json")).id == config.id

    def test_the_flat_dict_form_still_works(self) -> None:
        """Hand-written YAML names options directly rather than nesting them."""
        config = PipelineConfig(chunker={"name": "fixed_size", "chunk_size": 512, "overlap": 128})
        assert config.chunker.kwargs == {"chunk_size": 512, "overlap": 128}


class TestLoading:
    def test_reads_a_written_result(self, tmp_path: Path) -> None:
        write_run(tmp_path, make_run(retriever="hybrid:0.5"))
        (loaded,) = load_results(tmp_path)
        assert (loaded.chunker, loaded.embedder, loaded.retriever) == (
            "fixed_size:512:128",
            "minilm",
            "hybrid:0.5",
        )

    def test_a_broken_file_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """Result directories accumulate; one file from an older schema must
        not stop the others being charted."""
        write_run(tmp_path, make_run())
        (tmp_path / "20260808_130000_broken.json").write_text("{not json")
        assert len(load_results(tmp_path)) == 1

    def test_an_empty_directory_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert load_results(tmp_path) == []

    def test_the_flag_paths_layout_is_refused(self, tmp_path: Path) -> None:
        """It nests several retrievers under one file and writes a config of
        loose strings. Pydantic ignores what it does not recognise, so this
        parses into a row with no metrics and a retriever nobody chose — a
        heatmap of 0.000 that reads as a bad score rather than an unread file.
        """
        (tmp_path / "20260808_140000_fixed_size_512_128__minilm.json").write_text(
            json.dumps(
                {
                    "config": {"chunker": "fixed_size:512:128", "embedder": "minilm"},
                    "retrievers": {"dense": {"num_queries": 488, "means": {"ndcg@5": 0.43}}},
                }
            )
        )
        assert load_results(tmp_path) == []

    def test_a_file_with_no_metrics_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "20260808_150000_empty.json").write_text(
            json.dumps({"config": PipelineConfig().model_dump(mode="json"), "means": {}})
        )
        assert load_results(tmp_path) == []

    def test_the_timestamp_comes_from_the_filename(self, tmp_path: Path) -> None:
        path = write_run(tmp_path, make_run(stamp="20260101_093000"))
        assert load_result(path).timestamp == datetime(2026, 1, 1, 9, 30, 0)

    def test_an_unparseable_name_falls_back_to_the_mtime(self, tmp_path: Path) -> None:
        """Dropping a hand-named file from a chart would be worse than ordering
        it slightly wrong."""
        path = tmp_path / "hand-named.json"
        write_run(tmp_path, make_run())
        path.write_text((tmp_path / next(iter(p.name for p in tmp_path.glob("2026*")))).read_text())
        assert isinstance(load_result(path).timestamp, datetime)


class TestAlpha:
    @pytest.mark.parametrize(
        "retriever,expected",
        [("hybrid:0.0", 0.0), ("hybrid:0.3", 0.3), ("hybrid:0.5:rrf", 0.5), ("hybrid", None)],
    )
    def test_read_from_the_spec(self, retriever: str, expected: float | None) -> None:
        assert make_run(retriever=retriever).alpha == expected

    def test_a_non_hybrid_run_has_none(self) -> None:
        """Dense and BM25 have no weight, and charting them at alpha 0 would
        put them on a line they are not points of."""
        assert make_run(retriever="dense").alpha is None
        assert make_run(retriever="bm25").alpha is None

    @pytest.mark.parametrize(
        "retriever,expected", [("hybrid:0.5", "weighted"), ("hybrid:0.5:rrf", "rrf")]
    )
    def test_fusion_is_read_from_the_spec(self, retriever: str, expected: str) -> None:
        assert make_run(retriever=retriever).fusion == expected


class TestSelection:
    def test_a_rerun_replaces_its_predecessor(self) -> None:
        """Otherwise every chart draws each config twice, once with the numbers
        the re-run was meant to replace."""
        old = make_run(stamp="20260101_000000", **{"ndcg@5": 0.10})
        new = make_run(stamp="20260808_000000", **{"ndcg@5": 0.90})
        kept = latest_per_config([old, new])
        assert [run.metric("ndcg@5") for run in kept] == [0.90]

    def test_the_same_cell_in_two_experiments_is_kept_twice(self) -> None:
        """``sentence:5:1 | minilm | hybrid:0.5`` is both a grid cell and the
        midpoint of the sweep; collapsing them hands one chart the other's row.
        """
        grid = make_run(retriever="hybrid:0.5", experiment="grid_12")
        sweep = make_run(retriever="hybrid:0.5", experiment="alpha_sweep")
        assert len(latest_per_config([grid, sweep])) == 2

    def test_a_sweep_is_recognised_by_its_spread_of_weights(self) -> None:
        runs = [
            make_run(retriever=f"hybrid:{alpha}", experiment="alpha_sweep")
            for alpha in ("0.0", "0.5", "1.0")
        ] + [make_run(retriever="dense", experiment="grid_12")]
        assert sweep_experiments(runs) == {"alpha_sweep"}

    def test_one_weight_is_not_a_sweep(self) -> None:
        """A grid using hybrid:0.5 throughout is a comparison, not a sweep."""
        runs = [
            make_run(chunker=chunker, retriever="hybrid:0.5", experiment="grid_12")
            for chunker in ("fixed_size:512:128", "sentence:5:1")
        ]
        assert sweep_experiments(runs) == set()

    def test_an_unlabelled_run_is_never_a_sweep(self) -> None:
        """Older result files record no experiment, and a chart is better off
        treating those as a grid than dropping them."""
        runs = [make_run(retriever=f"hybrid:{a}", experiment="") for a in ("0.0", "1.0")]
        assert sweep_experiments(runs) == set()

    def test_varying_axes_names_only_what_differs(self) -> None:
        runs = [make_run(retriever=r) for r in ("dense", "hybrid:0.5")]
        assert varying_axes(runs) == ["retriever"]

    def test_available_metrics_are_the_shared_ones(self) -> None:
        """A column missing from one run would be a hole in the table."""
        full = make_run()
        partial = make_run(embedder="mpnet")
        partial.means.pop("coverage@5")
        assert "coverage@5" not in available_metrics([full, partial])
        assert "mrr" in available_metrics([full, partial])


class TestBaselineAndBest:
    def test_the_baseline_is_the_naive_config(self) -> None:
        """Not the weakest run — choosing the weakest would be picking the
        comparison that flatters the result."""
        runs = [
            make_run(chunker="sentence:5:1", **{"ndcg@5": 0.05}),
            make_run(**{"ndcg@5": 0.40}),
        ]
        assert plots.pick_baseline(runs).chunker == "fixed_size:512:128"

    def test_it_falls_back_to_the_weakest_when_absent(self) -> None:
        runs = [
            make_run(chunker="sentence:5:1", **{"ndcg@5": 0.40}),
            make_run(chunker="semantic:512:90", **{"ndcg@5": 0.05}),
        ]
        assert plots.pick_baseline(runs).chunker == "semantic:512:90"

    def test_the_best_is_by_the_named_metric(self) -> None:
        runs = [
            make_run(retriever="dense", **{"ndcg@5": 0.9, "mrr": 0.1}),
            make_run(retriever="hybrid:0.5", **{"ndcg@5": 0.1, "mrr": 0.9}),
        ]
        assert plots.pick_best(runs, "mrr").retriever == "hybrid:0.5"


class TestFigures:
    def test_the_heatmap_has_a_row_per_config(self) -> None:
        runs = [make_run(retriever=r) for r in ("dense", "hybrid:0.5")]
        figure = plots.metrics_heatmap(runs)
        labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
        assert labels == [run.label for run in runs]

    def test_the_heatmap_prints_true_values_not_normalized_ones(self) -> None:
        """Colors are scaled per column so configs are rankable; the numbers
        must stay the measured ones or the chart lies quietly."""
        runs = [
            make_run(retriever="dense", **{"mrr": 0.20}),
            make_run(retriever="hybrid:0.5", **{"mrr": 0.80}),
        ]
        printed = {text.get_text() for text in plots.metrics_heatmap(runs).axes[0].texts}
        assert {"0.200", "0.800"} <= printed

    def test_the_heatmap_drops_a_metric_no_run_has(self) -> None:
        figure = plots.metrics_heatmap([make_run()], metrics=("mrr", "invented@5"))
        assert [t.get_text() for t in figure.axes[0].get_xticklabels()] == ["MRR"]

    def test_dimension_impact_gets_a_panel_per_varying_axis(self) -> None:
        runs = [
            make_run(chunker=chunker, retriever=retriever)
            for chunker in ("fixed_size:512:128", "sentence:5:1")
            for retriever in ("dense", "hybrid:0.5")
        ]
        # Two panels plus nothing else: the embedder is constant here.
        assert len(plots.dimension_impact(runs).axes) == 2

    def test_dimension_impact_starts_the_axis_at_zero(self) -> None:
        """A truncated axis turns a 0.007 difference into a tall bar."""
        runs = [
            make_run(retriever=r, **{"ndcg@5": v}) for r, v in (("dense", 0.50), ("bm25", 0.51))
        ]
        assert plots.dimension_impact(runs).axes[0].get_ylim()[0] == 0

    def test_before_after_labels_the_delta(self) -> None:
        baseline = make_run(retriever="dense", **{"mrr": 0.600})
        improved = make_run(retriever="hybrid:0.5", **{"mrr": 0.750})
        figure = plots.before_after(baseline, improved, metrics=("mrr",))
        assert "+0.150" in {text.get_text() for text in figure.axes[0].texts}

    def test_before_after_marks_a_regression(self) -> None:
        """A negative delta has to read as one; a chart that only ever shows
        gains is not measuring anything."""
        worse = make_run(retriever="hybrid:0.5", **{"mrr": 0.400})
        figure = plots.before_after(make_run(**{"mrr": 0.600}), worse, metrics=("mrr",))
        assert "-0.200" in {text.get_text() for text in figure.axes[0].texts}

    def test_the_radar_axes_are_the_ones_the_judge_scores(self) -> None:
        """The radar reads the judge's ``means`` by name. If either side renames
        a dimension the chart does not fail — it draws four zeroes, which looks
        like a config that scored nothing rather than a wiring mistake.
        """
        from app.rag.evaluation.judge import DIMENSIONS

        drawn = tuple(axis.lower().replace(" ", "_") for axis in plots.GENERATION_AXES)
        assert drawn == DIMENSIONS

    def test_the_radar_closes_its_polygon(self) -> None:
        """An open polygon leaves a gap between the last axis and the first."""
        scores = {"a": dict.fromkeys(plots.GENERATION_AXES, 4.0)}
        line = plots.generation_quality_radar(scores).axes[0].lines[0]
        points = line.get_xydata()
        assert len(points) == len(plots.GENERATION_AXES) + 1
        assert points[0][1] == points[-1][1]

    def test_the_radar_needs_scores(self) -> None:
        with pytest.raises(ValueError, match="No judged scores"):
            plots.generation_quality_radar({})

    def test_latency_needs_per_query_samples(self) -> None:
        """Results scored before latency was recorded per query must say so
        rather than draw a distribution from a single mean."""
        with pytest.raises(ValueError, match="Re-run the grid"):
            plots.latency_distribution([make_run()])

    def test_latency_orders_fastest_at_the_top(self) -> None:
        slow = make_run(retriever="hybrid:0.5", latencies=[100.0, 110.0, 120.0])
        fast = make_run(retriever="dense", latencies=[4.0, 5.0, 6.0])
        figure = plots.latency_distribution([slow, fast])
        # Position 1 is the bottom of a horizontal violin plot.
        labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
        assert labels == [slow.label, fast.label]

    def test_the_boxplot_needs_per_query_samples_too(self) -> None:
        with pytest.raises(ValueError, match="Re-run the grid"):
            plots.latency_boxplot([make_run()])

    def test_the_boxplot_draws_a_box_per_config(self) -> None:
        runs = [
            make_run(retriever="hybrid:0.5", latencies=[9.0, 10.0, 11.0, 12.0]),
            make_run(retriever="dense", latencies=[4.0, 5.0, 6.0, 7.0]),
        ]
        figure = plots.latency_boxplot(runs)
        assert len([p for p in figure.axes[0].patches]) == 2

    def test_both_latency_charts_agree_on_the_order(self) -> None:
        """They are drawn from the same samples and sit next to each other in
        the report; a reader comparing them row by row must not be misled."""
        slow = make_run(retriever="hybrid:0.5", latencies=[100.0, 110.0, 120.0])
        fast = make_run(retriever="dense", latencies=[4.0, 5.0, 6.0])
        rows = [
            [text.get_text() for text in chart([fast, slow]).axes[0].get_yticklabels()]
            for chart in (plots.latency_distribution, plots.latency_boxplot)
        ]
        assert rows[0] == rows[1] == [slow.label, fast.label]

    def test_the_sweep_needs_two_weights(self) -> None:
        runs = [
            make_run(chunker=c, retriever="hybrid:0.5") for c in ("sentence:5:1", "semantic:512:90")
        ]
        with pytest.raises(ValueError, match="alpha_sweep"):
            plots.alpha_sweep(runs)

    def test_the_sweep_stays_on_one_corpus(self) -> None:
        """A grid cell at hybrid:0.5 over a different chunker shares the weight
        but not the corpus, and stacks into a spike at 0.5 if it is let in."""
        sweep = [
            make_run(chunker="sentence:5:1", retriever=f"hybrid:{alpha}")
            for alpha in ("0.0", "0.5", "1.0")
        ]
        intruder = make_run(chunker="fixed_size:512:128", retriever="hybrid:0.5")
        figure = plots.alpha_sweep([*sweep, intruder])
        assert len(figure.axes[0].lines[0].get_xydata()) == 3

    def test_the_sweep_plots_a_point_per_weight(self) -> None:
        runs = [
            make_run(chunker="sentence:5:1", retriever=f"hybrid:{alpha}")
            for alpha in ("0.0", "0.3", "0.5", "0.7", "1.0")
        ]
        xs = [x for x, _ in plots.alpha_sweep(runs).axes[0].lines[0].get_xydata()]
        assert xs == [0.0, 0.3, 0.5, 0.7, 1.0]
