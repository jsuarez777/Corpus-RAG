#!/usr/bin/env python3
"""Render the report's figures from the result files a grid run wrote.

Stage 6. Reads experiments/results/*.json and writes PNGs to
experiments/figures/. Charts nothing it does not have data for: a chart type
whose inputs are missing is reported as skipped, with what would produce it,
rather than drawn from placeholder numbers.

Usage:
    python app/visualize.py                       # every chart there is data for
    python app/visualize.py heatmap latency       # only these
    python app/visualize.py --list                # what can be drawn right now
    python app/visualize.py --metric hit_rate@5   # the metric the bar charts use

By default the newest run of each configuration is used. Result files are never
overwritten, so a re-scored grid leaves the old files in place; charting both
copies would draw every config twice, once with numbers the re-run replaced.
Pass ``--all-runs`` if the history is what you want to see.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):  # `python app/visualize.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matplotlib.figure import Figure  # noqa: E402

from app.rag.evaluation import plots  # noqa: E402
from app.rag.evaluation.results import (  # noqa: E402
    RunResult,
    latest_per_config,
    load_results,
    sweep_experiments,
)
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "experiments/results"
DEFAULT_FIGURES = PROJECT_ROOT / "experiments/figures"

#: Judged generation scores, written by the LLM judge. Absent until it runs.
DEFAULT_JUDGED = PROJECT_ROOT / "experiments/judged"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# The chart registry
# --------------------------------------------------------------------------- #
#
# Each entry knows how to build one figure from the loaded runs. A builder
# raises ValueError when its inputs are not there — the caller turns that into
# a skip with a reason, which is the difference between "this chart is not in
# the report yet" and "this chart silently came out wrong".


def _grid_runs(runs: list[RunResult]) -> list[RunResult]:
    """Runs from the comparison grids, with the weight sweeps dropped.

    A sweep holds the chunker and embedder fixed and varies only the fusion
    weight, so its cells would crowd a config comparison with rows that differ
    in one number — and its endpoints, pure BM25 and pure dense, would land in
    the same chart as the hybrid they are the components of.
    """
    sweeps = sweep_experiments(runs)
    return [run for run in runs if run.experiment not in sweeps] or runs


def _heatmap(runs: list[RunResult], args: argparse.Namespace) -> Figure:
    return plots.metrics_heatmap(_grid_runs(runs), normalize=not args.absolute_colors)


def _dimensions(runs: list[RunResult], args: argparse.Namespace) -> Figure:
    return plots.dimension_impact(_grid_runs(runs), metric=args.metric)


def _improvement(runs: list[RunResult], args: argparse.Namespace) -> Figure:
    grid = _grid_runs(runs)
    baseline = plots.pick_baseline(grid)
    best = plots.pick_best(grid, args.metric)
    if baseline.config_id == best.config_id:
        raise ValueError("The baseline is also the best run; nothing to compare")
    return plots.before_after(baseline, best)


def _generation(runs: list[RunResult], args: argparse.Namespace) -> Figure:
    scores = load_judged(args.judged)
    if not scores:
        raise ValueError(f"No judged answers in {_display(args.judged)} — run the LLM judge first")
    return plots.generation_quality_radar(scores)


def _latency(runs: list[RunResult], args: argparse.Namespace) -> Figure:
    return plots.latency_distribution(_grid_runs(runs))


def _latency_box(runs: list[RunResult], args: argparse.Namespace) -> Figure:
    return plots.latency_boxplot(_grid_runs(runs))


def _alpha(runs: list[RunResult], args: argparse.Namespace) -> Figure:
    sweeps = sweep_experiments(runs)
    return plots.alpha_sweep([run for run in runs if run.experiment in sweeps] or runs)


CHARTS: dict[str, tuple[str, Callable[[list[RunResult], argparse.Namespace], Figure]]] = {
    "heatmap": ("Retrieval metrics by configuration", _heatmap),
    "dimensions": ("Which configuration axis moved the metric", _dimensions),
    "improvement": ("Baseline vs best configuration", _improvement),
    "generation": ("Generation quality, judged", _generation),
    "latency": ("Query latency distribution", _latency),
    # The violin shows the shape and the box shows the quotable five numbers;
    # the spec asks for a histogram or a box plot, so the box is the compliant
    # one and the violin is the one worth looking at.
    "latency_box": ("Query latency quartiles", _latency_box),
    "alpha": ("Hybrid fusion weight sweep", _alpha),
}


def load_judged(judged_dir: Path) -> dict[str, dict[str, float]]:
    """Mean judge scores per config, from experiments/judged/*.json.

    Returns empty when the judge has not run, which is what makes the radar
    chart skip rather than fail. The shape it reads is ``{"config_id": ...,
    "means": {"relevance": ..., ...}}`` — the same layout the retrieval results
    use, so one loader shape covers both.
    """
    import json

    scores: dict[str, dict[str, float]] = {}
    if not Path(judged_dir).is_dir():
        return scores

    for path in sorted(Path(judged_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            log.warning(f"Skipping {path.name}: {error}")
            continue
        means = data.get("means", {})
        label = data.get("label") or data.get("config_id") or path.stem
        scores[label] = {
            axis: float(means.get(axis.lower().replace(" ", "_"), 0.0))
            for axis in plots.GENERATION_AXES
        }
    return scores


def render(names: list[str], runs: list[RunResult], args: argparse.Namespace) -> tuple[int, int]:
    """Draw and save each named chart. Returns (written, skipped)."""
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    for name in names:
        title, builder = CHARTS[name]
        try:
            figure = builder(runs, args)
        except ValueError as reason:
            log.warning(f"Skipping {name}: {reason}")
            skipped += 1
            continue
        path = args.out / f"{name}.png"
        figure.savefig(path)
        plt.close(figure)
        log.info(f"Wrote {_display(path)}  — {title}")
        written += 1

    return written, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the report's figures from scored experiment results.",
    )
    parser.add_argument(
        "charts",
        nargs="*",
        choices=[*CHARTS, []],
        metavar="CHART",
        help=f"which to draw; default: all. One or more of: {', '.join(CHARTS)}",
    )
    parser.add_argument("--list", action="store_true", help="show what can be drawn and exit")
    parser.add_argument(
        "--metric",
        default="ndcg@5",
        help="the metric the bar charts rank by; default: ndcg@5",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="chart every result file, not just the newest per config",
    )
    parser.add_argument(
        "--absolute-colors",
        action="store_true",
        help="one color scale across the heatmap instead of per metric",
    )
    parser.add_argument("-r", "--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("-j", "--judged", type=Path, default=DEFAULT_JUDGED)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("visualize")

    runs = load_results(args.results)
    if not runs:
        log.error(f"No result files in {_display(args.results)}. Run app/evaluate.py first.")
        return 1
    if not args.all_runs:
        runs = latest_per_config(runs)
    log.info(f"{len(runs)} run(s) from {_display(args.results)}")

    if args.list:
        for name, (title, _) in CHARTS.items():
            print(f"  {name:<12} {title}")
        return 0

    written, skipped = render(list(args.charts) or list(CHARTS), runs, args)
    log.info(f"{written} figure(s) in {_display(args.out)}, {skipped} skipped")
    # Skipping is a normal state until the judge exists; only writing nothing
    # at all is a failure worth a non-zero exit.
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
