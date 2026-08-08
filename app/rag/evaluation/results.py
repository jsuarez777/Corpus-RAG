"""Reading the result files a grid run wrote back into comparable records.

``app/evaluate.py`` writes one JSON per grid cell, timestamped, carrying the
full configuration next to the numbers. This module reads them back. It is the
seam between "a run happened" and "here is a chart of it", and it exists apart
from the plotting because the Streamlit UI and RESULTS.md want the same records
without wanting a figure.

Two things it handles that a bare ``json.load`` does not:

* **Re-runs.** Nothing overwrites; a second run of the same grid leaves twelve
  more files beside the first twelve. Charting all of them would plot each
  config twice, once with stale numbers, so :func:`latest_per_config` keeps the
  newest file per ``config_id`` and drops the rest.
* **The grid axes.** A chart of "which chunker won" needs the chunker as a
  value it can group by. The file stores the whole config, so the axes are read
  back off it rather than parsed out of the filename, which has already been
  slugged and cannot be reversed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.rag.config import PipelineConfig

log = logging.getLogger(__name__)

#: Result filenames are ``<YYYYmmdd>_<HHMMSS>_<config_id>.json``.
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class RunResult:
    """One scored grid cell: the config that produced it and what it scored."""

    path: Path
    config: PipelineConfig
    config_id: str
    means: dict[str, float]
    num_queries: int
    mean_latency_ms: float
    per_query: list[dict] = field(default_factory=list)
    corpus: str = ""
    #: The experiment file this cell came from, empty for older results. One
    #: config can belong to two experiments, so this is not derivable from the
    #: config and has to be recorded at the time the run happened.
    experiment: str = ""

    # --- the grid axes, for grouping ------------------------------------- #

    @property
    def chunker(self) -> str:
        return self.config.chunker.spec

    @property
    def embedder(self) -> str:
        return self.config.embedder.spec

    @property
    def retriever(self) -> str:
        return self.config.retriever.spec

    @property
    def alpha(self) -> float | None:
        """The hybrid dense/sparse weight, or None for a non-hybrid run.

        ``hybrid:0.3`` and ``hybrid:0.5:rrf`` both carry it positionally, and
        the dict form carries it by name; the sweep chart needs it as a number
        either way.
        """
        retriever = self.config.retriever
        if retriever.name != "hybrid":
            return None
        if "alpha" in retriever.options:
            return float(retriever.options["alpha"])
        first = str(retriever.options.get("_args", "")).partition(":")[0]
        try:
            return float(first)
        except ValueError:
            return None

    @property
    def fusion(self) -> str:
        """``weighted`` or ``rrf`` — how hybrid combined the two rankings."""
        retriever = self.config.retriever
        if "fusion" in retriever.options:
            return str(retriever.options["fusion"])
        parts = str(retriever.options.get("_args", "")).split(":")
        return parts[1] if len(parts) > 1 else "weighted"

    @property
    def timestamp(self) -> datetime:
        """When the run happened, from the filename.

        Falls back to the file's mtime for a hand-named file — a chart that
        silently dropped a result because its name did not parse would be
        worse than one that ordered it slightly wrong.
        """
        stem = self.path.stem
        try:
            return datetime.strptime("_".join(stem.split("_")[:2]), TIMESTAMP_FORMAT)
        except ValueError:
            return datetime.fromtimestamp(self.path.stat().st_mtime)

    @property
    def label(self) -> str:
        """Short axis label: the three things that vary across the grid."""
        return f"{self.chunker} | {self.embedder} | {self.retriever}"

    def latencies(self) -> list[float]:
        """Per-query latencies in ms, empty for a run scored before they were
        recorded."""
        return [row["latency_ms"] for row in self.per_query if "latency_ms" in row]

    def metric(self, name: str) -> float:
        return self.means.get(name, float("nan"))


def load_result(path: Path) -> RunResult:
    """Read one result file, rejecting a shape this cannot actually read.

    ``app/evaluate.py``'s flag-driven path writes a different layout — several
    retrievers nested under one file, and a ``config`` that is a handful of
    strings rather than a dumped :class:`PipelineConfig`. Pydantic ignores the
    keys it does not recognise, so that file parses without complaint into a
    record with no metrics and a retriever nobody chose: a heatmap row of
    0.000 across the board, which reads as a config that scored badly rather
    than one that was never read. Refusing it makes :func:`load_results` skip
    it and say why.
    """
    data = json.loads(Path(path).read_text())
    if "retrievers" in data:
        raise ValueError(
            "written by the flag-driven path, which stores several retrievers per "
            "file; re-score it with `evaluate.py -c <experiment.yaml>`"
        )
    if not data.get("means"):
        raise ValueError("no metrics in the file")
    config = PipelineConfig(**data["config"])
    return RunResult(
        path=Path(path),
        config=config,
        # Older files predate the field; the config can always regenerate it.
        config_id=data.get("config_id") or config.id,
        means=data.get("means", {}),
        num_queries=data.get("num_queries", 0),
        mean_latency_ms=data.get("mean_latency_ms", 0.0),
        per_query=data.get("per_query", []),
        corpus=data.get("corpus", ""),
        experiment=data.get("experiment", ""),
    )


def load_results(results_dir: Path, pattern: str = "*.json") -> list[RunResult]:
    """Every result file under ``results_dir``, oldest first.

    A file that does not parse is logged and skipped rather than raised on:
    result directories accumulate, and one experiment from an older schema
    should not stop the other eleven from being charted.
    """
    runs: list[RunResult] = []
    for path in sorted(Path(results_dir).glob(pattern)):
        try:
            runs.append(load_result(path))
        except Exception as error:
            log.warning(f"Skipping {path.name}: {error}")
    runs.sort(key=lambda run: run.timestamp)
    return runs


def latest_per_config(runs: list[RunResult]) -> list[RunResult]:
    """One run per config, keeping the most recent.

    Without this, re-running a grid doubles every chart: the same config
    appears twice, and the older copy carries whatever the code scored before
    the change that prompted the re-run.

    Keyed on the experiment as well as the config, because the same cell can
    legitimately appear in two experiments — collapsing those two would hand
    one experiment's chart a row belonging to the other.
    """
    newest: dict[tuple[str, str], RunResult] = {}
    for run in sorted(runs, key=lambda run: run.timestamp):
        newest[(run.experiment, run.config_id)] = run
    return list(newest.values())


def sweep_experiments(runs: list[RunResult]) -> set[str]:
    """Experiment names that vary the hybrid weight.

    A sweep's cells differ in one number, so they belong on the sweep's own
    line chart and nowhere near a config comparison. Unnamed runs are never
    counted as a sweep — an older result file with no experiment recorded is
    more likely to be a grid than not.
    """
    alphas: dict[str, set[float]] = {}
    for run in runs:
        if run.experiment and run.alpha is not None:
            alphas.setdefault(run.experiment, set()).add(run.alpha)
    return {name for name, values in alphas.items() if len(values) > 1}


def available_metrics(runs: list[RunResult]) -> list[str]:
    """Metric names present in every run, so a table has no holes."""
    if not runs:
        return []
    shared = set(runs[0].means)
    for run in runs[1:]:
        shared &= set(run.means)
    return sorted(shared)


def varying_axes(runs: list[RunResult]) -> list[str]:
    """Which of chunker/embedder/retriever actually differ across ``runs``.

    An axis held constant is not a finding, and a bar chart of one bar is
    noise. The alpha sweep varies only the retriever; the main grid varies all
    three.
    """
    axes = {"chunker": set(), "embedder": set(), "retriever": set()}
    for run in runs:
        axes["chunker"].add(run.chunker)
        axes["embedder"].add(run.embedder)
        axes["retriever"].add(run.retriever)
    return [name for name, values in axes.items() if len(values) > 1]
