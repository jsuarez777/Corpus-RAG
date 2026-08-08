"""The report's figures, one function per chart.

Each function takes records and returns a :class:`~matplotlib.figure.Figure`.
Nothing here writes a file or picks a filename — ``app/visualize.py`` does that,
and the Streamlit UI hands the same figure to ``st.pyplot`` instead. Keeping the
rendering separate from the saving is what lets one chart serve both.

Six charts, and what each one is for:

1. :func:`metrics_heatmap` — every config against every metric, the whole grid
   in one frame.
2. :func:`dimension_impact` — which axis of the grid actually moved the number.
3. :func:`before_after` — a baseline against the best config, with the deltas
   spelled out.
4. :func:`generation_quality_radar` — the judge's four scores per config.
5. :func:`latency_distribution` — what a query costs, as a distribution rather
   than a mean, and :func:`latency_boxplot`, the same numbers as quartiles.
6. :func:`alpha_sweep` — how much of the hybrid score should come from dense.

Two choices that are load-bearing and would otherwise look arbitrary:

**The heatmap normalizes its colors per metric.** Hit rate runs from about 0.64
to 0.95 here and precision@10 cannot exceed roughly 0.37 given how many chunks
are relevant per query, so a single color scale paints the precision columns
uniformly pale and hides every difference between configs — which is the one
comparison the chart exists to make. Colors rank configs within a column; the
printed number in each cell is always the real one.

**Latency is drawn from per-query samples, not means.** Two configs can share a
mean while one of them is occasionally very slow, and for a system answering
one query at a time the tail is what a user notices.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib

# No code path here opens a window: the CLI writes PNGs and Streamlit renders
# the figure object directly. Set before pyplot is imported, or it is ignored.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app.rag.evaluation.results import RunResult, varying_axes  # noqa: E402

log = logging.getLogger(__name__)

#: The spec asks for YlGnBu on the heatmap; the rest follow from it so the
#: figures read as one set.
HEATMAP_CMAP = "YlGnBu"
PALETTE = "YlGnBu_d"

#: Columns of the comparison heatmap. Ordered so the hit rates read as a curve
#: across k, which is the shape a reader is looking for.
HEATMAP_METRICS = (
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
    "hit_rate@10",
    "mrr",
    "ndcg@5",
    "precision@5",
    "coverage@5",
)

#: The four axes the LLM judge scores an answer on.
GENERATION_AXES = ("Relevance", "Accuracy", "Completeness", "Citation Quality")

#: The plainest thing anyone would build first, and so the honest "before".
BASELINE_CONFIG = ("fixed_size:512:128", "minilm", "dense")


def apply_style() -> None:
    """House style. Called by every chart, so a figure looks the same however
    it was produced."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "grid.alpha": 0.3,
        }
    )


def _pretty(metric: str) -> str:
    """``hit_rate@5`` -> ``Hit Rate@5``."""
    name, _, k = metric.partition("@")
    label = name.replace("_", " ").title().replace("Mrr", "MRR").replace("Ndcg", "NDCG")
    return f"{label}@{k}" if k else label


def pick_baseline(runs: Sequence[RunResult]) -> RunResult:
    """The run to measure improvement against.

    Prefers the naive default — fixed-size chunks, the small embedder, dense
    retrieval only — because that is what someone builds before running any
    experiments, which is what makes the comparison mean anything. Falls back to
    the weakest run when that config was not part of this grid; picking the
    weakest *by default* would be choosing the baseline to flatter the result.
    """
    if not runs:
        raise ValueError("No runs to pick a baseline from")
    chunker, embedder, retriever = BASELINE_CONFIG
    for run in runs:
        if (run.chunker, run.embedder, run.retriever) == (chunker, embedder, retriever):
            return run
    log.info(f"No {' | '.join(BASELINE_CONFIG)} run; using the weakest as the baseline")
    return min(runs, key=lambda run: run.metric("ndcg@5"))


def pick_best(runs: Sequence[RunResult], metric: str = "ndcg@5") -> RunResult:
    """The highest-scoring run on ``metric``."""
    if not runs:
        raise ValueError("No runs to pick a best from")
    return max(runs, key=lambda run: run.metric(metric))


# --------------------------------------------------------------------------- #
# 1. Retrieval metrics comparison heatmap
# --------------------------------------------------------------------------- #


def metrics_heatmap(
    runs: Sequence[RunResult],
    metrics: Sequence[str] = HEATMAP_METRICS,
    *,
    normalize: bool = True,
) -> Figure:
    """Every config against every metric, one cell each.

    ``normalize`` scales the color of each column to that column's own range,
    so the shading ranks configs on a metric rather than ranking the metrics
    against each other. Cell text is the true value either way. Pass ``False``
    for one absolute scale across the whole frame.
    """
    apply_style()
    runs = list(runs)
    metrics = [metric for metric in metrics if any(metric in run.means for run in runs)]
    values = np.array([[run.metric(metric) for metric in metrics] for run in runs])

    if normalize and len(runs) > 1:
        low = np.nanmin(values, axis=0)
        span = np.nanmax(values, axis=0) - low
        # A column where every config scored the same has no ranking to show;
        # 0.5 puts it mid-scale instead of dividing by zero.
        shaded = np.where(span > 0, (values - low) / np.where(span > 0, span, 1), 0.5)
        colorbar_label = "Relative to the best config on each metric"
    else:
        shaded = values
        colorbar_label = "Score"

    height = 1 + 0.42 * len(runs)
    figure, axes = plt.subplots(figsize=(1.35 * len(metrics) + 4, height))
    sns.heatmap(
        shaded,
        annot=values,
        fmt=".3f",
        cmap=HEATMAP_CMAP,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": colorbar_label},
        xticklabels=[_pretty(metric) for metric in metrics],
        yticklabels=[run.label for run in runs],
        ax=axes,
    )
    axes.set_title(f"Retrieval metrics by configuration ({runs[0].num_queries} queries)")
    axes.tick_params(axis="x", rotation=35)
    axes.tick_params(axis="y", rotation=0)
    for label in axes.get_xticklabels():
        label.set_horizontalalignment("right")
    return figure


# --------------------------------------------------------------------------- #
# 2. Configuration dimension impact
# --------------------------------------------------------------------------- #


def dimension_impact(runs: Sequence[RunResult], metric: str = "ndcg@5") -> Figure:
    """How much each axis of the grid moved ``metric``.

    Each bar averages every run sharing that value, holding nothing else fixed
    — the marginal effect of the choice across the rest of the grid, which is
    the question "does the chunker matter?" actually asks. The spread within a
    panel is the finding: a panel whose bars are level is an axis that did not
    matter, and the caption reports each one's range so that is legible without
    reading the y-axis.
    """
    apply_style()
    runs = list(runs)
    axes_present = varying_axes(runs) or ["retriever"]

    figure, panels = plt.subplots(
        1, len(axes_present), figsize=(4.2 * len(axes_present), 4.4), sharey=True
    )
    panels = np.atleast_1d(panels)
    spreads = []

    for panel, axis in zip(panels, axes_present, strict=True):
        grouped: dict[str, list[float]] = {}
        for run in runs:
            grouped.setdefault(getattr(run, axis), []).append(run.metric(metric))
        labels = sorted(grouped)
        heights = [float(np.mean(grouped[label])) for label in labels]
        spreads.append((axis, max(heights) - min(heights)))

        bars = panel.bar(
            labels, heights, color=sns.color_palette(PALETTE, n_colors=max(len(labels), 3))
        )
        panel.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        panel.set_title(axis.title())
        panel.tick_params(axis="x", rotation=20)
        for label in panel.get_xticklabels():
            label.set_horizontalalignment("right")

    panels[0].set_ylabel(_pretty(metric))
    # Headroom for the bar labels, and a floor at 0 so a small difference is
    # not magnified into a large-looking one by a truncated axis.
    panels[0].set_ylim(0, max(run.metric(metric) for run in runs) * 1.25)

    ranges = "   ".join(f"{axis}: {spread:+.3f}" for axis, spread in spreads)
    figure.suptitle(f"Which choice moved {_pretty(metric)}?", y=1.02)
    figure.text(0.5, -0.06, f"Range within each axis —  {ranges}", ha="center", fontsize=9)
    return figure


# --------------------------------------------------------------------------- #
# 3. Before / after improvement
# --------------------------------------------------------------------------- #


def before_after(
    baseline: RunResult,
    improved: RunResult,
    metrics: Sequence[str] = ("hit_rate@1", "hit_rate@5", "mrr", "ndcg@5"),
) -> Figure:
    """A baseline config beside a better one, with the gap labelled.

    The deltas are printed rather than left to the eye: the bars for a 3-point
    gain and a 15-point gain look similar at this scale, and the number is the
    claim being made.
    """
    apply_style()
    metrics = list(metrics)
    positions = np.arange(len(metrics))
    width = 0.36
    before = [baseline.metric(metric) for metric in metrics]
    after = [improved.metric(metric) for metric in metrics]
    colors = sns.color_palette(HEATMAP_CMAP, n_colors=6)

    figure, axes = plt.subplots(figsize=(1.9 * len(metrics) + 3, 4.8))
    left = axes.bar(positions - width / 2, before, width, label="Baseline", color=colors[2])
    right = axes.bar(positions + width / 2, after, width, label="Best", color=colors[4])
    axes.bar_label(left, fmt="%.3f", padding=2, fontsize=9)
    axes.bar_label(right, fmt="%.3f", padding=2, fontsize=9)

    for position, (low, high) in zip(positions, zip(before, after, strict=True), strict=True):
        delta = high - low
        axes.annotate(
            f"{delta:+.3f}",
            xy=(position, max(low, high)),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#1a7a3c" if delta >= 0 else "#b3261e",
        )

    axes.set_xticks(positions, [_pretty(metric) for metric in metrics])
    axes.set_ylabel("Score")
    axes.set_ylim(0, max(before + after) * 1.28)
    axes.set_title("Baseline vs best configuration")
    axes.legend(loc="lower right", frameon=True)
    figure.text(
        0.5,
        -0.04,
        f"Baseline: {baseline.label}\nBest: {improved.label}",
        ha="center",
        fontsize=9,
    )
    return figure


# --------------------------------------------------------------------------- #
# 4. Generation quality radar
# --------------------------------------------------------------------------- #


def generation_quality_radar(
    scores: dict[str, dict[str, float]],
    axes_names: Sequence[str] = GENERATION_AXES,
    *,
    scale: float = 5.0,
) -> Figure:
    """The judge's scores per config, one polygon each.

    ``scores`` maps a config label to ``{axis: score}``. A radar is the wrong
    chart for ranking — the area it encloses depends on which order the axes
    happen to be in — but it is the right one for shape, and shape is the
    question here: whether a config that answers accurately also cites well.
    """
    apply_style()
    if not scores:
        raise ValueError("No judged scores to plot")

    axes_names = list(axes_names)
    # The polygon has to close, so the first angle is repeated at the end.
    angles = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    angles += angles[:1]

    figure, axes = plt.subplots(figsize=(6.8, 6.2), subplot_kw={"projection": "polar"})
    # First axis at twelve o'clock, running clockwise — matplotlib's polar
    # default starts at three and runs the other way, which reads as arbitrary.
    axes.set_theta_offset(np.pi / 2)
    axes.set_theta_direction(-1)
    colors = sns.color_palette(PALETTE, n_colors=max(len(scores), 3))

    for color, (label, values) in zip(colors, scores.items(), strict=False):
        points = [values.get(axis, 0.0) for axis in axes_names]
        points += points[:1]
        axes.plot(angles, points, linewidth=2, label=label, color=color)
        axes.fill(angles, points, alpha=0.12, color=color)

    axes.set_xticks(angles[:-1], axes_names)
    axes.set_ylim(0, scale)
    axes.set_yticks(np.linspace(0, scale, int(scale) + 1)[1:])
    axes.set_rlabel_position(180 / len(axes_names))
    axes.set_title(f"Generation quality (judged 1–{scale:.0f})", pad=24)
    axes.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    return figure


# --------------------------------------------------------------------------- #
# 5. Query latency distribution
# --------------------------------------------------------------------------- #


#: Tick positions for the latency axis, in ms. Only the ones inside the
#: measured range are drawn.
LATENCY_TICKS = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000)


def _timed(runs: Sequence[RunResult]) -> list[RunResult]:
    """Runs carrying per-query latencies, slowest first.

    Slowest first because both latency charts are horizontal and position 1 is
    the bottom row, so this is what makes them read fastest-to-slowest down the
    page — and makes the two charts agree on the order.
    """
    timed = [run for run in runs if run.latencies()]
    if not timed:
        raise ValueError(
            "No per-query latencies in these results — they were scored before "
            "latency_ms was recorded per query. Re-run the grid."
        )
    return sorted(timed, key=lambda run: -float(np.median(run.latencies())))


def _latency_ticks(samples: Sequence[Sequence[float]]) -> list[int]:
    """The round tick values that fall inside the measured range."""
    low = min(min(values) for values in samples)
    high = max(max(values) for values in samples)
    return [tick for tick in LATENCY_TICKS if low / 1.5 <= tick <= high * 1.5]


def latency_distribution(runs: Sequence[RunResult], *, log_scale: bool = True) -> Figure:
    """Per-query retrieval latency, one distribution per config, fastest first.

    Log scale by default: dense search over a flat index and hybrid search that
    also scores the whole corpus with BM25 differ by an order of magnitude, and
    on a linear axis the faster configs collapse onto the baseline.

    The kernel density is estimated on the logged values rather than on the raw
    ones, which is not cosmetic. A KDE fitted in linear space and then drawn on
    a log axis smears its lower tail across most of the plot — a 4 ms config
    grows a skirt reaching down toward zero that no measurement supports. Doing
    the estimate in the space the reader is looking at keeps the shape honest.

    The p95 marker is drawn because the mean is not what a waiting user
    experiences.
    """
    apply_style()
    runs = _timed(runs)
    samples = [run.latencies() for run in runs]
    drawn = [np.log10(values) if log_scale else values for values in samples]

    def place(value: float) -> float:
        return float(np.log10(value)) if log_scale else float(value)

    figure, axes = plt.subplots(figsize=(9.5, 0.55 * len(runs) + 3))
    parts = axes.violinplot(drawn, orientation="horizontal", showextrema=False, widths=0.85)
    for body, color in zip(
        parts["bodies"], sns.color_palette(PALETTE, n_colors=max(len(runs), 3)), strict=False
    ):
        body.set_facecolor(color)
        body.set_alpha(0.65)

    positions = np.arange(1, len(runs) + 1)
    medians = [place(float(np.median(values))) for values in samples]
    p95s = [place(float(np.percentile(values, 95))) for values in samples]
    axes.scatter(
        medians, positions, color="white", edgecolor="#222", zorder=3, s=28, label="median"
    )
    axes.scatter(p95s, positions, color="#b3261e", marker="|", s=180, zorder=3, label="p95")

    axes.set_yticks(positions, [run.label for run in runs])
    axes.set_xlabel("Latency per query (ms)" + (", log scale" if log_scale else ""))
    if log_scale:
        # The axis is linear in log10(ms), so the ticks are placed by hand and
        # labelled in milliseconds — matplotlib's log formatter would print
        # 4x10^0 where the reader wants 4.
        shown = _latency_ticks(samples)
        axes.set_xticks([place(tick) for tick in shown], [str(tick) for tick in shown])

    axes.set_title(f"Query latency distribution ({len(samples[0])} queries)")
    axes.legend(loc="lower right", frameon=True)
    axes.grid(axis="y", visible=False)
    return figure


def latency_boxplot(runs: Sequence[RunResult], *, log_scale: bool = True) -> Figure:
    """The same latencies as quartiles and whiskers, one box per config.

    Kept alongside :func:`latency_distribution` rather than instead of it. The
    violin shows the shape of the distribution, which is what tells a bimodal
    config apart from a merely wide one; the box shows the five numbers a
    reader can quote, and is what the spec asks for. They are drawn from the
    same samples in the same order, so the pair is consistent by construction.

    Whiskers reach the 5th and 95th percentiles rather than the conventional
    1.5x IQR. At this sample size the IQR rule flags dozens of ordinary queries
    as outliers and buries the box in dots; p5–p95 says something a reader
    wants anyway — where all but the extreme twentieth of queries landed.

    Unlike the violin this one uses a real log axis: quartiles are order
    statistics, so they are the same points whether they are computed before or
    after the transform, and there is no density estimate to distort.
    """
    apply_style()
    runs = _timed(runs)
    samples = [run.latencies() for run in runs]

    figure, axes = plt.subplots(figsize=(9.5, 0.5 * len(runs) + 3))
    boxes = axes.boxplot(
        samples,
        orientation="horizontal",
        widths=0.6,
        patch_artist=True,
        whis=(5, 95),
        showfliers=False,
        medianprops={"color": "#222", "linewidth": 1.6},
    )
    for patch, color in zip(
        boxes["boxes"], sns.color_palette(PALETTE, n_colors=max(len(runs), 3)), strict=False
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    axes.set_yticks(np.arange(1, len(runs) + 1), [run.label for run in runs])
    axes.set_xlabel("Latency per query (ms)" + (", log scale" if log_scale else ""))
    if log_scale:
        axes.set_xscale("log")
        shown = _latency_ticks(samples)
        axes.set_xticks(shown, [str(tick) for tick in shown])
        axes.minorticks_off()

    axes.set_title(
        f"Query latency by configuration ({len(samples[0])} queries, box = IQR, whiskers = p5–p95)"
    )
    axes.grid(axis="y", visible=False)
    return figure


# --------------------------------------------------------------------------- #
# 6. Hybrid fusion weight sweep
# --------------------------------------------------------------------------- #


def alpha_sweep(
    runs: Sequence[RunResult],
    metrics: Sequence[str] = ("ndcg@5", "hit_rate@5", "mrr"),
) -> Figure:
    """Score against the hybrid weight, from pure sparse to pure dense.

    The endpoints are the single-retriever baselines: alpha 0 is BM25 alone and
    alpha 1 is dense alone, so the sweep shows both and the interior answers
    whether combining them beats either. Runs using rank fusion instead of score
    fusion have no comparable weight and are drawn as separate markers.

    Narrowed to a single (chunker, embedder) first. Any grid cell using
    ``hybrid:0.5`` shares an alpha with the sweep's midpoint while retrieving
    over a different corpus, and left in, they stack into a vertical spike at
    0.5 that reads as variance in the weight rather than what it is.
    """
    apply_style()
    hybrid = [run for run in runs if run.alpha is not None]
    corpora: dict[tuple[str, str], set[float]] = {}
    for run in hybrid:
        corpora.setdefault((run.chunker, run.embedder), set()).add(run.alpha)
    if not corpora or max(len(values) for values in corpora.values()) < 2:
        raise ValueError(
            "Fewer than two hybrid weights scored on any one chunker and embedder, "
            "so there is no sweep to draw; run config/experiments/alpha_sweep.yaml"
        )
    corpus = max(corpora, key=lambda key: len(corpora[key]))
    hybrid = [run for run in hybrid if (run.chunker, run.embedder) == corpus]

    weighted = sorted(
        (run for run in hybrid if run.fusion == "weighted"), key=lambda run: run.alpha
    )
    rrf = [run for run in hybrid if run.fusion == "rrf"]

    alphas = [run.alpha for run in weighted]
    figure, axes = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette(PALETTE, n_colors=max(len(metrics), 3))

    for color, metric in zip(colors, metrics, strict=False):
        values = [run.metric(metric) for run in weighted]
        axes.plot(alphas, values, marker="o", linewidth=2, color=color, label=_pretty(metric))
        peak = int(np.argmax(values))
        axes.annotate(
            f"{values[peak]:.3f} @ α={alphas[peak]:g}",
            xy=(alphas[peak], values[peak]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=color,
        )

    for run in rrf:
        axes.axhline(
            run.metric(metrics[0]),
            linestyle="--",
            linewidth=1.2,
            color="#b3261e",
            label=f"RRF — {_pretty(metrics[0])}",
        )

    axes.set_xlabel("α — share of the hybrid score from dense retrieval")
    axes.set_ylabel("Score")
    axes.set_xticks(alphas)
    axes.margins(y=0.12)  # room for the peak labels above the topmost line
    axes.set_title("Hybrid fusion weight sweep")
    axes.legend(loc="lower center", ncol=2, frameon=True)
    figure.text(
        0.5, -0.04, f"{weighted[0].chunker} | {weighted[0].embedder}", ha="center", fontsize=9
    )
    return figure
