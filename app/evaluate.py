#!/usr/bin/env python3
"""Score a retrieval config against the benchmark's qrels.

Stage 5. Runs every scoreable query through one or more retrievers and writes a
result JSON carrying the full configuration alongside the numbers — a metric
without the config that produced it cannot be compared to anything.

Usage:
    python app/evaluate.py                                # default config, all retrievers
    python app/evaluate.py -s sentence:5:1 -r hybrid
    python app/evaluate.py --alpha-sweep 0.3 0.5 0.7
    python app/evaluate.py --all-specs                    # every chunker on disk

Results land in experiments/results/<timestamp>_<config>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # `python app/evaluate.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.base import BaseRetriever  # noqa: E402
from app.rag.chunking import config_slug, load_chunks  # noqa: E402
from app.rag.embedding import DEFAULT_EMBEDDER, EMBEDDERS, get_embedder  # noqa: E402
from app.rag.evaluation import DEFAULT_KS, EvaluationResult, build_relevance, evaluate  # noqa: E402
from app.rag.evaluation.qrels import load_benchmark  # noqa: E402
from app.rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from app.rag.stores import config_id, index_dir, open_store  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHUNKS = DATA_DIR / "chunks"
DEFAULT_INDICES = DATA_DIR / "indices"
DEFAULT_BENCHMARK = DATA_DIR / "open_ragbench/pdf/arxiv"
DEFAULT_RESULTS = PROJECT_ROOT / "experiments/results"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def available_specs(chunks_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        try:
            found[json.loads(path.read_text())["chunker"]] = path
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return found


def build_retrievers(
    spec: str,
    embedder_name: str,
    chunks: list,
    indices_dir: Path,
    *,
    alphas: list[float],
    fusion: str,
) -> dict[str, BaseRetriever]:
    """Every retriever to be scored on this (chunker, embedder) pair."""
    target = index_dir(indices_dir, spec, embedder_name)
    if not target.is_dir():
        raise SystemExit(
            f"No index at {_display(target)} — run `python app/index.py {spec} -e {embedder_name}`."
        )

    dense = DenseRetriever(open_store(target), get_embedder(embedder_name))
    sparse = BM25Retriever(chunks)

    retrievers: dict[str, BaseRetriever] = {"dense": dense, "bm25": sparse}
    for alpha in alphas:
        retrievers[f"hybrid@{alpha:g}"] = HybridRetriever(dense, sparse, alpha=alpha, fusion=fusion)
    return retrievers


def write_result(
    results_dir: Path,
    spec: str,
    embedder_name: str,
    fusion: str,
    scored: dict[str, EvaluationResult],
    relevance_summary: str,
    ks: tuple[int, ...],
) -> Path:
    """One JSON per run, config included, so a results table can be rebuilt."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = results_dir / f"{stamp}_{config_id(spec, embedder_name)}.json"

    target.write_text(
        json.dumps(
            {
                "config": {
                    "chunker": spec,
                    "embedder": embedder_name,
                    "fusion": fusion,
                    "ks": list(ks),
                },
                "corpus": relevance_summary,
                "retrievers": {
                    name: {
                        "num_queries": result.num_queries,
                        "mean_latency_ms": round(result.mean_latency_ms, 2),
                        "mean_relevant_chunks": round(result.mean_relevant_chunks, 1),
                        "means": {k: round(v, 4) for k, v in result.means.items()},
                        "per_query": result.per_query,
                    }
                    for name, result in scored.items()
                },
            },
            indent=2,
        )
    )
    return target


def print_table(scored: dict[str, EvaluationResult], ks: tuple[int, ...]) -> None:
    """The comparison the whole stage exists to produce."""
    columns = ["hit_rate@1", "hit_rate@5", "mrr", "ndcg@5", "precision@5", f"coverage@{max(ks)}"]
    header = f"{'retriever':<16}" + "".join(f"{name:>13}" for name in columns) + f"{'ms':>8}"
    print(f"\n{header}\n{'-' * len(header)}")
    for name, result in scored.items():
        row = f"{name:<16}" + "".join(f"{result.means.get(c, 0):>13.4f}" for c in columns)
        print(f"{row}{result.mean_latency_ms:>8.0f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score retrieval configs against the benchmark qrels.",
    )
    parser.add_argument("-s", "--spec", help="chunker spec; default: every one indexed")
    parser.add_argument("--all-specs", action="store_true", help="score every chunker on disk")
    parser.add_argument("-e", "--embedder", default=DEFAULT_EMBEDDER, choices=sorted(EMBEDDERS))
    parser.add_argument(
        "-r", "--retriever", help="score only this retriever (dense, bm25, hybrid@0.5)"
    )
    parser.add_argument(
        "--alpha-sweep",
        type=float,
        nargs="+",
        default=[0.5],
        metavar="A",
        help="hybrid alphas to score; default: 0.5",
    )
    parser.add_argument("--fusion", default="weighted", choices=["weighted", "rrf"])
    parser.add_argument("--limit", type=int, help="score only the first N queries")
    parser.add_argument("-i", "--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("-x", "--indices", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("-b", "--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--no-write", action="store_true", help="print only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("evaluate")
    log.info(f"Logging to {log_file}")

    if not args.benchmark.is_dir():
        log.error(f"No benchmark at {_display(args.benchmark)}.")
        return 1
    specs = available_specs(args.chunks)
    if not specs:
        log.error(f"{_display(args.chunks)} holds no chunks — run `python app/chunk.py` first.")
        return 1

    if args.all_specs:
        targets = sorted(specs)
    elif args.spec:
        if args.spec not in specs:
            log.error(f"No chunks for {args.spec!r}. Available: {', '.join(sorted(specs))}")
            return 1
        targets = [args.spec]
    else:
        targets = [s for s in sorted(specs) if index_dir(args.indices, s, args.embedder).is_dir()]
        if not targets:
            log.error(f"No index built for {args.embedder!r} — run `python app/index.py`.")
            return 1

    queries, qrels, answers = load_benchmark(args.benchmark)

    for spec in targets:
        chunks = load_chunks(args.chunks / f"{config_slug(spec)}.json")
        relevance, report = build_relevance(chunks, queries, qrels, answers)
        if not relevance:
            log.error(f"{spec}: no scoreable queries — has `python app/align.py` been run?")
            continue
        if args.limit:
            relevance = relevance[: args.limit]

        retrievers = build_retrievers(
            spec,
            args.embedder,
            chunks,
            args.indices,
            alphas=args.alpha_sweep,
            fusion=args.fusion,
        )
        if args.retriever:
            if args.retriever not in retrievers:
                log.error(f"Unknown retriever {args.retriever!r}. Have: {', '.join(retrievers)}")
                return 1
            retrievers = {args.retriever: retrievers[args.retriever]}

        print(f"\n=== {spec} | {args.embedder} ===\n{report.summary()}")
        scored = {
            name: evaluate(retriever, relevance, ks=DEFAULT_KS)
            for name, retriever in retrievers.items()
        }
        print_table(scored, DEFAULT_KS)

        if not args.no_write:
            path = write_result(
                args.out, spec, args.embedder, args.fusion, scored, report.summary(), DEFAULT_KS
            )
            log.info(f"Wrote {_display(path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
