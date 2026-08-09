#!/usr/bin/env python3
"""Run the retrieval stage on its own: an experiment grid -> experiments/rankings/.

Stage 4a. Every scoreable query goes through every cell of the grid, and the
ranked chunk ids come out — no metrics, no LLM, no spend.

Split out because retrieval is the only stage that needs an index and an
embedder, and three later stages were each repeating it. `app/evaluate.py`
scores the ids, `app/generate_answers.py` builds prompts from them, and a
reranker experiment reorders them; before this, the answer stage re-ran the
same 488 searches the grid had already run. Retrieve once, read many.

It also decides where the process boundary falls. `app/__init__.py` keeps faiss
and torch alive together by pinning `OMP_NUM_THREADS=1`, which holds because
nothing runs concurrently. Confining both libraries to this script means the
stages that follow are pure API traffic and can use a worker pool without
touching that constraint.

Usage:
    python app/retrieve.py                              # the default grid
    python app/retrieve.py -c config/experiments/grid_12.yaml
    python app/retrieve.py -c ... --limit 20            # a cheap smoke run

Rankings land in experiments/rankings/<config_id>.json, one file per cell,
overwritten on a re-run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # `python app/retrieve.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluate import (  # noqa: E402
    DEFAULT_BENCHMARK,
    DEFAULT_CHUNKS,
    DEFAULT_INDICES,
    DEFAULT_RANKINGS,
    PROJECT_ROOT,
    _display,
    group_by_index,
    retrieve_grid,
)
from app.rag.config import load_grid, read_yaml  # noqa: E402
from app.rag.evaluation.qrels import load_benchmark  # noqa: E402
from app.rag.evaluation.rankings import write_rankings  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = PROJECT_ROOT / "config/experiments/grid_12.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve every cell of an experiment grid and save the ranked ids.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        metavar="YAML",
        help=f"experiment file; default: {DEFAULT_EXPERIMENT.name}",
    )
    parser.add_argument("--limit", type=int, help="retrieve only the first N queries")
    parser.add_argument("-i", "--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("-x", "--indices", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("-b", "--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_RANKINGS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("retrieve")
    log.info(f"Logging to {log_file}")

    if not args.config.is_file():
        log.error(f"No config at {_display(args.config)}.")
        return 1
    if not args.benchmark.is_dir():
        log.error(f"No benchmark at {_display(args.benchmark)}.")
        return 1

    configs = load_grid(args.config)
    experiment = read_yaml(args.config).get("name") or args.config.stem
    log.info(
        f"{experiment}: {len(configs)} cell(s) over {len(group_by_index(configs))} index/indices"
    )

    queries, qrels, answers = load_benchmark(args.benchmark)
    started = time.perf_counter()
    written = 0

    for config, _relevance, rankings, _corpus in retrieve_grid(
        configs,
        queries,
        qrels,
        answers,
        chunks_dir=args.chunks,
        indices_dir=args.indices,
        limit=args.limit,
    ):
        path = write_rankings(
            args.out,
            rankings,
            config_id=config.id,
            label=config.label,
            index_id=config.index_id,
            experiment=experiment,
            top_k=config.top_k,
            config=config.model_dump(mode="json"),
        )
        mean_ms = sum(r.latency_ms for r in rankings) / len(rankings) if rankings else 0.0
        print(f"  {config.retriever.spec:<14} {len(rankings)} queries | {mean_ms:.0f}ms/query")
        log.debug(f"  wrote {_display(path)}")
        written += 1

    if not written:
        log.error("No cell produced a ranking.")
        return 1

    log.info(f"\n{written} cell(s) in {time.perf_counter() - started:.0f}s")
    log.info(f"Rankings under {_display(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
