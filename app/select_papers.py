#!/usr/bin/env python3
"""Choose the working set: a seeded random sample of benchmark papers.

The full Open RAG Benchmark is already on disk — 1,000 PDFs and 1,000 parsed
`corpus/*.json`, of which **396 papers carry the 3,045 qrels queries**. Nothing
downloads. What this stage decides is which subset every later number is
computed over, which makes it worth being deliberate about.

Two decisions are baked in:

* **Random, not top-by-query-count.** Picking the papers with the most queries
  correlates with long, section-dense papers, which is exactly the property
  retrieval metrics are sensitive to — it would flatter the configs that
  chunk finely, for reasons that have nothing to do with retrieval quality.

* **Seeded, and the sample is written to a manifest.** The spec's
  reproducibility check re-runs a config and expects the metrics back; that
  only holds if the corpus is identical, so the selection is recorded rather
  than merely repeatable.

Sizing is a trade against the grid, not against ingestion. Ingestion is minutes
either way; evaluation is 12 configs x N queries at ~110 ms per dense query, so
the whole 396 papers (3,045 queries) is a ~12-hour grid while 60 papers
(~450 queries) is ~10 minutes. 60 is enough that one query flipping moves
hit@5 by 0.2 points instead of the 2.8 it moved at 10 papers.

Usage:
    python app/select_papers.py                    # 60 papers, seed 7
    python app/select_papers.py -n 100 --seed 42
    python app/select_papers.py --dry-run          # report, copy nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):  # `python app/select_papers.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.evaluation.qrels import load_benchmark  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BENCHMARK_DIR = DATA_DIR / "open_ragbench" / "pdf" / "arxiv"
PDF_DIR = DATA_DIR / "open_ragbench" / "data" / "papers"
WORKING_SET = DATA_DIR / "working_set"

MANIFEST_FILE = "manifest.json"

DEFAULT_SIZE = 60
DEFAULT_SEED = 7


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def eligible_papers(qrels: dict, pdf_dir: Path, corpus_dir: Path) -> dict[str, int]:
    """Papers that can actually be scored, mapped to their query count.

    A paper qualifies only with all three of a PDF (to ingest), a corpus JSON
    (to align sections against) and at least one query (to be scored on).
    Anything missing one of those adds index size and retrieval competition
    but can never appear in a metric.
    """
    pdfs = {path.stem for path in pdf_dir.glob("*.pdf")}
    corpus = {path.stem for path in corpus_dir.glob("*.json")}
    queries_per_paper = Counter(label["doc_id"] for label in qrels.values())
    return {
        paper_id: count
        for paper_id, count in queries_per_paper.items()
        if paper_id in pdfs and paper_id in corpus
    }


def sample_papers(eligible: dict[str, int], size: int, seed: int) -> list[str]:
    """Draw ``size`` papers, reproducibly.

    Sorted before sampling because ``random.sample`` walks the sequence it is
    given: an unordered set would make the seed meaningless across runs.
    """
    candidates = sorted(eligible)
    if size >= len(candidates):
        log.info(f"Requested {size} papers, only {len(candidates)} eligible — taking all.")
        return candidates
    return sorted(random.Random(seed).sample(candidates, size))


# --------------------------------------------------------------------------- #
# Materializing the working set
# --------------------------------------------------------------------------- #


def sync_working_set(
    papers: list[str], pdf_dir: Path, target: Path, dry_run: bool = False
) -> tuple[int, int, int]:
    """Make ``target`` hold exactly ``papers``. Returns (copied, kept, removed).

    Removal is safe: this directory is a working copy, and every file in it
    still exists under ``pdf_dir``. Leaving strays behind would be the actual
    hazard — they would be extracted, chunked and indexed, competing in every
    retrieval without ever being scoreable.
    """
    target.mkdir(parents=True, exist_ok=True)
    wanted = set(papers)
    present = {path.stem: path for path in target.glob("*.pdf")}

    copied = kept = removed = 0
    for stem, path in sorted(present.items()):
        if stem not in wanted:
            log.info(f"  - {path.name} (not in this sample)")
            if not dry_run:
                path.unlink()
            removed += 1

    for paper_id in papers:
        if paper_id in present:
            kept += 1
            continue
        if not dry_run:
            shutil.copy2(pdf_dir / f"{paper_id}.pdf", target / f"{paper_id}.pdf")
        copied += 1

    return copied, kept, removed


def write_manifest(
    path: Path, papers: list[str], eligible: dict[str, int], seed: int, queries: int
) -> None:
    """Record the sample so a later run can be compared against this one."""
    path.write_text(
        json.dumps(
            {
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "seed": seed,
                "num_papers": len(papers),
                "num_queries": queries,
                "eligible_papers": len(eligible),
                "papers": papers,
            },
            indent=2,
        )
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample benchmark papers into data/working_set/.",
        epilog="Every paper is already on disk; this only chooses which ones to ingest.",
    )
    parser.add_argument(
        "-n", "--num-papers", type=int, default=DEFAULT_SIZE, help=f"default: {DEFAULT_SIZE}"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"sampling seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=BENCHMARK_DIR, help=f"default: {BENCHMARK_DIR}"
    )
    parser.add_argument("--pdfs", type=Path, default=PDF_DIR, help=f"default: {PDF_DIR}")
    parser.add_argument(
        "-o", "--out", type=Path, default=WORKING_SET, help=f"default: {WORKING_SET}"
    )
    parser.add_argument("--dry-run", action="store_true", help="report the sample, change nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("select_papers")
    log.info(f"Logging to {log_file}")

    if not args.benchmark.is_dir():
        log.error(f"No benchmark at {args.benchmark}")
        return 1

    _, qrels, _ = load_benchmark(args.benchmark)
    eligible = eligible_papers(qrels, args.pdfs, args.benchmark / "corpus")
    if not eligible:
        log.error("No paper has a PDF, a corpus JSON and a query all at once.")
        return 1

    papers = sample_papers(eligible, args.num_papers, args.seed)
    queries = sum(eligible[paper_id] for paper_id in papers)

    log.info(
        f"\n{len(eligible)} eligible papers carrying {sum(eligible.values()):,} queries; "
        f"sampled {len(papers)} (seed {args.seed}) -> {queries:,} queries "
        f"({queries / len(papers):.1f} per paper)"
    )

    copied, kept, removed = sync_working_set(papers, args.pdfs, args.out, args.dry_run)
    if not args.dry_run:
        write_manifest(args.out / MANIFEST_FILE, papers, eligible, args.seed, queries)

    verb = "would be" if args.dry_run else "in"
    log.info(
        f"\n{_display(args.out)}: {copied} copied, {kept} already present, "
        f"{removed} removed — {len(papers)} papers {verb} the working set."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
