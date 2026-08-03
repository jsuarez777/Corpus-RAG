#!/usr/bin/env python3
"""Label chunks with the corpus sections they overlap, so qrels can score them.

Runs between chunking and indexing. Rewrites data/chunks/<config>.json in place
with ``section_indices`` filled in, and reports the coverage figure that belongs
in RESULTS.md — alignment is a measured property of the corpus, not a step that
either works or throws.

Usage:
    python app/align.py                       # interactive menu
    python app/align.py fixed_size:512:128
    python app/align.py --all                 # every chunk config on disk
    python app/align.py --all --dry-run       # report coverage, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python app/align.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import load_chunks, save_chunks  # noqa: E402
from app.rag.evaluation import align_corpus  # noqa: E402
from app.rag.loaders import load_documents  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DOCS = DATA_DIR / "extracted"
DEFAULT_CHUNKS = DATA_DIR / "chunks"
DEFAULT_CORPUS = DATA_DIR / "open_ragbench/pdf/arxiv/corpus"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _ask(prompt: str, default: str) -> str:
    try:
        return input(f"{prompt} [{default}]: ").strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Cancelled.") from None


def available_specs(chunks_dir: Path) -> dict[str, Path]:
    """Chunker specs on disk, keyed by the verbatim spec string."""
    found: dict[str, Path] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        try:
            found[json.loads(path.read_text())["chunker"]] = path
        except (json.JSONDecodeError, KeyError, OSError):
            log.warning(f"Skipping unreadable chunk file {_display(path)}")
    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label chunks with the corpus sections they overlap.",
        epilog="With no SPEC and no --all, an interactive menu lists what is on disk.",
    )
    parser.add_argument("spec", nargs="?", help="chunker spec, e.g. fixed_size:512:128")
    parser.add_argument("--all", action="store_true", help="align every chunk config on disk")
    parser.add_argument("--dry-run", action="store_true", help="report coverage, write nothing")
    parser.add_argument("-d", "--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("-i", "--in", dest="chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("-c", "--corpus", type=Path, default=DEFAULT_CORPUS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("align")
    log.info(f"Logging to {log_file}")

    if not args.corpus.is_dir():
        log.error(f"No corpus at {_display(args.corpus)} — alignment needs the labelled sections.")
        return 1
    documents = load_documents(args.docs)
    if not documents:
        log.error(f"No documents at {_display(args.docs)} — run `python app/extract.py` first.")
        return 1

    specs = available_specs(args.chunks)
    if not specs:
        log.error(f"{_display(args.chunks)} holds no chunk files — run `python app/chunk.py`.")
        return 1

    if args.all:
        targets = sorted(specs)
    elif args.spec:
        if args.spec not in specs:
            log.error(f"No chunks for {args.spec!r}. Available: {', '.join(sorted(specs))}")
            return 1
        targets = [args.spec]
    else:
        options = sorted(specs)
        print("\nWhich chunk config?")
        for number, option in enumerate(options, start=1):
            print(f"  {number}. {option}")
        print(f"  {len(options) + 1}. all")
        choice = _ask("Choice", "1")
        targets = (
            options
            if choice == str(len(options) + 1)
            else [options[int(choice) - 1]]
            if choice.isdigit() and 1 <= int(choice) <= len(options)
            else [choice]
            if choice in specs
            else []
        )
        if not targets:
            log.error("Not a listed option.")
            return 1

    log.info(f"{len(documents)} documents | corpus {_display(args.corpus)}\n")

    unlabelled_total = 0
    for spec in targets:
        chunks = load_chunks(specs[spec])
        report = align_corpus(documents, chunks, args.corpus)
        log.info(f"{spec}\n  {report.summary()}")

        unlabelled = report.chunks_total - report.chunks_labelled
        unlabelled_total += unlabelled
        if report.papers_missing_corpus:
            log.warning(
                f"  {report.papers_missing_corpus} paper(s) have no corpus entry — "
                "their chunks can never be scored"
            )
        if unlabelled:
            # Not fatal: unlabelled chunks stay in the index as distractors.
            log.warning(f"  {unlabelled} chunk(s) matched no section")

        if not args.dry_run:
            save_chunks(chunks, args.chunks, spec)
            log.info(f"  wrote {_display(specs[spec])}")

    if args.dry_run:
        log.info("\nDry run — nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
