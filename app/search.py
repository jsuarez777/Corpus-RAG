#!/usr/bin/env python3
"""Query an index from the command line — dense, BM25, or hybrid.

Stage 4, and the first point where the pipeline answers a question. No metrics
here: this is for looking at what comes back and deciding whether the numbers
that follow are worth trusting.

Usage:
    python app/search.py                                  # interactive
    python app/search.py "how are cells tracked?"
    python app/search.py "..." -r bm25
    python app/search.py "..." -r hybrid --alpha 0.7
    python app/search.py "..." -r hybrid --fusion rrf --compare

`--compare` runs all three retrievers on the same query and prints them side by
side, which is the quickest way to see where dense and BM25 disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # `python app/search.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.base import BaseRetriever  # noqa: E402
from app.rag.chunking import chunk_file, load_chunks  # noqa: E402
from app.rag.embedding import DEFAULT_EMBEDDER, EMBEDDERS, get_embedder  # noqa: E402
from app.rag.models import RetrievalResult  # noqa: E402
from app.rag.retrieval import (  # noqa: E402
    FUSIONS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from app.rag.stores import index_dir, open_store  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHUNKS = DATA_DIR / "chunks"
DEFAULT_INDICES = DATA_DIR / "indices"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _ask(prompt: str, default: str = "") -> str:
    try:
        return input(f"{prompt}{f' [{default}]' if default else ''}: ").strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Cancelled.") from None


def available_specs(chunks_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        try:
            found[json.loads(path.read_text())["chunker"]] = path
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return found


def build_retrievers(
    spec: str, embedder_name: str, chunks_dir: Path, indices_dir: Path
) -> dict[str, BaseRetriever]:
    """Dense, BM25 and hybrid over the same chunk set.

    BM25 is fitted here rather than loaded: it depends only on the chunker, not
    on the embedding model, so it does not belong in the per-(chunker, embedder)
    index directory, and refitting costs a fraction of a second.
    """
    target = index_dir(indices_dir, spec, embedder_name)
    if not target.is_dir():
        raise SystemExit(
            f"No index at {_display(target)} — run `python app/index.py {spec} -e {embedder_name}`."
        )

    path = chunk_file(chunks_dir, spec, embedder_name)
    if not path.is_file():
        raise SystemExit(f"No chunks at {_display(path)} — run `python app/chunk.py {spec}`.")

    embedder = get_embedder(embedder_name)
    dense = DenseRetriever(open_store(target), embedder)

    started = time.perf_counter()
    sparse = BM25Retriever(load_chunks(path))
    log.info(f"BM25 fitted in {time.perf_counter() - started:.2f}s")

    return {"dense": dense, "bm25": sparse, "hybrid": HybridRetriever(dense, sparse)}


def show(results: list[RetrievalResult], label: str, width: int = 100) -> None:
    """Print a ranked list compactly enough to compare two of them by eye."""
    print(f"\n{label} — {len(results)} result(s)")
    if not results:
        print("  (nothing matched)")
        return
    for rank, result in enumerate(results, start=1):
        metadata = result.chunk.metadata
        page = f" p{metadata.page_number}" if metadata.page_number else ""
        sections = getattr(metadata, "section_indices", []) or []
        where = f"{metadata.source}{page}"
        tag = f" §{','.join(str(s) for s in sections)}" if sections else " §-"
        snippet = " ".join(result.chunk.content.split())[:width]
        print(f"  {rank}. {result.score:7.4f}  {where}{tag}\n     {snippet}...")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search an index with dense, BM25 or hybrid retrieval.",
        epilog="With no QUERY, prompts for one and keeps asking until you enter a blank line.",
    )
    parser.add_argument("query", nargs="?", help="the question to retrieve for")
    parser.add_argument("-s", "--spec", help="chunker spec; default: the only one indexed")
    parser.add_argument("-e", "--embedder", default=DEFAULT_EMBEDDER, choices=sorted(EMBEDDERS))
    parser.add_argument("-r", "--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5, help="dense share, hybrid only")
    parser.add_argument("--fusion", default="weighted", choices=list(FUSIONS))
    parser.add_argument("--compare", action="store_true", help="run all three retrievers")
    parser.add_argument("-i", "--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("-x", "--indices", type=Path, default=DEFAULT_INDICES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("search")
    log.info(f"Logging to {log_file}")

    specs = available_specs(args.chunks)
    if not specs:
        log.error(f"{_display(args.chunks)} holds no chunks — run `python app/chunk.py` first.")
        return 1

    spec = args.spec
    if spec is None:
        indexed = [s for s in specs if index_dir(args.indices, s, args.embedder).is_dir()]
        if not indexed:
            log.error(f"No index built for embedder {args.embedder!r} — run `python app/index.py`.")
            return 1
        spec = indexed[0]
        if len(indexed) > 1:
            log.info(f"Several indices available; using {spec}. Pass --spec to choose.")

    retrievers = build_retrievers(spec, args.embedder, args.chunks, args.indices)
    retrievers["hybrid"] = HybridRetriever(
        retrievers["dense"], retrievers["bm25"], alpha=args.alpha, fusion=args.fusion
    )
    log.info(f"{spec} | {args.embedder} | {len(retrievers['bm25'])} chunks")

    queries = [args.query] if args.query else None
    while True:
        query = queries.pop(0) if queries else _ask("\nQuery (blank to quit)")
        if not query:
            return 0

        chosen = ["dense", "bm25", "hybrid"] if args.compare else [args.retriever]
        for name in chosen:
            started = time.perf_counter()
            results = retrievers[name].retrieve(query, top_k=args.top_k)
            elapsed = (time.perf_counter() - started) * 1000
            label = name if name != "hybrid" else f"hybrid (alpha={args.alpha}, {args.fusion})"
            show(results, f"{label}  [{elapsed:.0f}ms]")

        if args.query and not queries:
            return 0


if __name__ == "__main__":
    sys.exit(main())
