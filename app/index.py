#!/usr/bin/env python3
"""Run the indexing stage on its own: data/chunks/<config>.json -> data/indices/<config_id>/.

Stage 3, split out for the same reason as `app/extract.py` and `app/chunk.py`:
embedding a corpus costs real time and a model download, and the result is
reused by every retrieval experiment run against it.

Usage:
    python app/index.py                                  # interactive menu
    python app/index.py fixed_size:512:128
    python app/index.py fixed_size:512:128 -e mpnet
    python app/index.py fixed_size:512:128 -q "how are cells tracked?"

`-q` embeds a query against the freshly built index and prints the top hits —
a smoke test that the vectors mean something, before any retriever exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # `python app/index.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.base import BaseEmbedder  # noqa: E402
from app.rag.chunking import chunk_file, load_chunks  # noqa: E402
from app.rag.embedding import DEFAULT_EMBEDDER, EMBEDDERS, get_embedder  # noqa: E402
from app.rag.models import Chunk  # noqa: E402
from app.rag.stores import get_store, index_dir  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_IN = DATA_DIR / "chunks"
DEFAULT_OUT = DATA_DIR / "indices"


def _display(path: Path) -> str:
    """Project-relative when it can be, absolute otherwise."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _ask(prompt: str, default: str) -> str:
    """Prompt with a default. Ctrl-C / EOF reads as 'cancel', not a traceback."""
    try:
        return input(f"{prompt} [{default}]: ").strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Cancelled.") from None


# --------------------------------------------------------------------------- #
# Interactive menu
# --------------------------------------------------------------------------- #


def available_specs(chunks_dir: Path) -> dict[str, Path]:
    """Chunker specs already on disk, keyed by spec, from data/chunks/*.json."""
    import json

    found: dict[str, Path] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        try:
            # The spec is stored verbatim next to the lossy slug, so the menu
            # can offer `fixed_size:512:25%` rather than `fixed_size_512_25pct`.
            found[json.loads(path.read_text())["chunker"]] = path
        except (json.JSONDecodeError, KeyError, OSError):
            log.warning(f"Skipping unreadable chunk file {_display(path)}")
    return found


def _choose(prompt: str, options: list[str]) -> str:
    """Numbered pick-one, accepting either the number or the name."""
    print(f"\n{prompt}")
    for number, option in enumerate(options, start=1):
        print(f"  {number}. {option}")
    while True:
        choice = _ask("Choice", "1")
        if choice in options:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("  Pick a listed number or name.")


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def embed_chunks(chunks: list[Chunk], embedder: BaseEmbedder, batch: int = 512) -> float:
    """Attach embeddings in batches, logging progress. Returns seconds taken.

    Batched so a long corpus reports progress and peak memory stays bounded;
    the embedder does its own micro-batching inside each call.
    """
    started = time.perf_counter()
    for offset in range(0, len(chunks), batch):
        window = chunks[offset : offset + batch]
        embedder.embed_chunks(window)
        done = offset + len(window)
        log.info(f"  embedded {done}/{len(chunks)}")
    return time.perf_counter() - started


def preview(store, embedder: BaseEmbedder, query: str, top_k: int = 5) -> None:
    """Search the index and print what came back — does this look like sense?"""
    results = store.search(embedder.embed_query(query), top_k=top_k)
    print(f"\nTop {len(results)} for {query!r}:")
    for rank, result in enumerate(results, start=1):
        snippet = " ".join(result.chunk.content.split())[:110]
        page = result.chunk.metadata.page_number
        where = f"{result.chunk.metadata.source} p{page}" if page else result.chunk.metadata.source
        print(f"  {rank}. {result.score:.3f}  {where}\n     {snippet}...")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunks and build a FAISS index under data/indices/<config_id>/.",
        epilog="With no SPEC, an interactive menu lists the chunk files already on disk.",
    )
    parser.add_argument("spec", nargs="?", help="chunker spec, e.g. fixed_size:512:128")
    parser.add_argument(
        "-e",
        "--embedder",
        default=DEFAULT_EMBEDDER,
        choices=sorted(EMBEDDERS),
        help=f"default: {DEFAULT_EMBEDDER}",
    )
    parser.add_argument(
        "-i", "--in", dest="input", type=Path, default=DEFAULT_IN, help=f"default: {DEFAULT_IN}"
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}"
    )
    parser.add_argument("-q", "--query", help="search the new index and print the top hits")
    parser.add_argument("--limit", type=int, help="stop after N chunks (for a quick trial)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the index exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interactive = args.spec is None

    log_file = setup_logging("index")
    log.info(f"Logging to {log_file}")

    if not args.input.is_dir():
        log.error(f"No chunks at {args.input} — run `python app/chunk.py` first.")
        return 1
    specs = available_specs(args.input)
    if not specs:
        log.error(f"{args.input} holds no chunk files — run `python app/chunk.py` first.")
        return 1

    spec = args.spec or _choose("Which chunker config?", sorted(specs))
    embedder_name = (
        args.embedder if not interactive else _choose("Which embedder?", sorted(EMBEDDERS))
    )

    # Resolved through chunk_file, not the `chunker` field: a config that names
    # an embedder has two files carrying the same spec, so that field cannot
    # tell them apart.
    path = chunk_file(args.input, spec, embedder_name)
    if not path.is_file():
        log.error(
            f"No chunks at {_display(path)} — run `python app/chunk.py {spec} -e {embedder_name}`."
        )
        return 1

    target = index_dir(args.out, spec, embedder_name)
    if target.exists() and not args.force:
        log.info(f"{_display(target)} already exists — pass --force to rebuild.")
        return 0

    chunks = load_chunks(path)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        log.error(f"{_display(path)} holds no chunks.")
        return 1

    embedder = get_embedder(embedder_name)
    log.info(f"\n{len(chunks)} chunks | {embedder!r} -> {_display(target)}")
    if interactive and _ask("Proceed? (y/n)", "y").lower() not in {"y", "yes"}:
        log.info("Cancelled.")
        return 0

    elapsed = embed_chunks(chunks, embedder)
    log.info(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s ({len(chunks) / elapsed:.0f}/s)")

    store = get_store()
    store.add(chunks)
    store.save(target)
    log.info(f"\n{store!r} -> {_display(target)}")

    query = args.query
    if interactive and not query:
        query = _ask("Try a query? (blank to skip)", "")
    if query:
        preview(store, embedder, query)
    return 0


if __name__ == "__main__":
    sys.exit(main())
