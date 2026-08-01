#!/usr/bin/env python3
"""Run the chunking stage on its own: data/extracted/ -> data/chunks/<config>.json.

Stage 2 of `app/ingest.py`, split out for the same reason as `app/extract.py`:
chunking runs once per chunker config and is then reused by every embedding
model in the grid, so it is worth doing — and inspecting — on its own.

Usage:
    python app/chunk.py                             # interactive menu
    python app/chunk.py fixed_size:512:128
    python app/chunk.py recursive:256:64 --in data/extracted --out data/chunks

The spec is `name[:arg[:arg]]`, with arguments in each strategy's own
vocabulary — `fixed_size:512:128` is (chunk_size, overlap), `sentence:5:1` is
(sentences_per_chunk, overlap).
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from inspect import Parameter, signature
from pathlib import Path

if __package__ in (None, ""):  # `python app/chunk.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.base import BaseChunker  # noqa: E402
from app.rag.chunking import CHUNKERS, chunker_from_spec, config_slug, save_chunks  # noqa: E402
from app.rag.loaders import load_documents  # noqa: E402
from app.rag.models import Chunk, Document  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_IN = DATA_DIR / "extracted"
DEFAULT_OUT = DATA_DIR / "chunks"


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


def choose_spec() -> str:
    """Pick a strategy, then fill in its parameters, and return the spec string."""
    names = sorted(CHUNKERS)

    print("\nWhich chunking strategy?")
    for number, name in enumerate(names, start=1):
        summary = (CHUNKERS[name].__doc__ or "").strip().splitlines()[0]
        print(f"  {number}. {name:<16} {summary}")

    while True:
        choice = _ask("Choice", "1")
        if choice in names:
            name = choice
            break
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            name = names[int(choice) - 1]
            break
        print("  Pick a listed number or name.")

    # Ask for each constructor parameter in turn, defaults preselected, so a
    # new parameter on any chunker shows up here without touching this code.
    params = [
        param
        for param in signature(CHUNKERS[name].__init__).parameters.values()
        if param.name != "self" and param.default is not Parameter.empty
    ]
    print(f"\n{name} parameters (Enter accepts the default):")
    values = [_ask(f"  {param.name}", str(param.default)) for param in params]

    # Trim trailing defaults so the spec stays readable in filenames and logs.
    while values and values[-1] == str(params[len(values) - 1].default):
        values.pop()
    return ":".join([name, *values])


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def chunk_documents(documents: list[Document], chunker: BaseChunker) -> list[Chunk]:
    """Chunk every document, logging one line each."""
    chunks: list[Chunk] = []
    for number, document in enumerate(documents, start=1):
        produced = chunker.chunk(document)
        prefix = f"[{number}/{len(documents)}] {document.metadata.source}:"
        if produced:
            tokens = [getattr(c.metadata, "num_tokens", 0) or 0 for c in produced]
            log.info(f"{prefix} {len(produced)} chunks, mean {statistics.mean(tokens):.0f} tokens")
        else:
            log.info(f"{prefix} no chunks (empty text?)")
        chunks.extend(produced)
    return chunks


def summarize(chunks: list[Chunk]) -> str:
    """Size distribution — the number that decides whether a config is sane."""
    if not chunks:
        return "no chunks"
    tokens = sorted(getattr(c.metadata, "num_tokens", 0) or 0 for c in chunks)
    chars = [len(c.content) for c in chunks]
    return (
        f"{len(chunks)} chunks | tokens min {tokens[0]} "
        f"p50 {tokens[len(tokens) // 2]} max {tokens[-1]} "
        f"mean {statistics.mean(tokens):.0f} | mean {statistics.mean(chars):.0f} chars"
    )


def verify_offsets(documents: list[Document], chunks: list[Chunk]) -> int:
    """Count chunks whose offsets do not reproduce their own text.

    The invariant every later stage assumes, checked on real output rather than
    only in tests: a citation resolves by slicing the document with these
    offsets, so a drift here surfaces as a subtly wrong quote much later.
    """
    by_id = {document.id: document for document in documents}
    broken = 0
    for chunk in chunks:
        document = by_id.get(chunk.metadata.document_id)
        if document is None:
            broken += 1
            continue
        span = document.content[chunk.metadata.start_char : chunk.metadata.end_char]
        broken += span != chunk.content
    return broken


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk extracted Documents into data/chunks/<config>.json.",
        epilog="With no SPEC, an interactive menu asks which strategy and its parameters.",
    )
    parser.add_argument(
        "spec", nargs="?", help="chunker spec, e.g. fixed_size:512:128 or sentence:5:1"
    )
    parser.add_argument(
        "-i", "--in", dest="input", type=Path, default=DEFAULT_IN, help=f"default: {DEFAULT_IN}"
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}"
    )
    parser.add_argument("--limit", type=int, help="stop after N documents")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interactive = args.spec is None

    log_file = setup_logging("chunk")
    log.info(f"Logging to {log_file}")

    if not args.input.is_dir():
        log.error(f"No extracted documents at {args.input} — run `python app/extract.py` first.")
        return 1
    documents = load_documents(args.input)
    if not documents:
        log.error(f"{args.input} holds no documents — run `python app/extract.py` first.")
        return 1
    if args.limit:
        documents = documents[: args.limit]

    spec = args.spec or choose_spec()
    try:
        chunker = chunker_from_spec(spec)
    except (ValueError, TypeError) as exc:
        log.error(f"Bad chunker spec {spec!r}: {exc}")
        return 1

    target = args.out / f"{config_slug(spec)}.json"
    log.info(f"\n{len(documents)} document(s) via {chunker!r} -> {_display(target)}")
    if interactive and _ask("Proceed? (y/n)", "y").lower() not in {"y", "yes"}:
        log.info("Cancelled.")
        return 0

    chunks = chunk_documents(documents, chunker)
    if not chunks:
        log.error("No chunks produced.")
        return 1

    broken = verify_offsets(documents, chunks)
    if broken:
        # Loud, and a failing exit code: every downstream citation is suspect.
        log.error(f"OFFSET MISMATCH on {broken}/{len(chunks)} chunks — do not use this output.")

    path = save_chunks(chunks, args.out, spec)
    log.info(f"\n{summarize(chunks)}")
    log.info(f"Wrote {_display(path)}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
