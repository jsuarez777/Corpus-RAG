#!/usr/bin/env python3
"""Run the loader stage on its own: PDFs -> data/extracted/<paper_id>.json.

This is stage 1 of `app/ingest.py` split out as a one-off, because extraction
is the slow, config-independent step: it runs once per corpus, and every
chunker/embedder config downstream reads its cached output.

Usage:
    python app/extract.py                          # interactive menu
    python app/extract.py data/working_set         # every PDF in a folder
    python app/extract.py paper.pdf --loader pymupdf --force

With no path argument it prompts, mp3-style, so it is usable without
remembering any flags.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python app/extract.py` runs this as a script
    # Python puts only the script's own directory on the path; `app.rag` needs
    # the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.base import BaseLoader  # noqa: E402
from app.rag.loaders import DEFAULT_LOADER, LOADERS, get_loader, save_document  # noqa: E402
from app.rag.preprocessing import DEFAULT_PREPROCESSOR, PREPROCESSORS  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT = DATA_DIR / "extracted"

# Menu shortcuts, in preference order. Only ones that exist get listed.
SOURCES: list[tuple[str, Path]] = [
    ("Working set", DATA_DIR / "working_set"),
    ("Downloaded papers", DATA_DIR / "papers"),
    ("Raw PDFs", DATA_DIR / "raw"),
    ("Benchmark papers", DATA_DIR / "open_ragbench" / "data" / "papers"),
]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def collect_pdfs(path: Path, recursive: bool = False) -> list[Path]:
    """Every PDF at ``path``, which may be a single file or a folder."""
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    # Glob everything and filter on the lowercased suffix: glob patterns are
    # case-sensitive on Linux, and .PDF files are common enough to matter.
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")


# --------------------------------------------------------------------------- #
# Interactive menu
# --------------------------------------------------------------------------- #


def _display(path: Path) -> str:
    """Project-relative when it can be, absolute otherwise — ``-o`` may point
    anywhere on disk."""
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


def choose_source(recursive: bool) -> Path:
    """Numbered menu of the known PDF folders, plus a free-text path."""
    options = [(label, path) for label, path in SOURCES if path.is_dir()]

    print("\nWhich PDFs?")
    for number, (label, path) in enumerate(options, start=1):
        found = len(collect_pdfs(path, recursive))
        print(f"  {number}. {label:<18} {_display(path):<42} ({found} PDF{'s' * (found != 1)})")
    print(f"  {len(options) + 1}. Enter a path")

    while True:
        choice = _ask("Choice", "1")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][1]
        # Anything that is not a menu number is treated as a path, so pasting
        # one works whether or not you picked the "enter a path" item first.
        raw = _ask("Path to a PDF or folder", "") if choice == str(len(options) + 1) else choice
        candidate = Path(raw).expanduser().resolve()
        if candidate.exists():
            return candidate
        print(f"  Not found: {candidate}")


def choose_loader() -> str:
    """Numbered menu of the registered loaders."""
    names = sorted(LOADERS)
    if len(names) == 1:
        return names[0]

    print("\nWhich loader?")
    for number, name in enumerate(names, start=1):
        summary = (LOADERS[name].__doc__ or "").strip().splitlines()[0]
        print(f"  {number}. {name:<14} {summary}")

    while True:
        choice = _ask("Choice", "1")
        if choice in names:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("  Pick a listed number or name.")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract(
    pdfs: list[Path], loader: BaseLoader, out_dir: Path, force: bool = False
) -> tuple[int, int, int]:
    """Extract each PDF to ``out_dir``. Returns (written, skipped, failed)."""
    written = skipped = failed = 0

    for number, pdf in enumerate(pdfs, start=1):
        target = out_dir / f"{pdf.stem}.json"
        if target.exists() and not force:
            log.info(f"[{number}/{len(pdfs)}] {pdf.name}: cached, skipping (--force to redo)")
            skipped += 1
            continue

        try:
            document = loader.load(pdf)
        except Exception as exc:  # one unreadable PDF must not end the run
            log.error(f"[{number}/{len(pdfs)}] {pdf.name}: FAILED ({type(exc).__name__}: {exc})")
            failed += 1
            continue

        path = save_document(document, out_dir)
        log.info(
            f"[{number}/{len(pdfs)}] {pdf.name}: "
            f"{document.metadata.page_count} pages, {len(document.content):,} chars "
            f"-> {_display(path)}"
        )
        written += 1

    return written, skipped, failed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PDFs into Documents under data/extracted/.",
        epilog="With no PATH, an interactive menu asks which PDFs and which loader.",
    )
    parser.add_argument("path", nargs="?", type=Path, help="a PDF file or a folder of PDFs")
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT, help=f"output dir (default: {DEFAULT_OUT})"
    )
    parser.add_argument("--loader", choices=sorted(LOADERS), help=f"default: {DEFAULT_LOADER}")
    parser.add_argument(
        "--clean",
        choices=list(PREPROCESSORS),
        default=DEFAULT_PREPROCESSOR,
        help=f"text cleaning applied per page (default: {DEFAULT_PREPROCESSOR})",
    )
    parser.add_argument(
        "--layout",
        choices=["blocks", "text"],
        default="blocks",
        help="blocks keeps paragraph boundaries, which reflow needs (default: blocks)",
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="descend into subfolders")
    parser.add_argument("--limit", type=int, help="stop after N PDFs")
    parser.add_argument("--force", action="store_true", help="re-extract PDFs already cached")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interactive = args.path is None

    log_file = setup_logging("extract")
    log.info(f"Logging to {log_file}")

    source = args.path.expanduser().resolve() if args.path else choose_source(args.recursive)
    if not source.exists():
        log.error(f"No such path: {source}")
        return 1

    pdfs = collect_pdfs(source, args.recursive)
    if not pdfs:
        scope = "folder or its subfolders" if args.recursive else "folder"
        log.error(f"No PDFs in that {scope}: {source}")
        return 1
    if args.limit:
        pdfs = pdfs[: args.limit]

    loader_name = args.loader or (choose_loader() if interactive else DEFAULT_LOADER)
    loader = get_loader(loader_name, preprocess=args.clean, layout=args.layout)

    log.info(
        f"\n{len(pdfs)} PDF(s) from {source} via {loader_name} "
        f"(layout={args.layout}, clean={args.clean}) -> {args.out}"
    )
    if interactive and _ask("Proceed? (y/n)", "y").lower() not in {"y", "yes"}:
        log.info("Cancelled.")
        return 0

    written, skipped, failed = extract(pdfs, loader, args.out, args.force)
    log.info(f"\nDone: {written} written, {skipped} cached, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
