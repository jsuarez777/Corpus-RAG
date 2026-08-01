"""Text cleaning — the stage that makes extracted PDF text read like prose.

Runs inside the loader, per page, **before** ``page_starts`` is recorded, so
every offset in the system indexes the cleaned text. Cleaning a ``Document``
after the fact would silently invalidate every chunk offset and nothing would
raise; you would just get citations that quote the wrong words.

The load-bearing step is :func:`reflow`. A PDF stores glyphs at coordinates and
has no notion of a paragraph, so a sentence typeset over five lines arrives
with four newlines in it — and pysbd treats a newline as a sentence boundary.
Un-reflowed, sentence chunking segments *lines*, not sentences: measured on
this corpus, median 9 tokens per "sentence" against an expected ~28.

Reflow needs to know where paragraphs really end, which is why the loader
extracts PyMuPDF *blocks* and separates them with a blank line: raw
``get_text("text")`` output contains no blank lines at all, so there is nothing
for a text-only heuristic to key on.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Typographic ligatures, which academic PDFs emit as single glyphs. Left as an
# explicit map rather than NFKC normalization: NFKC also rewrites superscripts
# and fraction glyphs, which mangles the mathematics these papers are made of.
LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

# Zero-width and formatting characters that survive extraction and then split
# words for the tokenizer and BM25 alike.
INVISIBLE = dict.fromkeys(
    ["­", "​", "‌", "‍", "﻿", "⁠"],
    "",
)

_LIGATURE_TABLE = str.maketrans(LIGATURES | INVISIBLE)

# A hyphen at the end of a line, between two word characters: a word broken
# across the line break.
_LINE_HYPHEN = re.compile(r"(\w)[-‐‑]\n(?=\w)")

# One or more blank lines: the paragraph separator the loader emits.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACES = re.compile(r"[ \t]+")

# A finished sentence: terminal punctuation, optionally inside a closing quote
# or bracket. Anything else at a paragraph end means the thought continues.
_SENTENCE_END = re.compile(r"[.!?][\"'’”)\]]*$")


def normalize_characters(text: str) -> str:
    """Expand ligatures and drop invisible and control characters."""
    return _CONTROL.sub("", text.translate(_LIGATURE_TABLE))


def dehyphenate(text: str) -> str:
    """Rejoin a word split across a line break: ``real-\\ntime`` -> ``realtime``.

    Genuinely hyphenated compounds that happen to fall at a line end
    (``well-\\nknown``) lose their hyphen here, which is the known cost of the
    simple rule. In justified academic text the split-word case dominates by
    roughly an order of magnitude, and a lost hyphen costs far less at
    retrieval time than a word cut in half.
    """
    return _LINE_HYPHEN.sub(r"\1", text)


def reflow(text: str) -> str:
    """Join wrapped lines within each paragraph; keep paragraph breaks.

    Paragraphs are blank-line separated — which holds because the loader emits
    one blank line between PyMuPDF blocks. Without that structure this would
    join a heading onto the body text that follows it.
    """
    paragraphs = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        lines = [line.strip() for line in paragraph.splitlines()]
        joined = " ".join(line for line in lines if line)
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def merge_continuations(text: str) -> str:
    """Join paragraphs whose predecessor did not finish a sentence.

    PyMuPDF blocks are not paragraphs on this corpus — measured on one paper,
    66% are under 80 characters, and a single paragraph routinely arrives as
    several blocks (an inline equation splits it, or the line spacing shifts).
    Trusting those boundaries leaves sentences cut in half.

    The rule: a paragraph break is real only after terminal punctuation.
    A heading, which ends without punctuation, is therefore glued onto the
    paragraph below it — the accepted cost, and much the cheaper error. A few
    extra words at the head of a chunk are harmless; a sentence severed
    mid-clause is embedded as something nobody wrote.
    """
    merged: list[str] = []
    for paragraph in text.split("\n\n"):
        if merged and not _SENTENCE_END.search(merged[-1].rstrip()):
            merged[-1] = f"{merged[-1].rstrip()} {paragraph.lstrip()}"
        else:
            merged.append(paragraph)
    return "\n\n".join(merged)


def collapse_spaces(text: str) -> str:
    """Squeeze runs of spaces and tabs, which PDF layout produces in quantity."""
    return _SPACES.sub(" ", text)


def clean_text(text: str) -> str:
    """The default pipeline. Order matters.

    ``dehyphenate`` must run before ``reflow`` — both key on the newline, and
    once reflow has replaced it with a space the broken word is unrecoverable.
    ``merge_continuations`` must run after ``reflow``, since it reasons about
    whole paragraphs.
    """
    for step in (
        normalize_characters,
        dehyphenate,
        reflow,
        merge_continuations,
        collapse_spaces,
    ):
        text = step(text)
    return text


def reflow_only(text: str) -> str:
    """Reflow within the extractor's own paragraph boundaries, trusting them.

    The A/B control for :func:`merge_continuations`: same line rejoining, but
    block boundaries are taken at face value.
    """
    return reflow(dehyphenate(text))


# Named so a YAML config can select one without importing a callable.
PREPROCESSORS: dict[str, Callable[[str], str] | None] = {
    "default": clean_text,
    "reflow": reflow_only,
    "none": None,
}

DEFAULT_PREPROCESSOR = "default"


def resolve_preprocessor(
    preprocess: str | Callable[[str], str] | None = DEFAULT_PREPROCESSOR,
) -> Callable[[str], str] | None:
    """Accept a registry name, a callable, or None, and return the callable.

    Strings are what configs and CLI flags carry; callables are what tests and
    calling code pass. ``None`` means leave the extracted text untouched.
    """
    if preprocess is None or callable(preprocess):
        return preprocess
    try:
        return PREPROCESSORS[preprocess]
    except KeyError:
        available = ", ".join(PREPROCESSORS)
        raise ValueError(f"Unknown preprocessor {preprocess!r}. Available: {available}") from None
