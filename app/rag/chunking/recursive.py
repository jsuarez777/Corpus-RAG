"""Recursive character splitting: respect the author's structure where it fits.

Descends a list of separators from coarsest to finest, splitting only what is
still too large: paragraphs first, then lines, then sentence-ish boundaries,
then words. A section that already fits stays whole, so unlike ``fixed_size``
the boundaries land where the author put them, and unlike ``sentence`` a short
paragraph is not forced to merge with the next one.

Two properties are maintained throughout, and they are what most of this code
is for:

* **Pieces tile their range exactly.** Separators stay attached to the piece
  before them, so no character is lost between two chunks and a merged chunk is
  always one contiguous slice.
* **A piece that no separator can shrink is token-split.** Without that
  fallback a 900-token unbroken table would silently exceed ``chunk_size``.
"""

from __future__ import annotations

from app.rag.base import BaseChunker
from app.rag.chunking._spans import Span, drop_blank, spans_to_chunks
from app.rag.chunking._tokens import count_tokens, resolve_overlap, token_starts
from app.rag.models import Chunk, Document

# Coarse to fine. The trailing space is the last structural boundary before
# characters; splitting mid-word is what the token fallback is for.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


def _split_on(text: str, start: int, end: int, separator: str) -> list[tuple[int, int]]:
    """Ranges of ``[start, end)`` split at ``separator``, tiling exactly.

    The separator belongs to the piece it terminates, so concatenating the
    pieces reproduces the range character for character.
    """
    pieces, cursor = [], start
    while (found := text.find(separator, cursor, end)) != -1:
        pieces.append((cursor, found + len(separator)))
        cursor = found + len(separator)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def _token_split(text: str, start: int, end: int, size: int) -> list[Span]:
    """Last resort: cut a still-oversized range into ``size``-token windows."""
    tokens, starts = token_starts(text[start:end])
    return [
        Span(
            start=start + starts[cursor],
            end=start + starts[min(cursor + size, len(tokens))],
            num_tokens=min(cursor + size, len(tokens)) - cursor,
        )
        for cursor in range(0, len(tokens), size)
    ]


def _atomic_spans(
    text: str, start: int, end: int, size: int, separators: tuple[str, ...]
) -> list[Span]:
    """Split ``[start, end)`` until every piece fits in ``size`` tokens."""
    if end <= start:
        return []
    tokens = count_tokens(text[start:end])
    if tokens <= size:
        return [Span(start=start, end=end, num_tokens=tokens)]

    for index, separator in enumerate(separators):
        pieces = _split_on(text, start, end, separator)
        if len(pieces) < 2:
            continue  # this separator does not occur here; try a finer one
        # Finer separators only from here down: re-trying a coarser one on a
        # piece it just produced cannot split it again.
        remaining = separators[index + 1 :]
        return [
            span
            for piece_start, piece_end in pieces
            for span in _atomic_spans(text, piece_start, piece_end, size, remaining)
        ]

    return _token_split(text, start, end, size)


def _merge_spans(pieces: list[Span], size: int, overlap: int) -> list[Span]:
    """Greedily pack adjacent pieces up to ``size`` tokens, carrying ``overlap``
    tokens of trailing pieces into the next chunk."""
    spans: list[Span] = []
    index = 0
    while index < len(pieces):
        total, cursor = 0, index
        # `cursor == index` forces at least one piece in, so an oversized
        # single piece still emits rather than looping forever.
        while cursor < len(pieces) and (
            cursor == index or total + pieces[cursor].num_tokens <= size
        ):
            total += pieces[cursor].num_tokens
            cursor += 1

        spans.append(Span(start=pieces[index].start, end=pieces[cursor - 1].end, num_tokens=total))
        if cursor >= len(pieces):
            break

        # Walk back over trailing pieces that fit inside the overlap budget,
        # never past the first piece of the chunk just emitted -- the chunk has
        # to advance by at least one piece.
        back, step = 0, cursor
        while step > index + 1 and back + pieces[step - 1].num_tokens <= overlap:
            step -= 1
            back += pieces[step].num_tokens
        index = step
    return spans


def recursive_spans(
    text: str,
    size: int,
    overlap: int,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[Span]:
    """Split ``text`` by descending ``separators``, then pack back up to ``size``."""
    return _merge_spans(_atomic_spans(text, 0, len(text), size, separators), size, overlap)


class RecursiveChunker(BaseChunker):
    """Structure-aware splitting with a token budget."""

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int | str = 64,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if not separators:
            raise ValueError("separators must not be empty")
        self.chunk_size = chunk_size
        self.overlap = resolve_overlap(overlap, chunk_size)
        self.separators = tuple(separators)

    def chunk(self, document: Document) -> list[Chunk]:
        spans = recursive_spans(document.content, self.chunk_size, self.overlap, self.separators)
        return spans_to_chunks(document, drop_blank(document.content, spans))

    def __repr__(self) -> str:
        return f"RecursiveChunker(chunk_size={self.chunk_size}, overlap={self.overlap})"
