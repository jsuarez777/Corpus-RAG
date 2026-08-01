"""Fixed-size token windows — the baseline strategy.

Cuts every ``chunk_size`` tokens with ``overlap`` tokens carried forward,
ignoring sentence and paragraph structure entirely. That is the point: it is
the control every other strategy is measured against, and the one whose chunk
sizes are perfectly uniform.
"""

from __future__ import annotations

from app.rag.base import BaseChunker
from app.rag.chunking._spans import Span, drop_blank, spans_to_chunks
from app.rag.chunking._tokens import resolve_overlap, token_starts
from app.rag.models import Chunk, Document


def fixed_size_spans(text: str, size: int, overlap: int) -> list[Span]:
    """Overlapping token windows over ``text``, as char spans."""
    tokens, starts = token_starts(text)
    step = size - overlap  # > 0, guaranteed by resolve_overlap

    spans, cursor = [], 0
    while cursor < len(tokens):
        end = min(cursor + size, len(tokens))
        spans.append(Span(start=starts[cursor], end=starts[end], num_tokens=end - cursor))
        if end >= len(tokens):
            break  # the last window is short; stepping again would re-emit its tail
        cursor += step
    return spans


class FixedSizeChunker(BaseChunker):
    """Fixed token windows with a fixed overlap."""

    name = "fixed_size"

    def __init__(self, chunk_size: int = 512, overlap: int | str = 128) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = chunk_size
        self.overlap = resolve_overlap(overlap, chunk_size)

    def chunk(self, document: Document) -> list[Chunk]:
        spans = fixed_size_spans(document.content, self.chunk_size, self.overlap)
        return spans_to_chunks(document, drop_blank(document.content, spans))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(chunk_size={self.chunk_size}, overlap={self.overlap})"
