"""The span -> Chunk adapter every chunker shares.

Each strategy is written as a pure function from text to :class:`Span` offsets;
this module is the single place that turns those into ``Chunk`` objects. One
adapter means one place that can violate the contract ``base.py`` states
(``document_id``, ``chunk_index``, ``start_char``, ``end_char`` on every chunk),
and one place that has to be right about page numbers.

Chunk content is always an exact slice of ``document.content`` — never a
stripped or rejoined copy — so ``content[start_char:end_char] == chunk.content``
holds for every chunk in the system. Tests assert it; retrieval highlighting
and citation snippets depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.loaders import page_of
from app.rag.models import Chunk, ChunkMetadata, Document


@dataclass(frozen=True)
class Span:
    """A half-open char range of a document, plus what the chunker knows about it.

    ``num_tokens`` and ``num_sentences`` are optional because not every strategy
    computes both; whichever are set ride onto the chunk metadata as extras.
    """

    start: int
    end: int
    num_tokens: int | None = None
    num_sentences: int | None = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Span end ({self.end}) precedes start ({self.start})")


def spans_to_chunks(document: Document, spans: list[Span]) -> list[Chunk]:
    """Build the ``Chunk`` objects for ``spans``, in order, numbering from 0."""
    page_starts = getattr(document.metadata, "page_starts", None) or []
    paper_id = getattr(document.metadata, "paper_id", None)

    chunks = []
    for index, span in enumerate(spans):
        content = document.content[span.start : span.end]
        extras = {
            # end_page differs from page_number for a chunk that straddles a
            # page break -- worth keeping for citation display. max() guards
            # the empty span, whose end_char is its own start.
            "end_page": page_of(page_starts, max(span.end - 1, span.start)),
            "num_chars": len(content),
        }
        if span.num_tokens is not None:
            extras["num_tokens"] = span.num_tokens
        if span.num_sentences is not None:
            extras["num_sentences"] = span.num_sentences
        if paper_id is not None:
            # qrels keys on the arXiv id, not on the in-process UUID.
            extras["paper_id"] = paper_id

        chunks.append(
            Chunk(
                content=content,
                metadata=ChunkMetadata(
                    document_id=document.id,
                    source=document.metadata.source,
                    start_char=span.start,
                    end_char=span.end,
                    chunk_index=index,
                    page_number=page_of(page_starts, span.start),
                    **extras,
                ),
            )
        )
    return chunks


def drop_blank(text: str, spans: list[Span]) -> list[Span]:
    """Discard spans that are empty or all whitespace.

    A run of blank lines between sections can otherwise become a chunk with no
    retrievable content, which costs an embedding call and pollutes recall@k.
    """
    return [span for span in spans if text[span.start : span.end].strip()]
