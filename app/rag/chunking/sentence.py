"""Sentence-boundary chunking: N sentences per chunk, with sentence overlap.

Chunks never split mid-sentence, so an embedded chunk is always readable text.
Two guards from mp3 that earn their keep on this corpus:

* a **token cap** derived from the sentence count, so one flattened table
  cannot inflate a chunk to ten times its neighbours;
* an optional **token floor** (mp3's ``sentence-dynamic-min``), which keeps
  absorbing sentences past the count until the floor is met, so a run of short
  fragments packs into one usable chunk instead of several useless ones.
"""

from __future__ import annotations

from app.rag.base import BaseChunker
from app.rag.chunking._sentences import (
    Sentence,
    segment_sentences,
    sentence_cap_tokens,
    sentence_min_tokens,
)
from app.rag.chunking._spans import Span, drop_blank, spans_to_chunks
from app.rag.models import Chunk, Document


def group_sentences(
    sentences: list[Sentence],
    sentences_per_chunk: int,
    overlap: int,
    cap: int,
    min_tokens: int | None = None,
) -> list[Span]:
    """Group ``sentences`` into overlapping spans of ``sentences_per_chunk``."""
    spans: list[Span] = []
    index = 0
    while index < len(sentences):
        group: list[Sentence] = []
        tokens = 0
        cursor = index
        while cursor < len(sentences):
            sentence = sentences[cursor]
            if group:
                if tokens + sentence.num_tokens > cap:
                    break  # the cap always wins: this sentence opens the next chunk
                # Close once we have enough sentences AND, if a floor is set,
                # enough tokens to be worth retrieving.
                if len(group) >= sentences_per_chunk and (
                    min_tokens is None or tokens >= min_tokens
                ):
                    break
            group.append(sentence)
            tokens += sentence.num_tokens
            cursor += 1

        spans.append(
            Span(
                start=group[0].start,
                end=group[-1].end,
                num_tokens=tokens,
                num_sentences=len(group),
            )
        )
        if cursor >= len(sentences):
            break
        # Advance by what this chunk emitted, less the overlap — but always at
        # least one sentence, or a cap-forced single-sentence chunk would loop.
        index += max(1, len(group) - overlap)
    return spans


class SentenceChunker(BaseChunker):
    """Groups whole sentences into chunks.

    With ``dynamic_min=True`` this is mp3's ``sentence-dynamic-min`` strategy;
    the two are one class because they differ only in whether the floor applies.
    """

    name = "sentence"

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap: int = 1,
        dynamic_min: bool = False,
    ) -> None:
        if sentences_per_chunk < 1:
            raise ValueError(f"sentences_per_chunk must be >= 1, got {sentences_per_chunk}")
        if not 0 <= overlap < sentences_per_chunk:
            raise ValueError(
                f"overlap ({overlap}) must be >= 0 and less than "
                f"sentences_per_chunk ({sentences_per_chunk})"
            )
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap
        self.dynamic_min = dynamic_min
        self.cap = sentence_cap_tokens(sentences_per_chunk)
        self.min_tokens = sentence_min_tokens(sentences_per_chunk) if dynamic_min else None

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = segment_sentences(document.content, self.cap)
        spans = group_sentences(
            sentences, self.sentences_per_chunk, self.overlap, self.cap, self.min_tokens
        )
        # num_tokens is the sum over the group's sentences, which can differ by
        # a token or two from re-encoding the joined span (the whitespace
        # between sentences merges into neighbouring tokens). Not worth a
        # second pass over every chunk; it is a size statistic, not a boundary.
        return spans_to_chunks(document, drop_blank(document.content, spans))

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(sentences_per_chunk={self.sentences_per_chunk}, "
            f"overlap={self.overlap}, dynamic_min={self.dynamic_min})"
        )
