"""Sentence segmentation with trustworthy offsets.

Shared by the sentence and semantic chunkers — both need the same guarantee:
every sentence knows exactly where it sits in the source text.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.chunking._tokens import token_starts

# Sentence-chunk sizing, calibrated against an academic-prose corpus
# (mean ~34.5 words/sentence, ~1.3 cl100k tokens/word for English prose).
SENT_AVG_WORDS = 35
SENT_MIN_WORDS = 15
SENT_TOKENS_PER_WORD = 1.3
SENT_CAP_BUFFER = 0.20


@dataclass(frozen=True)
class Sentence:
    """One sentence, located in the document it came from."""

    start: int
    end: int
    num_tokens: int


def sentence_cap_tokens(sentences_per_chunk: int) -> int:
    """Token ceiling a sentence chunk may reach before it is closed early.

    Derived from the sentence count rather than configured separately, so the
    two knobs cannot drift into contradicting each other.
    """
    return int(sentences_per_chunk * SENT_AVG_WORDS * SENT_TOKENS_PER_WORD * (1 + SENT_CAP_BUFFER))


def sentence_min_tokens(sentences_per_chunk: int) -> int:
    """Token floor for the dynamic-minimum mode: a chunk keeps absorbing
    sentences past its sentence count until it reaches this, so a run of short
    fragments (a shredded table) packs together instead of becoming several
    near-useless chunks."""
    return int(sentences_per_chunk * SENT_MIN_WORDS * SENT_TOKENS_PER_WORD)


def segment_sentences(text: str, cap: int) -> list[Sentence]:
    """Segment ``text`` with pysbd in char_span mode.

    Offsets come straight from the segmenter rather than from a substring
    search: pysbd can alter sentence text even with ``clean=False`` (swapping a
    comma and a space, say), so searching for the returned text drifts, and can
    lock onto a duplicate far ahead — which produces enormous mis-sliced chunks
    that look plausible until you read one.

    Any single "sentence" over ``cap`` tokens is almost always a flattened
    table or chart with no real boundary; it is hard-split into cap-sized token
    windows so no downstream chunk can be oversized.
    """
    import pysbd  # local import: pulled in only when sentence chunking runs

    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)

    sentences: list[Sentence] = []
    previous_end = 0
    for span in segmenter.segment(text):
        # pysbd spans occasionally overlap by a character or two; clamp so
        # offsets stay monotonic and no two sentences share text.
        start = max(span.start, previous_end)
        end = max(span.end, start)
        previous_end = end
        if not text[start:end].strip():
            continue

        tokens, starts = token_starts(text[start:end])
        if len(tokens) <= cap:
            sentences.append(Sentence(start=start, end=end, num_tokens=len(tokens)))
            continue
        for cursor in range(0, len(tokens), cap):
            stop = min(cursor + cap, len(tokens))
            piece_start, piece_end = start + starts[cursor], start + starts[stop]
            if piece_end > piece_start:
                sentences.append(
                    Sentence(start=piece_start, end=piece_end, num_tokens=stop - cursor)
                )
    return sentences
