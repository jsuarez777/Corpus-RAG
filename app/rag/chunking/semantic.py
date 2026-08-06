"""Semantic chunking: cut where the meaning shifts, not where the tokens run out.

Every sentence is embedded, and the cosine distance between each consecutive
pair scores how far the topic moved across that boundary. The largest of those
shifts — the top ``100 - breakpoint_percentile`` percent — become *seams*, and a
chunk closes at a seam or when the next sentence would breach ``max_tokens``,
whichever comes first. The cap always wins, so no chunk can be oversized.

**This is the one chunker that depends on an embedder**, which has two
consequences worth stating plainly:

* ``semantic:512:90`` produces *different chunks* under MiniLM than under
  mpnet, unlike every other strategy. ``data/indices/`` is already keyed on
  ``<chunker>__<embedder>``, so nothing collides — but the grid can no longer
  assume the chunking pass is shared across embedding models.
* The embedder is the *retrieval* one, not a separate hosted embedding API.
  Boundaries drawn in one vector space while the retriever searches another are
  optimized for a geometry nobody queries; using the retrieval model means
  chunks break where that model thinks the topic changed. It is also free,
  offline, and stable across runs, which the reproducibility check needs.

The percentile is computed per document. A corpus-wide threshold would make one
document's chunking depend on which other documents happened to be in the
working set, which is not a property you want when the working set is a sample.
"""

from __future__ import annotations

import numpy as np

from app.rag.base import BaseChunker, BaseEmbedder
from app.rag.chunking._sentences import Sentence, segment_sentences
from app.rag.chunking._spans import Span, drop_blank, spans_to_chunks
from app.rag.models import Chunk, Document

DEFAULT_MAX_TOKENS = 512
# Top 10% of boundaries become seams; the cap does most of the sizing work in
# practice. A float so that spec strings coerce on the parameter's default --
# `semantic:512:87.5` has to survive, and an int default would hand back the
# string unconverted.
DEFAULT_PERCENTILE = 90.0


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, leaving zero rows alone.

    The grid models already emit unit vectors, but the chunker takes any
    ``BaseEmbedder`` and cosine distance is only a dot product once the rows
    are unit length.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def adjacent_distances(vectors: list[list[float]]) -> np.ndarray:
    """Cosine distance across each consecutive pair. Length is ``len(vectors) - 1``."""
    if len(vectors) < 2:
        return np.empty(0, dtype=np.float32)
    matrix = unit_rows(np.asarray(vectors, dtype=np.float32))
    similarities = np.sum(matrix[:-1] * matrix[1:], axis=1)
    # Clip before subtracting: float error can push a similarity a hair past
    # 1.0 and hand back a negative distance.
    return 1.0 - np.clip(similarities, -1.0, 1.0)


def seam_threshold(distances: np.ndarray, percentile: float) -> float | None:
    """Distance at or above which a boundary counts as a seam.

    ``None`` when there is nothing to cut on: fewer than two sentences, or
    every boundary equally distant — in which case no shift is *larger* than
    the others and cutting at all of them would be arbitrary, not semantic.
    """
    if distances.size == 0 or float(distances.max() - distances.min()) == 0.0:
        return None
    return float(np.percentile(distances, percentile))


def group_by_seams(
    sentences: list[Sentence],
    distances: np.ndarray,
    threshold: float | None,
    max_tokens: int,
) -> list[Span]:
    """Group sentences into spans, closing at each seam or at the token cap."""
    spans: list[Span] = []
    group: list[Sentence] = []
    tokens = 0

    def flush() -> None:
        nonlocal group, tokens
        if group:
            spans.append(
                Span(
                    start=group[0].start,
                    end=group[-1].end,
                    num_tokens=tokens,
                    num_sentences=len(group),
                )
            )
        group, tokens = [], 0

    for index, sentence in enumerate(sentences):
        # The cap closes the chunk *before* the sentence that would overflow
        # it, so that sentence opens the next one whole.
        if group and tokens + sentence.num_tokens > max_tokens:
            flush()
        group.append(sentence)
        tokens += sentence.num_tokens
        # distances[index] scores the boundary between this sentence and the
        # next, so a seam there closes the chunk after this sentence.
        if threshold is not None and index < distances.size and distances[index] >= threshold:
            flush()

    flush()  # whatever the last seam left behind
    return spans


class SemanticChunker(BaseChunker):
    """Groups sentences into chunks by embedding-space topic shift."""

    name = "semantic"

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        breakpoint_percentile: float = DEFAULT_PERCENTILE,
        *,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if not 0 < breakpoint_percentile < 100:
            raise ValueError(
                f"breakpoint_percentile must be in (0, 100), got {breakpoint_percentile}"
            )
        self.max_tokens = max_tokens
        self.breakpoint_percentile = breakpoint_percentile
        # Constructed, not loaded: SentenceTransformerEmbedder defers the
        # weights to first use, so a chunker built by the registry for a
        # `--help` listing costs nothing.
        self.embedder = embedder if embedder is not None else self._default_embedder()

    @staticmethod
    def _default_embedder() -> BaseEmbedder:
        # Local import: the embedding registry is only needed when nobody
        # injected one, and importing it eagerly would put it on the path of
        # every chunker.
        from app.rag.embedding import get_embedder

        return get_embedder()

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = segment_sentences(document.content, self.max_tokens)
        if not sentences:
            return []

        vectors = self.embedder.embed_texts(
            [document.content[sentence.start : sentence.end] for sentence in sentences]
        )
        distances = adjacent_distances(vectors)
        spans = group_by_seams(
            sentences,
            distances,
            seam_threshold(distances, self.breakpoint_percentile),
            self.max_tokens,
        )
        return spans_to_chunks(document, drop_blank(document.content, spans))

    def __repr__(self) -> str:
        return (
            f"SemanticChunker(max_tokens={self.max_tokens}, "
            f"breakpoint_percentile={self.breakpoint_percentile}, "
            f"embedder={getattr(self.embedder, 'alias', self.embedder)!r})"
        )
