"""FAISS vector store — exact cosine search over chunk embeddings.

``IndexFlatIP`` computes an inner product and nothing else, so cosine
similarity is obtained by scaling every vector to unit length first: the
denominator of ``dot(a, b) / (|a| |b|)`` becomes ``1 * 1``. That scaling lives
here rather than in the embedder because it is a requirement of *this index's*
metric, not of any model — see :mod:`app.rag.embedding.sentence_transformer`.
Normalizing an already-unit vector is a no-op, so the store is correct with
every embedder, including ones that normalize themselves.

Flat, not IVF or HNSW: this corpus runs to tens of thousands of chunks, where
approximate search saves milliseconds nobody is waiting on, and its recall
error would land inside the very numbers the experiment grid is comparing.

Row order is the identity map. FAISS returns row positions, and row ``i`` is
``self._chunks[i]`` — ordered, cheap to serialize, and stable across
save/load, which an ``IndexIDMap`` keyed on UUIDs would not be for free.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from app.rag.base import BaseVectorStore
from app.rag.models import Chunk, RetrievalResult, RetrieverType

log = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row, so an inner product reads as cosine similarity.

    A zero vector has no direction to normalize towards, so it is left at zero
    rather than dividing by zero into NaN. It then scores 0.0 against every
    query, which is the honest answer for a chunk with no signal in it.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _matrix(vectors: list[list[float]]) -> np.ndarray:
    """Vectors as the contiguous float32 array FAISS requires."""
    return np.ascontiguousarray(vectors, dtype=np.float32)


class FaissStore(BaseVectorStore):
    """Exact cosine nearest-neighbour search over chunks.

    ``dimension`` is inferred from the first batch added, so callers do not
    have to load an embedding model just to construct the store; pass it
    explicitly when building an empty index that must match a known model.
    """

    name = "faiss"

    def __init__(self, dimension: int | None = None) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension) if dimension else None
        self._chunks: list[Chunk] = []

    @property
    def dimension(self) -> int | None:
        """Vector length this index holds, or None while it is still empty."""
        return self._dimension

    def add(self, chunks: list[Chunk]) -> None:
        """Index ``chunks``, which must all carry an embedding."""
        if not chunks:
            return

        missing = next((c for c in chunks if c.embedding is None), None)
        if missing is not None:
            # Naming the chunk beats "NoneType is not iterable" thrown from
            # inside numpy three frames down.
            raise ValueError(
                f"Chunk {missing.id} from {missing.metadata.source} has no embedding — "
                "run the embedding stage before adding to the store."
            )

        matrix = _matrix([chunk.embedding for chunk in chunks])
        if self._index is None:
            self._dimension = matrix.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)
        elif matrix.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension {matrix.shape[1]} does not match this index's "
                f"{self._dimension} — the chunks were embedded by a different model."
            )

        self._index.add(unit_rows(matrix))
        self._chunks.extend(chunks)

    def search(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """Return the ``top_k`` nearest chunks, highest cosine first."""
        if self._index is None or not self._chunks:
            return []
        if top_k <= 0:
            return []

        query = _matrix([embedding])
        if query.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match this index's "
                f"{self._dimension} — query and chunks must share an embedder."
            )

        # FAISS pads a short result set with row -1; asking for no more than we
        # hold avoids having to filter those out of the common case.
        scores, rows = self._index.search(unit_rows(query), min(top_k, len(self._chunks)))
        return [
            RetrievalResult(
                chunk=self._chunks[row],
                score=float(score),
                # A store is not a retriever, but the contract returns
                # RetrievalResult, which must name one. Dense is what this is;
                # a hybrid retriever restamps it when it fuses result sets.
                retriever_type=RetrieverType.DENSE,
            )
            for score, row in zip(scores[0], rows[0], strict=True)
            if row != -1
        ]

    def save(self, path: Path) -> None:
        """Write the index, its chunks and a manifest under ``path``.

        The chunks are stored **without** their embeddings: those are already
        in ``index.faiss``, and re-encoding them as JSON floats roughly triples
        the directory for nothing. Chunks therefore come back from
        :meth:`load` with ``embedding`` unset, while search is unaffected.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            raise ValueError("Nothing to save: the store is empty.")

        faiss.write_index(self._index, str(path / INDEX_FILE))
        (path / CHUNKS_FILE).write_text(
            json.dumps(
                [chunk.model_dump(mode="json", exclude={"embedding"}) for chunk in self._chunks],
                ensure_ascii=False,
            )
        )
        (path / META_FILE).write_text(
            json.dumps(
                {
                    "store": self.name,
                    "index": "IndexFlatIP",
                    "metric": "cosine",
                    "dimension": self._dimension,
                    "num_chunks": len(self._chunks),
                },
                indent=2,
            )
        )
        log.info(f"Wrote {len(self._chunks)} chunks (dim {self._dimension}) to {path}")

    def load(self, path: Path) -> None:
        """Restore an index written by :meth:`save`, replacing any contents."""
        path = Path(path)
        index = faiss.read_index(str(path / INDEX_FILE))
        chunks = [
            Chunk.model_validate(record) for record in json.loads((path / CHUNKS_FILE).read_text())
        ]
        if index.ntotal != len(chunks):
            # Row i must be chunks[i]; if the two files disagree, every result
            # this store returns would cite the wrong text.
            raise ValueError(
                f"Corrupt index at {path}: {index.ntotal} vectors against {len(chunks)} chunks."
            )
        self._index, self._chunks, self._dimension = index, chunks, index.d

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self._dimension}, chunks={len(self._chunks)})"
