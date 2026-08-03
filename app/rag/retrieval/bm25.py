"""BM25 — lexical retrieval, the counterweight to the dense side.

Dense retrieval fails in a specific way: it matches meaning, so a query naming
an exact identifier the model never learned — a dataset name, an equation
label, ``all-mpnet-base-v2`` — retrieves things that are merely *about* the
same topic. BM25 matches the string. The two fail on different queries, which
is the entire argument for the hybrid retriever.

**The index is rebuilt on load rather than pickled.** rank-bm25 objects pickle
fine until the library version moves under them, at which point a stale index
either explodes or, worse, loads and scores subtly differently. Fitting is a
single pass over the corpus — milliseconds for the tens of thousands of chunks
here — so the saved form is the chunks and the parameters, and the index is
derived from them. Persisted state you can regenerate is state that cannot rot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.rag.base import BaseRetriever
from app.rag.models import Chunk, RetrievalResult, RetrieverType
from app.rag.retrieval._tokenize import tokenize

log = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"

# rank-bm25's own defaults, named here so they reach the manifest and the
# results table rather than sitting implicit inside the library.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Retriever(BaseRetriever):
    """Okapi BM25 over chunk text.

    ``k1`` controls how fast term frequency saturates, ``b`` how strongly long
    documents are penalized for their length. Both are exposed because chunk length
    varies by an order of magnitude across the chunking configs being compared,
    and ``b`` is precisely the knob that responds to that.
    """

    name = "bm25"

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        stem: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.stem = stem
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None
        if chunks:
            self.fit(chunks)

    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.BM25

    def fit(self, chunks: list[Chunk]) -> None:
        """Build the index over ``chunks``, replacing anything already held."""
        self._chunks = list(chunks)
        corpus = [tokenize(chunk.content, stem=self.stem) for chunk in self._chunks]
        # A chunk that tokenizes to nothing — pure equations, a page of figure
        # labels — would make BM25's average-length term undefined. One
        # placeholder token keeps the arithmetic well formed and leaves the
        # chunk unretrievable, which is the honest outcome for text with no
        # searchable words in it.
        corpus = [tokens or ["\x00empty"] for tokens in corpus]
        self._index = BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None
        log.info(f"BM25 fitted over {len(self._chunks)} chunks (stem={self.stem})")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the ``top_k`` best-scoring chunks, highest BM25 score first."""
        if self._index is None or not self._chunks or top_k <= 0:
            return []
        terms = tokenize(query, stem=self.stem)
        if not terms:
            # Every query word was a stopword. Returning nothing beats returning
            # an arbitrary slice of the corpus with all-zero scores.
            return []

        scores = self._index.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda row: scores[row], reverse=True)
        return [
            RetrievalResult(
                chunk=self._chunks[row],
                score=float(scores[row]),
                retriever_type=RetrieverType.BM25,
            )
            for row in ranked[:top_k]
            # A zero score means the chunk shares no term with the query. It is
            # not a weak match, it is not a match; padding the list to top_k
            # with them would inflate every precision figure.
            if scores[row] > 0
        ]

    def save(self, path: Path) -> None:
        """Write the chunks and parameters the index is rebuilt from."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / CHUNKS_FILE).write_text(
            json.dumps(
                [chunk.model_dump(mode="json", exclude={"embedding"}) for chunk in self._chunks],
                ensure_ascii=False,
            )
        )
        (path / META_FILE).write_text(
            json.dumps(
                {
                    "retriever": self.name,
                    "k1": self.k1,
                    "b": self.b,
                    "stem": self.stem,
                    "num_chunks": len(self._chunks),
                },
                indent=2,
            )
        )
        log.info(f"Wrote BM25 corpus ({len(self._chunks)} chunks) to {path}")

    def load(self, path: Path) -> None:
        """Restore parameters and chunks, then refit."""
        path = Path(path)
        meta = json.loads((path / META_FILE).read_text())
        self.k1, self.b, self.stem = meta["k1"], meta["b"], meta["stem"]
        chunks = [
            Chunk.model_validate(record) for record in json.loads((path / CHUNKS_FILE).read_text())
        ]
        self.fit(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(k1={self.k1}, b={self.b}, chunks={len(self._chunks)})"
