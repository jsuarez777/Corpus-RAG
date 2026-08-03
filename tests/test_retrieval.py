"""Tests for the three retrievers.

The properties that matter:

* BM25 returns nothing rather than padding to ``top_k`` with zero-score chunks —
  a chunk sharing no term with the query is not a weak match, and counting it
  would inflate every precision figure.
* Hybrid at alpha=1 must reproduce dense exactly, and at alpha=0 sparse exactly.
  If the extremes do not degenerate, the alpha sweep is measuring noise.
* Fusion pools candidates past ``top_k``, so a chunk ranked poorly by one
  retriever and well by the other can still surface.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.rag.base import BaseEmbedder, BaseRetriever
from app.rag.models import Chunk, ChunkMetadata, RetrieverType
from app.rag.retrieval import (
    DEFAULT_RETRIEVER,
    RETRIEVERS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    get_retriever,
    normalize_scores,
    tokenize,
)
from app.rag.stores import FaissStore

CORPUS = [
    "Cell division was tracked with live imaging microscopy across the embryo.",
    "The XGBoost model predicts the distribution of active cells per frame.",
    "Ripley's K function measures spatial clustering of point processes.",
    "The treaty of Vienna was signed in 1815 by the assembled powers.",
    "Gradient boosting outperformed the linear baseline on every replicate.",
]


def make_chunk(text: str, index: int, embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        content=text,
        embedding=embedding,
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source="paper.pdf",
            start_char=index * 100,
            end_char=index * 100 + len(text),
            chunk_index=index,
        ),
    )


class WordCountEmbedder(BaseEmbedder):
    """Vectors from counts of four marker words — deterministic, no download.

    Enough structure that dense retrieval returns a *meaningful* order, so the
    hybrid tests are fusing two real rankings rather than one real and one
    arbitrary.
    """

    MARKERS = ("cell", "model", "spatial", "treaty")

    @property
    def dimension(self) -> int:
        return len(self.MARKERS)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(text.lower().count(marker)) + 0.01 for marker in self.MARKERS] for text in texts
        ]


@pytest.fixture
def chunks() -> list[Chunk]:
    return [make_chunk(text, index) for index, text in enumerate(CORPUS)]


@pytest.fixture
def embedder() -> BaseEmbedder:
    return WordCountEmbedder()


@pytest.fixture
def dense(chunks: list[Chunk], embedder: BaseEmbedder) -> DenseRetriever:
    store = FaissStore()
    store.add(embedder.embed_chunks([make_chunk(c.content, i) for i, c in enumerate(chunks)]))
    return DenseRetriever(store, embedder)


@pytest.fixture
def sparse(chunks: list[Chunk]) -> BM25Retriever:
    return BM25Retriever(chunks)


class TestTokenize:
    def test_drops_stopwords(self) -> None:
        assert "the" not in tokenize("The model and the data")

    def test_stems_to_a_common_root(self) -> None:
        """Why stemming is on: otherwise these three never match each other."""
        assert len({tokenize("retrieving")[0], tokenize("retrieval")[0]}) == 1

    def test_keeps_numbers(self) -> None:
        """Model names, years and equation references carry real signal here."""
        assert "2024" in tokenize("published in 2024")
        assert "bm25" in tokenize("the BM25 baseline")

    def test_drops_single_characters(self) -> None:
        assert tokenize("a b cd") == ["cd"]

    def test_stem_can_be_turned_off(self) -> None:
        assert tokenize("embeddings", stem=False) == ["embeddings"]

    def test_empty_text(self) -> None:
        assert tokenize("") == []

    def test_a_query_of_only_stopwords(self) -> None:
        assert tokenize("what is the of and") == []


class TestDenseRetriever:
    def test_is_a_retriever(self, dense: DenseRetriever) -> None:
        assert isinstance(dense, BaseRetriever)
        assert dense.retriever_type is RetrieverType.DENSE

    def test_returns_at_most_top_k(self, dense: DenseRetriever) -> None:
        assert len(dense.retrieve("cell imaging", top_k=2)) == 2

    def test_results_descend_by_score(self, dense: DenseRetriever) -> None:
        scores = [r.score for r in dense.retrieve("cell model", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_an_empty_query_returns_nothing(self, dense: DenseRetriever) -> None:
        assert dense.retrieve("   ") == []

    def test_finds_the_semantically_closest(self, dense: DenseRetriever) -> None:
        assert "treaty" in dense.retrieve("treaty", top_k=1)[0].chunk.content.lower()


class TestBM25Retriever:
    def test_is_a_retriever(self, sparse: BM25Retriever) -> None:
        assert isinstance(sparse, BaseRetriever)
        assert sparse.retriever_type is RetrieverType.BM25

    def test_an_exact_term_wins(self, sparse: BM25Retriever) -> None:
        assert "Ripley" in sparse.retrieve("Ripley K function", top_k=1)[0].chunk.content

    def test_matches_across_morphology(self, sparse: BM25Retriever) -> None:
        """Stemming is what makes 'clustered' find 'clustering'."""
        assert sparse.retrieve("clustered points", top_k=1)[0].chunk.content.startswith("Ripley")

    def test_results_descend_by_score(self, sparse: BM25Retriever) -> None:
        scores = [r.score for r in sparse.retrieve("model cells", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_zero_score_chunks_are_not_returned(self, sparse: BM25Retriever) -> None:
        """The list is short rather than padded: a chunk sharing no term is not
        a weak match, and counting it would inflate precision."""
        results = sparse.retrieve("Vienna treaty", top_k=5)
        assert 0 < len(results) < 5
        assert all(result.score > 0 for result in results)

    def test_a_query_of_only_stopwords_returns_nothing(self, sparse: BM25Retriever) -> None:
        assert sparse.retrieve("what is the") == []

    def test_a_query_matching_nothing_returns_nothing(self, sparse: BM25Retriever) -> None:
        assert sparse.retrieve("photosynthesis chlorophyll") == []

    def test_non_positive_top_k(self, sparse: BM25Retriever) -> None:
        assert sparse.retrieve("cells", top_k=0) == []

    def test_an_unfitted_retriever_returns_nothing(self) -> None:
        assert BM25Retriever().retrieve("anything") == []

    def test_a_chunk_with_no_searchable_words_does_not_break_fitting(
        self, chunks: list[Chunk]
    ) -> None:
        """Pure-equation chunks tokenize to nothing; BM25's average-length term
        would be undefined without a placeholder. The chunk is indexed but
        unretrievable, which is the honest outcome for text with no words."""
        equations = make_chunk("$$ 1 + 1 = 2 $$", len(chunks))
        retriever = BM25Retriever([*chunks, equations])

        assert len(retriever) == len(chunks) + 1
        results = retriever.retrieve("Ripley clustering", top_k=5)
        assert results and results[0].chunk.content.startswith("Ripley")
        assert equations.id not in {result.chunk.id for result in results}

    def test_a_term_in_half_the_corpus_scores_zero(self) -> None:
        """Not a defect — Okapi IDF is log(N-df+0.5) - log(df+0.5), which is 0
        at df = N/2. Worth pinning down: on a small corpus it makes common
        terms silently worthless, and that shapes how BM25 results read.
        """
        retriever = BM25Retriever([make_chunk("cell imaging", 0), make_chunk("cell counting", 1)])
        assert retriever.retrieve("cell", top_k=2) == []

    def test_refitting_replaces_the_corpus(self, sparse: BM25Retriever) -> None:
        sparse.fit([make_chunk("an entirely new document about turbines", 0)])
        assert len(sparse) == 1
        assert sparse.retrieve("Ripley") == []

    def test_parameters_are_exposed(self) -> None:
        """k1 and b belong in the results table, not implicit in the library."""
        retriever = BM25Retriever(k1=1.2, b=0.4)
        assert (retriever.k1, retriever.b) == (1.2, 0.4)


class TestBM25Persistence:
    def test_round_trip_preserves_ranking(self, sparse: BM25Retriever, tmp_path: Path) -> None:
        before = sparse.retrieve("cell imaging model", top_k=5)
        sparse.save(tmp_path / "bm25")

        restored = BM25Retriever()
        restored.load(tmp_path / "bm25")
        after = restored.retrieve("cell imaging model", top_k=5)

        assert [r.chunk.content for r in after] == [r.chunk.content for r in before]
        assert [round(r.score, 6) for r in after] == [round(r.score, 6) for r in before]

    def test_round_trip_preserves_parameters(self, tmp_path: Path) -> None:
        BM25Retriever([make_chunk("text", 0)], k1=1.1, b=0.3, stem=False).save(tmp_path / "bm25")
        restored = BM25Retriever()
        restored.load(tmp_path / "bm25")
        assert (restored.k1, restored.b, restored.stem) == (1.1, 0.3, False)

    def test_the_index_is_rebuilt_not_pickled(self, sparse: BM25Retriever, tmp_path: Path) -> None:
        """Only chunks and parameters are written; a stale pickle of a library
        object would rot silently across versions."""
        sparse.save(tmp_path / "bm25")
        assert {p.name for p in (tmp_path / "bm25").iterdir()} == {"chunks.json", "meta.json"}


class TestNormalizeScores:
    @staticmethod
    def _results(chunks: list[Chunk], scores: list[float]) -> list:
        from app.rag.models import RetrievalResult

        return [
            RetrievalResult(chunk=chunk, score=score, retriever_type=RetrieverType.BM25)
            for chunk, score in zip(chunks, scores, strict=True)
        ]

    def test_maps_onto_zero_one(self, chunks: list[Chunk]) -> None:
        scores = normalize_scores(self._results(chunks[:3], [1.0, 3.0, 5.0]))
        assert sorted(scores.values()) == [0.0, 0.5, 1.0]

    def test_a_perfect_tie_does_not_divide_by_zero(self, chunks: list[Chunk]) -> None:
        assert set(normalize_scores(self._results(chunks[:2], [0.5, 0.5])).values()) == {1.0}

    def test_a_single_result(self, chunks: list[Chunk]) -> None:
        assert normalize_scores(self._results(chunks[:1], [7.0])) == {chunks[0].id: 1.0}

    def test_empty_input(self) -> None:
        assert normalize_scores([]) == {}


class TestHybridRetriever:
    def test_is_a_retriever(self, dense: DenseRetriever, sparse: BM25Retriever) -> None:
        hybrid = HybridRetriever(dense, sparse)
        assert isinstance(hybrid, BaseRetriever)
        assert hybrid.retriever_type is RetrieverType.HYBRID

    def test_results_are_tagged_hybrid(self, dense: DenseRetriever, sparse: BM25Retriever) -> None:
        results = HybridRetriever(dense, sparse).retrieve("cell imaging", top_k=3)
        assert all(r.retriever_type is RetrieverType.HYBRID for r in results)

    def test_alpha_one_reproduces_dense(self, dense: DenseRetriever, sparse: BM25Retriever) -> None:
        """The extremes must degenerate, or the alpha sweep measures noise."""
        hybrid = HybridRetriever(dense, sparse, alpha=1.0)
        assert [r.chunk.content for r in hybrid.retrieve("cell model", top_k=3)] == [
            r.chunk.content for r in dense.retrieve("cell model", top_k=3)
        ]

    def test_alpha_zero_reproduces_sparse(
        self, dense: DenseRetriever, sparse: BM25Retriever
    ) -> None:
        """Including the length: hybrid must not pad with dense candidates that
        BM25 gave no weight, or alpha=0 is not pure sparse retrieval."""
        hybrid = HybridRetriever(dense, sparse, alpha=0.0)
        expected = [r.chunk.content for r in sparse.retrieve("Ripley clustering", top_k=5)]
        assert [r.chunk.content for r in hybrid.retrieve("Ripley clustering", top_k=5)] == expected

    def test_results_descend_by_score(self, dense: DenseRetriever, sparse: BM25Retriever) -> None:
        scores = [r.score for r in HybridRetriever(dense, sparse).retrieve("cells", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_finds_what_only_one_retriever_ranks_well(
        self, dense: DenseRetriever, sparse: BM25Retriever
    ) -> None:
        """The whole argument for hybrid: an exact lexical match the dense
        embedder has no marker for still surfaces."""
        results = HybridRetriever(dense, sparse, alpha=0.5).retrieve("Ripley", top_k=3)
        assert any("Ripley" in r.chunk.content for r in results)

    def test_rrf_fusion_ignores_scale(self, dense: DenseRetriever, sparse: BM25Retriever) -> None:
        results = HybridRetriever(dense, sparse, fusion="rrf").retrieve("cell model", top_k=3)
        assert results
        # RRF scores are sums of 1/(60+rank), so they live well below 1.
        assert all(0 < r.score < 0.1 for r in results)

    def test_both_retrievers_empty_returns_nothing(
        self, dense: DenseRetriever, sparse: BM25Retriever
    ) -> None:
        assert HybridRetriever(dense, sparse).retrieve("   ", top_k=3) == []

    def test_non_positive_top_k(self, dense: DenseRetriever, sparse: BM25Retriever) -> None:
        assert HybridRetriever(dense, sparse).retrieve("cells", top_k=0) == []

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_alpha_outside_zero_one_is_rejected(
        self, dense: DenseRetriever, sparse: BM25Retriever, alpha: float
    ) -> None:
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            HybridRetriever(dense, sparse, alpha=alpha)

    def test_an_unknown_fusion_is_rejected(
        self, dense: DenseRetriever, sparse: BM25Retriever
    ) -> None:
        with pytest.raises(ValueError, match="fusion must be one of"):
            HybridRetriever(dense, sparse, fusion="magic")


class TestRegistry:
    def test_all_three_are_registered(self) -> None:
        assert set(RETRIEVERS) == {"dense", "bm25", "hybrid"}

    def test_default_is_registered(self) -> None:
        assert DEFAULT_RETRIEVER in RETRIEVERS

    def test_positional_arguments_reach_the_constructor(self, chunks: list[Chunk]) -> None:
        assert len(get_retriever("bm25", chunks)) == len(chunks)

    def test_unknown_name_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="Unknown retriever 'splade'"):
            get_retriever("splade")
