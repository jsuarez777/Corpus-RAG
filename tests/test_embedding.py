"""Tests for the embedding stage.

Most of these never load a model: construction must stay cheap, because the
registry and the CLI's argument parsing both build embedders they may never
use. The tests that do load one are marked ``slow`` — they download weights.

    pytest -m "not slow"      # skip the download
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.rag.base import BaseEmbedder
from app.rag.embedding import (
    DEFAULT_EMBEDDER,
    EMBEDDERS,
    MODELS,
    SentenceTransformerEmbedder,
    get_embedder,
)
from app.rag.models import Chunk, ChunkMetadata


def make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        content=text,
        metadata=ChunkMetadata(
            document_id=uuid4(),
            source="paper.pdf",
            start_char=index * 100,
            end_char=index * 100 + len(text),
            chunk_index=index,
        ),
    )


class FakeEmbedder(BaseEmbedder):
    """Deterministic vectors with no model behind them.

    Exercises the concrete helpers :class:`BaseEmbedder` provides, which every
    real embedder inherits, without a download.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) + i) for i in range(self._dimension)] for t in texts]


class TestBaseEmbedderHelpers:
    def test_embed_chunks_attaches_a_vector_to_each(self) -> None:
        chunks = [make_chunk("alpha", 0), make_chunk("beta", 1)]
        returned = FakeEmbedder().embed_chunks(chunks)

        assert returned is chunks  # in place, so callers keep their list
        assert all(len(c.embedding) == 4 for c in chunks)
        assert chunks[0].embedding != chunks[1].embedding

    def test_embed_chunks_preserves_pairing(self) -> None:
        """A vector attached to the wrong chunk is silent and unrecoverable."""
        chunks = [make_chunk("a", 0), make_chunk("bbbb", 1)]
        FakeEmbedder(dimension=1).embed_chunks(chunks)
        assert [c.embedding[0] for c in chunks] == [1.0, 4.0]

    def test_a_backend_returning_the_wrong_count_raises(self) -> None:
        class Truncating(FakeEmbedder):
            def embed_texts(self, texts):
                return super().embed_texts(texts)[:-1]

        with pytest.raises(ValueError, match="argument 2 is shorter"):
            Truncating().embed_chunks([make_chunk("a", 0), make_chunk("b", 1)])

    def test_embed_query_defaults_to_the_text_path(self) -> None:
        assert FakeEmbedder().embed_query("alpha") == FakeEmbedder().embed_texts(["alpha"])[0]


class TestSentenceTransformerEmbedder:
    def test_is_an_embedder(self) -> None:
        assert isinstance(SentenceTransformerEmbedder(), BaseEmbedder)

    def test_construction_does_not_load_the_model(self) -> None:
        """Weights are hundreds of MB; the registry must not pay for them."""
        embedder = SentenceTransformerEmbedder("mpnet")
        assert "_model" not in embedder.__dict__

    def test_aliases_resolve_to_model_ids(self) -> None:
        assert SentenceTransformerEmbedder("minilm").model_id == MODELS["minilm"]

    def test_an_unknown_name_passes_through_as_a_model_id(self) -> None:
        """Any HuggingFace id stays usable without editing the alias table."""
        assert SentenceTransformerEmbedder("BAAI/bge-small-en").model_id == "BAAI/bge-small-en"

    def test_repr_names_the_alias(self) -> None:
        assert repr(SentenceTransformerEmbedder("mpnet")) == "SentenceTransformerEmbedder('mpnet')"


class TestRegistry:
    def test_default_is_registered(self) -> None:
        assert DEFAULT_EMBEDDER in EMBEDDERS

    def test_both_grid_models_are_available(self) -> None:
        assert {"minilm", "mpnet"} <= set(EMBEDDERS)

    def test_the_alias_reaches_the_constructor(self) -> None:
        assert get_embedder("mpnet").model_id == MODELS["mpnet"]

    def test_unknown_name_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedder 'bert'. Available: minilm, mpnet"):
            get_embedder("bert")

    def test_kwargs_reach_the_constructor(self) -> None:
        assert get_embedder("minilm", batch_size=8).batch_size == 8


@pytest.fixture(scope="module")
def minilm() -> BaseEmbedder:
    """The real MiniLM, loaded once for the whole module."""
    return get_embedder("minilm")


@pytest.mark.slow
class TestRealModel:
    """The one test that proves the vectors mean something. Downloads MiniLM."""

    def test_dimension_matches_the_vectors(self, minilm: BaseEmbedder) -> None:
        vectors = minilm.embed_texts(["a sentence"])
        assert len(vectors[0]) == minilm.dimension == 384

    def test_embedding_nothing_returns_nothing(self, minilm: BaseEmbedder) -> None:
        assert minilm.embed_texts([]) == []

    def test_this_model_already_returns_unit_vectors(self, minilm: BaseEmbedder) -> None:
        """Both grid models ship a Normalize layer, so the store's scaling is a
        no-op for them. That is the point: the store is correct either way, and
        does not need to know which models normalize themselves. A model that
        did not — bge, e5, a raw transformer — would still get cosine scores.
        """
        norm = sum(v * v for v in minilm.embed_texts(["a sentence"])[0]) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    def test_related_text_scores_above_unrelated(self, minilm: BaseEmbedder) -> None:
        """Cosine over the store's normalization, end to end."""
        from app.rag.stores import FaissStore

        chunks = [
            make_chunk("Cell division was tracked with live imaging microscopy.", 0),
            make_chunk("The treaty was signed in Vienna in 1815.", 1),
        ]
        store = FaissStore()
        store.add(minilm.embed_chunks(chunks))

        results = store.search(minilm.embed_query("how were cells imaged?"), top_k=2)
        assert results[0].chunk.content.startswith("Cell division")
        assert results[0].score > results[1].score
