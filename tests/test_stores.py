"""Tests for the FAISS vector store.

The invariants downstream retrieval rests on: row ``i`` of the index is
``_chunks[i]`` and stays that way across save/load, and scores are cosine
similarities — which holds only because the store normalizes, so a store fed
raw model output must still return 1.0 for an exact match.

No model is loaded here. Vectors are handmade, which keeps the store's
behaviour separable from any embedder's.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.rag.base import BaseVectorStore
from app.rag.models import Chunk, ChunkMetadata, RetrieverType
from app.rag.stores import (
    DEFAULT_STORE,
    STORES,
    FaissStore,
    config_id,
    get_store,
    index_dir,
    open_store,
    unit_rows,
)
from app.rag.stores.faiss_store import CHUNKS_FILE, META_FILE


def make_chunk(text: str, embedding: list[float] | None, index: int = 0, **extra) -> Chunk:
    """A chunk with just enough metadata to satisfy the model."""
    return Chunk(
        content=text,
        embedding=embedding,
        metadata=ChunkMetadata(
            document_id=extra.pop("document_id", uuid4()),
            source=extra.pop("source", "paper.pdf"),
            start_char=index * 100,
            end_char=index * 100 + len(text),
            chunk_index=index,
            **extra,
        ),
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    """Three chunks on the axes of a 3-d space, so similarity is by hand.

    Deliberately un-normalized: the second has length 5. If the store did not
    normalize, its score against its own unit query would be 5.0, not 1.0.
    """
    return [
        make_chunk("first", [1.0, 0.0, 0.0], 0),
        make_chunk("second", [0.0, 5.0, 0.0], 1),
        make_chunk("third", [0.0, 0.0, 1.0], 2),
    ]


@pytest.fixture
def store(chunks: list[Chunk]) -> FaissStore:
    populated = FaissStore()
    populated.add(chunks)
    return populated


class TestUnitRows:
    def test_scales_rows_to_length_one(self) -> None:
        rows = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
        assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)

    def test_preserves_direction(self) -> None:
        rows = unit_rows(np.array([[3.0, 4.0]], dtype=np.float32))
        assert np.allclose(rows, [[0.6, 0.8]])

    def test_a_zero_vector_stays_zero_rather_than_nan(self) -> None:
        rows = unit_rows(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        assert np.allclose(rows[0], [0.0, 0.0])
        assert not np.isnan(rows).any()

    def test_is_idempotent(self) -> None:
        """An embedder that normalizes already must not be penalized twice."""
        once = unit_rows(np.array([[3.0, 4.0]], dtype=np.float32))
        assert np.allclose(unit_rows(once), once)

    def test_does_not_mutate_its_input(self) -> None:
        original = np.array([[3.0, 4.0]], dtype=np.float32)
        unit_rows(original)
        assert np.allclose(original, [[3.0, 4.0]])


class TestAdd:
    def test_is_a_vector_store(self) -> None:
        assert isinstance(FaissStore(), BaseVectorStore)

    def test_infers_dimension_from_the_first_batch(self, store: FaissStore) -> None:
        assert store.dimension == 3
        assert len(store) == 3

    def test_adding_nothing_is_a_no_op(self) -> None:
        empty = FaissStore()
        empty.add([])
        assert len(empty) == 0
        assert empty.dimension is None

    def test_appends_across_calls(self, store: FaissStore) -> None:
        store.add([make_chunk("fourth", [1.0, 1.0, 0.0], 3)])
        assert len(store) == 4

    def test_a_chunk_without_an_embedding_names_itself(self) -> None:
        with pytest.raises(ValueError, match="has no embedding"):
            FaissStore().add([make_chunk("unembedded", None)])

    def test_mismatched_dimension_is_rejected(self, store: FaissStore) -> None:
        with pytest.raises(ValueError, match="different model"):
            store.add([make_chunk("wrong width", [1.0, 0.0], 3)])

    def test_declared_dimension_is_enforced(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            FaissStore(dimension=8).add([make_chunk("too narrow", [1.0, 0.0], 0)])


class TestSearch:
    def test_an_exact_match_scores_one(self, store: FaissStore) -> None:
        """Cosine, not raw inner product: the matching chunk has length 5."""
        results = store.search([0.0, 1.0, 0.0], top_k=1)
        assert results[0].chunk.content == "second"
        assert math.isclose(results[0].score, 1.0, abs_tol=1e-5)

    def test_results_are_sorted_by_descending_score(self, store: FaissStore) -> None:
        results = store.search([0.9, 0.4, 0.1], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].chunk.content == "first"

    def test_an_orthogonal_query_scores_zero(self, store: FaissStore) -> None:
        assert math.isclose(store.search([1.0, 0.0, 0.0], top_k=3)[-1].score, 0.0, abs_tol=1e-6)

    def test_results_are_tagged_dense(self, store: FaissStore) -> None:
        assert store.search([1.0, 0.0, 0.0], top_k=1)[0].retriever_type is RetrieverType.DENSE

    def test_top_k_larger_than_the_corpus_returns_what_exists(self, store: FaissStore) -> None:
        assert len(store.search([1.0, 0.0, 0.0], top_k=50)) == 3

    def test_an_empty_store_returns_nothing(self) -> None:
        assert FaissStore().search([1.0, 0.0, 0.0]) == []

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_non_positive_top_k_returns_nothing(self, store: FaissStore, top_k: int) -> None:
        assert store.search([1.0, 0.0, 0.0], top_k=top_k) == []

    def test_a_query_of_the_wrong_width_is_rejected(self, store: FaissStore) -> None:
        with pytest.raises(ValueError, match="share an embedder"):
            store.search([1.0, 0.0])


class TestPersistence:
    def test_round_trip_preserves_order_and_scores(self, store: FaissStore, tmp_path: Path) -> None:
        query = [0.9, 0.4, 0.1]
        before = store.search(query, top_k=3)

        store.save(tmp_path / "idx")
        restored = open_store(tmp_path / "idx")
        after = restored.search(query, top_k=3)

        assert [r.chunk.id for r in after] == [r.chunk.id for r in before]
        assert [r.chunk.content for r in after] == [r.chunk.content for r in before]
        for old, new in zip(before, after, strict=True):
            assert math.isclose(old.score, new.score, rel_tol=1e-6)

    def test_metadata_survives_the_round_trip(self, store: FaissStore, tmp_path: Path) -> None:
        store.save(tmp_path / "idx")
        restored = open_store(tmp_path / "idx")
        original, recovered = store._chunks[1], restored._chunks[1]

        assert recovered.metadata.document_id == original.metadata.document_id
        assert recovered.metadata.start_char == original.metadata.start_char
        assert recovered.metadata.chunk_index == original.metadata.chunk_index

    def test_saved_chunks_carry_no_embeddings(self, store: FaissStore, tmp_path: Path) -> None:
        """The vectors live in index.faiss; writing them twice buys nothing."""
        store.save(tmp_path / "idx")
        records = json.loads((tmp_path / "idx" / CHUNKS_FILE).read_text())

        assert all("embedding" not in record for record in records)
        assert all(chunk.embedding is None for chunk in open_store(tmp_path / "idx")._chunks)

    def test_the_manifest_records_the_metric(self, store: FaissStore, tmp_path: Path) -> None:
        store.save(tmp_path / "idx")
        meta = json.loads((tmp_path / "idx" / META_FILE).read_text())

        assert meta == {
            "store": "faiss",
            "index": "IndexFlatIP",
            "metric": "cosine",
            "dimension": 3,
            "num_chunks": 3,
        }

    def test_saving_an_empty_store_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            FaissStore().save(tmp_path / "idx")

    def test_a_vector_count_mismatch_is_caught(self, store: FaissStore, tmp_path: Path) -> None:
        """Row i must be chunks[i] — disagreement means every citation is wrong."""
        store.save(tmp_path / "idx")
        records = json.loads((tmp_path / "idx" / CHUNKS_FILE).read_text())
        (tmp_path / "idx" / CHUNKS_FILE).write_text(json.dumps(records[:2]))

        with pytest.raises(ValueError, match="Corrupt index"):
            open_store(tmp_path / "idx")

    def test_load_replaces_rather_than_appends(self, store: FaissStore, tmp_path: Path) -> None:
        store.save(tmp_path / "idx")
        store.load(tmp_path / "idx")
        assert len(store) == 3


class TestRegistry:
    def test_default_is_registered(self) -> None:
        assert isinstance(get_store(), STORES[DEFAULT_STORE])

    def test_unknown_name_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="Unknown store 'nope'. Available: faiss"):
            get_store("nope")

    def test_kwargs_reach_the_constructor(self) -> None:
        assert get_store("faiss", dimension=384).dimension == 384


class TestConfigId:
    def test_names_both_halves_of_the_pair(self) -> None:
        assert config_id("fixed_size:512:128", "minilm") == "fixed_size_512_128__minilm"

    def test_the_embedder_distinguishes_otherwise_identical_indices(self) -> None:
        """Same chunks, different model, different index — one must not
        overwrite the other."""
        assert config_id("recursive:512:64", "minilm") != config_id("recursive:512:64", "mpnet")

    def test_index_dir_hangs_it_off_the_base(self, tmp_path: Path) -> None:
        assert index_dir(tmp_path, "sentence:5:1", "mpnet") == tmp_path / "sentence_5_1__mpnet"
