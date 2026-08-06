"""Tests for the semantic chunker.

Two things separate this strategy from the other four and get their own
coverage:

* **It is the only chunker that depends on an embedder**, so the injection has
  to work through both entry points and must not leak into the strategies that
  take no embedder.
* **The seam logic has a degenerate case the percentile hides.** If every
  boundary is equally distant, no shift is larger than the others, and cutting
  at all of them would produce one chunk per sentence for no semantic reason.

The embedder is stubbed throughout so the suite stays fast and offline; the
one test that loads real weights is marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.base import BaseEmbedder
from app.rag.chunking import CHUNKERS, chunker_from_spec, get_chunker
from app.rag.chunking._sentences import Sentence
from app.rag.chunking.semantic import (
    DEFAULT_PERCENTILE,
    SemanticChunker,
    adjacent_distances,
    group_by_seams,
    seam_threshold,
    unit_rows,
)
from app.rag.loaders import join_pages
from app.rag.models import Document, DocumentMetadata

# Two topics, four sentences each. A working chunker cuts at the seam between
# them, not inside either half.
TOPIC_A = (
    "The retriever selects passages relevant to the question. "
    "Dense retrieval embeds the query and the passages into one vector space. "
    "Sparse retrieval scores exact term overlap instead. "
    "Hybrid retrieval combines both scores. "
)
TOPIC_B = (
    "Volcanoes form where tectonic plates diverge or collide. "
    "Magma rises through the crust and collects in a chamber. "
    "Pressure builds until the surrounding rock fractures. "
    "The eruption that follows can last for months. "
)


def make_document(text: str) -> Document:
    content, page_starts = join_pages([text])
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source="test.pdf",
            page_count=len(page_starts),
            page_starts=page_starts,
            paper_id="2401.00000v1",
        ),
    )


class StubEmbedder(BaseEmbedder):
    """Embeds by keyword, so topic shift is exactly where the fixture puts it.

    Sentences mentioning magma/eruption land on one axis, everything else on
    another, giving a distance of ~1.0 at the seam and ~0.0 elsewhere.
    """

    def __init__(self) -> None:
        self.calls = 0

    @property
    def dimension(self) -> int:
        return 2

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        geology = ("volcano", "magma", "eruption", "crust", "tectonic", "rock")
        return [
            [0.0, 1.0] if any(word in text.lower() for word in geology) else [1.0, 0.0]
            for text in texts
        ]


class ConstantEmbedder(StubEmbedder):
    """Every sentence identical: no boundary is more of a shift than any other."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def sentences(count: int, tokens: int = 10) -> list[Sentence]:
    return [Sentence(start=i * 50, end=i * 50 + 40, num_tokens=tokens) for i in range(count)]


class TestUnitRows:
    def test_scales_rows_to_length_one(self) -> None:
        rows = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
        assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)

    def test_a_zero_row_stays_zero_rather_than_becoming_nan(self) -> None:
        rows = unit_rows(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        assert np.array_equal(rows[0], [0.0, 0.0])


class TestAdjacentDistances:
    def test_identical_vectors_are_distance_zero(self) -> None:
        assert adjacent_distances([[1.0, 0.0], [1.0, 0.0]]) == pytest.approx([0.0], abs=1e-6)

    def test_orthogonal_vectors_are_distance_one(self) -> None:
        assert adjacent_distances([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx([1.0], abs=1e-6)

    def test_length_is_one_less_than_the_input(self) -> None:
        assert adjacent_distances([[1.0, 0.0]] * 5).size == 4

    def test_a_single_vector_has_no_boundaries(self) -> None:
        assert adjacent_distances([[1.0, 0.0]]).size == 0

    def test_never_negative_despite_float_error(self) -> None:
        """Clipping matters: a similarity a hair over 1.0 would otherwise give
        a negative distance and skew the percentile."""
        assert (adjacent_distances([[1.0, 0.0]] * 10) >= 0).all()


class TestSeamThreshold:
    def test_none_when_there_are_no_boundaries(self) -> None:
        assert seam_threshold(np.empty(0), 90) is None

    def test_none_when_every_boundary_is_equally_distant(self) -> None:
        """The degenerate case. With no shift larger than the others, cutting
        everywhere would be arbitrary rather than semantic."""
        assert seam_threshold(np.array([0.4, 0.4, 0.4]), 90) is None

    def test_sits_at_the_requested_percentile(self) -> None:
        threshold = seam_threshold(np.array([0.0, 0.1, 0.2, 0.9]), 90)
        assert 0.2 < threshold <= 0.9


class TestGroupBySeams:
    def test_a_seam_closes_the_chunk_after_that_sentence(self) -> None:
        spans = group_by_seams(sentences(4), np.array([0.0, 0.9, 0.0]), 0.5, max_tokens=500)
        assert [span.num_sentences for span in spans] == [2, 2]

    def test_no_threshold_means_one_chunk_up_to_the_cap(self) -> None:
        spans = group_by_seams(sentences(4), np.array([0.9, 0.9, 0.9]), None, max_tokens=500)
        assert [span.num_sentences for span in spans] == [4]

    def test_the_cap_wins_over_the_seams(self) -> None:
        """A sentence that would overflow opens the next chunk whole, so no
        chunk can exceed max_tokens however far away the next seam is."""
        spans = group_by_seams(sentences(6, tokens=40), np.zeros(5), None, max_tokens=100)
        assert all(span.num_tokens <= 100 for span in spans)
        assert [span.num_sentences for span in spans] == [2, 2, 2]

    def test_trailing_sentences_after_the_last_seam_are_kept(self) -> None:
        spans = group_by_seams(sentences(3), np.array([0.9, 0.0]), 0.5, max_tokens=500)
        assert sum(span.num_sentences for span in spans) == 3

    def test_no_sentences(self) -> None:
        assert group_by_seams([], np.empty(0), 0.5, max_tokens=500) == []


class TestSemanticChunker:
    def test_cuts_at_the_topic_change(self) -> None:
        document = make_document(TOPIC_A + TOPIC_B)
        chunks = SemanticChunker(embedder=StubEmbedder()).chunk(document)
        assert len(chunks) == 2
        assert "magma" not in chunks[0].content
        assert "retriever" not in chunks[1].content

    def test_offsets_slice_back_to_the_source(self) -> None:
        """The invariant every strategy shares: no rejoined or stripped text."""
        document = make_document(TOPIC_A + TOPIC_B)
        for chunk in SemanticChunker(embedder=StubEmbedder()).chunk(document):
            start, end = chunk.metadata.start_char, chunk.metadata.end_char
            assert document.content[start:end] == chunk.content

    def test_uniform_text_is_not_shredded_into_one_chunk_per_sentence(self) -> None:
        """What the degenerate-threshold guard buys, at the chunker level."""
        document = make_document(TOPIC_A)
        chunks = SemanticChunker(embedder=ConstantEmbedder()).chunk(document)
        assert len(chunks) == 1

    def test_embeds_once_per_document(self) -> None:
        embedder = StubEmbedder()
        SemanticChunker(embedder=embedder).chunk(make_document(TOPIC_A + TOPIC_B))
        assert embedder.calls == 1

    def test_empty_document(self) -> None:
        assert SemanticChunker(embedder=StubEmbedder()).chunk(make_document("   ")) == []

    def test_rejects_an_impossible_cap(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            SemanticChunker(max_tokens=0, embedder=StubEmbedder())

    @pytest.mark.parametrize("percentile", [0, 100, -5, 101])
    def test_rejects_a_percentile_outside_the_open_interval(self, percentile: int) -> None:
        with pytest.raises(ValueError, match="breakpoint_percentile"):
            SemanticChunker(breakpoint_percentile=percentile, embedder=StubEmbedder())

    def test_repr_names_the_embedder(self) -> None:
        assert "max_tokens=512" in repr(SemanticChunker(embedder=StubEmbedder()))


class TestRegistration:
    def test_registered_under_its_name(self) -> None:
        assert CHUNKERS["semantic"] is SemanticChunker

    def test_spec_maps_onto_cap_and_percentile(self) -> None:
        chunker = chunker_from_spec("semantic:256:80", embedder=StubEmbedder())
        assert (chunker.max_tokens, chunker.breakpoint_percentile) == (256, 80.0)

    def test_a_fractional_percentile_survives_the_spec_string(self) -> None:
        """Why DEFAULT_PERCENTILE is a float: coercion reads the parameter's
        default, and an int default would hand back '87.5' unconverted."""
        chunker = chunker_from_spec("semantic:512:87.5", embedder=StubEmbedder())
        assert chunker.breakpoint_percentile == 87.5

    def test_the_spec_string_cannot_set_the_embedder(self) -> None:
        """It is keyword-only, so positional mapping stops at two arguments."""
        with pytest.raises(ValueError, match="at most 2"):
            chunker_from_spec("semantic:512:90:minilm", embedder=StubEmbedder())

    def test_injection_works_through_get_chunker_too(self) -> None:
        embedder = StubEmbedder()
        assert get_chunker("semantic", embedder=embedder).embedder is embedder

    @pytest.mark.parametrize("spec", ["fixed_size:512:128", "sentence:5:1", "recursive:512:64"])
    def test_the_embedder_is_not_forced_onto_strategies_that_take_none(self, spec: str) -> None:
        """Filtering by signature is what makes one injection call safe for
        every strategy — passing it blindly would be a TypeError on four."""
        assert chunker_from_spec(spec, embedder=StubEmbedder()) is not None

    def test_defaults_to_the_registry_embedder_when_none_is_injected(self) -> None:
        """Constructing must not load weights — the model arrives on first use."""
        assert SemanticChunker().embedder.alias == "minilm"


@pytest.mark.slow
class TestWithRealWeights:
    def test_a_real_model_finds_the_same_topic_boundary(self) -> None:
        from app.rag.embedding import get_embedder

        document = make_document(TOPIC_A + TOPIC_B)
        chunker = SemanticChunker(breakpoint_percentile=DEFAULT_PERCENTILE)
        chunker.embedder = get_embedder("minilm")
        chunks = chunker.chunk(document)
        assert len(chunks) >= 2
        assert "magma" not in chunks[0].content
