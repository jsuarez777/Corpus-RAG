"""Tests for the chunking stage.

The invariant that matters most is the round trip:
``document.content[c.start_char:c.end_char] == c.content`` for every chunk of
every strategy. Citations, highlighting and section alignment all slice the
document by those offsets, and a drift there is invisible until someone reads
a quote that doesn't match its source. It is asserted for each strategy below,
plus once over a real extracted paper.
"""

from pathlib import Path

import pytest

from app.chunk import chunk_documents, summarize, verify_offsets
from app.rag.base import BaseChunker
from app.rag.chunking import (
    CHUNKERS,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    SlidingWindowChunker,
    Span,
    chunker_from_spec,
    config_slug,
    count_tokens,
    get_chunker,
    load_chunks,
    save_chunks,
    spans_to_chunks,
    token_starts,
)
from app.rag.chunking._sentences import segment_sentences, sentence_cap_tokens
from app.rag.chunking._tokens import resolve_overlap
from app.rag.chunking.recursive import DEFAULT_SEPARATORS, _split_on, recursive_spans
from app.rag.loaders import join_pages
from app.rag.models import Document, DocumentMetadata

PROSE = (
    "Retrieval augmented generation grounds a language model in retrieved text. "
    "The retriever selects passages that are relevant to the question. "
    "The generator then conditions its answer on those passages. "
    "Citations let a reader verify each claim against its source. "
    "Evaluation measures whether the retrieved passages were the right ones. "
)


def make_document(text: str, pages: list[str] | None = None) -> Document:
    """A Document with a real page map, as the loaders produce."""
    content, page_starts = join_pages(pages if pages is not None else [text])
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source="test.pdf",
            page_count=len(page_starts),
            page_starts=page_starts,
            paper_id="2401.00000v1",
        ),
    )


ALL_CHUNKERS = [
    FixedSizeChunker(chunk_size=32, overlap=8),
    SentenceChunker(sentences_per_chunk=2, overlap=1),
    SentenceChunker(sentences_per_chunk=2, overlap=0, dynamic_min=True),
    SlidingWindowChunker(window_size=32, overlap_ratio=0.5),
    RecursiveChunker(chunk_size=32, overlap=8),
]


@pytest.fixture
def document() -> Document:
    return make_document(PROSE * 4)


# --------------------------------------------------------------------------- #
# Cross-strategy contracts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: repr(c))
class TestEveryChunker:
    def test_offsets_reproduce_the_content(self, chunker: BaseChunker, document: Document) -> None:
        for chunk in chunker.chunk(document):
            span = document.content[chunk.metadata.start_char : chunk.metadata.end_char]
            assert span == chunk.content

    def test_chunk_indices_are_sequential(self, chunker: BaseChunker, document: Document) -> None:
        chunks = chunker.chunk(document)
        assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunks_advance_and_cover_the_document(
        self, chunker: BaseChunker, document: Document
    ) -> None:
        chunks = chunker.chunk(document)
        starts = [c.metadata.start_char for c in chunks]
        assert starts == sorted(starts)  # never goes backwards
        assert starts[0] == 0
        assert chunks[-1].metadata.end_char == len(document.content)

    def test_provenance_is_stamped(self, chunker: BaseChunker, document: Document) -> None:
        for chunk in chunker.chunk(document):
            assert chunk.metadata.document_id == document.id
            assert chunk.metadata.source == "test.pdf"
            assert chunk.metadata.paper_id == "2401.00000v1"
            assert chunk.metadata.page_number == 1

    def test_no_blank_chunks(self, chunker: BaseChunker, document: Document) -> None:
        assert all(c.content.strip() for c in chunker.chunk(document))

    def test_empty_document_yields_nothing(self, chunker: BaseChunker) -> None:
        assert chunker.chunk(make_document("")) == []

    def test_whitespace_only_document_yields_nothing(self, chunker: BaseChunker) -> None:
        assert chunker.chunk(make_document("   \n\n   \n")) == []


# --------------------------------------------------------------------------- #
# Token helpers
# --------------------------------------------------------------------------- #


class TestTokenStarts:
    def test_starts_slice_the_text_without_loss(self) -> None:
        tokens, starts = token_starts(PROSE)
        assert len(starts) == len(tokens) + 1
        assert starts[0] == 0
        assert starts[-1] == len(PROSE)
        assert starts == sorted(starts)
        # Consecutive token slices reassemble the original exactly.
        assert "".join(PROSE[a:b] for a, b in zip(starts, starts[1:])) == PROSE

    def test_multibyte_characters_are_never_split(self) -> None:
        """A token beginning mid-character must map to that character's own
        offset, or slicing would cut a character in half."""
        text = "café — naïve — 日本語のテキスト — emoji 🧪 tail"
        tokens, starts = token_starts(text)
        assert "".join(text[a:b] for a, b in zip(starts, starts[1:])) == text
        assert starts[-1] == len(text)

    def test_empty_text(self) -> None:
        assert token_starts("") == ([], [0])

    def test_special_token_text_does_not_raise(self) -> None:
        """A paper quoting '<|endoftext|>' must count as tokens, not explode."""
        assert count_tokens("the literal <|endoftext|> marker") > 0


class TestResolveOverlap:
    def test_percentage(self) -> None:
        assert resolve_overlap("25%", 512) == 128

    def test_plain_int_passes_through(self) -> None:
        assert resolve_overlap(0, 512) == 0

    @pytest.mark.parametrize("bad", [512, 600, -1])
    def test_rejects_non_advancing_overlap(self, bad: int) -> None:
        with pytest.raises(ValueError):
            resolve_overlap(bad, 512)

    def test_rejects_unparseable(self) -> None:
        with pytest.raises(ValueError, match="int or a percentage"):
            resolve_overlap("half", 512)


# --------------------------------------------------------------------------- #
# Per-strategy behaviour
# --------------------------------------------------------------------------- #


class TestFixedSize:
    def test_respects_the_token_budget(self, document: Document) -> None:
        for chunk in FixedSizeChunker(chunk_size=64, overlap=16).chunk(document):
            assert count_tokens(chunk.content) <= 64

    def test_overlap_repeats_text(self, document: Document) -> None:
        chunks = FixedSizeChunker(chunk_size=64, overlap=16).chunk(document)
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.metadata.start_char < earlier.metadata.end_char

    def test_zero_overlap_tiles_exactly(self, document: Document) -> None:
        chunks = FixedSizeChunker(chunk_size=64, overlap=0).chunk(document)
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.metadata.start_char == earlier.metadata.end_char

    def test_document_shorter_than_one_window_is_one_chunk(self) -> None:
        chunks = FixedSizeChunker(chunk_size=512, overlap=128).chunk(make_document("Short text."))
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_rejects_impossible_configuration(self) -> None:
        with pytest.raises(ValueError, match="less than chunk size"):
            FixedSizeChunker(chunk_size=100, overlap=100)


class TestSlidingWindow:
    def test_ratio_becomes_a_token_overlap(self) -> None:
        chunker = SlidingWindowChunker(window_size=200, overlap_ratio=0.25)
        assert chunker.overlap == 50
        assert chunker.window_size == 200

    def test_is_fixed_size_with_the_same_numbers(self, document: Document) -> None:
        """Same algorithm, different vocabulary — the outputs must agree."""
        sliding = SlidingWindowChunker(window_size=64, overlap_ratio=0.5).chunk(document)
        fixed = FixedSizeChunker(chunk_size=64, overlap=32).chunk(document)
        assert [c.content for c in sliding] == [c.content for c in fixed]

    def test_rejects_a_window_that_never_advances(self) -> None:
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\)"):
            SlidingWindowChunker(window_size=200, overlap_ratio=1.0)


class TestSentence:
    def test_chunks_end_on_sentence_boundaries(self, document: Document) -> None:
        for chunk in SentenceChunker(sentences_per_chunk=2, overlap=0).chunk(document):
            assert chunk.content.rstrip().endswith((".", "?", "!"))

    def test_sentence_count_is_recorded(self, document: Document) -> None:
        chunks = SentenceChunker(sentences_per_chunk=2, overlap=0).chunk(document)
        assert all(c.metadata.num_sentences <= 2 for c in chunks)
        assert chunks[0].metadata.num_sentences == 2

    def test_overlap_repeats_a_sentence(self, document: Document) -> None:
        chunks = SentenceChunker(sentences_per_chunk=3, overlap=1).chunk(document)
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.metadata.start_char < earlier.metadata.end_char

    def test_dynamic_min_produces_larger_chunks(self) -> None:
        """A run of short fragments should pack together rather than become
        one tiny chunk each."""
        fragments = make_document("Yes. No. Maybe. Ok. Fine. Sure. Right. Left. Up. Down. " * 6)
        plain = SentenceChunker(sentences_per_chunk=2, overlap=0).chunk(fragments)
        packed = SentenceChunker(sentences_per_chunk=2, overlap=0, dynamic_min=True).chunk(
            fragments
        )
        assert len(packed) < len(plain)

    def test_an_oversized_pseudo_sentence_is_split(self) -> None:
        """A flattened table has no sentence boundary; the cap must still hold."""
        table = "col " * 4000  # no terminal punctuation anywhere
        cap = sentence_cap_tokens(2)
        assert all(s.num_tokens <= cap for s in segment_sentences(table, cap))

    def test_rejects_overlap_that_cannot_advance(self) -> None:
        with pytest.raises(ValueError, match="less than"):
            SentenceChunker(sentences_per_chunk=2, overlap=2)


class TestRecursive:
    def test_split_on_tiles_the_range_exactly(self) -> None:
        text = "alpha\n\nbeta\n\ngamma"
        pieces = _split_on(text, 0, len(text), "\n\n")
        assert "".join(text[a:b] for a, b in pieces) == text
        assert [text[a:b] for a, b in pieces] == ["alpha\n\n", "beta\n\n", "gamma"]

    def test_prefers_paragraph_boundaries(self) -> None:
        """Paragraphs that fit stay whole rather than being cut at a token count."""
        paragraphs = [
            f"Paragraph number {n}. It has a couple of sentences in it." for n in range(6)
        ]
        document = make_document("\n\n".join(paragraphs))
        chunks = RecursiveChunker(chunk_size=32, overlap=0).chunk(document)
        assert all(c.content.strip().startswith("Paragraph number") for c in chunks)

    def test_falls_back_to_token_split_when_no_separator_helps(self) -> None:
        """An unbroken run of characters still cannot exceed the budget."""
        spans = recursive_spans("x" * 4000, size=64, overlap=0)
        assert len(spans) > 1
        assert all(s.num_tokens <= 64 for s in spans)

    def test_respects_the_budget_on_prose(self, document: Document) -> None:
        for chunk in RecursiveChunker(chunk_size=64, overlap=0).chunk(document):
            # A single atomic piece may exceed the budget only if no separator
            # and no token split could shrink it, which prose never triggers.
            assert count_tokens(chunk.content) <= 64

    def test_overlap_carries_pieces_forward(self) -> None:
        document = make_document(PROSE * 4)
        with_overlap = RecursiveChunker(chunk_size=64, overlap=32).chunk(document)
        without = RecursiveChunker(chunk_size=64, overlap=0).chunk(document)
        assert len(with_overlap) > len(without)

    def test_custom_separators(self) -> None:
        spans = recursive_spans("a|b|c|d", size=1, overlap=0, separators=("|",))
        assert all(s.end > s.start for s in spans)

    def test_rejects_empty_separators(self) -> None:
        with pytest.raises(ValueError, match="separators"):
            RecursiveChunker(separators=())

    def test_default_separators_go_coarse_to_fine(self) -> None:
        assert DEFAULT_SEPARATORS == ("\n\n", "\n", ". ", " ")


# --------------------------------------------------------------------------- #
# Page mapping, registry, persistence
# --------------------------------------------------------------------------- #


class TestPageNumbers:
    def test_chunks_get_the_page_they_start_on(self) -> None:
        document = make_document("", pages=[PROSE, PROSE, PROSE])
        chunks = FixedSizeChunker(chunk_size=24, overlap=0).chunk(document)

        assert {c.metadata.page_number for c in chunks} == {1, 2, 3}
        for chunk in chunks:
            assert chunk.metadata.page_number <= chunk.metadata.end_page

    def test_a_straddling_chunk_records_both_pages(self) -> None:
        document = make_document("", pages=[PROSE, PROSE])
        chunks = FixedSizeChunker(chunk_size=512, overlap=0).chunk(document)
        assert chunks[0].metadata.page_number == 1
        assert chunks[0].metadata.end_page == 2

    def test_a_document_without_a_page_map_has_no_page_numbers(self) -> None:
        document = Document(content=PROSE, metadata=DocumentMetadata(source="x.pdf"))
        chunk = FixedSizeChunker(chunk_size=32, overlap=0).chunk(document)[0]
        assert chunk.metadata.page_number is None


class TestSpansToChunks:
    def test_rejects_a_backwards_span(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            Span(start=10, end=4)

    def test_extras_are_omitted_when_unknown(self) -> None:
        document = make_document(PROSE)
        chunk = spans_to_chunks(document, [Span(start=0, end=10)])[0]
        assert not hasattr(chunk.metadata, "num_tokens")
        assert chunk.metadata.num_chars == 10


class TestRegistry:
    def test_every_registered_chunker_builds_with_defaults(self) -> None:
        for name in CHUNKERS:
            assert isinstance(get_chunker(name), BaseChunker)

    def test_unknown_name_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunker 'nope'"):
            get_chunker("nope")

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("fixed_size", {"chunk_size": 512, "overlap": 128}),
            ("fixed_size:256", {"chunk_size": 256, "overlap": 128}),
            ("fixed_size:256:32", {"chunk_size": 256, "overlap": 32}),
            ("fixed_size:256:25%", {"chunk_size": 256, "overlap": 64}),
        ],
    )
    def test_spec_maps_arguments_positionally(self, spec: str, expected: dict) -> None:
        chunker = chunker_from_spec(spec)
        assert {key: getattr(chunker, key) for key in expected} == expected

    def test_spec_uses_each_strategys_own_vocabulary(self) -> None:
        sentence = chunker_from_spec("sentence:4:2")
        assert (sentence.sentences_per_chunk, sentence.overlap) == (4, 2)

        sliding = chunker_from_spec("sliding_window:128:0.25")
        assert (sliding.window_size, sliding.overlap) == (128, 32)

    def test_spec_parses_booleans(self) -> None:
        assert chunker_from_spec("sentence:3:1:true").dynamic_min is True
        assert chunker_from_spec("sentence:3:1:false").dynamic_min is False

    def test_spec_rejects_too_many_arguments(self) -> None:
        with pytest.raises(ValueError, match="takes at most"):
            chunker_from_spec("fixed_size:1:2:3:4")

    def test_spec_rejects_an_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunker"):
            chunker_from_spec("nope:1")


class TestStore:
    def test_round_trip_preserves_chunks(self, document: Document, tmp_path: Path) -> None:
        chunks = FixedSizeChunker(chunk_size=64, overlap=16).chunk(document)
        path = save_chunks(chunks, tmp_path, "fixed_size:64:16")
        restored = load_chunks(path)

        assert path.name == "fixed_size_64_16.json"
        assert len(restored) == len(chunks)
        assert [c.content for c in restored] == [c.content for c in chunks]
        assert restored[0].id == chunks[0].id
        assert restored[0].metadata.paper_id == "2401.00000v1"

    def test_slug_is_filename_safe(self) -> None:
        assert config_slug("fixed_size:512:25%") == "fixed_size_512_25pct"
        assert config_slug("sentence:5:1") == "sentence_5_1"


class TestChunkCLI:
    def test_chunk_documents_spans_the_corpus(self, document: Document) -> None:
        other = make_document("A second paper entirely. With its own sentences.")
        chunks = chunk_documents([document, other], FixedSizeChunker(chunk_size=32, overlap=0))
        assert {c.metadata.document_id for c in chunks} == {document.id, other.id}

    def test_verify_offsets_passes_on_real_output(self, document: Document) -> None:
        chunks = FixedSizeChunker(chunk_size=32, overlap=8).chunk(document)
        assert verify_offsets([document], chunks) == 0

    def test_verify_offsets_catches_drift(self, document: Document) -> None:
        chunks = FixedSizeChunker(chunk_size=32, overlap=8).chunk(document)
        chunks[1].metadata.start_char += 3  # exactly the bug it exists to catch
        assert verify_offsets([document], chunks) == 1

    def test_summarize_handles_the_empty_case(self) -> None:
        assert summarize([]) == "no chunks"

    def test_summarize_reports_the_distribution(self, document: Document) -> None:
        chunks = FixedSizeChunker(chunk_size=32, overlap=0).chunk(document)
        assert "chunks | tokens min" in summarize(chunks)
