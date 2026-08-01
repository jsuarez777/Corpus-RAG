"""Tests for the pipeline's typed payloads.

These cover the constraints the models are supposed to enforce, not Pydantic
itself: field requiredness, the range checks that catch off-by-one metadata,
and JSON round-tripping (every result file and index sidecar is written and
re-read through these models).
"""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.rag.models import (
    Chunk,
    ChunkMetadata,
    Citation,
    Document,
    DocumentMetadata,
    QAResponse,
    RetrievalResult,
    RetrieverType,
)


@pytest.fixture
def document() -> Document:
    return Document(
        content="Attention is all you need. " * 10,
        metadata=DocumentMetadata(
            source="1706.03762v7.pdf", title="Attention Is All You Need", page_count=15
        ),
    )


@pytest.fixture
def chunk(document: Document) -> Chunk:
    return Chunk(
        content="Attention is all you need.",
        metadata=ChunkMetadata(
            document_id=document.id,
            source=document.metadata.source,
            page_number=1,
            start_char=0,
            end_char=26,
            chunk_index=0,
            section_index=3,
        ),
    )


class TestDocument:
    def test_ids_are_unique_per_instance(self) -> None:
        """A shared default would give every document the same id."""
        metadata = DocumentMetadata(source="a.pdf")
        assert Document(content="x", metadata=metadata).id != (
            Document(content="x", metadata=metadata).id
        )

    def test_optional_metadata_defaults_to_none(self) -> None:
        metadata = DocumentMetadata(source="a.pdf")
        assert (metadata.title, metadata.author, metadata.page_count) == (None, None, None)

    def test_unknown_metadata_is_preserved(self) -> None:
        """Loaders surface per-format fields we do not enumerate."""
        metadata = DocumentMetadata(source="a.pdf", arxiv_version="v2")
        assert metadata.arxiv_version == "v2"
        assert "arxiv_version" in metadata.model_dump()

    def test_source_is_required(self) -> None:
        with pytest.raises(ValidationError, match="source"):
            DocumentMetadata()

    def test_negative_page_count_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DocumentMetadata(source="a.pdf", page_count=-1)


class TestChunk:
    def test_embedding_defaults_to_none_and_is_assignable(self, chunk: Chunk) -> None:
        """The embedding stage attaches vectors after construction."""
        assert chunk.embedding is None
        chunk.embedding = [0.1, 0.2, 0.3]
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_empty_embedding_rejected(self, chunk: Chunk) -> None:
        """None means 'not embedded yet'; [] is a zero-length vector, i.e. a bug."""
        with pytest.raises(ValidationError, match="non-empty vector"):
            Chunk(content="x", metadata=chunk.metadata, embedding=[])

    def test_end_char_before_start_char_rejected(self, document: Document) -> None:
        with pytest.raises(ValidationError, match="precedes start_char"):
            ChunkMetadata(
                document_id=document.id,
                source="a.pdf",
                start_char=50,
                end_char=10,
                chunk_index=0,
            )

    def test_zero_length_span_allowed(self, document: Document) -> None:
        """start == end is degenerate but not malformed."""
        metadata = ChunkMetadata(
            document_id=document.id, source="a.pdf", start_char=7, end_char=7, chunk_index=0
        )
        assert metadata.start_char == metadata.end_char

    @pytest.mark.parametrize("page_number", [0, -1])
    def test_page_numbers_are_one_indexed(self, document: Document, page_number: int) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ChunkMetadata(
                document_id=document.id,
                source="a.pdf",
                page_number=page_number,
                start_char=0,
                end_char=1,
                chunk_index=0,
            )

    @pytest.mark.parametrize("field", ["start_char", "chunk_index", "section_index"])
    def test_negative_offsets_rejected(self, document: Document, field: str) -> None:
        kwargs = {
            "document_id": document.id,
            "source": "a.pdf",
            "start_char": 0,
            "end_char": 10,
            "chunk_index": 0,
        }
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ChunkMetadata(**{**kwargs, field: -1})

    def test_section_index_optional(self, chunk: Chunk) -> None:
        """Alignment against the corpus can fail; that is a coverage finding,
        not a hard error."""
        metadata = ChunkMetadata(**{**chunk.metadata.model_dump(), "section_index": None})
        assert metadata.section_index is None

    def test_document_id_must_be_a_uuid(self) -> None:
        with pytest.raises(ValidationError, match="valid UUID"):
            ChunkMetadata(
                document_id="not-a-uuid", source="a.pdf", start_char=0, end_char=1, chunk_index=0
            )

    def test_uuid_string_is_coerced(self, document: Document) -> None:
        metadata = ChunkMetadata(
            document_id=str(document.id), source="a.pdf", start_char=0, end_char=1, chunk_index=0
        )
        assert isinstance(metadata.document_id, UUID)
        assert metadata.document_id == document.id


class TestRetrievalResult:
    def test_retriever_type_accepts_its_string_value(self, chunk: Chunk) -> None:
        """Configs and result files carry plain strings."""
        result = RetrievalResult(chunk=chunk, score=0.9, retriever_type="hybrid")
        assert result.retriever_type is RetrieverType.HYBRID
        assert result.retriever_type == "hybrid"

    def test_unknown_retriever_type_rejected(self, chunk: Chunk) -> None:
        with pytest.raises(ValidationError, match="'dense', 'bm25' or 'hybrid'"):
            RetrievalResult(chunk=chunk, score=0.9, retriever_type="sparse")

    @pytest.mark.parametrize("score", [-1.0, 0.0, 42.5])
    def test_scores_are_unbounded(self, chunk: Chunk, score: float) -> None:
        """Cosine is [-1, 1], BM25 is unbounded positive, rerankers differ again."""
        assert RetrievalResult(chunk=chunk, score=score, retriever_type="dense").score == score

    def test_serializes_enum_as_string(self, chunk: Chunk) -> None:
        result = RetrievalResult(chunk=chunk, score=0.5, retriever_type=RetrieverType.BM25)
        assert result.model_dump(mode="json")["retriever_type"] == "bm25"


class TestCitation:
    def test_from_chunk_copies_provenance(self, chunk: Chunk) -> None:
        citation = Citation.from_chunk(chunk, snippet="snippet", relevance_score=0.8)
        assert citation.chunk_id == chunk.id
        assert citation.source == chunk.metadata.source
        assert citation.page_number == chunk.metadata.page_number
        assert citation.text_snippet == "snippet"

    def test_from_chunk_defaults_snippet_to_full_content(self, chunk: Chunk) -> None:
        assert Citation.from_chunk(chunk).text_snippet == chunk.content

    def test_from_chunk_carries_missing_page_number(self, document: Document) -> None:
        pageless = Chunk(
            content="x",
            metadata=ChunkMetadata(
                document_id=document.id,
                source="a.pdf",
                start_char=0,
                end_char=1,
                chunk_index=0,
            ),
        )
        assert Citation.from_chunk(pageless).page_number is None

    def test_chunk_id_is_required(self) -> None:
        """A citation points at an existing chunk; there is nothing to generate."""
        with pytest.raises(ValidationError, match="chunk_id"):
            Citation(source="a.pdf", text_snippet="x")


class TestQAResponse:
    def test_collections_default_to_empty(self) -> None:
        response = QAResponse(query="q?", answer="a")
        assert (response.citations, response.chunks_used, response.confidence) == ([], [], None)

    def test_collection_defaults_are_not_shared(self) -> None:
        """A shared mutable default would leak citations between responses."""
        first, second = QAResponse(query="q", answer="a"), QAResponse(query="q", answer="a")
        first.citations.append(Citation(chunk_id=uuid4(), source="a.pdf", text_snippet="x"))
        assert second.citations == []

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_clamped_to_unit_interval(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            QAResponse(query="q", answer="a", confidence=confidence)

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_bounds_are_inclusive(self, confidence: float) -> None:
        assert QAResponse(query="q", answer="a", confidence=confidence).confidence == confidence


class TestRoundTrip:
    """Result files and index sidecars are written and re-read as JSON."""

    def test_qa_response_round_trips(self, chunk: Chunk) -> None:
        chunk.embedding = [0.1, 0.2, 0.3]
        original = QAResponse(
            query="What is attention?",
            answer="A weighting over positions [1].",
            citations=[Citation.from_chunk(chunk, snippet="snip", relevance_score=0.7)],
            chunks_used=[chunk],
            confidence=0.42,
        )
        assert QAResponse.model_validate_json(original.model_dump_json()) == original

    def test_document_round_trips_with_extra_metadata(self, document: Document) -> None:
        document.metadata.arxiv_version = "v7"
        restored = Document.model_validate_json(document.model_dump_json())
        assert restored == document
        assert restored.metadata.arxiv_version == "v7"

    def test_retrieval_result_round_trips(self, chunk: Chunk) -> None:
        original = RetrievalResult(chunk=chunk, score=0.87, retriever_type=RetrieverType.DENSE)
        assert RetrievalResult.model_validate_json(original.model_dump_json()) == original
