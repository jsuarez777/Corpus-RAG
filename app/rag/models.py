"""Typed payloads passed between pipeline stages.

These are the interfaces the components are graded on: every chunker, embedder,
retriever, reranker and LLM consumes and returns these models rather than plain
dicts, so implementations swap without callers changing.

Scores are deliberately left unbounded — cosine similarity over normalized
vectors is [-1, 1], BM25 is unbounded positive, and rerankers use their own
scale. Only ``confidence``, which is a probability-like quantity, is clamped.
"""

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrieverType(str, Enum):
    """Which retrieval path produced a result. Inherits str so the value
    serializes as a plain string in result files and YAML configs."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class DocumentMetadata(BaseModel):
    """Provenance for a source PDF.

    ``extra="allow"`` because loaders surface per-format fields (creation date,
    arXiv version, producer) that are worth keeping but not worth enumerating.
    """

    model_config = ConfigDict(extra="allow")

    source: str = Field(description="Filename or path the document was loaded from")
    title: str | None = None
    author: str | None = None
    page_count: int | None = Field(default=None, ge=0)


class ChunkMetadata(BaseModel):
    """Where a chunk came from, and how to score it against qrels.

    ``section_indices`` is assigned during ingestion by matching chunk text
    against ``corpus/{PAPER_ID}.json``; relevance is
    ``qrels[query].section_id in section_indices``.

    It is a list, not a single index, because chunk boundaries and section
    boundaries are set by different processes and do not line up: measured on
    this corpus, 4–14% of chunks straddle a section boundary depending on the
    chunker. Forcing a straddling chunk into one section leaves the other
    section with no chunks at all, and any query pointing at it then scores
    zero recall however good the retriever is.

    An empty list means alignment found nothing — a coverage figure to report,
    not an error to raise.
    """

    model_config = ConfigDict(extra="allow")

    document_id: UUID
    source: str = Field(description="Filename of the parent document")
    page_number: int | None = Field(default=None, ge=1, description="1-indexed")
    start_char: int = Field(ge=0, description="Offset into the parent document's content")
    end_char: int = Field(ge=0)
    chunk_index: int = Field(ge=0, description="Position of this chunk within its document")
    section_indices: list[int] = Field(
        default_factory=list, description="Every corpus section this chunk overlaps"
    )

    @model_validator(mode="after")
    def _check_span(self) -> "ChunkMetadata":
        if self.end_char < self.start_char:
            raise ValueError(f"end_char ({self.end_char}) precedes start_char ({self.start_char})")
        return self


class Document(BaseModel):
    """A whole source PDF after text extraction."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    """A retrievable span of a document, optionally with its vector attached.

    ``embedding`` is populated by the embedding stage rather than at
    construction, so it is mutable and defaults to None.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None

    @model_validator(mode="after")
    def _check_embedding(self) -> "Chunk":
        if self.embedding is not None and not self.embedding:
            raise ValueError("embedding must be None or a non-empty vector, not []")
        return self


class RetrievalResult(BaseModel):
    """A chunk plus the score the named retriever gave it."""

    chunk: Chunk
    score: float
    retriever_type: RetrieverType


class Citation(BaseModel):
    """One `[N]` marker in a generated answer, resolved back to its chunk."""

    chunk_id: UUID
    source: str = Field(description="Filename, for display to the reader")
    page_number: int | None = Field(default=None, ge=1)
    text_snippet: str
    relevance_score: float | None = None

    @classmethod
    def from_chunk(
        cls, chunk: Chunk, snippet: str | None = None, relevance_score: float | None = None
    ) -> "Citation":
        """Build a citation from the chunk a `[N]` marker referred to."""
        return cls(
            chunk_id=chunk.id,
            source=chunk.metadata.source,
            page_number=chunk.metadata.page_number,
            text_snippet=snippet if snippet is not None else chunk.content,
            relevance_score=relevance_score,
        )


class QAResponse(BaseModel):
    """The end-to-end answer for one query, with the evidence behind it."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks_used: list[Chunk] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
