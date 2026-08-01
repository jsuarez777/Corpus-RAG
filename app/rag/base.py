"""Abstract base classes for the swappable pipeline components.

Every stage is defined by one of these interfaces, so a YAML config can name a
concrete class and the caller never changes. Implementations live under
``rag/chunking``, ``rag/embedding``, ``rag/stores``, ``rag/retrieval``,
``rag/reranking`` and ``rag/generation``.

The contracts these impose, which the concrete classes are expected to honour:

* Loaders finalize ``Document.content`` — cleaning included — before recording
  ``page_starts``, so every downstream offset indexes the same string.
* Chunkers set ``document_id``, ``chunk_index``, ``start_char`` and ``end_char``
  on every chunk they emit — the evaluation stage aligns on those offsets.
* Embedders return one vector per input, in order, all of ``dimension`` length.
* Retrievers return results sorted by descending ``score``, at most ``top_k``.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from app.rag.models import Chunk, Document, RetrievalResult, RetrieverType


class BaseLoader(ABC):
    """Extracts a source file into a Document with page-offset provenance.

    Not in the spec's ABC list, but "PDF Loader" is a component the spec
    requires to be swappable, and swapping needs an interface to swap behind.
    """

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Read one file into a Document.

        Implementations set ``page_starts`` (char offset of each page) and
        ``paper_id`` on the metadata — ``DocumentMetadata`` allows extras — so
        chunk offsets map back to pages and chunks map back to qrels. Any text
        cleaning happens *before* those offsets are taken; cleaning afterwards
        would silently invalidate every offset in the system.
        """

    def load_many(self, paths: list[Path]) -> list[Document]:
        """Load a corpus. Override when a backend can batch across files."""
        return [self.load(path) for path in paths]


class BaseChunker(ABC):
    """Splits a document's text into retrievable chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` into chunks carrying offsets back into its content."""

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a corpus. Override when a strategy can batch across documents."""
        return [chunk for document in documents for chunk in self.chunk(document)]


class BaseEmbedder(ABC):
    """Turns text into dense vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this embedder produces."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings, returning one vector per input, in order."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query. Override for models with asymmetric
        query/passage prefixes, where a query is encoded differently."""
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Attach an embedding to each chunk, in place, and return them."""
        vectors = self.embed_texts([chunk.content for chunk in chunks])
        # strict: a backend returning the wrong count is a bug worth raising on,
        # not something to paper over by silently dropping the tail.
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        return chunks


class BaseVectorStore(ABC):
    """Persists chunk vectors and answers nearest-neighbour queries."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Index ``chunks``; each must already carry an embedding."""

    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """Return the ``top_k`` nearest chunks, highest score first."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write the index and its chunk payloads under ``path``."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore an index previously written by :meth:`save`."""

    def __len__(self) -> int:
        """Number of indexed chunks. Override when the count is not tracked
        by a ``_chunks`` attribute."""
        return len(getattr(self, "_chunks", ()))


class BaseRetriever(ABC):
    """Finds the chunks most relevant to a query."""

    @property
    @abstractmethod
    def retriever_type(self) -> RetrieverType:
        """Tag stamped onto every result, so mixed result sets stay attributable."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return at most ``top_k`` results, sorted by descending score."""


class BaseReranker(ABC):
    """Reorders first-stage results with a stronger, slower model."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Rescore and reorder ``results``, keeping at most ``top_k``.

        Implementations replace ``score`` with the reranker's own scale, so
        reranked scores are not comparable with the retriever's.
        """


class BaseLLM(ABC):
    """Generates text. Wraps ``openai_client`` or any other provider."""

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.0, **kwargs) -> str:
        """Complete ``prompt`` and return the raw text response."""
