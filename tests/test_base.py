"""Tests for the component interfaces.

Two things matter here: an incomplete implementation must fail loudly at
construction rather than at some later call, and the concrete helpers the ABCs
provide (``chunk_many``, ``embed_query``, ``embed_chunks``) must behave for any
subclass that only implements the abstract methods.
"""

from pathlib import Path

import pytest

from app.rag.base import (
    BaseChunker,
    BaseEmbedder,
    BaseLLM,
    BaseLoader,
    BaseReranker,
    BaseRetriever,
    BaseVectorStore,
)
from app.rag.models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    RetrievalResult,
    RetrieverType,
)

ALL_ABCS = [
    BaseLoader,
    BaseChunker,
    BaseEmbedder,
    BaseVectorStore,
    BaseRetriever,
    BaseReranker,
    BaseLLM,
]


class WordChunker(BaseChunker):
    """Minimal chunker: one chunk per word, with real offsets."""

    def chunk(self, document: Document) -> list[Chunk]:
        chunks, cursor = [], 0
        for index, word in enumerate(document.content.split()):
            start = document.content.index(word, cursor)
            cursor = start + len(word)
            chunks.append(
                Chunk(
                    content=word,
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        source=document.metadata.source,
                        start_char=start,
                        end_char=cursor,
                        chunk_index=index,
                    ),
                )
            )
        return chunks


class LengthEmbedder(BaseEmbedder):
    """Deterministic stand-in: a vector derived from the text's length."""

    @property
    def dimension(self) -> int:
        return 2

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class MiscountingEmbedder(LengthEmbedder):
    """Returns too few vectors — the failure ``embed_chunks`` must not hide."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return super().embed_texts(texts)[:-1]


@pytest.fixture
def document() -> Document:
    return Document(content="alpha beta gamma", metadata=DocumentMetadata(source="a.pdf"))


@pytest.mark.parametrize("abc", ALL_ABCS, ids=lambda cls: cls.__name__)
def test_abstract_classes_cannot_be_instantiated(abc: type) -> None:
    with pytest.raises(TypeError, match="abstract"):
        abc()


def test_incomplete_implementation_fails_at_construction() -> None:
    """Missing methods must surface immediately, not mid-pipeline."""

    class HalfChunker(BaseChunker):
        pass

    with pytest.raises(TypeError, match="chunk"):
        HalfChunker()


class TestBaseLoader:
    """``load_many`` is the only concrete behaviour the ABC provides."""

    class StemLoader(BaseLoader):
        """Stand-in loader: the filename stem becomes the content."""

        def load(self, path: Path) -> Document:
            return Document(
                content=path.stem, metadata=DocumentMetadata(source=path.name, page_count=1)
            )

    def test_load_many_preserves_order(self) -> None:
        documents = self.StemLoader().load_many([Path("b.pdf"), Path("a.pdf")])
        assert [d.content for d in documents] == ["b", "a"]

    def test_load_many_of_nothing_is_empty(self) -> None:
        assert self.StemLoader().load_many([]) == []


class TestBaseChunker:
    def test_chunk_produces_usable_offsets(self, document: Document) -> None:
        chunks = WordChunker().chunk(document)
        assert [c.content for c in chunks] == ["alpha", "beta", "gamma"]
        for chunk in chunks:
            span = document.content[chunk.metadata.start_char : chunk.metadata.end_char]
            assert span == chunk.content

    def test_chunk_indices_are_sequential(self, document: Document) -> None:
        chunks = WordChunker().chunk(document)
        assert [c.metadata.chunk_index for c in chunks] == [0, 1, 2]

    def test_chunk_many_concatenates_in_document_order(self, document: Document) -> None:
        other = Document(content="delta", metadata=DocumentMetadata(source="b.pdf"))
        chunks = WordChunker().chunk_many([document, other])
        assert [c.content for c in chunks] == ["alpha", "beta", "gamma", "delta"]

    def test_chunk_many_of_nothing_is_empty(self) -> None:
        assert WordChunker().chunk_many([]) == []


class TestBaseEmbedder:
    def test_embed_query_returns_a_single_vector(self) -> None:
        assert LengthEmbedder().embed_query("abcd") == [4.0, 1.0]

    def test_embed_chunks_attaches_vectors_in_place(self, document: Document) -> None:
        chunks = WordChunker().chunk(document)
        returned = LengthEmbedder().embed_chunks(chunks)

        assert returned is chunks  # documented as in-place
        assert [c.embedding for c in chunks] == [[5.0, 1.0], [4.0, 1.0], [5.0, 1.0]]

    def test_embed_chunks_of_nothing_is_empty(self) -> None:
        assert LengthEmbedder().embed_chunks([]) == []

    def test_vector_count_mismatch_raises(self, document: Document) -> None:
        """A backend dropping a vector would otherwise silently truncate."""
        chunks = WordChunker().chunk(document)
        with pytest.raises(ValueError, match="argument 2 is shorter"):
            MiscountingEmbedder().embed_chunks(chunks)


class TestBaseVectorStore:
    def test_len_defaults_to_tracked_chunks(self, document: Document) -> None:
        class MemoryStore(BaseVectorStore):
            def __init__(self) -> None:
                self._chunks: list[Chunk] = []

            def add(self, chunks: list[Chunk]) -> None:
                self._chunks.extend(chunks)

            def search(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
                return [
                    RetrievalResult(chunk=c, score=1.0, retriever_type=RetrieverType.DENSE)
                    for c in self._chunks[:top_k]
                ]

            def save(self, path: Path) -> None: ...

            def load(self, path: Path) -> None: ...

        store = MemoryStore()
        assert len(store) == 0
        store.add(WordChunker().chunk(document))
        assert len(store) == 3
        assert len(store.search([0.0], top_k=2)) == 2


class TestBaseRetriever:
    def test_retriever_type_tags_results(self, document: Document) -> None:
        class StubRetriever(BaseRetriever):
            @property
            def retriever_type(self) -> RetrieverType:
                return RetrieverType.BM25

            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                chunks = WordChunker().chunk(document)[:top_k]
                return [
                    RetrievalResult(
                        chunk=c, score=float(len(chunks) - i), retriever_type=self.retriever_type
                    )
                    for i, c in enumerate(chunks)
                ]

        results = StubRetriever().retrieve("alpha", top_k=2)
        assert len(results) == 2
        assert all(r.retriever_type is RetrieverType.BM25 for r in results)
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


class TestBaseRerankerAndLLM:
    def test_reranker_can_reorder_and_truncate(self, document: Document) -> None:
        class ReverseReranker(BaseReranker):
            def rerank(
                self, query: str, results: list[RetrievalResult], top_k: int | None = None
            ) -> list[RetrievalResult]:
                flipped = list(reversed(results))
                return flipped if top_k is None else flipped[:top_k]

        chunks = WordChunker().chunk(document)
        results = [
            RetrievalResult(chunk=c, score=float(i), retriever_type=RetrieverType.DENSE)
            for i, c in enumerate(chunks)
        ]
        reranked = ReverseReranker().rerank("q", results, top_k=2)
        assert [r.chunk.content for r in reranked] == ["gamma", "beta"]

    def test_llm_generate_signature_is_keyword_only_for_temperature(self) -> None:
        class EchoLLM(BaseLLM):
            def generate(self, prompt: str, *, temperature: float = 0.0, **kwargs) -> str:
                return f"{prompt}@{temperature}"

        assert EchoLLM().generate("hi") == "hi@0.0"
        assert EchoLLM().generate("hi", temperature=0.7) == "hi@0.7"
