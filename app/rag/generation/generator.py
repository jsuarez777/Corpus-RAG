"""Retrieve, prompt, generate, resolve citations — one query to one QAResponse.

This is the seam the rest of the system talks to: the CLI, the Streamlit app
and the judge all call :meth:`AnswerGenerator.answer` and get back the same
model. Which retriever, which LLM and which prompt version are constructor
arguments, so swapping any of them is a config change, not a code change.

Two cases never reach the API. An empty result set has nothing to answer from,
and asking anyway invites the model to answer from its own training data — the
one failure a RAG evaluation must not let through. And when the model replies
``INSUFFICIENT_CONTEXT``, that is recorded as an abstention at confidence 0.0
rather than passed on as prose, so the evaluation can count abstentions apart
from wrong answers.
"""

from __future__ import annotations

import logging
import time

from app.rag.base import BaseLLM, BaseReranker, BaseRetriever
from app.rag.generation.citations import (
    DEFAULT_SNIPPET_CHARS,
    parse_citations,
    unresolved_markers,
)
from app.rag.generation.prompts import (
    DEFAULT_PASSAGE_CHARS,
    INSUFFICIENT,
    AnswerPrompt,
    build_context,
    load_prompt,
)
from app.rag.models import QAResponse, RetrievalResult

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

#: How many candidates to fetch for a reranker to choose from, when the caller
#: does not say. Four times the passages actually kept: a reranker earns its
#: cost by narrowing, and fetching exactly what you keep leaves it nothing to
#: narrow. Measured on 488 queries, reranking 10 candidates down to 1 moved
#: hit@1 by +0.033 while reranking 10 down to 10 moved it by exactly zero.
DEFAULT_RERANK_DEPTH_FACTOR = 4

#: Answer returned without calling the model when retrieval comes back empty.
NO_CONTEXT_ANSWER = "No relevant passages were retrieved for this question."

#: Answer recorded when the model declines. Kept distinct from the above: one
#: means retrieval found nothing, the other means what it found did not answer.
ABSTAINED_ANSWER = "The retrieved passages do not answer this question."


class AnswerGenerator:
    """Answers a question from retrieved context, with resolvable citations."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        *,
        top_k: int = DEFAULT_TOP_K,
        prompt: AnswerPrompt | None = None,
        reranker: BaseReranker | None = None,
        rerank_depth: int | None = None,
        passage_chars: int = DEFAULT_PASSAGE_CHARS,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
        temperature: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.prompt = prompt or load_prompt()
        self.reranker = reranker
        self.rerank_depth = rerank_depth
        self.passage_chars = passage_chars
        self.snippet_chars = snippet_chars
        self.temperature = temperature

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve, and rerank when a reranker is configured.

        With a reranker the retriever is asked for more than will be kept, so
        the reranker has candidates to choose between. Fetching exactly ``k``
        and reranking to ``k`` returns the same passages in a different order —
        the model reads all of them either way, so the reordering is nearly all
        the effect there is.
        """
        k = self.top_k if top_k is None else top_k
        depth = k
        if self.reranker is not None:
            depth = max(k, self.rerank_depth or k * DEFAULT_RERANK_DEPTH_FACTOR)

        results = self.retriever.retrieve(query, top_k=depth)
        if self.reranker is not None and results:
            results = self.reranker.rerank(query, results, top_k=k)
        return results

    def answer(self, query: str, *, top_k: int | None = None) -> QAResponse:
        """Retrieve context for ``query`` and answer from it."""
        query = query.strip()
        if not query:
            raise ValueError("query is empty")

        started = time.perf_counter()
        results = self.retrieve(query, top_k)
        log.info(
            f"Retrieved {len(results)} passage(s) in {(time.perf_counter() - started) * 1000:.0f}ms"
        )
        return self.answer_from(query, results)

    def answer_from(self, query: str, results: list[RetrievalResult]) -> QAResponse:
        """Answer from results already in hand.

        The path the experiment grid takes: retrieval metrics are computed over
        these same results, so re-running the retriever to generate would risk
        scoring one result set and answering from another.
        """
        if not results:
            log.warning(f"No passages retrieved for {query!r}; skipping the model call")
            return QAResponse(query=query, answer=NO_CONTEXT_ANSWER, confidence=0.0)

        context = build_context(results, passage_chars=self.passage_chars)
        user_prompt = self.prompt.format(query, context)
        log.debug(f"Prompt ({self.prompt.version}, {len(results)} passages):\n{user_prompt}")

        started = time.perf_counter()
        raw = self.llm.generate(
            user_prompt,
            system=self.prompt.system,
            **({} if self.temperature is None else {"temperature": self.temperature}),
        )
        log.info(f"Generated in {time.perf_counter() - started:.2f}s")

        chunks = [result.chunk for result in results]
        if raw.strip().upper().startswith(INSUFFICIENT):
            log.info(f"Model abstained on {query!r}")
            return QAResponse(
                query=query, answer=ABSTAINED_ANSWER, chunks_used=chunks, confidence=0.0
            )

        stray = unresolved_markers(raw, results)
        if stray:
            log.warning(f"Answer cites passages that do not exist: {stray}")
        citations = parse_citations(raw, results, snippet_chars=self.snippet_chars)
        if not citations:
            log.warning(f"Answer for {query!r} cites nothing")

        # confidence is left None: nothing here measures it. The LLM judge
        # scores answers in stage 6, and a made-up number now would be mistaken
        # for that one later.
        return QAResponse(query=query, answer=raw, citations=citations, chunks_used=chunks)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.retriever!r}, {self.llm!r}, "
            f"top_k={self.top_k}, prompt={self.prompt.version!r})"
        )
