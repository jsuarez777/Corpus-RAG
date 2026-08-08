"""Cohere's hosted reranker.

The API takes the query and the candidate documents in one call and returns
indices with relevance scores, already in ``[0, 1]`` and already sorted. So
unlike the local cross-encoder there is nothing to squash — but the two are
deliberately put on the same scale so a chart can hold both.

Two operational details that are not optional in practice:

* **Trial keys are rate limited per rolling minute.** A grid sweep hits that
  ceiling almost immediately, and the useful response is to wait out the window
  rather than to fail the run — a 429 here is a pause, not an error.
* **The key is read from the environment, and from ``~/.profile`` if it is not
  there.** Exporting a key in a shell profile and then finding that a launched
  process cannot see it is a common half-hour, and reading the file directly
  costs one function.
"""

from __future__ import annotations

import logging
import os
import re
import time
from functools import cached_property
from pathlib import Path

from app.rag.base import BaseReranker
from app.rag.models import RetrievalResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "rerank-v3.5"

#: Checked in order. CO_API_KEY is what the older SDK documented.
KEY_VARIABLES = ("COHERE_API_KEY", "CO_API_KEY")

#: Where to look when the environment has nothing.
PROFILE_FILES = ("~/.profile", "~/.bash_profile", "~/.zshrc")

DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 60.0

_EXPORT_RE = re.compile(
    r"^\s*(?:export\s+)?(COHERE_API_KEY|CO_API_KEY)\s*=\s*[\"']?([^\"'\s#]+)", re.MULTILINE
)


def find_api_key() -> str | None:
    """The Cohere key from the environment, falling back to shell profiles."""
    for variable in KEY_VARIABLES:
        value = os.environ.get(variable)
        if value:
            return value

    for candidate in PROFILE_FILES:
        path = Path(candidate).expanduser()
        try:
            match = _EXPORT_RE.search(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if match:
            log.debug(f"Read Cohere key from {candidate}")
            return match.group(2)
    return None


def is_rate_limit(error: Exception) -> bool:
    """True for a 429, without importing Cohere's exception hierarchy.

    The SDK renames these classes between majors, and matching on the status
    code keeps this from breaking on an upgrade that changes nothing else.
    """
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status == 429:
        return True
    return "429" in str(error) or "rate limit" in str(error).lower()


class CohereReranker(BaseReranker):
    """Rerank via Cohere's ``/rerank`` endpoint.

    Needs ``COHERE_API_KEY``; construction succeeds without one so a registry
    sweep does not require credentials, and the error arrives on first use
    naming the variable to set.
    """

    name = "cohere"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self.alias = model
        self.model_id = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    @cached_property
    def _client(self):
        try:
            import cohere
        except ImportError:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The Cohere reranker needs the `cohere` package: pip install cohere"
            ) from None

        key = self.api_key or find_api_key()
        if not key:
            raise RuntimeError(
                f"No Cohere API key. Set one of {', '.join(KEY_VARIABLES)}, "
                "or use the local reranker: cross_encoder"
            )
        return cohere.ClientV2(api_key=key)

    def _call(self, query: str, documents: list[str], top_n: int):
        """One rerank call, waiting out trial-key rate limits."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._client.rerank(
                    model=self.model_id, query=query, documents=documents, top_n=top_n
                )
            except Exception as error:
                if not is_rate_limit(error) or attempt == self.max_retries:
                    raise
                # The trial limit is per rolling minute, so the wait is a fixed
                # window rather than an exponential backoff — doubling would
                # sleep far past the point the quota came back.
                log.warning(
                    f"Cohere rate limit (attempt {attempt}/{self.max_retries}); "
                    f"waiting {self.backoff_seconds:.0f}s"
                )
                time.sleep(self.backoff_seconds)
        raise RuntimeError("unreachable")  # pragma: no cover

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Rescore and reorder ``results`` using Cohere's relevance scores."""
        if not results:
            return []

        documents = [result.chunk.content for result in results]
        response = self._call(query, documents, top_n=top_k or len(results))

        return [
            results[item.index].model_copy(
                update={
                    "score": float(item.relevance_score),
                    "original_rank": results[item.index].original_rank or item.index + 1,
                }
            )
            for item in response.results
        ]

    def __repr__(self) -> str:
        return f"CohereReranker({self.alias!r})"
