"""``BaseLLM`` over ``openai_client``.

Thin on purpose: the prompt lives in :mod:`app.rag.generation.prompts`, the
answer assembly in :mod:`app.rag.generation.generator`. What this class owns is
the two things that only make sense next to the API call — retries, and the
per-call cost accounting ``openai_client.pricing`` makes available.

**Retries are the SDK's, deliberately.** ``openai_client`` defaults to
``max_retries=0`` and leaves retrying to the application; this class opts back
in, because the decision needs information that does not reach this layer. A 429
carries ``retry-after-ms`` telling the caller how long to wait, and the SDK is
the only code holding the response headers — everything above it sees a
``RateLimitError`` with no timing in it. Given the header it waits exactly that
long, refuses to retry at all past two minutes, and jitters its own fallback
backoff so concurrent callers do not retry in lockstep.

That last point is why this changed. Batch answering and judging run several
calls at once, and a hand-rolled fixed backoff had every worker hit the limit
together, sleep the same two seconds, and retry together. Retrying blind is
worse than not retrying in a pool.

The ``openai`` package is imported lazily, inside the methods that need it, so
that importing this module (and the tests for context building and citation
parsing) does not require the SDK or an API key to be present.
"""

from __future__ import annotations

import logging
from typing import Any

from app.rag.base import BaseLLM

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"

#: Deterministic by default. Answers feed the judge and the experiment grid,
#: and a config that cannot be re-run to the same numbers cannot be compared.
DEFAULT_TEMPERATURE = 0.0

#: Transport retries, performed by the SDK. Five attempts is enough to ride out
#: the rate limiting a pool of workers causes itself; each wait is whatever the
#: server asked for, or a jittered ``0.5 * 2**n`` when it asked for nothing.
DEFAULT_MAX_RETRIES = 4


def _transient_errors() -> tuple[type[Exception], ...]:
    """OpenAI exception types worth retrying. Imported lazily with the SDK."""
    import openai

    return (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)


class OpenAILLM(BaseLLM):
    """Generates text with an OpenAI model through ``MyOpenAIClient``.

    Tracks cumulative token usage and cost across calls, so a grid run can
    report what it spent without every caller threading a counter through.
    """

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        from openai_client.openai_client import MyOpenAIClient

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        # Temperature is passed per call rather than held on the client, so a
        # judge at 0.0 and an answer at 0.3 can share one instance.
        self._client = MyOpenAIClient(model=model, api_key=api_key, max_retries=max_retries)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Complete ``prompt`` and return the response text.

        ``system`` becomes a system message ahead of the prompt. Remaining
        keyword arguments (``max_output_tokens``, ``seed``, ...) pass through
        to the responses API untouched.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._call(
            messages,
            self.temperature if temperature is None else temperature,
            kwargs,
        )
        self._record_usage(response)
        return (response.output_text or "").strip()

    def _call(self, messages: list[dict[str, str]], temperature: float, kwargs: dict[str, Any]):
        """One API call. Retries happen inside the SDK; this names the failure.

        No loop of its own: the SDK has already retried
        :attr:`max_retries` times by the time an exception reaches here, so a
        second loop around it would multiply the attempts and compound the
        sleeps. What is left worth doing is saying which model failed, since the
        SDK's message does not, and a grid run has several in flight.
        """
        try:
            return self._client.query(
                input=messages, model=self.model, temperature=temperature, **kwargs
            )
        except _transient_errors() as error:
            raise RuntimeError(
                f"{self.model} failed after {self.max_retries} SDK retries: {error}"
            ) from error

    def _record_usage(self, response: Any) -> None:
        """Accumulate tokens from one response. Absent usage is not an error —
        it costs nothing to answer, and losing an answer over a missing counter
        would be the wrong trade."""
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            log.debug("Response carried no usage; cost total is now an undercount")
            return
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        self.input_tokens += (getattr(usage, "input_tokens", 0) or 0) - cached
        self.cached_input_tokens += cached
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    @property
    def cost_usd(self) -> float | None:
        """Cumulative spend, or None when the model is absent from the pricing CSV."""
        from openai_client.pricing import cost_usd

        try:
            return cost_usd(
                self.model, self.input_tokens, self.output_tokens, self.cached_input_tokens
            )
        except KeyError:
            log.debug(f"No pricing entry for {self.model}; cost unavailable")
            return None

    def usage_summary(self) -> str:
        """One line for a log or a result file."""
        total_in = self.input_tokens + self.cached_input_tokens
        cost = self.cost_usd
        priced = "cost unknown" if cost is None else f"${cost:.4f}"
        return (
            f"{self.model}: {self.calls} call(s), "
            f"{total_in:,} in / {self.output_tokens:,} out tokens, {priced}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model!r}, temperature={self.temperature})"
