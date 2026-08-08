"""LLM-as-judge: scoring generated answers on four dimensions, 1-5.

Retrieval metrics say whether the right passage was found. They say nothing
about what the model did with it — an answer can cite the correct chunk and
still misread it, pad it, or drop the half that mattered. This module is the
other half of the measurement: relevance, accuracy, completeness and citation
quality, scored by a model given the same passages the answer was written from.

Three choices worth stating:

* **The judge sees the context, not just the answer.** Accuracy and citation
  quality are claims about a relationship between the answer and the passages,
  and cannot be scored from the answer alone. This roughly doubles the tokens
  per query; there is no cheaper way to ask the question.
* **The reference answer is optional.** The benchmark ships ground-truth
  answers, and they sharpen accuracy scoring, but the judge has to work on
  ad-hoc questions too, so it grades against the passages and treats the
  reference as corroboration when present.
* **Structured output, not prose parsed after the fact.** The schema at
  :data:`SCORE_SCHEMA` constrains the model to four integers and a rationale,
  so a malformed score is a request the API rejects rather than a regex that
  quietly returns 3. Dimensions are ``enum`` rather than ``minimum``/
  ``maximum`` because strict JSON schema mode supports the former.

Judging is a second LLM call per answer, at temperature 0. Scores from two
different judge models are not comparable, so the model name travels with the
report the same way the prompt version does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from app.rag.base import BaseLLM
from app.rag.generation.prompts import (
    DEFAULT_PASSAGE_CHARS,
    PROJECT_ROOT,
    AnswerPrompt,
    build_chunk_context,
    load_prompt,
)
from app.rag.models import QAResponse

log = logging.getLogger(__name__)

PROMPTS_DIR = PROJECT_ROOT / "prompts" / "judge"

#: Scored dimensions, in the order the spec lists and the radar chart plots.
DIMENSIONS = ("relevance", "accuracy", "completeness", "citation_quality")

SCALE = (1, 2, 3, 4, 5)

#: Below this mean, the answer prompt is not asking for citations clearly
#: enough — a prompt fix, not a retrieval fix.
CITATION_QUALITY_FLOOR = 3.5

#: The judge grades; it does not write. Temperature 0 so a re-run of the same
#: configuration reproduces the same scores.
DEFAULT_TEMPERATURE = 0.0

#: Strict JSON schema for one score. Hand-written rather than derived from
#: :class:`JudgeScore` because strict mode rejects the ``minimum``/``maximum``
#: keywords Pydantic emits for a bounded int.
SCORE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rationale", *DIMENSIONS],
    "properties": {
        # Asked for first so the model states its reasoning before committing
        # to numbers rather than justifying numbers it has already written.
        "rationale": {
            "type": "string",
            "description": "One or two sentences naming what set these scores.",
        },
        **{name: {"type": "integer", "enum": list(SCALE)} for name in DIMENSIONS},
    },
}

RESPONSE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "answer_score",
        "strict": True,
        "schema": SCORE_SCHEMA,
    }
}


class JudgeError(RuntimeError):
    """The judge returned something that is not a valid score."""


class JudgeScore(BaseModel):
    """One judged answer: four 1-5 scores and the reason behind them."""

    relevance: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    citation_quality: int = Field(ge=1, le=5)
    rationale: str = ""

    @property
    def average(self) -> float:
        """Mean of the four dimensions — the single number for a leaderboard."""
        return sum(getattr(self, name) for name in DIMENSIONS) / len(DIMENSIONS)

    @property
    def confidence(self) -> float:
        """The average rescaled to 0-1, for ``QAResponse.confidence``.

        1 is the floor of the scale, not zero quality, so the mapping is
        ``(average - 1) / 4``: a straight division would call a uniformly
        terrible answer 20% confident.
        """
        return (self.average - 1) / (len(SCALE) - 1)

    def as_dict(self) -> dict:
        return {**{name: getattr(self, name) for name in DIMENSIONS}, "average": self.average}


@dataclass
class JudgeReport:
    """Mean scores over a set of judged answers, with the detail behind them."""

    num_scored: int = 0
    means: dict[str, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)
    model: str = ""
    prompt_version: str = ""

    @property
    def citation_quality_is_low(self) -> bool:
        """True when the spec's 3.5 floor is breached.

        An unscored report is not a failing one — nothing was measured, so the
        floor has nothing to be below.
        """
        if not self.num_scored:
            return False
        return self.means.get("citation_quality", 0.0) < CITATION_QUALITY_FLOOR

    def summary(self) -> str:
        scores = " ".join(f"{name[:4]} {self.means.get(name, 0):.2f}" for name in DIMENSIONS)
        line = (
            f"{self.num_scored} answers judged | {scores} | avg {self.means.get('average', 0):.2f}"
        )
        if self.citation_quality_is_low:
            line += f" | citation quality below {CITATION_QUALITY_FLOOR} — revise the answer prompt"
        return line


class LLMJudge:
    """Scores answers with an LLM, one call per answer."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        prompt: AnswerPrompt | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        passage_chars: int = DEFAULT_PASSAGE_CHARS,
    ) -> None:
        self.llm = llm
        self.prompt = prompt or load_prompt(prompts_dir=PROMPTS_DIR)
        self.temperature = temperature
        self.passage_chars = passage_chars

    def score(self, response: QAResponse, *, reference: str | None = None) -> JudgeScore:
        """Score one answer. ``reference`` is the ground-truth answer, if any."""
        context = build_chunk_context(response.chunks_used, passage_chars=self.passage_chars)
        user_prompt = self.prompt.render(
            question=response.query,
            context=context or "(no passages were retrieved)",
            answer=response.answer,
            reference=f"\nReference answer:\n\n{reference.strip()}\n" if reference else "",
        )
        raw = self.llm.generate(
            user_prompt,
            system=self.prompt.system,
            temperature=self.temperature,
            text=RESPONSE_FORMAT,
        )
        return self._parse(raw)

    def score_all(
        self, responses: list[QAResponse], references: dict[str, str] | None = None
    ) -> JudgeReport:
        """Score every answer and average the results.

        ``references`` maps query text to a ground-truth answer. One bad score
        does not sink a grid run: a query the judge fails on is logged and left
        out of the means, because a partial report is still comparable and a
        crashed one is not.
        """
        references = references or {}
        scored: list[JudgeScore] = []
        per_query: list[dict] = []

        for response in responses:
            try:
                score = self.score(response, reference=references.get(response.query))
            except JudgeError as error:
                log.warning(f"Judge failed on {response.query!r}: {error}")
                continue
            scored.append(score)
            per_query.append(
                {"query": response.query, **score.as_dict(), "rationale": score.rationale}
            )

        means = (
            {name: sum(getattr(s, name) for s in scored) / len(scored) for name in DIMENSIONS}
            if scored
            else {}
        )
        if scored:
            means["average"] = sum(s.average for s in scored) / len(scored)

        return JudgeReport(
            num_scored=len(scored),
            means=means,
            per_query=per_query,
            model=getattr(self.llm, "model", ""),
            prompt_version=self.prompt.version,
        )

    @staticmethod
    def _parse(raw: str) -> JudgeScore:
        """Validate the model's JSON into a score, or say why it is not one."""
        try:
            return JudgeScore.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as error:
            raise JudgeError(f"Unusable score {raw[:200]!r}: {error}") from None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.llm!r}, prompt={self.prompt.version!r})"
