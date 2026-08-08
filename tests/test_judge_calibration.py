"""Does the judge actually detect the flaw it is shown? Billed, opt-in.

``tests/test_judge.py`` proves the plumbing: that a score is parsed, averaged
and reported correctly. It cannot prove the thing that matters — that a 2 means
a worse answer than a 4. Only a real model can, and only against answers whose
defects are known in advance.

So each case below is an answer written to fail on named dimensions and hold up
on the rest, over synthetic passages whose content is fixed here rather than
retrieved. Knowing exactly what the context supports is the whole point: with
real passages, a disagreement between the judge and the expectation cannot be
told apart from a misreading of the source.

What is asserted is **direction, not calibration**. The planted defect must
score at or below 3 and the good baseline must score at or above 4 everywhere.
Untargeted dimensions are deliberately not asserted: the judge couples them —
it reads a brief answer as less relevant, and a misattributed marker as an
accuracy fault — and pinning that down here would turn a known, tolerable
weakness into a test that fails on rubric edits made for other reasons. See
``scratch/judge_calibration_findings.md`` for the measurements behind that.

Run with::

    JUDGE_CALIBRATION=1 .venv/bin/python -m pytest tests/test_judge_calibration.py

Costs roughly $0.004 per run on gpt-4.1-mini, and needs OPENAI_API_KEY.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.rag.evaluation.judge import DIMENSIONS, JudgeScore, LLMJudge
from app.rag.models import Chunk, ChunkMetadata, QAResponse

pytestmark = [
    pytest.mark.paid,
    pytest.mark.skipif(
        os.getenv("JUDGE_CALIBRATION") != "1",
        reason="billed API calls; set JUDGE_CALIBRATION=1 to run",
    ),
]

#: At or below this, a dimension is reporting a defect.
LOW = 3

#: At or above this, a dimension is reporting no defect.
HIGH = 4

QUESTION = "How were cells tracked across frames, and how accurate was the tracking?"

PASSAGES = [
    "Cells were segmented in each frame with a U-Net and linked across frames by "
    "nearest-neighbour matching within a 12-pixel radius. Tracks shorter than five "
    "frames were discarded.",
    "Live imaging was performed at 3-minute intervals for 18 hours on a spinning-disk "
    "confocal microscope with a 40x water immersion objective.",
    "Track accuracy was evaluated against 240 manually annotated trajectories, giving "
    "a linking precision of 0.91 and a recall of 0.87.",
    "Dividing cells were handled by allowing one-to-two assignments in the linking "
    "step, and each daughter cell inherited a new track identity.",
    "The XGBoost classifier predicting active cells per frame reached an F1 of 0.78 "
    "on held-out wells.",
]

GOOD_ANSWER = (
    "Cells were segmented in each frame with a U-Net, then linked across frames by "
    "nearest-neighbour matching within a 12-pixel radius, with tracks shorter than "
    "five frames discarded [1]. Division was handled by allowing one-to-two "
    "assignments during linking, with each daughter cell given a new track identity "
    "[4]. Imaging ran at 3-minute intervals for 18 hours on a spinning-disk confocal "
    "microscope [2]. Against 240 manually annotated trajectories the linking reached "
    "a precision of 0.91 and a recall of 0.87 [3]."
)

#: ``(label, answer, the dimensions the answer was written to fail)``.
CASES = [
    (
        # Accurate and correctly cited, but about the classifier, not tracking.
        "off-topic",
        "An XGBoost classifier was used to predict the number of active cells in each "
        "frame, and it reached an F1 of 0.78 on held-out wells [5]. Imaging was carried "
        "out on a spinning-disk confocal microscope with a 40x water immersion objective "
        "[2].",
        ("relevance", "completeness"),
    ),
    (
        # Fluent and well-formed, with a method and a number found in no passage.
        "hallucinated detail",
        "Cells were tracked with a Kalman filter initialised from the U-Net segmentation "
        "and refined by a Hungarian assignment step over 30 frames [1]. The tracker ran "
        "on 1,400 annotated trajectories and reached a precision of 0.97 [3].",
        ("accuracy",),
    ),
    (
        # The good answer with every marker stripped out.
        "correct but uncited",
        "Cells were segmented in each frame with a U-Net and linked by nearest-neighbour "
        "matching within a 12-pixel radius, discarding tracks shorter than five frames. "
        "Dividing cells were handled with one-to-two assignments, each daughter taking a "
        "new track identity. Evaluated against 240 manually annotated trajectories, "
        "linking precision was 0.91 and recall 0.87.",
        ("citation_quality",),
    ),
    (
        # True, cited, on topic, and one sentence where the passages support four.
        "thin but true",
        "Cells were linked across frames by nearest-neighbour matching [1].",
        ("completeness",),
    ),
    (
        # Right content, markers pointing elsewhere, including a passage that does
        # not exist — the failure the positional citation contract has to catch.
        "misattributed markers",
        "Cells were segmented with a U-Net and linked by nearest-neighbour matching "
        "within a 12-pixel radius [5]. Linking precision was 0.91 and recall was 0.87 "
        "[2], measured against 240 annotated trajectories [7].",
        ("citation_quality",),
    ),
    (
        "bad on everything",
        "The study shows that cell tracking is a well-understood problem and that modern "
        "deep learning approaches now solve it almost perfectly in most laboratory "
        "settings, so accuracy is rarely a concern.",
        DIMENSIONS,
    ),
]


def make_response(answer: str) -> QAResponse:
    """The same five passages behind every case, numbered [1]..[5]."""
    chunks = [
        Chunk(
            content=text,
            metadata=ChunkMetadata(
                document_id=uuid4(),
                source="tracking_methods.pdf",
                page_number=3 + index // 2,
                start_char=index * 400,
                end_char=index * 400 + len(text),
                chunk_index=index,
            ),
        )
        for index, text in enumerate(PASSAGES)
    ]
    return QAResponse(query=QUESTION, answer=answer, chunks_used=chunks)


def report(label: str, score: JudgeScore) -> str:
    """Scores and the judge's reasoning, so a failure says why it failed."""
    scores = " ".join(f"{name} {getattr(score, name)}" for name in DIMENSIONS)
    return f"{label}: {scores}\n  {score.rationale}"


@pytest.fixture(scope="module")
def judge() -> LLMJudge:
    pytest.importorskip("openai")
    from app.rag.generation import OpenAILLM
    from openai_client.openai_client import load_env_var_from_profile

    # Resolved the way MyOpenAIClient resolves it: the environment first, then
    # ~/.profile. Checking only the environment skips on a working setup.
    if not (os.getenv("OPENAI_API_KEY") or load_env_var_from_profile("OPENAI_API_KEY")):
        pytest.skip("no OPENAI_API_KEY in the environment or ~/.profile")
    return LLMJudge(OpenAILLM())


def test_a_good_answer_scores_well_everywhere(judge: LLMJudge) -> None:
    """Without this, a judge that returns 1s for everything would pass the rest."""
    score = judge.score(make_response(GOOD_ANSWER))
    weak = [name for name in DIMENSIONS if getattr(score, name) < HIGH]
    assert not weak, report("good baseline", score)


@pytest.mark.parametrize(("label", "answer", "expected_low"), CASES, ids=[c[0] for c in CASES])
def test_the_planted_defect_is_detected(
    judge: LLMJudge, label: str, answer: str, expected_low: tuple[str, ...]
) -> None:
    score = judge.score(make_response(answer))
    missed = [name for name in expected_low if getattr(score, name) > LOW]
    assert not missed, f"scored {missed} above {LOW} — {report(label, score)}"
