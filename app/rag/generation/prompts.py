"""Context assembly and the versioned prompt files behind it.

The citation contract lives here: retrieved results are numbered ``[1]``,
``[2]``, ... in rank order, and that ordering is the *only* thing tying a
marker in the answer back to a chunk. So context building and citation parsing
must be handed the same list, in the same order — :mod:`app.rag.generation.citations`
takes the results list rather than a mapping for exactly that reason.

Numbered context is mp4's Challenge 5 Approach 2. The alternative — asking for
``(filename, page)`` and regexing it back out — depends on the model copying a
string correctly; a small integer index does not.

Prompt text is kept in ``prompts/answer/vN/`` rather than in this module so a
prompt revision is a diffable file with a version number that can be recorded
in a result file, not a code change buried in a string literal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.models import RetrievalResult

log = logging.getLogger(__name__)

# app/rag/generation/prompts.py -> miniproject4/ is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "answer"

SYSTEM_FILE = "answer_system.prompt"
USER_FILE = "answer_user.template"

#: The model says exactly this when the context does not answer the question.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

#: Characters of chunk text per passage. Roughly 500 tokens, so a top_k of 10
#: still leaves a 4.1-mini call well inside its window with room for the answer.
DEFAULT_PASSAGE_CHARS = 2000

_VERSION_RE = re.compile(r"v(\d+)")


@dataclass(frozen=True)
class AnswerPrompt:
    """One version of the answer prompt, loaded from disk.

    ``version`` travels with the answer into result files: a judge score is
    only comparable across runs that used the same prompt.
    """

    version: str
    system: str
    user_template: str

    def format(self, query: str, context: str) -> str:
        """Fill the user template. ``system`` is passed to the LLM separately."""
        return self.user_template.format(context=context, query=query)


def latest_version(prompts_dir: Path = PROMPTS_DIR) -> str:
    """Highest-numbered ``vN`` directory under ``prompts_dir``."""
    versions = [d for d in prompts_dir.glob("v*") if d.is_dir() and _VERSION_RE.fullmatch(d.name)]
    if not versions:
        raise FileNotFoundError(f"No prompt version directories (v1, v2, ...) in {prompts_dir}")
    return max(versions, key=lambda d: int(d.name[1:])).name


def load_prompt(version: str | None = None, prompts_dir: Path = PROMPTS_DIR) -> AnswerPrompt:
    """Load ``prompts/answer/<version>/``; newest version when unspecified."""
    version = version or latest_version(prompts_dir)
    version_dir = prompts_dir / version
    try:
        system = (version_dir / SYSTEM_FILE).read_text(encoding="utf-8").strip()
        user_template = (version_dir / USER_FILE).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise FileNotFoundError(f"Incomplete prompt version at {version_dir}: {error}") from None
    return AnswerPrompt(version=version, system=system, user_template=user_template)


def passage_label(result: RetrievalResult, number: int) -> str:
    """The header line for one passage: ``[3] paper.pdf p7``.

    Source and page are shown even though the model cites by number — they give
    it something to reason about ("both passages come from the same paper") and
    they make a logged prompt readable when an answer looks wrong.
    """
    metadata = result.chunk.metadata
    page = f" p{metadata.page_number}" if metadata.page_number else ""
    return f"[{number}] {metadata.source}{page}"


def build_context(
    results: list[RetrievalResult], *, passage_chars: int = DEFAULT_PASSAGE_CHARS
) -> str:
    """Render ranked results as numbered passages, ``[1]`` first.

    Whitespace is collapsed: chunk text carries PDF line breaks mid-sentence,
    and leaving them in costs tokens while making the passage harder to read.
    Over-long passages are truncated rather than dropped — a truncated passage
    still supports a citation, a missing one silently renumbers everything
    after it and breaks the marker-to-chunk mapping.
    """
    passages = []
    for number, result in enumerate(results, start=1):
        text = " ".join(result.chunk.content.split())
        if passage_chars and len(text) > passage_chars:
            log.debug(f"Passage [{number}] truncated: {len(text)} -> {passage_chars} chars")
            text = text[:passage_chars].rstrip() + " ..."
        passages.append(f"{passage_label(result, number)}\n{text}")
    return "\n\n".join(passages)
