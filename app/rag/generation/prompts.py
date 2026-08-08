"""Context assembly and the versioned prompt files behind it.

The citation contract lives here: retrieved results are numbered ``[1]``,
``[2]``, ... in rank order, and that ordering is the *only* thing tying a
marker in the answer back to a chunk. So context building and citation parsing
must be handed the same list, in the same order — :mod:`app.rag.generation.citations`
takes the results list rather than a mapping for exactly that reason.

Numbering is what makes that contract cheap to honour. The alternative — asking
for ``(filename, page)`` and regexing it back out — depends on the model copying
a string correctly; a small integer index does not.

Prompt text is kept in ``prompts/answer/vN/`` rather than in this module so a
prompt revision is a diffable file with a version number that can be recorded
in a result file, not a code change buried in a string literal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.models import Chunk, RetrievalResult

log = logging.getLogger(__name__)

# app/rag/generation/prompts.py -> the project root is four levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "answer"

#: A version directory holds one of each: ``<name>_system.prompt`` and
#: ``<name>_user.template``. Matched by suffix rather than by exact filename so
#: the answer prompt and the judge prompt share this loader while keeping names
#: that say which they are when one is open in an editor.
SYSTEM_SUFFIX = "_system.prompt"
USER_SUFFIX = "_user.template"

#: The model says exactly this when the context does not answer the question.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

#: Characters of chunk text per passage — a backstop against an anomalous
#: chunk, not a routine trim. The chunker already bounds passage size, in
#: tokens and per spec, so a cap that fires on ordinary chunks would silently
#: undo that: retrieval scores the whole chunk, and the truncated tail may be
#: the part that matched. This corpus runs ~3.3 chars/token, and the longest
#: chunk any current spec produces is near 3,000 characters, so 5,000 leaves
#: room for a denser document without cutting anything the ranking counted.
DEFAULT_PASSAGE_CHARS = 5000

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
        return self.render(context=context, query=query)

    def render(self, **fields: str) -> str:
        """Fill the user template from named fields.

        The answer template takes ``context`` and ``query``; the judge template
        takes more. Both live under ``prompts/`` with the same version layout,
        so they share the loader and differ only in what they substitute.
        """
        return self.user_template.format(**fields)


def latest_version(prompts_dir: Path = PROMPTS_DIR) -> str:
    """Highest-numbered ``vN`` directory under ``prompts_dir``."""
    versions = [d for d in prompts_dir.glob("v*") if d.is_dir() and _VERSION_RE.fullmatch(d.name)]
    if not versions:
        raise FileNotFoundError(f"No prompt version directories (v1, v2, ...) in {prompts_dir}")
    return max(versions, key=lambda d: int(d.name[1:])).name


def _read_one(version_dir: Path, suffix: str) -> str:
    """The single ``*suffix`` file in ``version_dir``, as text."""
    matches = sorted(version_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} file in {version_dir}")
    if len(matches) > 1:
        # Picking one silently would make which prompt ran depend on sort order.
        raise FileNotFoundError(
            f"{len(matches)} *{suffix} files in {version_dir}: {[m.name for m in matches]}"
        )
    return matches[0].read_text(encoding="utf-8").strip()


def load_prompt(version: str | None = None, prompts_dir: Path = PROMPTS_DIR) -> AnswerPrompt:
    """Load ``<prompts_dir>/<version>/``; newest version when unspecified."""
    version = version or latest_version(prompts_dir)
    version_dir = prompts_dir / version
    try:
        system = _read_one(version_dir, SYSTEM_SUFFIX)
        user_template = _read_one(version_dir, USER_SUFFIX)
    except (OSError, FileNotFoundError) as error:
        raise FileNotFoundError(f"Incomplete prompt version at {version_dir}: {error}") from None
    return AnswerPrompt(version=version, system=system, user_template=user_template)


def chunk_label(chunk: Chunk, number: int) -> str:
    """The header line for one passage: ``[3] paper.pdf p7``.

    Source and page are shown even though the model cites by number — they give
    it something to reason about ("both passages come from the same paper") and
    they make a logged prompt readable when an answer looks wrong.
    """
    metadata = chunk.metadata
    page = f" p{metadata.page_number}" if metadata.page_number else ""
    return f"[{number}] {metadata.source}{page}"


def passage_label(result: RetrievalResult, number: int) -> str:
    """:func:`chunk_label` for a retrieval result."""
    return chunk_label(result.chunk, number)


def build_chunk_context(chunks: list[Chunk], *, passage_chars: int = DEFAULT_PASSAGE_CHARS) -> str:
    """Render chunks as numbered passages, ``[1]`` first.

    Whitespace is collapsed: chunk text carries PDF line breaks mid-sentence,
    and leaving them in costs tokens while making the passage harder to read.
    Over-long passages are truncated rather than dropped — a truncated passage
    still supports a citation, a missing one silently renumbers everything
    after it and breaks the marker-to-chunk mapping.

    Taking chunks rather than results is what lets the judge rebuild the exact
    context an answer was written from: a ``QAResponse`` keeps ``chunks_used``,
    in rank order, but not the scores that ranked them.
    """
    passages = []
    for number, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.content.split())
        if passage_chars and len(text) > passage_chars:
            log.debug(f"Passage [{number}] truncated: {len(text)} -> {passage_chars} chars")
            text = text[:passage_chars].rstrip() + " ..."
        passages.append(f"{chunk_label(chunk, number)}\n{text}")
    return "\n\n".join(passages)


def build_context(
    results: list[RetrievalResult], *, passage_chars: int = DEFAULT_PASSAGE_CHARS
) -> str:
    """:func:`build_chunk_context` for ranked retrieval results."""
    return build_chunk_context([result.chunk for result in results], passage_chars=passage_chars)
