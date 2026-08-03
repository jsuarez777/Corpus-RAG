"""Resolve the ``[N]`` markers in a generated answer back to their chunks.

Parsing is deliberately lenient about *form* and strict about *range*. The
prompt asks for ``[1][3]``, but models write ``[1, 3]`` and ``[1-3]`` often
enough that rejecting those would throw away citations the model got right.
A marker pointing at a passage that was never in the context is a different
matter: it is a hallucinated source, and it is reported rather than resolved.

Both functions take the same ordered ``results`` list that
:func:`app.rag.generation.prompts.build_context` was given — the numbering is
positional, so a different list means different citations.
"""

from __future__ import annotations

import logging
import re

from app.rag.models import Citation, RetrievalResult

log = logging.getLogger(__name__)

#: Snippet length in a Citation. Enough to recognise the passage in a UI
#: without pasting the whole chunk into every result file.
DEFAULT_SNIPPET_CHARS = 300

# A bracketed run of numbers: [2], [1, 3], [1-3], [1][2] (matched separately).
# Non-numeric brackets — "[sic]", "[Smith et al.]" — are not markers.
_MARKER = re.compile(r"\[\s*(\d+(?:\s*[,;-]\s*\d+)*)\s*\]")
_SEPARATOR = re.compile(r"\s*[,;]\s*")
_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")

#: Bounds on what counts as an expandable range. "[1-3]" is three passages;
#: "[2020-2024]" is a date and "[1-40]" is a mangled reference, and expanding
#: either would invent citations. Outside these bounds the two endpoints are
#: kept as written — parse_citations then drops whichever fall past the end of
#: the results, and unresolved_markers reports them.
MAX_RANGE = 10
MAX_PASSAGE = 100


def parse_markers(answer: str) -> list[int]:
    """Every passage number cited, in order of first appearance, deduplicated."""
    seen: dict[int, None] = {}
    for match in _MARKER.finditer(answer):
        for part in _SEPARATOR.split(match.group(1)):
            span = _RANGE.match(part)
            if span:
                low, high = int(span.group(1)), int(span.group(2))
                plausible = 0 < high - low <= MAX_RANGE and high <= MAX_PASSAGE
                numbers = range(low, high + 1) if plausible else (low, high)
            else:
                numbers = (int(part),)
            for number in numbers:
                seen.setdefault(number, None)
    return list(seen)


def unresolved_markers(answer: str, results: list[RetrievalResult]) -> list[int]:
    """Cited numbers with no passage behind them — the model's inventions.

    Worth surfacing rather than silently dropping: a rising count here is the
    signal that a prompt revision or a larger ``top_k`` has made the numbering
    harder for the model to keep straight.
    """
    return [n for n in parse_markers(answer) if not 1 <= n <= len(results)]


def parse_citations(
    answer: str,
    results: list[RetrievalResult],
    *,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> list[Citation]:
    """Build a :class:`~app.rag.models.Citation` for each resolvable marker.

    Ordered by first appearance in the answer, not by retrieval rank, so the
    citation list reads in the same order as the prose it came from. Passages
    the model did not cite produce no citation, however well they scored.
    """
    citations = []
    for number in parse_markers(answer):
        if not 1 <= number <= len(results):
            log.warning(f"Answer cites [{number}] but only {len(results)} passages were given")
            continue
        result = results[number - 1]
        snippet = " ".join(result.chunk.content.split())
        if snippet_chars and len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rstrip() + "..."
        citations.append(
            Citation.from_chunk(result.chunk, snippet=snippet, relevance_score=result.score)
        )
    return citations
