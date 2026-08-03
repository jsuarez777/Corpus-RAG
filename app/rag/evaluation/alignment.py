"""Locating corpus sections inside our own extraction, so qrels can score chunks.

``qrels.json`` labels relevance as ``(doc_id, section_id)``, where ``section_id``
indexes the ``sections`` array of ``corpus/{PAPER_ID}.json``. Our chunks are
spans of PyMuPDF text with character offsets and no idea sections exist. This
module is the bridge: it finds where each section begins in
``Document.content``, and then labels every chunk with the sections it overlaps.

The corpus is a **labelling key only**. Its text is never chunked, embedded or
retrieved — using it as the retrieval source would mean grading a pipeline that
skips the PDF extraction the project is about. Nor are unaligned chunks dropped:
they stay in the index as unlabelled candidates, because pruning the corpus with
ground truth would delete exactly the distractors a real system must rank
against.

**Matching cannot be exact.** The corpus was produced by Mistral OCR and holds
LaTeX source (``$\\operatorname{Corr}(m,[k])$``); ours comes from PyMuPDF and
holds the rendered glyphs. Nothing turns one into the other, so math is stripped
from the section text before matching and probes are drawn only from surviving
prose.

**Assignment is by overlap, not by midpoint.** A chunk belongs to every section
it touches. Forcing one section per chunk left 7 of 31 sections with no chunks
under ``fixed_size:512:128``, and a query whose gold section holds no chunks
scores zero recall no matter what the retriever does.

Measured on the 10-paper working set, 152 sections:

===========================================  =========  ============
approach                                      located    note
===========================================  =========  ============
whole section text, unconstrained search         87%     LaTeX defeats the probes
math stripped from probes                        95%     table of contents captures headings
each section searched forward of the last        99%     the two misses are pure-equation proofs
===========================================  =========  ============

The residual 1% is two sections of one paper that are almost entirely
``\\begin{aligned}`` blocks — they contain no 24 consecutive characters of prose
to match on. Queries against those two sections are unscoreable, which belongs
in RESULTS.md as a stated limit rather than being papered over.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.models import Chunk, Document

log = logging.getLogger(__name__)

# Probe length is a trade-off: long enough that a hit is not coincidence, short
# enough to fit between the LaTeX fragments that pepper these papers. Twelve
# probes spread through the section means a few unlucky ones do not lose it.
PROBE_LEN = 40
NUM_PROBES = 12

# Fallback for sections that are almost entirely equations, where 40 consecutive
# prose characters do not exist. Short probes risk coincidence, so they are used
# only when the full-length pass found nothing, only from the section's opening
# (its heading, the most distinctive text it has), and the result is flagged
# weak. The forward-only search in `locate_sections` is the backstop: a short
# probe cannot match anything before the previous section began.
FALLBACK_PROBE_LEN = 24
FALLBACK_STARTS = 4

# A match resting on one or two probes is reported but flagged: it is usually a
# section whose prose is almost entirely equations.
WEAK_EVIDENCE = 2

_ALNUM = re.compile(r"[a-z0-9]")

# Display math, inline math, environments, bare commands, and the corpus's own
# <table_id> / <image_id> placeholders. All are corpus-side artifacts with no
# counterpart in extracted PDF text.
_MATH = re.compile(
    r"\$\$.*?\$\$|\$[^$]*\$|\\begin\{.*?\}.*?\\end\{.*?\}|\\[a-zA-Z]+\s*|<[^>]{1,60}>",
    re.DOTALL,
)


@dataclass(frozen=True)
class SectionSpan:
    """Where one corpus section sits in a Document's content.

    ``probe_hits`` is how many of the probes drawn from this section were found.
    High agreement, plus spans that ascend across the paper, is what separates a
    real match from a coincidental n-gram.
    """

    section_index: int
    start_char: int
    end_char: int
    probe_hits: int

    @property
    def is_weak(self) -> bool:
        return self.probe_hits <= WEAK_EVIDENCE


def strip_math(text: str) -> str:
    """Remove LaTeX and placeholder markup, leaving prose that PDF text can match."""
    return _MATH.sub(" ", text)


def normalize(text: str) -> tuple[str, list[int]]:
    """Lowercase alphanumerics only, plus each kept character's original offset.

    Discarding whitespace and punctuation is what makes two different
    extractions of the same sentence comparable; the offset list is what turns
    a match back into a real position in ``Document.content``.
    """
    kept: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text.lower()):
        if _ALNUM.match(char):
            kept.append(char)
            offsets.append(index)
    return "".join(kept), offsets


def _probes(prose: str) -> list[tuple[int, str]]:
    """Evenly spaced (offset, probe) pairs drawn from a section's prose."""
    if len(prose) < PROBE_LEN:
        return [(0, prose)] if prose else []
    step = max(1, (len(prose) - PROBE_LEN) // NUM_PROBES)
    return [
        (start, prose[start : start + PROBE_LEN])
        for start in range(0, len(prose) - PROBE_LEN + 1, step)
    ]


def _locate_start(section_text: str, haystack: str, search_from: int = 0) -> tuple[int, int] | None:
    """(normalized start offset, probe hits) for one section, or None.

    ``search_from`` is where the previous section began. Searching forward from
    there is what defeats the table of contents: a TOC lists every heading in
    order near the front of the paper, so an unconstrained search matches
    section 8's heading against its own TOC line rather than against section 8.
    Measured on this corpus, that accounted for six of the eight sections that
    first appeared to be unlocatable.

    The section's start is taken from the *first* probe that hits — the one
    drawn closest to its opening, which is a heading and first sentence and so
    the most distinctive text it has. Later probes count only as corroboration;
    they cannot fix the position, because ``str.find`` returns the earliest
    occurrence and a recurring phrase would drag the estimate backwards.
    """
    prose, _ = normalize(strip_math(section_text))
    anchor: int | None = None
    hits = 0
    for offset, probe in _probes(prose):
        if len(probe) != PROBE_LEN:
            continue
        found = haystack.find(probe, search_from)
        if found == -1:
            continue
        hits += 1
        if anchor is None:
            anchor = max(search_from, found - offset)
    if anchor is not None:
        return anchor, hits

    # Nothing matched at full length: a proof section whose prose never runs 40
    # characters without an equation. Try short probes from the opening only.
    for offset in range(FALLBACK_STARTS):
        probe = prose[offset : offset + FALLBACK_PROBE_LEN]
        if len(probe) < FALLBACK_PROBE_LEN:
            break
        found = haystack.find(probe, search_from)
        if found != -1:
            return max(search_from, found - offset), 1
    return None


def locate_sections(document: Document, sections: list[dict]) -> list[SectionSpan]:
    """Find each section of ``sections`` within ``document.content``.

    Returns only the sections that were located, in document order. A section
    runs until the next located one begins, so the spans tile the document
    without gaps — chunks in an unlocated section are absorbed by its
    predecessor rather than going unlabelled.
    """
    flat, offsets = normalize(document.content)
    if not flat:
        return []

    # Each section is searched forward of the one before it, so results ascend
    # by construction and the table of contents cannot capture a heading.
    ordered: list[tuple[int, int, int]] = []  # (section_index, start_char, probe_hits)
    cursor = 0
    for index, section in enumerate(sections):
        result = _locate_start(section.get("text", ""), flat, cursor)
        if result is None:
            log.debug(f"Section {index} not located in {document.metadata.source}")
            continue
        normalized_start, hits = result
        if normalized_start >= len(offsets):
            continue
        ordered.append((index, offsets[normalized_start], hits))
        # +1 so the next section cannot resolve to the same position; a section
        # of zero length would make the spans ambiguous.
        cursor = normalized_start + 1

    spans = []
    for position, (index, start, hits) in enumerate(ordered):
        end = ordered[position + 1][1] if position + 1 < len(ordered) else len(document.content)
        spans.append(SectionSpan(index, start, end, hits))
    return spans


def assign_sections(chunks: list[Chunk], spans: list[SectionSpan]) -> int:
    """Label each chunk with every section it overlaps. Returns chunks labelled.

    Mutates ``chunk.metadata.section_indices`` in place. Overlap rather than
    containment: a chunk that begins in section 3 and ends in section 4 is
    evidence for both, and a query on either one can legitimately be answered
    by it.
    """
    labelled = 0
    for chunk in chunks:
        start, end = chunk.metadata.start_char, chunk.metadata.end_char
        chunk.metadata.section_indices = [
            span.section_index for span in spans if start < span.end_char and end > span.start_char
        ]
        labelled += bool(chunk.metadata.section_indices)
    return labelled


def load_sections(paper_id: str, corpus_dir: Path) -> list[dict] | None:
    """Read ``corpus/{paper_id}.json``, or None when the paper is not in it."""
    path = Path(corpus_dir) / f"{paper_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())["sections"]


@dataclass
class AlignmentReport:
    """What alignment achieved on one corpus — the coverage figure to publish."""

    papers: int = 0
    papers_missing_corpus: int = 0
    sections_total: int = 0
    sections_located: int = 0
    sections_weak: int = 0
    sections_with_chunks: int = 0
    chunks_total: int = 0
    chunks_labelled: int = 0
    chunks_multi_section: int = 0

    @property
    def section_coverage(self) -> float:
        return self.sections_located / self.sections_total if self.sections_total else 0.0

    @property
    def chunk_coverage(self) -> float:
        return self.chunks_labelled / self.chunks_total if self.chunks_total else 0.0

    def summary(self) -> str:
        return (
            f"{self.papers} papers ({self.papers_missing_corpus} without corpus) | "
            f"sections {self.sections_located}/{self.sections_total} located "
            f"({self.section_coverage:.0%}), {self.sections_weak} weak, "
            f"{self.sections_with_chunks} with chunks | "
            f"chunks {self.chunks_labelled}/{self.chunks_total} labelled "
            f"({self.chunk_coverage:.0%}), {self.chunks_multi_section} span two sections"
        )


def align_corpus(
    documents: list[Document], chunks: list[Chunk], corpus_dir: Path
) -> AlignmentReport:
    """Align every document, label its chunks, and report the coverage.

    Chunks are matched to documents by ``source`` filename stem, which is the
    ``paper_id`` the corpus is keyed on.
    """
    report = AlignmentReport(papers=len(documents))
    by_paper: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_paper.setdefault(Path(chunk.metadata.source).stem, []).append(chunk)

    for document in documents:
        paper_id = (
            getattr(document.metadata, "paper_id", None) or Path(document.metadata.source).stem
        )
        sections = load_sections(paper_id, corpus_dir)
        if sections is None:
            report.papers_missing_corpus += 1
            log.warning(f"No corpus entry for {paper_id} — its chunks stay unlabelled")
            continue

        spans = locate_sections(document, sections)
        report.sections_total += len(sections)
        report.sections_located += len(spans)
        report.sections_weak += sum(span.is_weak for span in spans)

        paper_chunks = by_paper.get(paper_id, [])
        report.chunks_total += len(paper_chunks)
        report.chunks_labelled += assign_sections(paper_chunks, spans)
        report.chunks_multi_section += sum(
            len(chunk.metadata.section_indices) > 1 for chunk in paper_chunks
        )
        report.sections_with_chunks += len(
            {index for chunk in paper_chunks for index in chunk.metadata.section_indices}
        )

    return report
