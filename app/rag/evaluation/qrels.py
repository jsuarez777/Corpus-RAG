"""Turning the benchmark's labels into something a retriever can be scored on.

The dataset labels relevance as ``(query_id -> doc_id, section_id)``. A
retriever returns chunks. This module closes that gap: for each query, the set
of chunk ids that lie in the gold section of the gold paper — which is what the
alignment stage's ``section_indices`` was built to make possible.

**Unscoreable queries are excluded, not scored zero.** A query is unscoreable
when its paper is not in the index, or when no chunk carries its gold section
index because alignment failed to locate that section.

The second case is a *labelling* failure, not missing content. Alignment tiles
the document, so an unlocated section's text is absorbed into its predecessor's
span: the text is still there, still chunked, still retrievable — it just
carries the wrong section number. Measured on the working set, all six
distinctive words of one unlocated section appear in the extracted text. So the
retriever can return exactly the right passage and score zero, and what the
metric would be recording is our labelling, not its ranking.

Excluding is not free either. The sections that fail to align are the
proof-heavy ones — the hardest passages to retrieve — so dropping them biases
every metric optimistically. On the 10-paper working set this is 2 sections of
152 with no queries pointing at them, so the effect is nil; at corpus scale the
count belongs in RESULTS.md beside any number derived from it, which is why
:class:`RelevanceReport` carries it rather than logging and forgetting it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from app.rag.models import Chunk

log = logging.getLogger(__name__)

QUERIES_FILE = "queries.json"
QRELS_FILE = "qrels.json"
ANSWERS_FILE = "answers.json"


@dataclass(frozen=True)
class QueryRelevance:
    """One query, its gold section, and the chunks that realize it."""

    query_id: str
    query: str
    doc_id: str
    section_id: int
    relevant: frozenset[UUID]
    answer: str | None = None

    @property
    def is_scoreable(self) -> bool:
        return bool(self.relevant)


@dataclass
class RelevanceReport:
    """How much of the benchmark this working set can actually be scored on."""

    queries_total: int = 0
    queries_in_corpus: int = 0
    queries_scoreable: int = 0
    papers_indexed: int = 0
    unscoreable_sections: list[tuple[str, int]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.papers_indexed} papers indexed | "
            f"{self.queries_in_corpus}/{self.queries_total} queries target them | "
            f"{self.queries_scoreable} scoreable "
            f"({len(self.unscoreable_sections)} gold sections have no chunks)"
        )


def load_json(directory: Path, name: str) -> dict:
    return json.loads((Path(directory) / name).read_text())


def load_benchmark(directory: Path) -> tuple[dict, dict, dict]:
    """Read ``queries.json``, ``qrels.json`` and ``answers.json``."""
    return (
        load_json(directory, QUERIES_FILE),
        load_json(directory, QRELS_FILE),
        load_json(directory, ANSWERS_FILE),
    )


def index_chunks_by_section(chunks: list[Chunk]) -> dict[tuple[str, int], set[UUID]]:
    """Map (paper_id, section_index) to the chunks covering it.

    A chunk straddling a boundary appears under every section it overlaps,
    which is what makes both neighbours scoreable.
    """
    by_section: dict[tuple[str, int], set[UUID]] = {}
    for chunk in chunks:
        paper_id = Path(chunk.metadata.source).stem
        for section_index in getattr(chunk.metadata, "section_indices", []) or []:
            by_section.setdefault((paper_id, section_index), set()).add(chunk.id)
    return by_section


def build_relevance(
    chunks: list[Chunk],
    queries: dict,
    qrels: dict,
    answers: dict | None = None,
) -> tuple[list[QueryRelevance], RelevanceReport]:
    """Resolve every query that targets an indexed paper into relevant chunk ids.

    Returns only the scoreable queries, plus the report describing what was
    dropped and why.
    """
    by_section = index_chunks_by_section(chunks)
    indexed_papers = {paper_id for paper_id, _ in by_section}

    report = RelevanceReport(queries_total=len(qrels), papers_indexed=len(indexed_papers))
    scoreable: list[QueryRelevance] = []

    for query_id, label in qrels.items():
        doc_id, section_id = label["doc_id"], label["section_id"]
        if doc_id not in indexed_papers:
            continue
        report.queries_in_corpus += 1

        relevant = by_section.get((doc_id, section_id), set())
        if not relevant:
            # The paper is indexed but alignment never located this section, so
            # its chunks carry the preceding section's number. The text is
            # retrievable; the label is not on it.
            report.unscoreable_sections.append((doc_id, section_id))
            continue

        query_text = queries.get(query_id, {})
        scoreable.append(
            QueryRelevance(
                query_id=query_id,
                query=query_text.get("query", "") if isinstance(query_text, dict) else query_text,
                doc_id=doc_id,
                section_id=section_id,
                relevant=frozenset(relevant),
                answer=(answers or {}).get(query_id),
            )
        )

    report.queries_scoreable = len(scoreable)
    log.info(report.summary())
    return scoreable, report
