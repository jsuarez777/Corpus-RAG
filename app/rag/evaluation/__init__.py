"""Evaluation: aligning chunks to corpus sections, and scoring against qrels."""

from app.rag.evaluation.alignment import (
    AlignmentReport,
    SectionSpan,
    align_corpus,
    assign_sections,
    load_sections,
    locate_sections,
    normalize,
    strip_math,
)

__all__ = [
    "AlignmentReport",
    "SectionSpan",
    "align_corpus",
    "assign_sections",
    "load_sections",
    "locate_sections",
    "normalize",
    "strip_math",
]
