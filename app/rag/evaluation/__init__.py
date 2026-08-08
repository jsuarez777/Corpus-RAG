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
from app.rag.evaluation.judge import (
    CITATION_QUALITY_FLOOR,
    DIMENSIONS,
    JudgeError,
    JudgeReport,
    JudgeScore,
    LLMJudge,
)
from app.rag.evaluation.metrics import (
    DEFAULT_KS,
    EvaluationResult,
    evaluate,
    evaluate_query,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    section_coverage_at_k,
)
from app.rag.evaluation.qrels import (
    QueryRelevance,
    RelevanceReport,
    build_relevance,
    index_chunks_by_section,
    load_benchmark,
)

__all__ = [
    "DEFAULT_KS",
    "CITATION_QUALITY_FLOOR",
    "DIMENSIONS",
    "AlignmentReport",
    "EvaluationResult",
    "JudgeError",
    "JudgeReport",
    "JudgeScore",
    "LLMJudge",
    "QueryRelevance",
    "RelevanceReport",
    "SectionSpan",
    "align_corpus",
    "assign_sections",
    "build_relevance",
    "evaluate",
    "evaluate_query",
    "hit_rate_at_k",
    "index_chunks_by_section",
    "load_benchmark",
    "load_sections",
    "locate_sections",
    "ndcg_at_k",
    "normalize",
    "precision_at_k",
    "section_coverage_at_k",
    "reciprocal_rank",
    "strip_math",
]
