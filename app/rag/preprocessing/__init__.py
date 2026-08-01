"""Text cleaning applied inside the loader, before offsets are recorded."""

from app.rag.preprocessing.clean import (
    DEFAULT_PREPROCESSOR,
    LIGATURES,
    PREPROCESSORS,
    clean_text,
    collapse_spaces,
    dehyphenate,
    merge_continuations,
    normalize_characters,
    reflow,
    reflow_only,
    resolve_preprocessor,
)

__all__ = [
    "DEFAULT_PREPROCESSOR",
    "LIGATURES",
    "PREPROCESSORS",
    "clean_text",
    "collapse_spaces",
    "dehyphenate",
    "merge_continuations",
    "normalize_characters",
    "reflow",
    "reflow_only",
    "resolve_preprocessor",
]
