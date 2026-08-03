"""Chunking strategies, and the registry a config names one by.

Two ways in, both of which avoid importing a concrete class at a call site:

* ``get_chunker("recursive", chunk_size=256)`` — for YAML configs, which
  already have a dict of options.
* ``chunker_from_spec("recursive:256:64")`` — for CLI flags and the experiment
  grid, where a whole configuration has to survive as one string.
"""

from __future__ import annotations

from inspect import Parameter, signature

from app.rag.base import BaseChunker
from app.rag.chunking._sentences import Sentence, segment_sentences
from app.rag.chunking._spans import Span, spans_to_chunks
from app.rag.chunking._tokens import count_tokens, resolve_overlap, token_starts
from app.rag.chunking.fixed_size import FixedSizeChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.sentence import SentenceChunker
from app.rag.chunking.sliding_window import SlidingWindowChunker
from app.rag.chunking.store import config_slug, load_chunks, save_chunks

# semantic.py lands here once the embedding stage exists to inject into it.
CHUNKERS: dict[str, type[BaseChunker]] = {
    FixedSizeChunker.name: FixedSizeChunker,
    SentenceChunker.name: SentenceChunker,
    SlidingWindowChunker.name: SlidingWindowChunker,
    RecursiveChunker.name: RecursiveChunker,
}

DEFAULT_CHUNKER = FixedSizeChunker.name


def get_chunker(name: str = DEFAULT_CHUNKER, **kwargs) -> BaseChunker:
    """Build the chunker registered under ``name``."""
    try:
        chunker_cls = CHUNKERS[name]
    except KeyError:
        available = ", ".join(sorted(CHUNKERS))
        raise ValueError(f"Unknown chunker {name!r}. Available: {available}") from None
    return chunker_cls(**kwargs)


def chunker_from_spec(spec: str) -> BaseChunker:
    """Build a chunker from ``name[:arg[:arg...]]``.

    Arguments map positionally onto the constructor's own parameters, so the
    spec reads in each strategy's vocabulary — ``fixed_size:512:128`` is
    (chunk_size, overlap) while ``sentence:5:1`` is (sentences_per_chunk,
    overlap) — and adding a parameter to a chunker needs no change here.
    """
    name, _, arguments = spec.partition(":")
    try:
        chunker_cls = CHUNKERS[name]
    except KeyError:
        available = ", ".join(sorted(CHUNKERS))
        raise ValueError(f"Unknown chunker {name!r}. Available: {available}") from None
    if not arguments:
        return chunker_cls()

    values = arguments.split(":")
    params = [
        param
        for param in signature(chunker_cls.__init__).parameters.values()
        if param.name != "self" and param.kind is not Parameter.VAR_KEYWORD
    ]
    if len(values) > len(params):
        names = ", ".join(param.name for param in params)
        raise ValueError(f"{spec!r}: {chunker_cls.__name__} takes at most {len(params)} ({names})")

    return chunker_cls(**{p.name: _coerce(v, p.default) for v, p in zip(values, params)})


def _coerce(value: str, default: object) -> object:
    """Type a spec argument from the parameter's default."""
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        # Ints that are not ints are left alone: overlap also accepts "25%".
        return int(value) if value.lstrip("-").isdigit() else value
    if isinstance(default, float):
        return float(value)
    return value


__all__ = [
    "CHUNKERS",
    "DEFAULT_CHUNKER",
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "Sentence",
    "SentenceChunker",
    "SlidingWindowChunker",
    "Span",
    "chunker_from_spec",
    "config_slug",
    "count_tokens",
    "get_chunker",
    "load_chunks",
    "resolve_overlap",
    "save_chunks",
    "segment_sentences",
    "spans_to_chunks",
    "token_starts",
]
