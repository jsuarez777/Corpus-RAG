"""Chunking strategies, and the registry a config names one by.

Two ways in, both of which avoid importing a concrete class at a call site:

* ``get_chunker("recursive", chunk_size=256)`` — for YAML configs, which
  already have a dict of options.
* ``chunker_from_spec("recursive:256:64")`` — for CLI flags and the experiment
  grid, where a whole configuration has to survive as one string.
"""

from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path

from app.rag.base import BaseChunker
from app.rag.chunking._sentences import Sentence, segment_sentences
from app.rag.chunking._spans import Span, spans_to_chunks
from app.rag.chunking._tokens import count_tokens, resolve_overlap, token_starts
from app.rag.chunking.fixed_size import FixedSizeChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.semantic import SemanticChunker
from app.rag.chunking.sentence import SentenceChunker
from app.rag.chunking.sliding_window import SlidingWindowChunker
from app.rag.chunking.store import config_slug, load_chunks, save_chunks

CHUNKERS: dict[str, type[BaseChunker]] = {
    FixedSizeChunker.name: FixedSizeChunker,
    SentenceChunker.name: SentenceChunker,
    SlidingWindowChunker.name: SlidingWindowChunker,
    RecursiveChunker.name: RecursiveChunker,
    SemanticChunker.name: SemanticChunker,
}

DEFAULT_CHUNKER = FixedSizeChunker.name


def _resolve(name: str) -> type[BaseChunker]:
    try:
        return CHUNKERS[name]
    except KeyError:
        available = ", ".join(sorted(CHUNKERS))
        raise ValueError(f"Unknown chunker {name!r}. Available: {available}") from None


def needs_embedder(name: str) -> bool:
    """True for strategies whose boundaries depend on an embedding model.

    Only ``semantic`` does, and the consequence reaches disk: its chunks are
    not shared across embedders the way every other strategy's are, so its
    files need the embedder in the name. Callers ask this rather than testing
    the name, which would go stale the moment a second such strategy exists.
    """
    return "embedder" in signature(_resolve(name).__init__).parameters


def chunk_file(chunks_dir: Path, spec: str, embedder: str | None = None) -> Path:
    """Path to ``spec``'s chunk file, naming ``embedder`` only where it matters.

    Every stage that reads ``data/chunks/`` goes through here, so the rule for
    which configs name an embedder lives in one place: a caller that built the
    filename itself would silently read `semantic`'s minilm chunks while
    searching an mpnet index.
    """
    named = embedder if (embedder and needs_embedder(spec.partition(":")[0])) else None
    return Path(chunks_dir) / f"{config_slug(spec, named)}.json"


def _inject(chunker_cls: type[BaseChunker], embedder: object | None) -> dict:
    """Pass ``embedder`` only to strategies that declare one.

    Only ``semantic`` does. Filtering by signature rather than by name keeps
    the caller from having to know which strategy is which — and passing it
    unconditionally would be a TypeError on all four of the others.
    """
    if embedder is None:
        return {}
    takes_it = "embedder" in signature(chunker_cls.__init__).parameters
    return {"embedder": embedder} if takes_it else {}


def get_chunker(name: str = DEFAULT_CHUNKER, *, embedder: object | None = None, **kwargs):
    """Build the chunker registered under ``name``.

    ``kwargs`` stay strict — an unknown option is a config typo worth raising
    on. ``embedder`` is the one exception, being injected by the pipeline
    rather than written by a user.
    """
    chunker_cls = _resolve(name)
    return chunker_cls(**kwargs, **_inject(chunker_cls, embedder))


def chunker_from_spec(spec: str, *, embedder: object | None = None) -> BaseChunker:
    """Build a chunker from ``name[:arg[:arg...]]``.

    Arguments map positionally onto the constructor's own parameters, so the
    spec reads in each strategy's vocabulary — ``fixed_size:512:128`` is
    (chunk_size, overlap) while ``sentence:5:1`` is (sentences_per_chunk,
    overlap) — and adding a parameter to a chunker needs no change here.

    Keyword-only parameters are excluded from that mapping: declaring one is
    the author saying it cannot be positional, and ``semantic``'s embedder is
    injected, never spelled out in a spec string.
    """
    name, _, arguments = spec.partition(":")
    chunker_cls = _resolve(name)
    injected = _inject(chunker_cls, embedder)
    if not arguments:
        return chunker_cls(**injected)

    values = arguments.split(":")
    params = [
        param
        for param in signature(chunker_cls.__init__).parameters.values()
        if param.name != "self"
        and param.kind not in (Parameter.VAR_KEYWORD, Parameter.KEYWORD_ONLY)
    ]
    if len(values) > len(params):
        names = ", ".join(param.name for param in params)
        raise ValueError(f"{spec!r}: {chunker_cls.__name__} takes at most {len(params)} ({names})")

    mapped = {p.name: _coerce(v, p.default) for v, p in zip(values, params)}
    return chunker_cls(**mapped, **injected)


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
    "SemanticChunker",
    "Sentence",
    "SentenceChunker",
    "SlidingWindowChunker",
    "Span",
    "chunk_file",
    "chunker_from_spec",
    "config_slug",
    "count_tokens",
    "get_chunker",
    "load_chunks",
    "needs_embedder",
    "resolve_overlap",
    "save_chunks",
    "segment_sentences",
    "spans_to_chunks",
    "token_starts",
]
