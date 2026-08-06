"""YAML in, instantiated components out.

The spec's swappability criterion is "components swap without code changes",
which means exactly one thing in practice: nothing outside this module may
import a concrete class. The per-stage registries (``get_loader``,
``get_chunker``, ``get_embedder``, ``get_store``) already resolve a name to a
class; this module is what turns a *file* into a whole configured pipeline, and
what the experiment grid expands.

Every component is written one of two ways, and both mean the same thing::

    chunker: fixed_size:512:128                  # spec string
    chunker: {name: fixed_size, chunk_size: 512, overlap: 128}

The string form exists because a configuration has to survive as a filename and
a table cell — ``fixed_size_512_128__minilm`` is a directory under
``data/indices/``, and the grid carries it through result JSON into RESULTS.md.
The dict form exists because spec strings are positional, so the moment you
want the fourth parameter but not the third, you need names.

Retrievers are the one stage this module builds by hand rather than through a
registry, because they are the one stage whose constructors genuinely differ:
dense needs a store and an embedder, BM25 needs the chunks themselves, hybrid
needs the other two. A uniform ``build(name, **kwargs)`` would have to invent a
shared signature that none of them actually has.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.rag.base import (
    BaseChunker,
    BaseEmbedder,
    BaseLoader,
    BaseRetriever,
    BaseVectorStore,
)
from app.rag.chunking import DEFAULT_CHUNKER, chunker_from_spec, get_chunker
from app.rag.embedding import DEFAULT_EMBEDDER, get_embedder
from app.rag.loaders import DEFAULT_LOADER, get_loader
from app.rag.models import Chunk
from app.rag.preprocessing import DEFAULT_PREPROCESSOR
from app.rag.retrieval import DEFAULT_RETRIEVER, BM25Retriever, DenseRetriever, HybridRetriever
from app.rag.stores import DEFAULT_STORE, config_id, get_store

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# The grid axes. Anything else in a `grid:` block is a config error rather than
# a silently-ignored key, which is the failure mode that costs an overnight run.
GRID_AXES = ("loader", "chunker", "embedder", "store", "retriever", "top_k")


class Component(BaseModel):
    """One stage: a registered name plus whatever options it takes.

    ``spec`` is the round-trip back to the string form — it is what names the
    index directory and what appears in a results table, so it has to survive
    the dict form too.
    """

    name: str
    options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def parse(cls, value: Any, default: str) -> Component:
        """Accept a spec string, an options dict, or nothing at all."""
        if value is None:
            return cls(name=default)
        if isinstance(value, str):
            name, _, arguments = value.partition(":")
            # Positional arguments stay in the string: only the chunker
            # registry knows how they map onto a given strategy's parameters,
            # and re-deriving that here would be a second place to get it wrong.
            return cls(name=name, options={"_args": arguments} if arguments else {})
        if isinstance(value, dict):
            options = dict(value)
            name = options.pop("name", None)
            if not name:
                raise ValueError(f"Component {value!r} has no 'name'")
            return cls(name=name, options=options)
        raise TypeError(f"Component must be a string or a mapping, got {type(value).__name__}")

    @property
    def spec(self) -> str:
        """The string form, for naming things."""
        arguments = self.options.get("_args")
        return f"{self.name}:{arguments}" if arguments else self.name

    @property
    def kwargs(self) -> dict[str, Any]:
        """Options with the positional-argument carrier removed."""
        return {key: value for key, value in self.options.items() if key != "_args"}

    def __str__(self) -> str:
        return self.spec


class PipelineConfig(BaseModel):
    """A complete, named pipeline — everything one grid cell needs."""

    name: str = "default"
    loader: Component = Field(default_factory=lambda: Component(name=DEFAULT_LOADER))
    chunker: Component = Field(default_factory=lambda: Component(name=DEFAULT_CHUNKER))
    embedder: Component = Field(default_factory=lambda: Component(name=DEFAULT_EMBEDDER))
    store: Component = Field(default_factory=lambda: Component(name=DEFAULT_STORE))
    retriever: Component = Field(default_factory=lambda: Component(name=DEFAULT_RETRIEVER))
    top_k: int = DEFAULT_TOP_K
    clean: str = DEFAULT_PREPROCESSOR

    @field_validator("loader", "chunker", "embedder", "store", "retriever", mode="before")
    @classmethod
    def _as_component(cls, value: Any) -> Any:
        # Defaults are filled in by parse() only when the key is absent, so a
        # config naming three stages still gets a working pipeline.
        return value if isinstance(value, Component) else Component.parse(value, "")

    @property
    def index_id(self) -> str:
        """Directory name under ``data/indices/`` for this config's index.

        Keyed on (chunker, embedder) only — dense, BM25 and hybrid all read the
        same index, so varying the retriever must not rebuild it.
        """
        return config_id(self.chunker.spec, self.embedder.spec)

    @property
    def id(self) -> str:
        """Stable identifier for one grid cell, including the retriever."""
        return f"{self.index_id}__{self.retriever.spec.replace(':', '_')}"

    def summary(self) -> str:
        return (
            f"{self.chunker.spec} | {self.embedder.spec} | "
            f"{self.retriever.spec} | top_k={self.top_k}"
        )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def read_yaml(path: Path) -> dict:
    import yaml  # local: keeps the dependency off the import path of every CLI

    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_config(path: Path) -> PipelineConfig:
    """Read one pipeline config from a YAML file."""
    return PipelineConfig(**read_yaml(path))


def expand_grid(data: dict) -> list[PipelineConfig]:
    """Expand a ``grid:`` block into one config per combination.

    ``defaults:`` supplies every field the grid does not vary, so the axes stay
    readable::

        defaults: {top_k: 10}
        grid:
          chunker:   [fixed_size:512:128, sentence:5:1, semantic:512:90]
          embedder:  [minilm, mpnet]
          retriever: [dense, bm25]

    Order is the axis order above, which makes the resulting table group by
    chunker — the axis with the largest effect, so the grouping reads well.
    """
    defaults = dict(data.get("defaults") or {})
    grid = data.get("grid") or {}
    if not grid:
        return [PipelineConfig(**defaults, name=data.get("name", "default"))]

    unknown = set(grid) - set(GRID_AXES)
    if unknown:
        raise ValueError(
            f"Unknown grid axes: {', '.join(sorted(unknown))}. Available: {', '.join(GRID_AXES)}"
        )

    axes = [axis for axis in GRID_AXES if axis in grid]
    # A bare scalar on an axis is a pin, not a mistake — it reads naturally and
    # lets one axis be held fixed without moving it into `defaults`.
    values = [grid[axis] if isinstance(grid[axis], list) else [grid[axis]] for axis in axes]

    configs = []
    for combination in itertools.product(*values):
        settings = {**defaults, **dict(zip(axes, combination, strict=True))}
        config = PipelineConfig(**settings)
        config.name = config.id
        configs.append(config)
    return configs


def load_grid(path: Path) -> list[PipelineConfig]:
    """Read an experiment file, whether or not it declares a grid."""
    configs = expand_grid(read_yaml(path))
    log.info(f"{path}: {len(configs)} configuration(s)")
    return configs


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_loader(config: PipelineConfig) -> BaseLoader:
    return get_loader(config.loader.name, preprocess=config.clean, **config.loader.kwargs)


def build_embedder(config: PipelineConfig) -> BaseEmbedder:
    return get_embedder(config.embedder.name, **config.embedder.kwargs)


def build_chunker(config: PipelineConfig, embedder: BaseEmbedder | None = None) -> BaseChunker:
    """Build the chunker, injecting an embedder if the strategy takes one.

    Only ``semantic`` does, and passing it unconditionally would break every
    other constructor — so the injection is filtered by signature in
    ``chunker_from_spec``. Reusing the caller's embedder matters: building a
    second one would load a second copy of the weights.
    """
    overrides = {"embedder": embedder or build_embedder(config)}
    if config.chunker.kwargs:
        return get_chunker(config.chunker.name, **config.chunker.kwargs, **overrides)
    return chunker_from_spec(config.chunker.spec, **overrides)


def build_store(config: PipelineConfig, dimension: int | None = None) -> BaseVectorStore:
    options = dict(config.store.kwargs)
    if dimension is not None:
        options.setdefault("dimension", dimension)
    return get_store(config.store.name, **options)


def build_retriever(
    config: PipelineConfig,
    *,
    store: BaseVectorStore | None = None,
    chunks: list[Chunk] | None = None,
    embedder: BaseEmbedder | None = None,
) -> BaseRetriever:
    """Assemble the retriever this config names.

    ``store`` is required for anything with a dense half, ``chunks`` for
    anything with a sparse half; hybrid needs both. Raising here beats a
    ``NoneType`` error thirty seconds into a grid run.
    """
    name = config.retriever.name
    options = dict(config.retriever.kwargs)

    # Spec-string form: hybrid:0.3 pins alpha, hybrid:0.3:rrf also picks fusion.
    arguments = config.retriever.options.get("_args")
    if arguments:
        if name != "hybrid":
            raise ValueError(f"{config.retriever.spec!r}: only 'hybrid' takes spec arguments")
        parts = arguments.split(":")
        options.setdefault("alpha", float(parts[0]))
        if len(parts) > 1:
            options.setdefault("fusion", parts[1])

    def dense() -> DenseRetriever:
        if store is None:
            raise ValueError(f"Retriever {name!r} needs a vector store")
        return DenseRetriever(store, embedder or build_embedder(config))

    def sparse() -> BM25Retriever:
        if chunks is None:
            raise ValueError(f"Retriever {name!r} needs the chunks to fit BM25 on")
        return BM25Retriever(chunks)

    if name == "dense":
        return dense()
    if name == "bm25":
        return BM25Retriever(chunks, **options) if chunks is not None else sparse()
    if name == "hybrid":
        return HybridRetriever(dense(), sparse(), **options)
    raise ValueError(f"Unknown retriever {name!r}. Available: dense, bm25, hybrid")


__all__ = [
    "DEFAULT_TOP_K",
    "GRID_AXES",
    "Component",
    "PipelineConfig",
    "build_chunker",
    "build_embedder",
    "build_loader",
    "build_retriever",
    "build_store",
    "expand_grid",
    "load_config",
    "load_grid",
    "read_yaml",
]
