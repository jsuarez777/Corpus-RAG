"""Tests for the YAML -> component layer.

The expensive failure here is not a crash — it is a grid that runs to
completion and answers the wrong question. Two guards get the most attention:

* **A misspelled grid axis must raise.** Silently ignoring ``retrievers:``
  would run every cell at the default retriever and produce twelve results
  that look fine and mean nothing.
* **`index_id` must not include the retriever.** Dense, BM25 and hybrid all
  read one index; if the retriever leaked into the name, the grid would
  re-embed the whole corpus three times over for identical vectors.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.rag.chunking import FixedSizeChunker, SemanticChunker
from app.rag.config import (
    DEFAULT_TOP_K,
    Component,
    PipelineConfig,
    build_chunker,
    build_embedder,
    build_loader,
    build_retriever,
    build_store,
    expand_grid,
    load_config,
    load_grid,
)
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.loaders import PyMuPDFLoader
from app.rag.models import Chunk, ChunkMetadata
from app.rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from app.rag.stores import FaissStore

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def make_chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            content=f"retrieval augmented generation passage {index}",
            metadata=ChunkMetadata(
                document_id=uuid4(),
                source="p.pdf",
                start_char=index * 100,
                end_char=index * 100 + 50,
                chunk_index=index,
            ),
        )
        for index in range(count)
    ]


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


class TestComponent:
    def test_a_bare_name(self) -> None:
        component = Component.parse("minilm", "default")
        assert (component.name, component.options, component.spec) == ("minilm", {}, "minilm")

    def test_a_spec_string_keeps_its_arguments_for_the_registry_to_map(self) -> None:
        """Only the chunking registry knows how positional arguments land on a
        given strategy, so they are carried, not re-parsed here."""
        component = Component.parse("fixed_size:512:128", "default")
        assert component.name == "fixed_size"
        assert component.spec == "fixed_size:512:128"

    def test_a_dict_form_means_the_same_thing(self) -> None:
        component = Component.parse({"name": "hybrid", "alpha": 0.3}, "default")
        assert component.name == "hybrid"
        assert component.kwargs == {"alpha": 0.3}

    def test_missing_falls_back_to_the_default(self) -> None:
        assert Component.parse(None, "pymupdf").name == "pymupdf"

    def test_a_dict_without_a_name_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no 'name'"):
            Component.parse({"alpha": 0.3}, "default")

    def test_a_number_is_not_a_component(self) -> None:
        with pytest.raises(TypeError):
            Component.parse(7, "default")


class TestPipelineConfig:
    def test_every_stage_has_a_working_default(self) -> None:
        config = PipelineConfig()
        assert config.top_k == DEFAULT_TOP_K
        assert config.loader.name and config.chunker.name and config.embedder.name

    def test_naming_one_stage_leaves_the_others_at_their_defaults(self) -> None:
        config = PipelineConfig(chunker="sentence:5:1")
        assert config.chunker.spec == "sentence:5:1"
        assert config.embedder.name == PipelineConfig().embedder.name

    def test_index_id_pairs_chunker_with_embedder(self) -> None:
        config = PipelineConfig(chunker="fixed_size:512:128", embedder="minilm")
        assert config.index_id == "fixed_size_512_128__minilm"

    def test_index_id_ignores_the_retriever(self) -> None:
        """Dense, BM25 and hybrid share one index; including the retriever
        would re-embed the corpus once per retrieval method."""
        base = {"chunker": "sentence:5:1", "embedder": "mpnet"}
        assert (
            PipelineConfig(**base, retriever="dense").index_id
            == PipelineConfig(**base, retriever="hybrid:0.3").index_id
        )

    def test_id_distinguishes_the_retriever(self) -> None:
        base = {"chunker": "sentence:5:1", "embedder": "mpnet"}
        assert (
            PipelineConfig(**base, retriever="dense").id
            != PipelineConfig(**base, retriever="hybrid:0.3").id
        )


class TestExpandGrid:
    def test_multiplies_every_axis(self) -> None:
        configs = expand_grid(
            {"grid": {"chunker": ["a:1", "b:2"], "embedder": ["minilm", "mpnet"]}}
        )
        assert len(configs) == 4

    def test_defaults_fill_the_axes_the_grid_does_not_vary(self) -> None:
        configs = expand_grid({"defaults": {"top_k": 10}, "grid": {"embedder": ["minilm"]}})
        assert configs[0].top_k == 10

    def test_a_scalar_axis_pins_it(self) -> None:
        configs = expand_grid({"grid": {"embedder": ["minilm", "mpnet"], "store": "faiss"}})
        assert len(configs) == 2
        assert all(config.store.name == "faiss" for config in configs)

    def test_an_unknown_axis_raises_rather_than_being_ignored(self) -> None:
        """The failure this prevents: 'retrievers' runs the whole grid at the
        default retriever and reports twelve meaningless results."""
        with pytest.raises(ValueError, match="Unknown grid axes: retrievers"):
            expand_grid({"grid": {"retrievers": ["dense"]}})

    def test_no_grid_block_yields_the_single_config(self) -> None:
        configs = expand_grid({"defaults": {"chunker": "sentence:5:1"}})
        assert len(configs) == 1
        assert configs[0].chunker.spec == "sentence:5:1"

    def test_each_config_is_named_by_its_id(self) -> None:
        configs = expand_grid({"grid": {"embedder": ["minilm", "mpnet"]}})
        assert len({config.name for config in configs}) == 2
        assert configs[0].name == configs[0].id


class TestLoading:
    def test_reads_a_file(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "chunker: sentence:5:1\ntop_k: 3\n")
        config = load_config(path)
        assert (config.chunker.spec, config.top_k) == ("sentence:5:1", 3)

    def test_an_empty_file_is_the_default_pipeline(self, tmp_path: Path) -> None:
        assert load_config(write_yaml(tmp_path, "")).top_k == DEFAULT_TOP_K

    def test_a_top_level_list_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mapping"):
            load_config(write_yaml(tmp_path, "- one\n- two\n"))

    def test_the_shipped_default_config_loads(self) -> None:
        assert load_config(CONFIG_DIR / "default.yaml").chunker.name == "fixed_size"

    def test_the_shipped_grid_is_the_twelve_the_spec_asks_for(self) -> None:
        configs = load_grid(CONFIG_DIR / "experiments" / "grid_12.yaml")
        assert len(configs) == 12
        assert len({config.id for config in configs}) == 12

    def test_the_shipped_grid_builds_only_six_indices(self) -> None:
        """3 chunkers x 2 embedders. Twelve would mean the retriever leaked
        into the index name."""
        configs = load_grid(CONFIG_DIR / "experiments" / "grid_12.yaml")
        assert len({config.index_id for config in configs}) == 6

    def test_the_alpha_sweep_holds_everything_but_the_retriever_fixed(self) -> None:
        configs = load_grid(CONFIG_DIR / "experiments" / "alpha_sweep.yaml")
        assert len({config.index_id for config in configs}) == 1
        assert len({config.retriever.spec for config in configs}) == len(configs)


class TestBuilding:
    def test_builds_a_loader(self) -> None:
        assert isinstance(build_loader(PipelineConfig()), PyMuPDFLoader)

    def test_builds_an_embedder_by_alias(self) -> None:
        embedder = build_embedder(PipelineConfig(embedder="mpnet"))
        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder.alias == "mpnet"

    def test_builds_a_chunker_from_its_spec_string(self) -> None:
        chunker = build_chunker(PipelineConfig(chunker="fixed_size:256:64"))
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 256

    def test_builds_a_chunker_from_the_dict_form(self) -> None:
        chunker = build_chunker(
            PipelineConfig(chunker={"name": "fixed_size", "chunk_size": 256, "overlap": 32})
        )
        assert (chunker.chunk_size, chunker.overlap) == (256, 32)

    def test_the_semantic_chunker_gets_the_pipeline_embedder(self) -> None:
        """Not a second one: building another would load a second copy of the
        weights for vectors that are already in memory."""
        config = PipelineConfig(chunker="semantic:512:90", embedder="minilm")
        embedder = build_embedder(config)
        chunker = build_chunker(config, embedder)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.embedder is embedder

    def test_builds_a_store(self) -> None:
        assert isinstance(build_store(PipelineConfig(), dimension=384), FaissStore)

    def test_builds_a_dense_retriever(self) -> None:
        config = PipelineConfig(retriever="dense")
        retriever = build_retriever(config, store=FaissStore(dimension=384))
        assert isinstance(retriever, DenseRetriever)

    def test_builds_a_bm25_retriever(self) -> None:
        retriever = build_retriever(PipelineConfig(retriever="bm25"), chunks=make_chunks())
        assert isinstance(retriever, BM25Retriever)

    def test_builds_a_hybrid_retriever(self) -> None:
        retriever = build_retriever(
            PipelineConfig(retriever="hybrid"),
            store=FaissStore(dimension=384),
            chunks=make_chunks(),
        )
        assert isinstance(retriever, HybridRetriever)

    def test_a_spec_string_pins_alpha(self) -> None:
        retriever = build_retriever(
            PipelineConfig(retriever="hybrid:0.3"),
            store=FaissStore(dimension=384),
            chunks=make_chunks(),
        )
        assert retriever.alpha == 0.3

    def test_a_spec_string_can_also_pick_the_fusion(self) -> None:
        retriever = build_retriever(
            PipelineConfig(retriever="hybrid:0.5:rrf"),
            store=FaissStore(dimension=384),
            chunks=make_chunks(),
        )
        assert retriever.fusion == "rrf"

    def test_the_dict_form_configures_hybrid_too(self) -> None:
        retriever = build_retriever(
            PipelineConfig(retriever={"name": "hybrid", "alpha": 0.7}),
            store=FaissStore(dimension=384),
            chunks=make_chunks(),
        )
        assert retriever.alpha == 0.7

    def test_only_hybrid_takes_spec_arguments(self) -> None:
        with pytest.raises(ValueError, match="only 'hybrid'"):
            build_retriever(PipelineConfig(retriever="dense:0.3"), store=FaissStore(dimension=384))

    def test_a_missing_store_is_named_rather_than_a_nonetype_error(self) -> None:
        """This fires thirty seconds into a grid run otherwise."""
        with pytest.raises(ValueError, match="needs a vector store"):
            build_retriever(PipelineConfig(retriever="dense"))

    def test_a_missing_chunk_list_is_named_too(self) -> None:
        with pytest.raises(ValueError, match="needs the chunks"):
            build_retriever(PipelineConfig(retriever="bm25"))

    def test_an_unknown_retriever_lists_the_real_ones(self) -> None:
        with pytest.raises(ValueError, match="Available: dense, bm25, hybrid"):
            build_retriever(PipelineConfig(retriever="colbert"), store=FaissStore(dimension=384))
