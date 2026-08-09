"""Tests for the chunk-set stamp that pairs a chunk file with its index.

`Chunk.id` is a `uuid4` minted at construction, so re-running `app/chunk.py`
renames every chunk in the corpus while the filename stays put. Three artifacts
store those ids — the chunk file, the index, and a saved ranking — and only the
first is rewritten. What makes that dangerous rather than merely stale is hybrid
retrieval: dense results carry ids from the index, BM25 results carry ids from
the chunk file, and across two generations those sets are disjoint. The fusion
still runs and the metrics still come out. They are simply wrong.

So these check the guard fires on a mismatch, and equally on an artifact with
no stamp at all: unknown is not the same as safe, and unknown is the state every
artifact was in while the bug was possible.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.rag.chunking.store import chunk_set_id, load_chunks, new_chunk_set_id, save_chunks
from app.rag.models import Chunk, ChunkMetadata
from app.rag.stores.store import index_chunk_set_id, require_same_chunk_set


def make_chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            content=f"passage {i}",
            metadata=ChunkMetadata(
                document_id=uuid4(),
                source="paper.pdf",
                page_number=i + 1,
                start_char=i * 100,
                end_char=i * 100 + 10,
                chunk_index=i,
            ),
        )
        for i in range(count)
    ]


def write_index(path: Path, stamp: str | None) -> Path:
    """A directory shaped like a saved index, as far as the guard can see."""
    path.mkdir(parents=True, exist_ok=True)
    meta = {"store": "faiss", "dimension": 3, "num_chunks": 3}
    if stamp is not None:
        meta["chunk_set_id"] = stamp
    (path / "meta.json").write_text(json.dumps(meta))
    return path


class TestStamping:
    def test_every_save_stamps_a_new_id(self, tmp_path: Path) -> None:
        first = save_chunks(make_chunks(), tmp_path, "fixed_size:512:128")
        one = chunk_set_id(first)
        second = save_chunks(make_chunks(), tmp_path, "fixed_size:512:128")

        assert one and chunk_set_id(second) and chunk_set_id(second) != one

    def test_identical_content_is_still_a_new_chunk_set(self, tmp_path: Path) -> None:
        """The stamp tracks ids, not text. Re-chunking an unchanged corpus mints
        fresh uuid4s, so calling it the same set would defeat the whole check."""
        chunks = make_chunks()
        a = chunk_set_id(save_chunks(chunks, tmp_path / "a", "fixed_size:512:128"))
        b = chunk_set_id(save_chunks(chunks, tmp_path / "b", "fixed_size:512:128"))

        assert a != b

    def test_the_stamp_does_not_disturb_the_chunks(self, tmp_path: Path) -> None:
        chunks = make_chunks()
        path = save_chunks(chunks, tmp_path, "fixed_size:512:128")

        assert [c.id for c in load_chunks(path)] == [c.id for c in chunks]

    def test_an_explicit_id_is_honoured(self, tmp_path: Path) -> None:
        path = save_chunks(make_chunks(), tmp_path, "fixed_size:512:128", chunk_set="fixed")
        assert chunk_set_id(path) == "fixed"

    def test_ids_are_unique(self) -> None:
        assert len({new_chunk_set_id() for _ in range(100)}) == 100


class TestReading:
    def test_a_file_written_before_stamping_reads_as_unknown(self, tmp_path: Path) -> None:
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"chunker": "fixed_size:512:128", "chunks": []}))
        assert chunk_set_id(path) == ""

    def test_a_missing_file_reads_as_unknown(self, tmp_path: Path) -> None:
        assert chunk_set_id(tmp_path / "absent.json") == ""

    def test_an_index_without_a_manifest_reads_as_unknown(self, tmp_path: Path) -> None:
        assert index_chunk_set_id(tmp_path) == ""


class TestGuard:
    def test_a_mismatch_stops_the_run(self, tmp_path: Path) -> None:
        chunks = save_chunks(make_chunks(), tmp_path, "fixed_size:512:128")
        index = write_index(tmp_path / "idx", "some-other-run")

        with pytest.raises(SystemExit, match="chunks were rebuilt"):
            require_same_chunk_set(chunks, index)

    def test_the_message_names_the_command_that_fixes_it(self, tmp_path: Path) -> None:
        chunks = save_chunks(make_chunks(), tmp_path, "fixed_size:512:128")
        index = write_index(tmp_path / "idx", "stale")

        with pytest.raises(SystemExit, match="index.py --force"):
            require_same_chunk_set(chunks, index)

    def test_a_match_passes(self, tmp_path: Path) -> None:
        chunks = save_chunks(make_chunks(), tmp_path, "fixed_size:512:128")
        index = write_index(tmp_path / "idx", chunk_set_id(chunks))

        require_same_chunk_set(chunks, index)  # would raise

    def test_an_unstamped_index_is_refused(self, tmp_path: Path) -> None:
        """Unknown is not the same as safe. An index with no stamp cannot be
        shown to hold the chunks on disk, and unknown is exactly the state
        every artifact was in while the bug was possible."""
        chunks = save_chunks(make_chunks(), tmp_path, "fixed_size:512:128")
        index = write_index(tmp_path / "idx", None)

        with pytest.raises(SystemExit, match="no chunk_set_id"):
            require_same_chunk_set(chunks, index)

    def test_an_unstamped_chunk_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"chunker": "fixed_size:512:128", "chunks": []}))
        index = write_index(tmp_path / "idx", "anything")

        with pytest.raises(SystemExit, match="chunk.py"):
            require_same_chunk_set(path, index)

    def test_both_unstamped_is_still_refused(self, tmp_path: Path) -> None:
        """The state the whole corpus is in today, and the one that allowed a
        stale index to pair with fresh chunks unnoticed."""
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"chunker": "fixed_size:512:128", "chunks": []}))

        with pytest.raises(SystemExit):
            require_same_chunk_set(path, write_index(tmp_path / "idx", None))
