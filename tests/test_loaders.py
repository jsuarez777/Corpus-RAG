"""Tests for the loader stage.

The invariant everything downstream rests on: ``page_starts`` indexes the same
string as ``Document.content``, so any char offset resolves to the page it
really came from — before and after a JSON round-trip.
"""

from pathlib import Path

import pymupdf
import pytest

from app.extract import collect_pdfs, extract
from app.rag.base import BaseLoader
from app.rag.loaders import (
    DEFAULT_LOADER,
    LOADERS,
    PyMuPDFLoader,
    get_loader,
    join_pages,
    load_document,
    load_documents,
    page_of,
    save_document,
)
from app.rag.models import Document


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """Two small PDFs with known, distinct per-page text."""
    pages = {
        "2401.00001v1": ["Alpha page one text.", "Beta page two text.", "Gamma page three."],
        "2401.00002v2": ["Solo page."],
    }
    directory = tmp_path / "pdfs"
    directory.mkdir()
    for stem, page_texts in pages.items():
        doc = pymupdf.open()
        for text in page_texts:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.set_metadata({"title": f"Paper {stem}", "author": "A. Author"})
        doc.save(directory / f"{stem}.pdf")
        doc.close()
    return directory


@pytest.fixture
def document(pdf_dir: Path) -> Document:
    return PyMuPDFLoader().load(pdf_dir / "2401.00001v1.pdf")


class TestPageMap:
    """join_pages / page_of, independent of any PDF."""

    def test_starts_index_the_joined_string(self) -> None:
        pages = ["first", "second", "third"]
        content, starts = join_pages(pages, "\n\n")
        for text, start in zip(pages, starts, strict=True):
            assert content[start : start + len(text)] == text

    def test_empty_corpus(self) -> None:
        assert join_pages([]) == ("", [])

    def test_page_of_maps_every_offset(self) -> None:
        content, starts = join_pages(["aaa", "bbb"], "\n\n")
        assert page_of(starts, 0) == 1
        assert page_of(starts, 2) == 1
        assert page_of(starts, starts[1]) == 2
        # end_char is exclusive and may sit one past the content
        assert page_of(starts, len(content)) == 2

    def test_page_of_without_a_map_is_none(self) -> None:
        assert page_of([], 0) is None

    def test_blank_page_resolves_to_its_successor(self) -> None:
        """An empty page shares the next page's offset — bisect_right picks
        the page that actually owns the text."""
        _, starts = join_pages(["aaa", "", "ccc"], "")
        assert starts[1] == starts[2]
        assert page_of(starts, starts[1]) == 3


class TestPyMuPDFLoader:
    def test_is_a_loader(self) -> None:
        assert isinstance(PyMuPDFLoader(), BaseLoader)

    def test_extracts_every_page(self, document: Document) -> None:
        assert document.metadata.page_count == 3
        assert "Alpha" in document.content
        assert "Gamma" in document.content

    def test_records_provenance(self, document: Document) -> None:
        assert document.metadata.source == "2401.00001v1.pdf"
        assert document.metadata.paper_id == "2401.00001v1"
        assert document.metadata.loader == "pymupdf"
        assert document.metadata.title == "Paper 2401.00001v1"
        assert document.metadata.author == "A. Author"

    def test_page_starts_resolve_to_the_right_page(self, document: Document) -> None:
        starts = document.metadata.page_starts
        assert len(starts) == 3
        for expected_page, word in enumerate(("Alpha", "Beta", "Gamma"), start=1):
            assert page_of(starts, document.content.index(word)) == expected_page

    def test_preprocess_runs_before_offsets_are_taken(self, pdf_dir: Path) -> None:
        """Cleaning after the fact would invalidate every offset, so the hook
        is per page and the map is built from the cleaned text."""
        loader = PyMuPDFLoader(preprocess=lambda text: text.replace("Alpha", "A"))
        document = loader.load(pdf_dir / "2401.00001v1.pdf")

        assert "Alpha" not in document.content
        starts = document.metadata.page_starts
        assert page_of(starts, document.content.index("Beta")) == 2

    def test_cleans_by_default(self, pdf_dir: Path) -> None:
        """The default preprocessor runs unless it is turned off — extraction
        that leaves wrapped lines in place is what breaks sentence chunking."""
        document = PyMuPDFLoader().load(pdf_dir / "2401.00001v1.pdf")
        assert document.metadata.preprocess == "clean_text"

    def test_preprocessing_can_be_turned_off(self, pdf_dir: Path) -> None:
        document = PyMuPDFLoader(preprocess="none").load(pdf_dir / "2401.00001v1.pdf")
        assert document.metadata.preprocess is None

    def test_rejects_an_unknown_preprocessor(self) -> None:
        with pytest.raises(ValueError, match="Unknown preprocessor"):
            PyMuPDFLoader(preprocess="scrub")

    def test_rejects_an_unknown_layout(self) -> None:
        with pytest.raises(ValueError, match="layout must be"):
            PyMuPDFLoader(layout="paragraphs")

    def test_layout_is_recorded(self, pdf_dir: Path) -> None:
        document = PyMuPDFLoader(layout="text").load(pdf_dir / "2401.00001v1.pdf")
        assert document.metadata.layout == "text"

    def test_both_layouts_extract_the_same_words(self, pdf_dir: Path) -> None:
        """Block mode changes where the breaks fall, not what the text says."""
        blocks = PyMuPDFLoader(layout="blocks").load(pdf_dir / "2401.00001v1.pdf")
        plain = PyMuPDFLoader(layout="text").load(pdf_dir / "2401.00001v1.pdf")
        assert blocks.content.split() == plain.content.split()

    def test_absent_metadata_is_none_not_empty_string(self, tmp_path: Path) -> None:
        doc = pymupdf.open()
        doc.new_page()
        doc.save(tmp_path / "bare.pdf")
        doc.close()

        loaded = PyMuPDFLoader().load(tmp_path / "bare.pdf")
        assert loaded.metadata.title is None
        assert loaded.metadata.author is None

    def test_load_many_preserves_order(self, pdf_dir: Path) -> None:
        paths = sorted(pdf_dir.glob("*.pdf"))
        documents = PyMuPDFLoader().load_many(paths)
        assert [d.metadata.source for d in documents] == [p.name for p in paths]


class TestRegistry:
    def test_default_is_registered(self) -> None:
        assert isinstance(get_loader(), LOADERS[DEFAULT_LOADER])

    def test_unknown_name_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="Unknown loader 'nope'. Available: pymupdf"):
            get_loader("nope")

    def test_kwargs_reach_the_constructor(self, pdf_dir: Path) -> None:
        loader = get_loader("pymupdf", page_separator="<<<")
        assert "<<<" in loader.load(pdf_dir / "2401.00001v1.pdf").content


class TestStore:
    def test_round_trip_preserves_content_and_page_map(
        self, document: Document, tmp_path: Path
    ) -> None:
        path = save_document(document, tmp_path)
        restored = load_document(path)

        assert path.name == "2401.00001v1.json"
        assert restored.id == document.id
        assert restored.content == document.content
        assert restored.metadata.page_starts == document.metadata.page_starts
        assert restored.metadata.paper_id == document.metadata.paper_id

    def test_load_documents_reads_a_directory(self, pdf_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "extracted"
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            save_document(PyMuPDFLoader().load(pdf), out)

        assert [d.metadata.paper_id for d in load_documents(out)] == [
            "2401.00001v1",
            "2401.00002v2",
        ]


class TestExtractCLI:
    def test_collect_pdfs_from_a_folder_and_a_file(self, pdf_dir: Path) -> None:
        assert len(collect_pdfs(pdf_dir)) == 2
        assert collect_pdfs(pdf_dir / "2401.00001v1.pdf") == [pdf_dir / "2401.00001v1.pdf"]

    def test_collect_pdfs_ignores_non_pdfs_and_subfolders(self, pdf_dir: Path) -> None:
        (pdf_dir / "notes.txt").write_text("not a pdf")
        nested = pdf_dir / "nested"
        nested.mkdir()
        (nested / "deep.pdf").write_bytes(b"%PDF-1.4")

        assert len(collect_pdfs(pdf_dir)) == 2
        assert len(collect_pdfs(pdf_dir, recursive=True)) == 3

    def test_extract_writes_then_skips_cached(self, pdf_dir: Path, tmp_path: Path) -> None:
        pdfs = collect_pdfs(pdf_dir)
        out = tmp_path / "extracted"

        assert extract(pdfs, PyMuPDFLoader(), out) == (2, 0, 0)
        assert extract(pdfs, PyMuPDFLoader(), out) == (0, 2, 0)
        assert extract(pdfs, PyMuPDFLoader(), out, force=True) == (2, 0, 0)

    def test_a_broken_pdf_does_not_abort_the_run(self, pdf_dir: Path, tmp_path: Path) -> None:
        broken = pdf_dir / "corrupt.pdf"
        broken.write_bytes(b"not really a pdf")

        written, skipped, failed = extract(collect_pdfs(pdf_dir), PyMuPDFLoader(), tmp_path / "out")
        assert (written, skipped, failed) == (2, 0, 1)
