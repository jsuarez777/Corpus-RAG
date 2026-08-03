"""Tests for chunk-to-section alignment.

The property that matters: a chunk overlapping a section boundary is labelled
with both sections. Getting that wrong leaves sections with no chunks, and a
query on such a section scores zero recall regardless of the retriever — a
silent ceiling that looks like a bad retriever.

The other property is that matching survives the gap between two extractions of
the same paper: the corpus holds LaTeX and OCR text, ours holds rendered PDF
glyphs.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.rag.evaluation import (
    SectionSpan,
    align_corpus,
    assign_sections,
    load_sections,
    locate_sections,
    normalize,
    strip_math,
)
from app.rag.models import Chunk, ChunkMetadata, Document, DocumentMetadata

# Three sections of prose, concatenated, as our extraction would render them.
BODY_ONE = "Recent advances in live imaging let researchers watch embryogenesis unfold. "
BODY_TWO = "We trained a gradient boosted model on the segmented cell counts per frame. "
BODY_THREE = "Accuracy improved over the baseline across every held out replicate set. "


def make_document(content: str, paper_id: str = "2401.00001v1") -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(source=f"{paper_id}.pdf", paper_id=paper_id),
    )


def make_chunk(document: Document, start: int, end: int, index: int = 0) -> Chunk:
    return Chunk(
        content=document.content[start:end],
        metadata=ChunkMetadata(
            document_id=document.id,
            source=document.metadata.source,
            start_char=start,
            end_char=end,
            chunk_index=index,
        ),
    )


@pytest.fixture
def document() -> Document:
    """Three sections as PyMuPDF would render them: headings inline, no markup."""
    return make_document(
        "1. Introduction\n\n"
        + BODY_ONE * 3
        + "2. Methods\n\n"
        + BODY_TWO * 3
        + "3. Results\n\n"
        + BODY_THREE * 3
    )


@pytest.fixture
def sections() -> list[dict]:
    """The same three sections as the corpus stores them: markdown headings."""
    return [
        {"text": "# 1. Introduction\n\n" + BODY_ONE * 3},
        {"text": "# 2. Methods\n\n" + BODY_TWO * 3},
        {"text": "# 3. Results\n\n" + BODY_THREE * 3},
    ]


class TestStripMath:
    def test_removes_inline_math(self) -> None:
        assert "operatorname" not in strip_math("The space $\\operatorname{Corr}(m)$ is smooth.")

    def test_removes_display_math(self) -> None:
        assert "frac" not in strip_math("We obtain\n$$\n\\frac{a}{b}\n$$\nand conclude.")

    def test_removes_environments(self) -> None:
        stripped = strip_math("Given\n\\begin{aligned}\nx &= 1\n\\end{aligned}\nwe see.")
        assert "aligned" not in stripped and "we see" in stripped

    def test_removes_table_and_image_placeholders(self) -> None:
        assert "table_id1" not in strip_math("As shown in <table_id1> the trend holds.")

    def test_keeps_the_prose(self) -> None:
        """Everything the matcher has to work with lives in what survives."""
        stripped = strip_math("The metric $d^{orb}$ is continuous on the orbit space.")
        assert "The metric" in stripped
        assert "is continuous on the orbit space." in stripped


class TestNormalize:
    def test_keeps_only_lowercase_alphanumerics(self) -> None:
        flat, _ = normalize("Hello, World! (2024)")
        assert flat == "helloworld2024"

    def test_offsets_index_the_original(self) -> None:
        text = "A b, C!"
        flat, offsets = normalize(text)
        for position, char in enumerate(flat):
            assert text[offsets[position]].lower() == char

    def test_two_renderings_of_one_sentence_agree(self) -> None:
        """The whole point: different whitespace and punctuation, same string."""
        assert normalize("real-\ntime  movies")[0] == normalize("Realtime movies!")[0]

    def test_empty_input(self) -> None:
        assert normalize("") == ("", [])


class TestLocateSections:
    def test_finds_every_section(self, document: Document, sections: list[dict]) -> None:
        spans = locate_sections(document, sections)
        assert [span.section_index for span in spans] == [0, 1, 2]

    def test_spans_ascend_and_tile_the_document(
        self, document: Document, sections: list[dict]
    ) -> None:
        spans = locate_sections(document, sections)
        assert spans[0].start_char == 0
        for earlier, later in zip(spans, spans[1:], strict=False):
            assert earlier.end_char == later.start_char
        assert spans[-1].end_char == len(document.content)

    def test_offsets_point_at_the_right_text(
        self, document: Document, sections: list[dict]
    ) -> None:
        spans = locate_sections(document, sections)
        assert document.content[spans[1].start_char :].startswith("2. Methods")
        assert document.content[spans[2].start_char :].startswith("3. Results")

    def test_a_table_of_contents_does_not_capture_the_headings(self) -> None:
        """A TOC lists every heading, in order, near the front. Searching each
        section forward of the previous one is what stops section 2 matching
        its own contents line — six of eight failures on the real corpus.
        """
        toc = "Contents\n\n1. Introduction 1\n2. Methods 4\n3. Results 9\n\n"
        document = make_document(
            toc
            + "1. Introduction\n\n"
            + BODY_ONE * 3
            + "2. Methods\n\n"
            + BODY_TWO * 3
            + "3. Results\n\n"
            + BODY_THREE * 3
        )
        sections = [
            {"text": "# 1. Introduction\n\n" + BODY_ONE * 3},
            {"text": "# 2. Methods\n\n" + BODY_TWO * 3},
            {"text": "# 3. Results\n\n" + BODY_THREE * 3},
        ]
        spans = locate_sections(document, sections)

        assert len(spans) == 3
        # Every span must land past the table of contents, not inside it.
        assert all(span.start_char >= len(toc) - 2 for span in spans)
        assert document.content[spans[2].start_char :].startswith("3. Results")

    def test_a_repeated_phrase_does_not_drag_the_start_backwards(self) -> None:
        """`str.find` returns the earliest match, so a probe from deep inside a
        section lands on an earlier copy of a repeated phrase. Anchoring on the
        first probe is what stops that becoming a wrong section boundary."""
        boilerplate = "The same standard disclaimer sentence appears in both sections here. "
        document = make_document(
            "1. Intro\n\n" + boilerplate * 3 + "2. Methods\n\n" + boilerplate * 3
        )
        sections = [
            {"text": "# 1. Intro\n\n" + boilerplate * 3},
            {"text": "# 2. Methods\n\n" + boilerplate * 3},
        ]
        spans = locate_sections(document, sections)
        assert document.content[spans[1].start_char :].startswith("2. Methods")

    def test_records_probe_agreement(self, document: Document, sections: list[dict]) -> None:
        """Several probes hitting is what distinguishes a match from a
        coincidental n-gram."""
        assert all(span.probe_hits > 0 for span in locate_sections(document, sections))

    def test_a_section_that_is_not_present_is_skipped(self, document: Document) -> None:
        sections = [
            {"text": BODY_ONE * 3},
            {"text": "An entirely unrelated discussion of maritime insurance law. " * 3},
            {"text": BODY_THREE * 3},
        ]
        assert [span.section_index for span in locate_sections(document, sections)] == [0, 2]

    def test_math_only_sections_still_match_on_their_prose(self) -> None:
        """The failure mode that took v1 from 100% to 87%: the corpus stores
        LaTeX source where the PDF shows rendered glyphs."""
        prose = "Proof of Proposition 3.2. The metric spaces share the same topology here. " * 2
        document = make_document("B.1. " + prose)
        sections = [
            {
                "text": "# B.1. Proof of Proposition 3.2.\n\nThe metric spaces "
                "$(\\operatorname{Corr}(m,[k]), d^{\\text {orb }, k})$ "
                "share the same topology here. " + prose
            }
        ]
        assert len(locate_sections(document, sections)) == 1

    def test_an_empty_document_locates_nothing(self, sections: list[dict]) -> None:
        assert locate_sections(make_document(""), sections) == []

    def test_no_sections_locates_nothing(self, document: Document) -> None:
        assert locate_sections(document, []) == []


class TestAssignSections:
    SPANS = [SectionSpan(0, 0, 100, 10), SectionSpan(1, 100, 200, 10), SectionSpan(2, 200, 300, 10)]

    def _chunk(self, start: int, end: int) -> Chunk:
        document = make_document("x" * 300)
        return make_chunk(document, start, end)

    def test_a_contained_chunk_gets_one_section(self) -> None:
        chunk = self._chunk(10, 50)
        assign_sections([chunk], self.SPANS)
        assert chunk.metadata.section_indices == [0]

    def test_a_straddling_chunk_gets_both(self) -> None:
        """The reason the field is a list. A chunk beginning in section 0 and
        ending in section 1 can answer a query on either."""
        chunk = self._chunk(80, 120)
        assign_sections([chunk], self.SPANS)
        assert chunk.metadata.section_indices == [0, 1]

    def test_a_long_chunk_spans_three(self) -> None:
        chunk = self._chunk(50, 250)
        assign_sections([chunk], self.SPANS)
        assert chunk.metadata.section_indices == [0, 1, 2]

    def test_boundaries_are_half_open(self) -> None:
        """A chunk ending exactly where a section starts does not touch it."""
        chunk = self._chunk(0, 100)
        assign_sections([chunk], self.SPANS)
        assert chunk.metadata.section_indices == [0]

    def test_returns_the_labelled_count(self) -> None:
        chunks = [self._chunk(0, 50), self._chunk(400, 500)]
        assert assign_sections(chunks, self.SPANS) == 1
        assert chunks[1].metadata.section_indices == []

    def test_no_spans_leaves_everything_unlabelled(self) -> None:
        chunk = self._chunk(0, 50)
        assign_sections([chunk], [])
        assert chunk.metadata.section_indices == []

    def test_every_section_receives_a_chunk(self, document: Document, sections: list[dict]) -> None:
        """Overlap assignment exists so no section is left unscoreable. With
        chunks this coarse, midpoint assignment would lose one."""
        spans = locate_sections(document, sections)
        size = len(document.content) // 2
        chunks = [
            make_chunk(document, start, min(start + size, len(document.content)), index)
            for index, start in enumerate(range(0, len(document.content), size))
        ]
        assign_sections(chunks, spans)

        covered = {index for chunk in chunks for index in chunk.metadata.section_indices}
        assert covered == {0, 1, 2}


class TestLoadSections:
    def test_reads_a_corpus_file(self, tmp_path: Path) -> None:
        (tmp_path / "2401.00001v1.json").write_text(
            json.dumps({"title": "T", "sections": [{"text": "a"}, {"text": "b"}]})
        )
        assert len(load_sections("2401.00001v1", tmp_path)) == 2

    def test_a_missing_paper_is_none_not_an_error(self, tmp_path: Path) -> None:
        """Not every extracted paper is in the labelled corpus; that is a
        coverage figure, not a crash."""
        assert load_sections("nope", tmp_path) is None


class TestAlignCorpus:
    @pytest.fixture
    def corpus_dir(self, tmp_path: Path, sections: list[dict]) -> Path:
        directory = tmp_path / "corpus"
        directory.mkdir()
        (directory / "2401.00001v1.json").write_text(json.dumps({"sections": sections}))
        return directory

    def test_reports_full_coverage(self, document: Document, corpus_dir: Path) -> None:
        chunks = [make_chunk(document, 0, len(document.content))]
        report = align_corpus([document], chunks, corpus_dir)

        assert report.sections_located == report.sections_total == 3
        assert report.section_coverage == 1.0
        assert report.chunk_coverage == 1.0

    def test_counts_papers_without_a_corpus_entry(self, tmp_path: Path) -> None:
        (tmp_path / "corpus").mkdir()
        document = make_document(BODY_ONE, paper_id="9999.99999v1")
        report = align_corpus([document], [make_chunk(document, 0, 10)], tmp_path / "corpus")

        assert report.papers_missing_corpus == 1
        assert report.chunks_labelled == 0

    def test_counts_chunks_spanning_two_sections(
        self, document: Document, corpus_dir: Path
    ) -> None:
        chunks = [make_chunk(document, 0, len(document.content))]
        report = align_corpus([document], chunks, corpus_dir)
        assert report.chunks_multi_section == 1

    def test_summary_names_the_coverage(self, document: Document, corpus_dir: Path) -> None:
        report = align_corpus([document], [make_chunk(document, 0, 50)], corpus_dir)
        assert "3/3 located (100%)" in report.summary()

    def test_chunks_are_matched_to_documents_by_paper_id(
        self, document: Document, corpus_dir: Path
    ) -> None:
        """A chunk from another paper must not be labelled with this one's
        sections."""
        other = make_document(BODY_ONE, paper_id="9999.99999v1")
        stray = make_chunk(other, 0, 20)
        align_corpus([document], [make_chunk(document, 0, 50), stray], corpus_dir)
        assert stray.metadata.section_indices == []

    def test_an_empty_corpus_run(self, corpus_dir: Path) -> None:
        report = align_corpus([], [], corpus_dir)
        assert report.section_coverage == 0.0
        assert report.chunk_coverage == 0.0


class TestChunkIdentity:
    def test_alignment_does_not_disturb_offsets(self, document: Document, corpus_dir=None) -> None:
        """Labelling must not touch the offsets citations resolve through."""
        chunk = make_chunk(document, 10, 60)
        before = (chunk.metadata.start_char, chunk.metadata.end_char, chunk.content)
        assign_sections([chunk], [SectionSpan(0, 0, 100, 10)])
        assert (chunk.metadata.start_char, chunk.metadata.end_char, chunk.content) == before

    def test_uuid_is_untouched(self, document: Document) -> None:
        chunk = make_chunk(document, 0, 10)
        original = chunk.id
        assign_sections([chunk], [SectionSpan(0, 0, 100, 10)])
        assert chunk.id == original
        assert chunk.id != uuid4()
