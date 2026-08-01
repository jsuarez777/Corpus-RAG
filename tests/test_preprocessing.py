"""Tests for text cleaning.

The stage exists for one measurable reason: pysbd treats a newline as a
sentence boundary, so un-reflowed PDF text segments into lines rather than
sentences. The last test here is the one that matters — it asserts the effect
on segmentation, not just the string transformation.
"""

import pytest

from app.rag.chunking._sentences import segment_sentences, sentence_cap_tokens
from app.rag.preprocessing import (
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


class TestNormalizeCharacters:
    def test_expands_ligatures(self) -> None:
        assert normalize_characters("ﬁnal classiﬁcation eﬀect") == "final classification effect"

    def test_drops_soft_hyphens_and_zero_widths(self) -> None:
        assert normalize_characters("micro­scope​image") == "microscopeimage"

    def test_drops_control_characters_but_keeps_newlines_and_tabs(self) -> None:
        assert normalize_characters("a\x00b\x07c\nd\te") == "abc\nd\te"

    def test_leaves_ordinary_text_alone(self) -> None:
        assert (
            normalize_characters("Ripley's K-function, α = 0.05") == "Ripley's K-function, α = 0.05"
        )


class TestDehyphenate:
    def test_rejoins_a_word_split_across_lines(self) -> None:
        assert dehyphenate("real-\ntime movies") == "realtime movies"

    def test_handles_unicode_hyphens(self) -> None:
        assert dehyphenate("tran‐\nscription") == "transcription"

    def test_leaves_a_hyphen_inside_a_line(self) -> None:
        assert dehyphenate("well-known result") == "well-known result"

    def test_leaves_a_dash_before_a_paragraph_break(self) -> None:
        """A hyphen followed by a blank line is not a split word."""
        assert dehyphenate("dash -\n\nNext paragraph") == "dash -\n\nNext paragraph"


class TestReflow:
    def test_joins_wrapped_lines(self) -> None:
        wrapped = "However, genetic and\nlive imaging techniques have\noutpaced analysis."
        assert (
            reflow(wrapped)
            == "However, genetic and live imaging techniques have outpaced analysis."
        )

    def test_keeps_paragraph_breaks(self) -> None:
        assert reflow("First para\nwrapped.\n\nSecond para\nwrapped.") == (
            "First para wrapped.\n\nSecond para wrapped."
        )

    def test_collapses_runs_of_blank_lines(self) -> None:
        assert reflow("One.\n\n\n\nTwo.") == "One.\n\nTwo."

    def test_drops_empty_paragraphs(self) -> None:
        assert reflow("\n\n  \n\nOnly text.\n\n   ") == "Only text."

    def test_empty_input(self) -> None:
        assert reflow("") == ""


class TestMergeContinuations:
    """PyMuPDF blocks split paragraphs mid-sentence; a break is only real
    after terminal punctuation."""

    def test_joins_a_severed_sentence(self) -> None:
        blocks = "The model is employed to forecast\n\nthe count of active cells."
        assert (
            merge_continuations(blocks)
            == "The model is employed to forecast the count of active cells."
        )

    def test_keeps_a_break_after_a_finished_sentence(self) -> None:
        assert merge_continuations("First thought.\n\nSecond thought.") == (
            "First thought.\n\nSecond thought."
        )

    @pytest.mark.parametrize("ending", ["end.", "end!", "end?", 'end."', "end.)", "end.’"])
    def test_recognizes_terminal_punctuation_inside_closers(self, ending: str) -> None:
        assert merge_continuations(f"{ending}\n\nNext.") == f"{ending}\n\nNext."

    def test_glues_a_heading_to_its_body(self) -> None:
        """The documented cost of the rule: a heading has no terminal
        punctuation, so it reads as an unfinished sentence."""
        assert merge_continuations("2 Methods\n\nWe began by.") == "2 Methods We began by."

    def test_chains_several_fragments(self) -> None:
        assert merge_continuations("a\n\nb\n\nc.\n\nd.") == "a b c.\n\nd."

    def test_empty_input(self) -> None:
        assert merge_continuations("") == ""


class TestCleanText:
    def test_runs_dehyphenate_before_reflow(self) -> None:
        """Reflow replaces the newline with a space; once it has, the broken
        word can no longer be rejoined."""
        assert clean_text("real-\ntime") == "realtime"

    def test_full_pipeline(self) -> None:
        raw = "The clas-\nsiﬁcation  of  eﬀects\nwas   measured.\n\nA second\nparagraph."
        assert (
            clean_text(raw) == "The classification of effects was measured.\n\nA second paragraph."
        )

    def test_repairs_a_block_split_mid_sentence(self) -> None:
        """The whole pipeline on the shape PyMuPDF actually emits."""
        raw = "we developed a quan-\ntitative approach to\n\nmeasure the outputs."
        assert clean_text(raw) == "we developed a quantitative approach to measure the outputs."

    def test_collapse_spaces_squeezes_layout_padding(self) -> None:
        assert collapse_spaces("a     b\t\tc") == "a b c"

    def test_reflow_only_leaves_ligatures(self) -> None:
        """The A/B control must change exactly one thing."""
        assert reflow_only("ﬁnal\nresult") == "ﬁnal result"


class TestRegistry:
    def test_named_preprocessors_resolve(self) -> None:
        assert resolve_preprocessor("default") is clean_text
        assert resolve_preprocessor("reflow") is reflow_only
        assert resolve_preprocessor("none") is None

    def test_callable_passes_through(self) -> None:
        assert resolve_preprocessor(str.upper) is str.upper

    def test_none_passes_through(self) -> None:
        assert resolve_preprocessor(None) is None

    def test_unknown_name_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="Unknown preprocessor 'scrub'"):
            resolve_preprocessor("scrub")

    def test_every_registered_name_resolves(self) -> None:
        for name in PREPROCESSORS:
            resolved = resolve_preprocessor(name)
            assert resolved is None or callable(resolved)


class TestEffectOnSegmentation:
    """The reason this stage exists, asserted end to end."""

    WRAPPED = (
        "However, genetic and\n"
        "live imaging techniques have outpaced analysis approaches\n"
        "to harvest the bountiful information contained within real-\n"
        "time movies of transcriptional dynamics with modern meth-\n"
        "ods confined to static parameter cell and transcript tracking\n"
        "methods. To assess these mutant enhancer pheno-\n"
        "types systematically, we developed a quantitative approach.\n"
    )

    def test_wrapped_text_segments_into_lines(self) -> None:
        """Six lines, two sentences: un-reflowed, pysbd sees the lines."""
        segments = segment_sentences(self.WRAPPED, sentence_cap_tokens(5))
        assert len(segments) > 4

    def test_reflow_recovers_real_sentences(self) -> None:
        cleaned = clean_text(self.WRAPPED)
        segments = segment_sentences(cleaned, sentence_cap_tokens(5))

        assert len(segments) == 2
        # And the split words are whole again, so retrieval can match them.
        assert "realtime movies" in cleaned
        assert "phenotypes systematically" in cleaned

    def test_reflow_raises_median_segment_size(self) -> None:
        cap = sentence_cap_tokens(5)
        raw = [s.num_tokens for s in segment_sentences(self.WRAPPED, cap)]
        cleaned = [s.num_tokens for s in segment_sentences(clean_text(self.WRAPPED), cap)]
        assert min(cleaned) > max(raw)
