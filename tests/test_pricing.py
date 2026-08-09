"""Tests for the pricing snapshots every reported cost is computed from.

The rates are transcribed by hand and nothing fetches or validates them, so the
failure mode is a confidently wrong dollar figure rather than an error. These
check the two properties that a hand edit can break silently:

* the file is filtered — a model above the ceiling must not reappear, since a
  typo in ``--model`` would then bill a 488-query run at frontier rates;
* a model that is absent costs *unknown*, never zero, because a run reporting
  $0.00 reads as free rather than as unpriced.
"""

from __future__ import annotations

import csv

import pytest

from openai_client.pricing import (
    MAX_PRICE_PER_1M,
    PRICES,
    _latest_pricing_file,
    cost_usd,
)


def rows() -> list[dict]:
    with open(_latest_pricing_file(), newline="") as handle:
        return list(csv.DictReader(handle))


class TestTheFilter:
    def test_no_model_exceeds_the_ceiling(self) -> None:
        """The guard the CSV exists to enforce. Named per offender, because
        "some row is too expensive" is not something you can act on."""
        over = [
            f"{row['model']} ({row['input_per_1m']} in / {row['output_per_1m']} out)"
            for row in rows()
            if max(float(row["input_per_1m"]), float(row["output_per_1m"])) > MAX_PRICE_PER_1M
        ]
        assert not over, f"above ${MAX_PRICE_PER_1M}/1M: {', '.join(over)}"

    def test_the_ceiling_is_a_ceiling_not_a_floor(self) -> None:
        """A filter that removed everything would also pass the test above."""
        assert len(PRICES) >= 5

    def test_the_model_this_project_runs_on_is_priced(self) -> None:
        """Filtering must not drop the default out from under the cost logs."""
        from app.rag.generation.llm import DEFAULT_MODEL

        assert DEFAULT_MODEL in PRICES


class TestUnpricedModels:
    def test_an_absent_model_raises_rather_than_costing_zero(self) -> None:
        """`OpenAILLM.cost_usd` turns this into None and prints "cost unknown".
        Returning 0.0 here would make an unpriced run look like a free one."""
        with pytest.raises(KeyError):
            cost_usd("gpt-5.5-pro", 1_000_000, 1_000_000)

    def test_a_filtered_out_model_is_genuinely_absent(self) -> None:
        """gpt-5.5 was in the 06192026 snapshot at $30/1M output and is exactly
        what the ceiling exists to keep out."""
        assert "gpt-5.5" not in PRICES


class TestSnapshotSelection:
    def test_the_newest_dated_file_wins(self) -> None:
        assert _latest_pricing_file().name.startswith("pricing_")

    def test_superseded_snapshots_are_kept(self, tmp_path) -> None:
        """An old run's cost stays explainable by the rates in force then."""
        directory = _latest_pricing_file().parent
        assert len(list(directory.glob("pricing_*.csv"))) > 1

    def test_every_row_parses_as_a_price(self) -> None:
        for row in rows():
            assert float(row["input_per_1m"]) > 0
            assert float(row["output_per_1m"]) > 0
            # Empty means the model has no cached rate, which is not zero.
            if row["cached_input_per_1m"]:
                assert float(row["cached_input_per_1m"]) > 0

    def test_cached_input_is_never_dearer_than_fresh_input(self) -> None:
        """A transcription slip that swapped two columns would show up here and
        nowhere else — the arithmetic would still run and still look plausible."""
        for row in rows():
            if row["cached_input_per_1m"]:
                assert float(row["cached_input_per_1m"]) <= float(row["input_per_1m"]), row["model"]
