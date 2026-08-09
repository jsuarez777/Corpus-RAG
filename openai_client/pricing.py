"""
OpenAI model pricing — loaded from the most recent pricing CSV in the same directory.
Looks for pricing_<MMDDYYYY>.csv files; falls back to pricing.csv (treated as oldest).
CSV columns: model, input_per_1m, cached_input_per_1m, output_per_1m  (USD per 1M tokens)
Empty cached_input_per_1m means caching is not available for that model.

Rates are transcribed from https://platform.openai.com/docs/pricing; see
:func:`_latest_pricing_file` for how the snapshots are dated and superseded.
"""

import csv
import re
from datetime import date
from pathlib import Path

_DIR = Path(__file__).parent

#: Models above this on input *or* output are left out of the CSVs, in USD per
#: 1M tokens. Not a statement about the models — a guard on what this project
#: can accidentally spend. Every stage here bills per query across a 488-query
#: benchmark, twice over once judging runs, so a model at $30 output turns a
#: routine grid run into a three-figure one. An omitted model is not free: it
#: has no pricing entry, so ``cost_usd`` returns None and every report says
#: "cost unknown" rather than quietly showing $0.00. Raising the ceiling means
#: transcribing the extra rows, not changing code.
MAX_PRICE_PER_1M = 5.00


def _latest_pricing_file(directory: Path = _DIR) -> Path:
    """The newest dated pricing CSV, which is the one every cost figure uses.

    **Where the data comes from:** https://developers.openai.com/api/docs/pricing
    (``platform.openai.com/docs/pricing`` 301s there) — transcribed by hand, not
    fetched. The developer-docs page rather than the marketing one because it
    breaks out the cached-input rate per model, which is the third column here
    and the difference between a right and a wrong number on any workload that
    repeats a system prompt.

    **The CSVs are filtered, not complete.** Models priced above
    :data:`MAX_PRICE_PER_1M` on input or output are left out deliberately, so a
    typo in ``--model`` cannot quietly bill a 488-query run at frontier rates.
    A model that is missing reports "cost unknown", never $0.00.

    The date in the filename is the day the rates were copied, so it is a claim
    about the snapshot rather than anything verified. Nothing here notices when
    OpenAI changes a price: the newest file simply wins, and if it is stale then
    every reported cost is quietly stale with it. Refreshing means adding a new
    ``pricing_<MMDDYYYY>.csv`` beside the others — the old ones stay, so a cost
    recorded in a past run can still be explained by the rates in force then.
    """
    dated: list[tuple[date, Path]] = []
    for f in directory.glob("pricing_*.csv"):
        m = re.fullmatch(r"pricing_(\d{2})(\d{2})(\d{4})\.csv", f.name)
        if m:
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                dated.append((date(year, month, day), f))
            except ValueError:
                pass  # skip files with invalid dates

    if dated:
        return max(dated, key=lambda t: t[0])[1]

    fallback = directory / "pricing.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No pricing CSV found in {directory}")


def _load_prices(path: Path | None = None) -> dict[str, dict[str, float | None]]:
    if path is None:
        path = _latest_pricing_file()
    prices = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cached = row["cached_input_per_1m"].strip()
            prices[row["model"]] = {
                "input": float(row["input_per_1m"]),
                "cached_input": float(cached) if cached else None,
                "output": float(row["output_per_1m"]),
            }
    return prices


_PRICING_FILE = _latest_pricing_file()
PRICES: dict[str, dict[str, float | None]] = _load_prices(_PRICING_FILE)


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Return estimated cost in USD for a single call."""
    if model not in PRICES:
        raise KeyError(f"No pricing data for model '{model}'. Known models: {list(PRICES)}")
    p = PRICES[model]
    cached_rate = p["cached_input"] if p["cached_input"] is not None else p["input"]
    return (
        input_tokens * p["input"] + cached_input_tokens * cached_rate + output_tokens * p["output"]
    ) / 1_000_000


def cost_summary(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> str:
    """Return a human-readable cost breakdown string."""
    total = cost_usd(model, input_tokens, output_tokens, cached_input_tokens)
    p = PRICES[model]
    cached_rate = p["cached_input"]
    lines = [
        f"Model : {model}",
        f"Input : {input_tokens:,} tokens  @ ${p['input']:.4f}/1M = ${input_tokens * p['input'] / 1_000_000:.6f}",
    ]
    if cached_input_tokens:
        rate = cached_rate if cached_rate is not None else p["input"]
        lines.append(
            f"Cached: {cached_input_tokens:,} tokens  @ ${rate:.4f}/1M = ${cached_input_tokens * rate / 1_000_000:.6f}"
        )
    lines += [
        f"Output: {output_tokens:,} tokens  @ ${p['output']:.4f}/1M = ${output_tokens * p['output'] / 1_000_000:.6f}",
        f"Total : ${total:.6f}",
    ]
    return "\n".join(lines)
