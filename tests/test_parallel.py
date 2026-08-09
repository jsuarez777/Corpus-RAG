"""Tests for the worker pool the batch stages run their API calls through.

Speed is not what these check — a thread pool over sleeps proves little. What
matters is that going concurrent did not quietly weaken the two guarantees the
batch stages were built on:

* every result is handed back on the calling thread, so the append-and-flush
  that makes a run resumable needs no lock and still means what it meant
  serially;
* one failed item is dropped, not fatal, exactly as in the serial loop.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.rag.utils.parallel import DEFAULT_WORKERS, run_in_parallel


class TestResults:
    def test_every_item_is_processed(self) -> None:
        assert sorted(run_in_parallel(lambda n: n * 2, list(range(20)), workers=4)) == [
            n * 2 for n in range(20)
        ]

    def test_none_is_dropped_rather_than_collected(self) -> None:
        """The convention both batch stages already use for a query the model
        failed on — one bad item must not end a run that has been paid for."""
        out = run_in_parallel(lambda n: None if n % 2 else n, list(range(10)), workers=4)
        assert sorted(out) == [0, 2, 4, 6, 8]

    def test_an_empty_batch_does_nothing(self) -> None:
        assert run_in_parallel(lambda n: n, [], workers=4) == []

    def test_an_exception_is_not_swallowed(self) -> None:
        """Dropping a `None` is deliberate; dropping a raised error would hide
        a broken run behind a plausible-looking partial one."""

        def boom(n: int) -> int:
            if n == 3:
                raise RuntimeError("upstream is down")
            return n

        with pytest.raises(RuntimeError, match="upstream is down"):
            run_in_parallel(boom, list(range(10)), workers=4)


class TestCallbacks:
    def test_results_arrive_on_the_calling_thread(self) -> None:
        """This is what lets the batch stages write JSONL without a lock. If
        callbacks ever moved onto workers, both files could interleave lines."""
        caller = threading.get_ident()
        seen: list[int] = []

        run_in_parallel(
            lambda n: n,
            list(range(20)),
            workers=8,
            on_result=lambda n: seen.append(threading.get_ident()),
        )

        assert set(seen) == {caller}

    def test_progress_counts_failures_too(self) -> None:
        """Progress is about the run finishing, not about how much of it
        worked — a log that stalls at 400/488 because 88 queries failed reads
        as a hang."""
        counts: list[tuple[int, int]] = []

        run_in_parallel(
            lambda n: None if n < 5 else n,
            list(range(10)),
            workers=4,
            on_progress=lambda done, total: counts.append((done, total)),
        )

        assert counts[-1] == (10, 10)

    def test_the_callback_runs_once_per_result(self) -> None:
        seen: list[int] = []
        run_in_parallel(lambda n: n, list(range(30)), workers=6, on_result=seen.append)
        assert sorted(seen) == list(range(30))


class TestSerialPath:
    def test_one_worker_keeps_the_original_order(self) -> None:
        """`workers=1` is the plain loop, for anything that needs determinism."""
        seen: list[int] = []
        run_in_parallel(lambda n: n, list(range(10)), workers=1, on_result=seen.append)
        assert seen == list(range(10))

    def test_one_worker_still_drops_none(self) -> None:
        assert run_in_parallel(lambda n: None, [1, 2, 3], workers=1) == []

    def test_the_serial_path_reports_progress_the_same_way(self) -> None:
        counts: list[tuple[int, int]] = []
        run_in_parallel(
            lambda n: n, [1, 2, 3], workers=1, on_progress=lambda d, t: counts.append((d, t))
        )
        assert counts == [(1, 3), (2, 3), (3, 3)]


class TestConcurrency:
    def test_work_actually_overlaps(self) -> None:
        """The whole point: 8 blocking calls of 50ms must not take 400ms.

        Timed loosely — this asserts concurrency happened at all, not a
        throughput figure, since a loaded CI box can make any tighter bound
        flake."""
        started = time.perf_counter()
        run_in_parallel(lambda n: time.sleep(0.05) or n, list(range(8)), workers=8)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.2

    def test_the_default_is_modest(self) -> None:
        """`OpenAILLM` backs off on `RateLimitError`, so an oversized pool does
        not fail — it spends its time retrying, which looks like throughput and
        is not."""
        assert 1 < DEFAULT_WORKERS <= 16
