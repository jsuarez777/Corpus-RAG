"""Run one function over many items concurrently, handing results back as they land.

Both batch stages — answering and judging — are one API call per query in a loop,
and nearly all of that time is network wait. Answering 488 queries took ~25
minutes and judging them ~12, almost none of it compute.

Two properties the batch stages depend on, which is why this is a helper rather
than a bare :class:`~concurrent.futures.ThreadPoolExecutor`:

* **Results arrive on the calling thread.** ``on_result`` runs in the caller,
  one at a time, so appending to a JSONL file needs no lock and the flush after
  each line still means what it meant serially: a run killed at any point has
  every answer it paid for already on disk.
* **Order is not preserved.** Records are keyed by query id and resumed by id,
  so out-of-order arrival costs nothing and waiting for it would cost the
  speedup.

Threads rather than processes: the work is I/O, the OpenAI client is thread-safe,
and a process pool would pay to pickle a chunk-bearing record each way. Worth
knowing where this must *not* be pointed — anything that calls into faiss or
torch. ``app/__init__.py`` survives having both in one process by pinning
``OMP_NUM_THREADS=1``, and that pinning assumes nothing runs concurrently.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: Enough to hide the latency of a call that takes a second or two, low enough
#: that the OpenAI tier's rate limit is not the thing being measured.
#: ``OpenAILLM`` already retries ``RateLimitError`` with exponential backoff, so
#: an oversized pool does not fail — it just spends its time backing off, which
#: looks like throughput and is not.
DEFAULT_WORKERS = 8

#: Futures in flight at once. Submitting all 488 up front would be harmless for
#: memory but would make a Ctrl-C leave hundreds of queued calls to cancel;
#: keeping the window small means stopping is nearly immediate.
QUEUE_FACTOR = 4


def run_in_parallel(
    work: Callable[[T], R | None],
    items: Sequence[T],
    *,
    workers: int = DEFAULT_WORKERS,
    on_result: Callable[[R], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Apply ``work`` to every item, collecting what it returns.

    ``work`` returning ``None`` means "nothing usable here" and is dropped —
    the convention both batch stages already use for a query the model failed
    on, so one bad item does not end the run.

    ``workers=1`` runs the plain loop rather than a pool of one, which keeps
    the serial path exactly as it was for anything that needs deterministic
    order.
    """
    if not items:
        return []
    if workers <= 1:
        return _serial(work, items, on_result=on_result, on_progress=on_progress)

    collected: list[R] = []
    pending: set = set()
    queue = list(reversed(items))
    done_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while queue or pending:
                while queue and len(pending) < workers * QUEUE_FACTOR:
                    pending.add(pool.submit(work, queue.pop()))

                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    done_count += 1
                    result = future.result()
                    if result is None:
                        continue
                    collected.append(result)
                    if on_result:
                        on_result(result)
                    if on_progress:
                        on_progress(done_count, len(items))
        except BaseException:
            # Ctrl-C included: drop what has not started so the pool's exit
            # does not block for the whole backlog. Work already in flight
            # still finishes, and whatever `on_result` wrote is on disk.
            for future in pending:
                future.cancel()
            raise

    return collected


def _serial(
    work: Callable[[T], R | None],
    items: Iterable[T],
    *,
    on_result: Callable[[R], None] | None,
    on_progress: Callable[[int, int], None] | None,
) -> list[R]:
    collected: list[R] = []
    total = len(items) if isinstance(items, Sequence) else 0
    for position, item in enumerate(items, start=1):
        result = work(item)
        if result is None:
            continue
        collected.append(result)
        if on_result:
            on_result(result)
        if on_progress:
            on_progress(position, total)
    return collected
