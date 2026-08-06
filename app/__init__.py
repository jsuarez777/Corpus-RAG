"""Package init, kept almost empty on purpose — but not entirely.

FAISS and PyTorch each bundle their own OpenMP runtime, and the pipeline needs
both in one process: it embeds with torch and searches with faiss. On macOS the
two runtimes collide and kill the process — signal 11 or an abort, no
traceback, nothing in the log. It takes real work to trigger, so short-text
smoke tests pass and only a full run dies.

Three settings, all required, established by bisecting the crash:

* ``OMP_NUM_THREADS=1`` — the collision is over the thread pool. 2, 4 and 8 all
  still segfault; only one thread survives.
* ``KMP_DUPLICATE_LIB_OK`` — lets the second runtime load instead of aborting.
  It carries a documented risk of wrong results, which is why the pinning above
  is not optional: with one thread there is no race to lose. Checked rather
  than assumed — faiss search over 12,399 vectors agrees exactly with the same
  top-10 recomputed in numpy, rankings and scores, on all 50 probe queries.
* Importing both here, faiss first. Order matters and lazy loading is not
  enough: torch first aborts inside ``faiss.search``, and either one arriving
  after the other has run a parallel region crashes too. Settling it in the
  package init means no entry point can get it wrong.

The cost is single-threaded faiss search and about a second of import time on
`extract.py`, which has no other use for either library. Flat inner-product
search over this corpus is small and memory-bound, so it is not what the
experiment grid waits on; embedding runs on MPS and is untouched.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # noqa: E402, F401
import torch  # noqa: E402, F401
