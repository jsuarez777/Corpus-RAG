"""The rankings artifact: what a retriever returned, saved before anything scores it.

Retrieval is the only stage that needs an index, an embedder, and — on macOS —
faiss and torch alive in the same process. Everything after it works on ordered
chunk ids: metrics count how many came from the gold section, generation looks
the ids up in ``data/chunks/`` to build a prompt, judging never sees them at
all. Writing the ids down once turns the expensive stage into an input rather
than a step every consumer has to repeat.

Two things follow from that, and they are the reason this file exists:

* **A config is retrieved once.** Scoring it, answering from it, and rescoring
  it after a rubric change all read the same file. Before, the answer stage
  re-ran a search the grid had already run over the same 488 queries.
* **Downstream stages can be threaded.** ``app/__init__.py`` survives faiss and
  torch sharing a process by pinning ``OMP_NUM_THREADS=1``, which works because
  nothing runs in parallel. A consumer that only reads ids never enters either
  library, so its worker pool cannot violate that.

One file per config, overwritten rather than timestamped: a ranking is a
function of the config and the corpus, so a second run of the same cell is the
same file, and the consumers below want to name it without knowing when it ran.
Results keep their timestamps because a metric belongs to the moment it was
measured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from app.rag.evaluation.metrics import Ranking
from app.rag.models import Chunk, RetrievalResult, RetrieverType

log = logging.getLogger(__name__)


class StaleRankings(RuntimeError):
    """A ranking names chunk ids the current chunk file does not contain."""


@dataclass
class RankingSet:
    """Every ranking from one grid cell, with the config that produced them."""

    config_id: str
    rankings: list[Ranking] = field(default_factory=list)
    label: str = ""
    index_id: str = ""
    experiment: str = ""
    top_k: int = 0
    config: dict = field(default_factory=dict)
    retrieved_at: str = ""

    def by_id(self) -> dict[str, Ranking]:
        return {ranking.query_id: ranking for ranking in self.rankings}

    def __len__(self) -> int:
        return len(self.rankings)


def rankings_file(rankings_dir: Path, config_id: str) -> Path:
    return rankings_dir / f"{config_id}.json"


def write_rankings(
    rankings_dir: Path,
    rankings: list[Ranking],
    *,
    config_id: str,
    label: str = "",
    index_id: str = "",
    experiment: str = "",
    top_k: int = 0,
    config: dict | None = None,
) -> Path:
    """Save one cell's rankings, carrying the config that produced them.

    The config travels with the ids for the same reason it travels with a
    metric: a ranking that cannot be traced back to a chunker and a retriever
    cannot be compared to another one, or safely reused by the answer stage.
    """
    rankings_dir.mkdir(parents=True, exist_ok=True)
    path = rankings_file(rankings_dir, config_id)
    payload = {
        "config_id": config_id,
        "label": label,
        "index_id": index_id,
        "experiment": experiment,
        "top_k": top_k,
        "num_queries": len(rankings),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config or {},
        "rankings": [ranking.as_dict() for ranking in rankings],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_rankings(path: Path) -> RankingSet:
    """Read a rankings file back, or say why it is not one."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not readable as rankings: {error}") from None

    if "rankings" not in payload or "config_id" not in payload:
        raise ValueError(
            f"{path.name} has no rankings — expected a file written by app/retrieve.py"
        )

    return RankingSet(
        config_id=payload["config_id"],
        rankings=[Ranking.from_dict(row) for row in payload["rankings"]],
        label=payload.get("label", ""),
        index_id=payload.get("index_id", ""),
        experiment=payload.get("experiment", ""),
        top_k=int(payload.get("top_k", 0)),
        config=payload.get("config", {}),
        retrieved_at=payload.get("retrieved_at", ""),
    )


def as_results(
    ranking: Ranking,
    chunks_by_id: dict[UUID, Chunk],
    *,
    top_k: int | None = None,
    retriever_type: str = "",
) -> list[RetrievalResult]:
    """Rebuild the retriever's output from saved ids and the chunk set.

    Chunk ids are generated per chunking run, so a rankings file is only valid
    against the ``data/chunks/`` file it was retrieved from. Re-chunking makes
    every id in it a miss, which would otherwise show up as answers written
    from silently fewer passages — so an unresolvable id raises rather than
    being skipped.
    """
    wanted = ranking.retrieved if top_k is None else ranking.retrieved[:top_k]
    missing = [chunk_id for chunk_id in wanted if chunk_id not in chunks_by_id]
    if missing:
        raise StaleRankings(
            f"{len(missing)} of {len(wanted)} chunk ids for query {ranking.query_id} "
            f"are not in this chunk set — the chunks were rebuilt after retrieval. "
            f"Re-run `python app/retrieve.py`."
        )

    kind = ranking.retriever_type or retriever_type or RetrieverType.DENSE.value
    scores = ranking.scores or [0.0] * len(ranking.retrieved)
    return [
        RetrievalResult(
            chunk=chunks_by_id[chunk_id],
            score=scores[position] if position < len(scores) else 0.0,
            retriever_type=RetrieverType(kind),
        )
        for position, chunk_id in enumerate(wanted)
    ]


def load_rankings_dir(rankings_dir: Path) -> list[RankingSet]:
    """Every rankings file in a directory, skipping what does not parse."""
    sets: list[RankingSet] = []
    for path in sorted(rankings_dir.glob("*.json")):
        try:
            sets.append(load_rankings(path))
        except ValueError as error:
            log.warning(f"Skipping {path.name}: {error}")
    return sets
