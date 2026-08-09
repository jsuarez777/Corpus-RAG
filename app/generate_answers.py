#!/usr/bin/env python3
"""Generate answers for every benchmark query under one pipeline config.

Stage 6. `app/evaluate.py` scores what retrieval *found*; this scores nothing at
all — it produces the answers that `app/judge_answers.py` then grades, which is
what the Generation Quality Radar and the "average quality > 4.0" target are
measured from. Until this runs, `AnswerGenerator` is reachable only one question
at a time through `app/serve.py`.

Usage:
    python app/generate_answers.py --config-id fixed_size_512_128__mpnet__hybrid_0.5
    python app/generate_answers.py --config-id ... --limit 20        # pilot
    python app/generate_answers.py --config-id ... --resume          # after a stop
    python app/generate_answers.py --config-id ... --judge           # and grade it

This costs money: one API call per query, 488 of them at the current corpus.
`--limit` runs a prefix of the same query order `app/evaluate.py --limit` uses,
so a pilot here is answering the queries that pilot scored. `--resume` skips the
queries already in the output file, so a run that dies at 300 does not pay for
those 300 twice.

Answers land in experiments/answers/<config_id>.jsonl — one line of run header,
then one line per answer. The name carries no timestamp on purpose: `--resume`
has to find the file again, and a re-run of the same config is a continuation of
that config's answers rather than a new artifact to compare against the old one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # `python app/generate_answers.py` runs as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import chunk_file, load_chunks  # noqa: E402
from app.rag.config import PipelineConfig, build_retriever, load_grid  # noqa: E402
from app.rag.embedding import get_embedder  # noqa: E402
from app.rag.evaluation.qrels import QueryRelevance, build_relevance, load_benchmark  # noqa: E402
from app.rag.evaluation.rankings import as_results, load_rankings, rankings_file  # noqa: E402
from app.rag.generation import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    AnswerGenerator,
    OpenAILLM,
    load_prompt,
)
from app.rag.models import QAResponse, RetrievalResult  # noqa: E402
from app.rag.reranking import reranker_from_spec  # noqa: E402
from app.rag.retrieval import BM25Retriever, ReplayRetriever  # noqa: E402
from app.rag.stores import index_dir, open_store  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402
from app.rag.utils.parallel import DEFAULT_WORKERS, run_in_parallel  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHUNKS = DATA_DIR / "chunks"
DEFAULT_INDICES = DATA_DIR / "indices"
DEFAULT_BENCHMARK = DATA_DIR / "open_ragbench/pdf/arxiv"
DEFAULT_EXPERIMENT = PROJECT_ROOT / "config/experiments/grid_12.yaml"
DEFAULT_ANSWERS = PROJECT_ROOT / "experiments/answers"
DEFAULT_RANKINGS = PROJECT_ROOT / "experiments/rankings"

#: Passages given to the model. The grid retrieves 10 because its deepest metric
#: is @10; an answer prompt is a different question, and the spec's UI default is
#: 5. Overridable, but the config's own top_k is not the right default here.
DEFAULT_TOP_K = 5


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def label_for(config: PipelineConfig) -> str:
    """The axis label the retrieval charts use, so the radar lines up with them."""
    return config.label


def select_config(configs: list[PipelineConfig], config_id: str | None) -> PipelineConfig:
    """Pick one cell out of an experiment file.

    A grid with more than one cell and no ``--config-id`` is an error rather
    than a default, because the default would silently spend a few dollars on
    whichever cell happened to be first.
    """
    if config_id:
        for config in configs:
            if config.id == config_id:
                return config
        available = "\n  ".join(config.id for config in configs)
        raise SystemExit(f"No config {config_id!r} in this experiment. Available:\n  {available}")
    if len(configs) == 1:
        return configs[0]
    available = "\n  ".join(config.id for config in configs)
    raise SystemExit(
        f"This experiment declares {len(configs)} configs — pass --config-id to choose one:"
        f"\n  {available}"
    )


def build_live_retriever(config: PipelineConfig, *, chunks_dir: Path, indices_dir: Path):
    """Open the index this config names and build its retriever.

    The expensive path, and the only one that loads faiss and torch. Kept for
    the cases a saved ranking cannot cover — a config the grid never ran, or a
    depth deeper than it retrieved.

    Deliberately the same construction `app/evaluate.py` uses — `build_retriever`
    off the same `PipelineConfig` — so the passages behind an answer here are the
    ones the retrieval metrics were computed over.
    """
    chunker_spec, embedder_name = config.chunker.spec, config.embedder.spec
    target = index_dir(indices_dir, chunker_spec, embedder_name)
    if not target.is_dir():
        raise SystemExit(
            f"No index at {_display(target)} — "
            f"run `python app/index.py {chunker_spec} -e {embedder_name}`."
        )

    chunks = load_chunks(chunk_file(chunks_dir, chunker_spec, embedder_name))
    embedder = get_embedder(embedder_name)
    embedder.embed_query("warm up")  # weights load here, not inside the first query

    return build_retriever(
        config,
        store=open_store(target),
        chunks=chunks,
        embedder=embedder,
        sparse=BM25Retriever(chunks),
    )


def build_replay_retriever(
    config: PipelineConfig,
    items: list[QueryRelevance],
    *,
    rankings_path: Path,
    chunks_dir: Path,
    depth: int,
) -> ReplayRetriever:
    """Serve the ranking `app/retrieve.py` already saved for this config.

    The search for these queries has been run — scoring it is what produced the
    retrieval metrics. Running it again to write answers costs the same minutes
    and, more to the point, drags faiss and torch into a process that otherwise
    does nothing but wait on the API. Reading the ids back leaves a stage that
    parallelizes.

    ``depth`` is what the generator will ask for, which with a reranker is
    several times ``top_k``. Refused rather than quietly truncated if the saved
    ranking is shallower: answering from 5 passages where 20 were asked for is
    a different experiment, not a slightly worse one.
    """
    saved = load_rankings(rankings_path)
    if saved.config_id != config.id:
        raise SystemExit(
            f"{_display(rankings_path)} holds rankings for {saved.config_id!r}, not {config.id!r}."
        )
    if saved.top_k and saved.top_k < depth:
        raise SystemExit(
            f"{_display(rankings_path)} was retrieved at top_k={saved.top_k}, "
            f"but {depth} passages are wanted. Re-run `python app/retrieve.py` deeper, "
            f"or pass --live to retrieve now."
        )

    chunks = load_chunks(chunk_file(chunks_dir, config.chunker.spec, config.embedder.spec))
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    by_id = saved.by_id()

    by_query: dict[str, list[RetrievalResult]] = {}
    for item in items:
        ranking = by_id.get(item.query_id)
        if ranking is None:
            log.warning(f"No saved ranking for {item.query_id} — it will be skipped")
            continue
        by_query[item.query.strip()] = as_results(ranking, chunks_by_id)

    log.info(f"Replaying {len(by_query)} saved rankings from {_display(rankings_path)}")
    return ReplayRetriever(by_query, retriever_type=_retriever_kind(config, saved))


def _retriever_kind(config: PipelineConfig, saved) -> str:
    """What the saved ranking should call itself.

    The file records it per query; the config is the fallback for a file
    written before that field existed.
    """
    for ranking in saved.rankings:
        if ranking.retriever_type:
            return ranking.retriever_type
    return config.retriever.name


def why_serial(args: argparse.Namespace, saved: Path) -> str:
    """Why this run cannot use a worker pool, or "" if it can.

    All three reasons are the same one underneath: the work would call into
    faiss or torch from a worker thread. ``app/__init__.py`` keeps those two
    libraries from killing each other by pinning ``OMP_NUM_THREADS=1``, and
    that only holds while nothing runs concurrently — so the answer is to keep
    them out of the parallel stage, not to thread more carefully.

    Returned as a sentence rather than a bool because the caller has to be able
    to say *which* of the three it hit; "cannot be parallel" on its own is not
    something anyone can act on.
    """
    if args.rerank:
        return "reranking runs a cross-encoder (torch) inside each call"
    if args.live:
        return "--live retrieves for real, which loads faiss and torch"
    if not saved.is_file():
        return (
            f"no saved rankings at {_display(saved)}, so retrieval has to run live — "
            f"`python app/retrieve.py` writes them once and every later run replays them"
        )
    return ""


def build_generator(
    retriever,
    *,
    llm: OpenAILLM,
    top_k: int,
    prompt_version: str | None,
    rerank: str | None,
    rerank_depth: int | None,
) -> AnswerGenerator:
    """Wrap a retriever — live or replayed — in the answering pipeline."""
    return AnswerGenerator(
        retriever,
        llm,
        top_k=top_k,
        prompt=load_prompt(prompt_version),
        reranker=reranker_from_spec(rerank) if rerank else None,
        rerank_depth=rerank_depth,
    )


def load_benchmark_queries(
    config: PipelineConfig, *, benchmark: Path, chunks_dir: Path
) -> list[QueryRelevance]:
    """The scoreable queries for this config, in the order evaluate.py uses.

    Answers are generated for exactly the queries retrieval was scored on. A
    query whose gold section has no chunks is not answerable *against this
    benchmark* either — there is nothing to judge citations against.
    """
    queries, qrels, answers = load_benchmark(benchmark)
    chunks = load_chunks(chunk_file(chunks_dir, config.chunker.spec, config.embedder.spec))
    relevance, report = build_relevance(chunks, queries, qrels, answers)
    log.info(report.summary())
    return relevance


def already_answered(path: Path) -> set[str]:
    """Query ids present in an existing answers file, for ``--resume``."""
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "query_id" in record:
            done.add(record["query_id"])
    return done


def header(config: PipelineConfig, generator: AnswerGenerator, num_queries: int) -> dict:
    """First line of the answers file: everything needed to reproduce it.

    An answer without the prompt version and model that produced it cannot be
    compared to another one, the same reason the result files carry the config.
    """
    return {
        "config": config.model_dump(mode="json"),
        "config_id": config.id,
        "label": label_for(config),
        "model": generator.llm.model,
        "temperature": generator.llm.temperature,
        "prompt_version": generator.prompt.version,
        "top_k": generator.top_k,
        "reranker": repr(generator.reranker) if generator.reranker else "",
        "num_queries": num_queries,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def as_record(item: QueryRelevance, response: QAResponse, latency_ms: float) -> dict:
    """One answer, flattened for the judge and for a reader.

    ``chunks_used`` is dumped whole rather than by id: the judge scores an
    answer against the passages it was written from, and resolving ids back to
    text would mean re-loading the chunk set to grade a file.
    """
    return {
        "query_id": item.query_id,
        "query": item.query,
        "doc_id": item.doc_id,
        "section_id": item.section_id,
        "reference": item.answer,
        "latency_ms": round(latency_ms, 1),
        "response": response.model_dump(mode="json"),
    }


def answer_one(generator: AnswerGenerator, item: QueryRelevance) -> dict | None:
    """One answered query, or None when it failed.

    ``retrieve`` then ``answer_from`` rather than ``answer()``: the passages are
    wanted in the record, and one retrieval must serve both. A single bad query
    is logged and dropped rather than ending a run that has already been paid
    for.
    """
    started = time.perf_counter()
    try:
        results = generator.retrieve(item.query)
        response = generator.answer_from(item.query, results)
    except Exception as error:  # noqa: BLE001 — one bad query must not end the run
        log.warning(f"Failed on {item.query_id} {item.query!r}: {error}")
        return None
    return as_record(item, response, (time.perf_counter() - started) * 1000)


def generate(
    generator: AnswerGenerator,
    items: list[QueryRelevance],
    out_path: Path,
    *,
    config: PipelineConfig,
    resume: bool = False,
    workers: int = 1,
) -> int:
    """Answer every query, appending each one as it lands.

    Written a line at a time and flushed rather than collected and dumped at the
    end: 488 model calls is long enough that a crash at query 400 must not throw
    away 400 paid-for answers.

    ``workers`` above 1 is safe only when the passages come from saved rankings.
    A live retriever calls into faiss and torch, and ``app/__init__.py`` keeps
    those two from colliding by pinning ``OMP_NUM_THREADS=1`` — which holds
    because nothing runs concurrently. ``main`` refuses the combination rather
    than leaving it to be discovered as a segfault mid-run.
    """
    done = already_answered(out_path) if resume else set()
    pending = [item for item in items if item.query_id not in done]
    if done:
        log.info(f"Resuming: {len(done)} already answered, {len(pending)} to go")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if done else "w"

    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write(json.dumps(header(config, generator, len(items))) + "\n")

        def write(record: dict) -> None:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

        def progress(position: int, total: int) -> None:
            if position % 25 == 0 or position == total:
                log.info(f"  {position}/{total} answered | {generator.llm.usage_summary()}")

        written = run_in_parallel(
            lambda item: answer_one(generator, item),
            pending,
            workers=workers,
            on_result=write,
            on_progress=progress,
        )

    return len(written)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer every benchmark query under one pipeline config.",
        epilog="Costs one model call per query. Start with --limit.",
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--config-id", help="which cell of the experiment to answer with")
    parser.add_argument("-n", "--limit", type=int, help="answer only the first N queries")
    parser.add_argument(
        "--resume", action="store_true", help="skip queries already in the output file"
    )
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("-p", "--prompt-version", help="prompts/answer/vN; default: newest")
    parser.add_argument("--rerank", metavar="SPEC", help="cross_encoder[:model] or cohere[:model]")
    parser.add_argument("--rerank-depth", type=int, metavar="N")
    parser.add_argument(
        "-j", "--judge", action="store_true", help="grade the answers once they are written"
    )
    parser.add_argument("--judge-model", default=DEFAULT_MODEL, help="model that grades, with -j")
    parser.add_argument("-b", "--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("-i", "--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("-x", "--indices", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("-o", "--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument(
        "--rankings",
        type=Path,
        default=DEFAULT_RANKINGS,
        metavar="DIR",
        help="saved rankings to answer from; falls back to live retrieval if absent",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="retrieve now instead of replaying saved rankings (loads faiss and torch)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        # No default, so "you asked for 8" is distinguishable from "nobody said".
        # An explicit value is honoured or refused; an absent one is chosen.
        help=(
            f"concurrent answer calls. Default {DEFAULT_WORKERS} when replaying saved "
            f"rankings, 1 when retrieving live or reranking. Passing a value above 1 "
            f"where it is impossible is an error rather than a silent downgrade."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("generate_answers")
    log.info(f"Logging to {log_file}")

    if not args.config.is_file():
        log.error(f"No experiment file at {_display(args.config)}.")
        return 1
    if not args.benchmark.is_dir():
        log.error(f"No benchmark at {_display(args.benchmark)}.")
        return 1

    config = select_config(load_grid(args.config), args.config_id)
    items = load_benchmark_queries(config, benchmark=args.benchmark, chunks_dir=args.chunks)
    if not items:
        log.error("No scoreable queries — has `python app/align.py` been run?")
        return 1
    if args.limit:
        items = items[: args.limit]

    llm = OpenAILLM(model=args.model, temperature=args.temperature)

    # How deep the generator will ask, which a reranker multiplies.
    depth = args.top_k
    if args.rerank:
        depth = max(depth, args.rerank_depth or args.top_k * 4)

    saved = rankings_file(args.rankings, config.id)
    serial_because = why_serial(args, saved)

    # An explicit --workers is never quietly downgraded. Asking for eight and
    # silently getting one turns a four-minute run into twenty-five, and the
    # only place that shows up is the wall clock after it has finished.
    if serial_because and (args.workers or 0) > 1:
        log.error(f"--workers {args.workers} is not possible here: {serial_because}")
        return 1
    workers = 1 if serial_because else (args.workers or DEFAULT_WORKERS)
    if serial_because and args.workers is None:
        log.info(f"Running one query at a time: {serial_because}")

    if args.live or not saved.is_file():
        retriever = build_live_retriever(config, chunks_dir=args.chunks, indices_dir=args.indices)
    else:
        retriever = build_replay_retriever(
            config, items, rankings_path=saved, chunks_dir=args.chunks, depth=depth
        )

    generator = build_generator(
        retriever,
        llm=llm,
        top_k=args.top_k,
        prompt_version=args.prompt_version,
        rerank=args.rerank,
        rerank_depth=args.rerank_depth,
    )
    log.info(f"{config.summary()} | {args.model} | prompt {generator.prompt.version}")
    log.info(f"Answering {len(items)} queries with top_k={args.top_k} | {workers} worker(s)")

    out_path = args.answers / f"{config.id}.jsonl"
    written = generate(
        generator, items, out_path, config=config, resume=args.resume, workers=workers
    )
    log.info(f"Wrote {written} answer(s) to {_display(out_path)}")
    log.info(llm.usage_summary())

    if args.judge:
        from app.judge_answers import judge_file

        # --resume carries through: a generation resumed after a stop would
        # otherwise re-buy a score for every answer that already has one.
        # Judging never retrieves, so it threads regardless of how the passages
        # for these answers were obtained.
        judged = judge_file(
            out_path,
            model=args.judge_model,
            resume=args.resume,
            workers=args.workers or DEFAULT_WORKERS,
        )
        log.info(f"Judged into {_display(judged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
