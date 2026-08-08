#!/usr/bin/env python3
"""Ask a question and get a cited answer — stage 5, the first end-to-end path.

Where `app/search.py` stops at the ranked passages, this feeds them to the LLM
and prints the answer with its sources. The retrievers are built by search.py's
`build_retrievers`, so the passages behind an answer here are exactly the ones
that script would have shown for the same query and config.

Usage:
    python app/serve.py                                   # interactive
    python app/serve.py "how are cells tracked?"
    python app/serve.py "..." -r bm25 -k 8
    python app/serve.py "..." -m gpt-4.1 --show-context

Every `[N]` in the answer is listed under Sources with the passage it points
at; a marker with no source line is a citation the model invented.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python app/serve.py` runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.embedding import DEFAULT_EMBEDDER, EMBEDDERS  # noqa: E402
from app.rag.evaluation.judge import DIMENSIONS, JudgeError, LLMJudge  # noqa: E402
from app.rag.generation import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    AnswerGenerator,
    OpenAILLM,
    build_context,
    load_prompt,
    unresolved_markers,
)
from app.rag.models import QAResponse  # noqa: E402
from app.rag.retrieval import FUSIONS, HybridRetriever  # noqa: E402
from app.rag.stores import index_dir  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402
from app.search import (  # noqa: E402
    DEFAULT_CHUNKS,
    DEFAULT_INDICES,
    available_specs,
    build_retrievers,
)

log = logging.getLogger(__name__)


def show(response: QAResponse, results, width: int = 100) -> None:
    """Print the answer, then one line per cited passage."""
    print(f"\n{response.answer}\n")

    if not response.citations:
        if response.chunks_used:
            print("Sources: none cited.")
        return

    # Citations carry chunk ids, not passage numbers; recover the number from
    # the position of the chunk in the results list the answer was built from.
    numbers = {result.chunk.id: n for n, result in enumerate(results, start=1)}
    print("Sources")
    for citation in response.citations:
        page = f" p{citation.page_number}" if citation.page_number else ""
        score = f"  {citation.relevance_score:.4f}" if citation.relevance_score is not None else ""
        snippet = citation.text_snippet[:width]
        print(
            f"  [{numbers[citation.chunk_id]}] {citation.source}{page}{score}\n      {snippet}..."
        )

    stray = unresolved_markers(response.answer, results)
    if stray:
        print(f"\n  ! cites non-existent passage(s): {', '.join(f'[{n}]' for n in stray)}")


def show_score(judge: LLMJudge, response: QAResponse) -> None:
    """Judge the answer just printed and show the four scores."""
    try:
        score = judge.score(response)
    except JudgeError as error:
        log.warning(f"Judge returned no usable score: {error}")
        return

    response.confidence = score.confidence
    scores = "  ".join(f"{name.replace('_', ' ')} {getattr(score, name)}/5" for name in DIMENSIONS)
    print(f"\nJudge  {scores}  avg {score.average:.2f}")
    if score.rationale:
        print(f"       {score.rationale}")


def _log_usage(llm: OpenAILLM, judge: LLMJudge | None) -> None:
    """Answers and judging are billed separately, so report them separately."""
    log.info(llm.usage_summary())
    if judge:
        log.info(f"judge: {judge.llm.usage_summary()}")


def _ask(prompt: str) -> str:
    try:
        return input(f"{prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Cancelled.") from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a question from an index, with citations.",
        epilog="With no QUERY, prompts for one and keeps asking until you enter a blank line.",
    )
    parser.add_argument("query", nargs="?", help="the question to answer")
    parser.add_argument("-s", "--spec", help="chunker spec; default: the only one indexed")
    parser.add_argument("-e", "--embedder", default=DEFAULT_EMBEDDER, choices=sorted(EMBEDDERS))
    parser.add_argument("-r", "--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("-k", "--top-k", type=int, default=5, help="passages given to the model")
    parser.add_argument("--alpha", type=float, default=0.5, help="dense share, hybrid only")
    parser.add_argument("--fusion", default="weighted", choices=list(FUSIONS))
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("-p", "--prompt-version", help="prompts/answer/vN; default: newest")
    parser.add_argument(
        "-j", "--judge", action="store_true", help="score the answer 1-5 on four dimensions"
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_MODEL,
        help="model that grades, with --judge; a different one avoids self-grading",
    )
    parser.add_argument("--show-context", action="store_true", help="print the numbered passages")
    parser.add_argument("-i", "--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("-x", "--indices", type=Path, default=DEFAULT_INDICES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("serve")
    log.info(f"Logging to {log_file}")

    specs = available_specs(args.chunks)
    if not specs:
        log.error(f"{args.chunks} holds no chunks — run `python app/chunk.py` first.")
        return 1

    spec = args.spec
    if spec is None:
        indexed = [s for s in specs if index_dir(args.indices, s, args.embedder).is_dir()]
        if not indexed:
            log.error(f"No index built for embedder {args.embedder!r} — run `python app/index.py`.")
            return 1
        spec = indexed[0]
        if len(indexed) > 1:
            log.info(f"Several indices available; using {spec}. Pass --spec to choose.")

    retrievers = build_retrievers(spec, args.embedder, args.chunks, args.indices)
    retrievers["hybrid"] = HybridRetriever(
        retrievers["dense"], retrievers["bm25"], alpha=args.alpha, fusion=args.fusion
    )

    prompt = load_prompt(args.prompt_version)
    llm = OpenAILLM(model=args.model, temperature=args.temperature)
    generator = AnswerGenerator(retrievers[args.retriever], llm, top_k=args.top_k, prompt=prompt)
    log.info(
        f"{spec} | {args.embedder} | {args.retriever} | {args.model} | prompt {prompt.version}"
    )

    # The judge gets its own client so its tokens are costed apart from the
    # answers', and so it can run on a different model.
    judge = None
    if args.judge:
        judge = LLMJudge(OpenAILLM(model=args.judge_model))
        log.info(f"Judging with {args.judge_model} | prompt {judge.prompt.version}")

    queries = [args.query] if args.query else None
    while True:
        query = queries.pop(0) if queries else _ask("\nQuestion (blank to quit)")
        if not query:
            _log_usage(llm, judge)
            return 0

        results = generator.retrieve(query)
        if args.show_context:
            print(f"\n{build_context(results, passage_chars=generator.passage_chars)}")
        response = generator.answer_from(query, results)
        show(response, results)
        if judge:
            show_score(judge, response)

        if args.query and not queries:
            _log_usage(llm, judge)
            return 0


if __name__ == "__main__":
    sys.exit(main())
