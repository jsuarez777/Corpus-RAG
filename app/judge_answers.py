#!/usr/bin/env python3
"""Grade generated answers with the LLM judge.

Stage 6, second half. `app/generate_answers.py` writes the answers; this scores
them on relevance, accuracy, completeness and citation quality, and writes the
per-config means that `app/visualize.py generation` plots as the Generation
Quality Radar.

Usage:
    python app/judge_answers.py                       # every answers file
    python app/judge_answers.py experiments/answers/<config_id>.jsonl
    python app/judge_answers.py ... --limit 20 --resume

Kept separate from generation for one reason: the judge prompt is versioned and
will change, and re-grading 488 answers must not mean re-generating them. The
scores land beside the report as `<config_id>.scores.jsonl`, appended one at a
time so a re-grade after a stop resumes instead of paying twice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python app/judge_answers.py` runs as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.evaluation.judge import (  # noqa: E402
    DIMENSIONS,
    JudgeError,
    JudgeReport,
    JudgeScore,
    LLMJudge,
)
from app.rag.generation import DEFAULT_MODEL, OpenAILLM  # noqa: E402
from app.rag.models import QAResponse  # noqa: E402
from app.rag.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANSWERS = PROJECT_ROOT / "experiments/answers"
DEFAULT_JUDGED = PROJECT_ROOT / "experiments/judged"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_answers(path: Path) -> tuple[dict, list[dict]]:
    """Split an answers file into its run header and its answer records."""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{_display(Path(path))} is empty")
    head = json.loads(lines[0])
    if "config_id" not in head:
        raise ValueError(f"{_display(Path(path))} has no run header on its first line")
    return head, [json.loads(line) for line in lines[1:]]


def scored_ids(path: Path) -> dict[str, dict]:
    """Scores already on disk, keyed by query id, for ``--resume``."""
    if not path.is_file():
        return {}
    found: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "query_id" in row:
            found[row["query_id"]] = row
    return found


def report_from(rows: list[dict], *, model: str, prompt_version: str) -> JudgeReport:
    """Aggregate scored rows into the report the radar chart reads.

    Built here rather than by ``LLMJudge.score_all`` because scoring runs one
    answer at a time and appends as it goes; the means have to come from
    whatever is on disk, including a run resumed across two sessions.
    """
    means: dict[str, float] = {}
    if rows:
        for name in DIMENSIONS:
            means[name] = sum(row[name] for row in rows) / len(rows)
        means["average"] = sum(means[name] for name in DIMENSIONS) / len(DIMENSIONS)
    return JudgeReport(
        num_scored=len(rows),
        means=means,
        per_query=rows,
        model=model,
        prompt_version=prompt_version,
    )


def write_report(judged_dir: Path, head: dict, report: JudgeReport) -> Path:
    """The per-config summary `app/visualize.py`'s ``load_judged`` expects.

    ``label`` matches ``RunResult.label`` — ``chunker | embedder | retriever`` —
    so the radar's polygons carry the same names as the heatmap's rows.
    """
    judged_dir.mkdir(parents=True, exist_ok=True)
    target = judged_dir / f"{head['config_id']}.json"
    target.write_text(
        json.dumps(
            {
                "config_id": head["config_id"],
                "label": head.get("label", head["config_id"]),
                "config": head.get("config", {}),
                "answer_model": head.get("model", ""),
                "answer_prompt_version": head.get("prompt_version", ""),
                "judge_model": report.model,
                "judge_prompt_version": report.prompt_version,
                "num_scored": report.num_scored,
                "means": {name: round(value, 4) for name, value in report.means.items()},
                "per_query": report.per_query,
            },
            indent=2,
        )
    )
    return target


def judge_file(
    answers_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    judged_dir: Path = DEFAULT_JUDGED,
    limit: int | None = None,
    resume: bool = False,
    use_reference: bool = True,
) -> Path:
    """Score one answers file and write its report. Returns the report path."""
    head, records = read_answers(answers_path)
    if limit:
        records = records[:limit]

    judged_dir.mkdir(parents=True, exist_ok=True)
    scores_path = judged_dir / f"{head['config_id']}.scores.jsonl"
    existing = scored_ids(scores_path) if resume else {}
    if existing:
        log.info(f"Resuming: {len(existing)} already scored")

    judge = LLMJudge(OpenAILLM(model=model))
    log.info(f"Judging {head['config_id']} with {model} | prompt {judge.prompt.version}")

    rows = [existing[record["query_id"]] for record in records if record["query_id"] in existing]
    pending = [record for record in records if record["query_id"] not in existing]

    with scores_path.open("a" if existing else "w", encoding="utf-8") as handle:
        for position, record in enumerate(pending, start=1):
            row = _score_one(judge, record, use_reference=use_reference)
            if row is None:
                continue
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            if position % 25 == 0 or position == len(pending):
                log.info(f"  {position}/{len(pending)} scored | {judge.llm.usage_summary()}")

    report = report_from(rows, model=model, prompt_version=judge.prompt.version)
    log.info(report.summary())
    log.info(judge.llm.usage_summary())
    return write_report(judged_dir, head, report)


def _score_one(judge: LLMJudge, record: dict, *, use_reference: bool) -> dict | None:
    """One answer scored, or None when the judge gave nothing usable.

    A query the judge fails on is dropped from the means rather than sinking the
    run — the same trade ``LLMJudge.score_all`` makes, for the same reason: a
    partial report is still comparable and a crashed one is not.
    """
    response = QAResponse.model_validate(record["response"])
    reference = record.get("reference") if use_reference else None
    try:
        score: JudgeScore = judge.score(response, reference=reference)
    except (JudgeError, RuntimeError) as error:
        log.warning(f"Judge failed on {record['query_id']}: {error}")
        return None
    return {
        "query_id": record["query_id"],
        "query": record["query"],
        **score.as_dict(),
        "rationale": score.rationale,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated answers 1-5 on four dimensions.",
        epilog="One model call per answer. With no FILE, grades every answers file.",
    )
    parser.add_argument("files", nargs="*", type=Path, help="answers .jsonl files")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="the model that grades")
    parser.add_argument("-n", "--limit", type=int, help="score only the first N answers")
    parser.add_argument("--resume", action="store_true", help="skip answers already scored")
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="grade against the passages alone, ignoring the benchmark's answer",
    )
    parser.add_argument("-a", "--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("-o", "--judged", type=Path, default=DEFAULT_JUDGED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_file = setup_logging("judge_answers")
    log.info(f"Logging to {log_file}")

    files = args.files or sorted(args.answers.glob("*.jsonl"))
    if not files:
        log.error(f"No answers files in {_display(args.answers)} — run generate_answers.py first.")
        return 1

    for path in files:
        if not path.is_file():
            log.error(f"No such file: {_display(path)}")
            return 1
        target = judge_file(
            path,
            model=args.model,
            judged_dir=args.judged,
            limit=args.limit,
            resume=args.resume,
            use_reference=not args.no_reference,
        )
        log.info(f"Wrote {_display(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
