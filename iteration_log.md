# Iteration Log

One entry per change made in response to a measurement, including changes that
did not help. Design rationale lives in the docstring of the code that made the
decision; this file records only what changed, what motivated it, and what moved.

---

```
Iteration: 1
Change: Added four independence rules to prompts/judge/v1/judge_system.prompt
Reason: Judge marked down dimensions that were not at fault — 7 collateral low scores over 7 planted-defect answers, only 3 scored cleanly
Metric Before: defect detection 6/6, clean cases 3/7, collateral 7
Metric After:  defect detection 6/6, clean cases 4/7, collateral 7
Delta: clean cases +1; detection already at ceiling; collateral redistributed, not reduced. Kept — the 4 real coupling errors are mild (a 3 where a 5 belongs), so defect-vs-no-defect comparisons hold but single absolute dimensions do not. Pinned by tests/test_judge_calibration.py. Still self-grading unless --judge-model is passed.
```

```
Iteration: 2
Change: maxLength 1000 on SCORE_SCHEMA's rationale field; max_output_tokens 3500 on LLMJudge
Reason: 2 of 976 answers lost to degenerate repetition — the judge looped one control character until the reply hit the API ceiling and truncated the JSON mid-escape
Metric Before: 974/976 scored; a runaway ran to 32,768 output tokens and $0.0539, then failed
Metric After:  10/10 retries of the two lost answers scored; a runaway is bounded at ~2,900 tokens and ~$0.005, and parses
Delta: both lost scores recoverable, worst case 94% cheaper. The cap does not stop the runaway, it closes the string so the integers still parse. Sized at 1200 first and that was worse than no cap at all (5/8 vs 8/8): escaped control characters cost 6 wire characters each, so a runaway needs ~3,000 tokens to reach 1000 chars. Query-specific — the 2 failures ran away 10/10, twelve controls 0/60, both culprits dense LaTeX.
```

```
Iteration: 3
Change: Reorganize the pipeline so generating answers and judging run independently of retrieval, and parallelize the requests — retrieval split out of scoring and saved to experiments/rankings/, downstream stages read ranked ids from disk, --workers on both batch stages
Reason: Running generation and judge queries serially results in ~25 min per 488 answer run. In order to parallelize the requests, we need to remove dependency on retrieval from the generation and judge modules.
Metric Before: 12 cells scored in 116s, each re-running its search; 488 answers ~25 min; both batch stages serial
Metric After:  12 cells scored in 1.9s from saved rankings with no index opened; 488 answers in 125s on 30 workers
Delta: rescoring ~60x, generation 11.8-12.4x, cost unchanged at $0.66 per 488 answers. Metrics verified identical — 12 cells x 17 metrics, 0 mismatches at 4 decimal places.
```

```
Iteration: 4
Change: Use OpenAI client max_retries instead of our own rolled multi-thread with back-off
Reason: According to OpenAI docs, the max_retries will automatically wait the prescribed amount of time if a 429 is returned, so we dont have to try and handle that ourselves.
Metric Before: 3 attempts over a blind 6s, no jitter
Metric After:  5 attempts at the server-directed wait, jittered, refusing waits over 120s
Delta: none measured — no rate limit was hit at 30 workers to measure against
```

