# Iteration Log

One entry per change made in response to a measurement. Each records what was
changed, the observation that motivated it, and the metric before and after —
including the changes that did not help, since a lever that turned out to be
flat is as much a result as one that worked.

---

## Iteration 1 — Judge rubric: scoring dimensions independently

**Change:** Added four explicit independence rules to `prompts/judge/v1/judge_system.prompt`,
naming the confusions to avoid ("a brief answer is not irrelevant"; "a claim
cited to the wrong passage is a citation fault, not an accuracy fault").

**Reason:** The judge had only ever been run on two genuinely good answers,
both of which it scored 5/5/5/5. That is consistent with a working judge and
equally consistent with one that always returns 5, and nothing measured so far
could tell those apart. Seven answers were written with defects planted on
named dimensions — off-topic but accurate, hallucinated numbers, correct but
uncited, one true sentence where the passages support four, right content with
markers pointing at the wrong passages (including a `[7]` that does not exist),
and one bad on everything — and judged against synthetic passages whose content
is fixed, so what the context supports is known rather than inferred.

The first run answered the leniency question: scores spread 5.00 to 1.00 in the
right order, and the planted defect was detected in every degraded case. It
also exposed a different problem. The judge dragged down dimensions that were
not at fault — scoring the off-topic answer 3 on citation quality when its
markers were correct, and scoring the brief-but-accurate answer 2 on relevance
when it was squarely on topic.

**Metric before:** defect detection 6/6 · collateral low scores on untargeted
dimensions 7 · cases scored cleanly on every dimension 3/7

**Metric after:** defect detection 6/6 · collateral low scores 7 · cases scored
cleanly on every dimension 4/7

**Delta:** clean cases +1. Detection unchanged, already at ceiling. Collateral
count unchanged at 7, but redistributed rather than static: the off-topic case
was fully corrected (citation quality 3 → 5, exactly right, since every marker
it used was accurate), while a new mild coupling appeared on the brief answer
(citation quality 5 → 3). Relevance on that same brief answer moved 2 → 3, the
right direction and not far enough.

**Kept, with the weakness recorded rather than resolved.** Of the seven
collateral scores, three are defensible under the rubric as written — an answer
whose markers back invented claims genuinely does have poor citation quality.
Four are real coupling errors, and all four are mild: a dimension reading 3
where it should read 5, never a 1. A configuration comparison that turns on the
gap between "no defect" and "severe defect" is unaffected; one that turns on a
single dimension's absolute value is not yet trustworthy.

Locked in as `tests/test_judge_calibration.py`, which asserts direction only —
planted defect at or below 3, good baseline at or above 4 — and deliberately
does not assert independence, so a later rubric edit is free to change the
coupling without breaking a test that was never measuring it. Billed and
opt-in behind `JUDGE_CALIBRATION=1`, about $0.004 per run.

**Not addressed:** the judge model was the same one that writes answers
(`gpt-4.1-mini`), so a grid run self-grades unless `--judge-model` is passed.
Measuring whether that inflates scores needs a second model and is its own
iteration.

---

## Iteration 2 — Bounding judge runaways with the response schema

**Change:** Added `maxLength: 1000` to the `rationale` property of
`SCORE_SCHEMA`, and a `max_output_tokens` argument on `LLMJudge` defaulting to
3500.

**Reason:** The first full judging pass — 976 answers across two configurations
— lost two scores to the same failure. The judge fell into a degenerate
repetition, locking onto a single control character and emitting it until the
reply hit the API's own output ceiling and truncated the JSON mid-escape.
`JudgeScore.model_validate` correctly refused the fragment, so the query was
dropped from the means rather than scored wrongly, but the answer was paid for
twice over: once to generate, once to grade, and no score to show for it. Both
failures were on unicode-heavy mathematics papers.

Strict mode already guaranteed the *structure* of a score. It bounded nothing
about the *length* of a string, and every character in the loop was schema-legal
right up to the truncation. The rationale's `description` already asked for "one
or two sentences" and was ignored; a description shifts probabilities, while
`maxLength` is compiled into the decoding grammar server-side and removes the
tokens outright. Verified directly before relying on it: with `maxLength: 40` the
model returned exactly 40 characters, cut mid-word.

**Metric before:** 974/976 answers scored · a runaway ran to 32,768 output
tokens and $0.0539 before failing

**Metric after:** 10/10 retries of the two lost answers scored · a runaway is
bounded at ~2,900 output tokens and ~$0.005, and now yields a valid score

**Delta:** the two lost scores are recoverable, and the worst case costs 94%
less. The mechanism is worth stating precisely, because it is not what it looks
like: `maxLength` does not prevent the runaway. The model still locks up and
still emits its 1000 characters of garbage — the retries that recovered show a
rationale of exactly 1000 characters, the cap being hit. What the cap buys is
that the grammar *closes the string* at the limit, after which the model emits
the four integers and the document parses. A lost score becomes a valid one
whose rationale is mostly junk.

Mostly, not entirely, and the shape of it matters. Reading one back: 83
printable characters of a real, on-topic judgement — "The answer correctly
identifies the scaling property of the surreal delta function" — followed by 917
repetitions of `U+007F` to the cap. The model reaches an assessment and then
locks up mid-sentence while writing it down. The reasoning happens; the
transcription breaks. That is the most likely reason the four integers stay
sensible, and it is a different failure from one where the judge never formed a
view at all. The repeated character also varies — the two batch failures looped
on `U+0003`, this retry on `U+007F` — so this is generic degeneration rather
than anything specific to one codepoint.

Which raises the obvious question about those four numbers: does a model that
has just emitted a thousand control characters still score, or does it fall back
on 5s? Measured, six retries per query, separating the runaways from the clean
runs by rationale length:

    mpnet | hybrid   clean x5   5/5/3/5   avg 4.50   (identical every time)
                     runaway    5/4/3/3   avg 3.75
    minilm | dense   clean x5   4.50-5.00
                     runaway    5/5/4/4   avg 4.50

They do not default to 5, and the discriminating signal survives: completeness
is the dimension that separates configurations in this corpus, and the first
query's runaway returned 3 — exactly what all five of its clean runs returned.

Whether runaways skew low is a weaker claim than it first looked. Across four
observed runaways, each against the clean trials of the same query:

    delta function   clean 4.50 x5                    runaway 3.75   -0.75
    Z2 symmetry      clean 5.00 5.00 4.50 5.00 4.50   runaway 4.50   -0.30
    delta function   clean 4.75 4.50 4.50 4.50 4.25   runaway 4.50    0.00
    Z2 symmetry      clean 5.00 4.75 4.75 5.00 5.00   runaway 4.75   -0.15

Never higher, averaging -0.30, but three of the four sit within 0.3 and one is
exactly the clean mean. Two observations had suggested runaways land at or below
the clean *minimum*; the third falsifies that, coming in above its query's
lowest clean run. The honest reading is a slight downward trend on a small
sample, not a bias worth correcting for.

Kept, on the arithmetic: two scores in 976, biased low by at most 0.75, move a
mean by roughly 0.0008. If the rate ever climbed the right response would be to
discard runaway scores rather than average them in, and a rationale of exactly
`MAX_RATIONALE_CHARS` is the signal to detect them by — no runaway recovered so
far has come in under the cap.

**The token cap was sized wrong first, and the error is the instructive part.**
The two caps are in different units — characters of the decoded string against
tokens of the escaped JSON on the wire — and the conversion is not constant. A
1000-character rationale is 165 tokens of plain English, 555 of escaped
mathematics, and 3,006 of repeated control characters, which cost six wire
characters each. Sized against the legitimate rows, the cap went in at 1200.
Measured against the two real failures, that scored 5 of 8 retries where *no cap
at all* scored 8 of 8: the token limit fired mid-runaway, before the grammar
could reach 1000 characters and close the string, reintroducing the exact
truncation `maxLength` was added to prevent. The backstop has to sit above the
thing it is backing up. At 3500 both queries score 5/5.

`tests/test_judge.py::test_the_token_cap_sits_above_the_character_cap` encodes a
full-length runaway with tiktoken and asserts it fits under the cap, so the two
constants cannot be moved into collision again.

**The runaway is query-specific, not random.** Measured against twelve control
queries drawn at random from the same run, five trials each, counting any reply
containing control characters:

    the two batch failures    10/10 trials
    twelve control queries     0/60 trials

Both culprits are dense LaTeX — `$\tilde{\delta}(a x)$` and `$\mathbb{Z}_{2}$`.
It is not the paper: one control query came from `2411.03676v2`, the same paper
as a failure, phrased in plain prose, and stayed clean through all five trials.

Looking at degree rather than presence refines it further. Nearly every reply to
these two queries carries a few stray control characters — typically two to
eight out of ~500, invisible and harmless — and roughly one reply in six
escalates into a full loop that reaches the cap. So the trigger and the runaway
are separate events: notation-heavy content reliably makes the judge emit stray
control codepoints, and occasionally it latches onto one and repeats it. Scores
stayed between 4.25 and 5.00 across all of it, independent of how many leaked
in, which is further evidence the scoring survives the transcription fault.

**Not addressed:** how many of the 488 queries are susceptible. Two were
observed failing in a single pass, and a susceptible query only runs away about
one time in six, so the susceptible set is probably several times larger than
the failures seen. Establishing it would mean re-judging the corpus and counting
control characters rather than failures — cheap to do as a side effect of the
next full run, and pointless as a standalone exercise.

---

## Iteration 3 — Saving the ranking, so nothing retrieves twice

**Change:** Split retrieval out of scoring. `retrieve_all` produces ranked chunk
ids and `score_rankings` consumes them; `evaluate()` is now the composition of
the two. `app/retrieve.py` writes the ids to `experiments/rankings/`, and the
stages downstream read them — `app/evaluate.py --from-rankings` scores them,
`app/generate_answers.py` answers from them through a new `ReplayRetriever`.
Both batch stages gained `--workers`.

**Reason:** the same search was being run three times. The grid retrieved 488
queries per cell to compute metrics; answer generation retrieved the same 488
queries under the same config to build prompts; a reranker experiment would have
made three. Retrieval is also the only stage that needs an index and an
embedder, and that turned out to be what was blocking the batch stages from
running concurrently — the thing they actually needed, since a 488-query pass
was ~25 minutes to answer and ~12 to judge, almost all of it network wait.

The obstacle was not obvious. `app/__init__.py` already fixed the faiss/torch
collision by pinning `OMP_NUM_THREADS=1` and importing both in a fixed order,
and its own docstring says why that is safe: "with one thread there is no race
to lose". A `ThreadPoolExecutor` around a loop that retrieves puts the race
back, and with `KMP_DUPLICATE_LIB_OK` set the documented failure is wrong
results rather than a crash. So the choice was not "thread it or don't" — it was
whether the parallel stage touches those libraries at all. Reading ids off disk
means it does not.

**Metric before:** 12 cells rescored in 117s, every one re-running its search ·
answer generation re-retrieved 488 queries per config · both batch stages
serial

**Metric after:** 12 cells rescored in 1s from saved rankings · answers replay
the ranking the grid already produced · 24 answers in ~11s against 79.1s of
serial API time, and 24 judged in ~8s

**Delta:** rescoring is ~100x faster and needs no index at all, which is the
part that changes what is practical: adding a metric or a k to the grid is now
seconds rather than two minutes of re-running searches whose answers have not
changed. The batch stages are ~7x and ~5x on 8 workers — under the 8x ceiling,
as expected, since the tail of a batch is one straggler.

**Verified equal, not assumed.** The risk in splitting a function that both
retrieves and scores is that the two halves stop agreeing and every number in
`experiments/results/` shifts underneath. All twelve grid cells were rescored
from saved rankings and compared against the committed results: 17 metrics each,
0 mismatches to four decimal places.

**The artifact carries scores and the retriever kind, not just ids.** A
`RetrievalResult` cannot be constructed without both, so ids alone would have
forced the replay path to invent them — and a replayed hybrid result reporting
itself as dense would misattribute the passages behind every answer written from
it. Ids stay a plain list with scores parallel to them, because the qrels and the
charts only ever want the ids.

**Chunk ids are per chunking run, which makes rankings invalidatable.**
`Chunk.id` defaults to `uuid4()`, so re-running `app/chunk.py` renames every
chunk and orphans every saved ranking. Skipping unresolvable ids would answer
from fewer passages than asked for and read as a slightly worse config, so
`as_results` raises instead and names the command that fixes it. Same reasoning
behind refusing a ranking shallower than the depth asked for: a reranker wanting
20 candidates and silently getting 5 is a different experiment, not a degraded
one.

**Not addressed:** `app/evaluate.py`'s flag path still writes the second result
schema the charts cannot read, and it does not save rankings at all — only the
`-c` path does. The reranker is still absent from the grid runner, though the
rankings artifact is most of what wiring it in needs: a reranker reorders
candidates that are now on disk, so scoring one no longer means re-running the
retrieval it reranks.
