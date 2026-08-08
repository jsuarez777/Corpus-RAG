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
