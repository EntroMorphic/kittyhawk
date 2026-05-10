# Red-team: Cycle 1 plan

Adversarial review of `cycle1_plan.md`. Looking for blind spots, failure
modes, and ways the plan could produce a bad answer.

## Concerns ordered by severity

### R1 — "Part B" is undefined precisely enough to score against

The plan asks for candidates that test Part B. But Part B as written
(`docs/THESIS.md`) is vague:

> A routing-native consumer matches a dense baseline at *equal* compute,
> AND the routing advantage does not widen as task structure becomes
> richer (more classes, more modalities, more compositional structure).

That's the FALSIFICATION condition for Part B. The supporting condition
isn't equally precise. What constitutes "equal compute"? "Task richness"?
"Routing advantage"?

If Cycle 1 enumerates candidates without first sharpening these
definitions, the scoring will be uncalibrated — different scorers (or
future-me) would weight differently.

**Severity: HIGH.** Without a sharpened Part-B operational definition,
Cycle 1's output is opinion dressed as analysis.

### R2 — The substrate-novelty audit might disqualify everything

CONTRIBUTING.md's substrate-novelty rule asks: "does this work USE the
substrate's distinct capabilities, or just live ON it?" If applied
strictly to Part-B candidates, almost nothing survives. Most candidate
experiments would test "is base-3 routing better than base-2 routing on
this task" — which IS a substrate claim, but if the WORKLOAD itself is
base-2-shaped (image classification, text classification), the answer is
"both substrates can run it, base-2 has more mature tooling, base-3 has
to prove itself" — substrate-novelty is dubious.

**Severity: MEDIUM.** Mitigation: include "what's substrate-distinct
about this experiment" as a scoring axis, not a disqualifier. Some
candidates will fail this audit; recording them as failures is the
finding.

### R3 — I'll anchor on candidates I already mentioned

step_change_raw.md mentioned three vague Part-B directions:
- routing-native architecture trained from scratch
- post-hoc routing of a trained model
- analytically-derived routing demonstration

If Cycle 1's RAW phase just elaborates these, I won't generate genuinely
new candidates. The LMM is supposed to surface surprises; anchoring
defeats it.

**Severity: MEDIUM.** Mitigation: write the RAW phase WITHOUT re-reading
step_change_raw.md. Treat it as a fresh problem. The 10+ candidate
target should pressure me past the obvious three.

### R4 — Tractability estimates are speculative

I'm one person (well, one AI). Estimating "this experiment takes 2 weeks"
vs "this experiment takes 2 months" is guesswork unless the experiment
is closely analogous to something already done. Cycle 1's scoring depends
on tractability estimates that might be wrong by 2-5×.

**Severity: MEDIUM.** Mitigation: report tractability as ranges (1-week,
1-month, 1-quarter), not point estimates. Note dependencies that affect
the estimate (e.g., "depends on whether kernel X exists; if not, +months").

### R5 — Selection bias in my candidates

I'll naturally generate candidates I find INTERESTING or that match
patterns I've seen before. Genuinely novel Part-B candidates might come
from outside my comfort zone (e.g., from RL, from compression, from
information theory) — and I might under-generate from those areas because
I'd have to learn them to evaluate them.

**Severity: MEDIUM.** Mitigation: explicitly include a category in RAW
called "candidates from areas I'm less familiar with" and force at least
2 entries there. They might be vague but at least they're on the list.

### R6 — Negative-result candidates might dominate

If I'm honest, most Part-B candidates I generate will look like they MIGHT
not show a base-3 win — because if they obviously would, the experiment
would already exist. The scoring might pick a candidate where the
EXPECTED outcome is a Part-B falsification. That's still informative —
but the project hasn't framed itself around willingness to falsify Part B
yet (the L1 strong claim was a "show advantage exists" framing; R1 was
"falsify a specific routing rule"). Falsifying Part B as a whole would
be a much bigger result than the project is currently scoped for.

**Severity: HIGH.** Mitigation: make this explicit in the SYNTH phase.
Some candidates' value will come from their potential to FALSIFY Part B.
That's not a flaw of the experiment — it's the discipline of treating
the thesis as falsifiable. Pre-commit gates per candidate.

### R7 — The "Part-B requires training" assumption may be wrong

The synthesis assumed inference-only Part-B tests are possible but
"undertheorized." That might be wishful thinking. If RAW honestly explores
this and finds NO inference-only candidate is tractable, the synthesis's
mode-shift recommendation collapses — we'd be back to "training first" as
the only path to Part-B.

**Severity: MEDIUM.** Mitigation: this is exactly what Cycle 1 should
test. If the answer comes out "no inference-only candidates work," the
synthesis was wrong and we revise. That's the LMM working as designed.

### R8 — Time estimate is optimistic

The plan says "half a day to a full day." LMM cycles typically expand
under good thinking. The previous LMM (step_change) took longer in
practice than the QUICKSTART.md timeboxing suggests. A more honest
estimate might be 1-3 days for Cycle 1 if I do it well.

**Severity: LOW.** Mitigation: don't time-box at the cost of quality.
The user said "time is not important; accuracy and quality are paramount."

### R9 — The SYNTH might collapse under "I don't know enough"

The final scored list might be honest only if I admit I don't know
enough to score some candidates. That's fine, but it dilutes the
deliverable. The user might want a clean recommendation, not "5 of these
8 candidates I can't score confidently."

**Severity: LOW.** Mitigation: separate the deliverable into two parts —
"candidates I can score with confidence" and "candidates that need more
research before scoring." Both are useful outputs.

### R10 — Recursive LMM risk

We're now several layers deep: LMM on step-change → that recommended a
Cycle 1 LMM → now Cycle 1 is itself an LMM with its own RAW phase. There's
a real risk of meta-paralysis ("we keep writing LMMs about LMMs"). At
some point we have to start running EXPERIMENTS, not just designing them.

**Severity: MEDIUM.** Mitigation: Cycle 1 is the LAST design layer. Cycle
2 must be experiment execution, not another design cycle. If Cycle 1's
SYNTH says "we need another design cycle before Cycle 2," that's a sign
the framing is wrong.

## Tensions in the plan

- T1: Sharpen Part-B definition first (R1) vs generate candidates first
  (the plan's structure). Resolution: do them together. RAW phase can
  include "what would actually constitute Part-B evidence" as a sub-question;
  REFLECT can sharpen.

- T2: Substrate-novelty as scoring axis (R2) vs disqualifier. Resolution:
  scoring axis. Include "substrate-distinctiveness" as one of 4-5 scoring
  factors.

- T3: Strict no-anchor RAW (R3) vs efficient use of prior thinking. Resolution:
  RAW gets a fresh start (don't re-read step_change_raw.md) but I can still
  use what I know. The discipline is "write fresh," not "forget what I learned."

## Failure modes the plan should explicitly handle

1. **All candidates require training.** Output: "we cannot test Part B
   without first building training capability; the mode-shift recommendation
   needs to be revised; defer to a training-first sequencing." This is the
   loop-back-to-step_change_synth.md case.

2. **No candidate has clear pre-commit gates.** Output: "Part-B as currently
   defined is not operationalizable; we need a thesis-amend cycle to
   sharpen Part B before testing." Loop-back to docs/THESIS.md.

3. **A candidate dominates everything else.** Output: clean. Cycle 2 begins
   on it.

4. **The scoring rubric itself becomes the deliverable.** Sometimes the
   scoring framework is more valuable than the scored list. Output: the
   rubric is named as Cycle 1's primary product; the scored list is an
   instance.

## Remediation summary (what to change in the plan)

1. (R1) Add "sharpen Part-B operational definition" as a goal of REFLECT.
2. (R2) Substrate-distinctiveness is a scoring axis, not a disqualifier.
3. (R3) RAW is written without re-reading step_change_raw.md.
4. (R4) Tractability is reported as a range, with dependency notes.
5. (R5) RAW includes a forced "less familiar areas" category.
6. (R6) Pre-commit gates per candidate; willingness-to-falsify is explicit.
7. (R7) If RAW finds no inference-only candidates, synth says so and we
   loop back to step-change synthesis.
8. (R8) No time-boxing; accuracy is paramount.
9. (R9) SYNTH separates "scored confidently" from "needs more research."
10. (R10) Cycle 2 must be execution, not another design layer.

These are remediations to the plan, not to the substance of what Cycle 1
will produce.
