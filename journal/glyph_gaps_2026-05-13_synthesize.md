# SYNTHESIZE — Glyph gaps action plan

## Headline

The substrate-eviction arc is a corollary of claim 3. Claims 1 and
2 have no measurement loop. The next unit of work that moves Glyph
forward is **claim 2's first measurement loop (math-expression →
signature bridge)**, not another eviction battery. The eviction
arc finishes either via a background N=50-100 battery or via an
explicit "inconclusive, parked" stopping point.

## Decisions

1. **Background-run the eviction battery at N=50.** Use the
   existing `redteam_b_harness.py` driver, swap in 50 prompts (the
   20 already tokenized plus 30 new). Run in the background — do
   not foreground-block on it. Whatever it says is the closing
   answer for the eviction territory.
2. **Foreground attack: claim 2's bridge.** Design a minimal
   toy that takes a single algebraic expression and derives a
   signature. The first measurement loop is "does the bridge
   produce stable signatures under semantic equivalence?" (e.g.,
   `x+y` and `y+x` should produce the same signature).
3. **Hygiene in cheap parallel cycles.** c_dump_v3 provenance
   audit, journal current-state index, README/CONTRIBUTING
   resync after the Phase ζ retraction, memory entry review.
4. **Defer:** claim 1 primitive-set closure audit (depends on
   claim 2 outputs to know what expressions matter), qsigdist
   unit tests (research mode, not load-bearing yet), per-prompt
   coherence audit of no_evict baseline (not blocking).

## Execution plan

### Track A — Claim 2 bridge (foreground, ~3-5 days estimate, not a deadline)

**A1. Specification (1 day).** Write a short spec for the bridge:
- Input grammar: arithmetic expressions over variables, constants,
  and a fixed primitive set. Start with `{+, -, *}` to keep the
  first iteration tractable.
- Output: a signature in the substrate's encoding (trit vector at
  some dimension, e.g. 128).
- Derivation rule: must be deterministic and respect at least one
  semantic equivalence (e.g., commutativity of `+`).
- Falsifiable test: equivalent expressions produce equal signatures;
  semantically-different expressions produce L1-distant signatures.

Output as `journal/claim2_bridge_spec.md`.

**A2. Toy implementation (1-2 days).** Implement the spec in
Python. Place in `experiments/claim2_bridge/`. No production C
yet — research code.

**A3. First measurement loop (1 day).** Test set: 20-50
expressions of varying complexity. Measure:
- equivalence preservation rate (target: 100% on commutative pairs).
- semantic-distance correlation: do expressions that humans rate as
  "more similar" produce L1-closer signatures? Use a small hand-
  labeled set first; consider crowdsourcing later if it matters.
- determinism: same expression always same signature.

Output as `journal/claim2_first_measurement_2026-MM-DD.md`.

**Gate to next step:** if A3 produces a measurable property of the
bridge (positive or negative), claim 2 has its first measurement
loop and the foundational gap is no longer "no infrastructure
exists." If A3 produces nothing measurable, A1's spec needs revision.

### Track B — Eviction settling (background)

**B1. Tokenize 30 more diverse prompts** (~30 min) extending the
existing 20-prompt set. Cover more topic categories: code, poetry,
multi-turn dialog, error messages, technical jargon.

**B2. Run N=50 harness battery at window=16** in the background.
Expect ~1-3 hours wall-clock with OS throttling.

**B3. Aggregate + report.** If qsigdist's Δ vs random reaches
significance (CI excludes zero), the territory verdict closes
positive — substrate eviction is a real (if modest) win. If CI
still straddles zero at N=50, accept "inconclusive with positive
trend" as the closing position and write a closeout journal.

### Track C — Hygiene (parallel cheap cycles)

**C1. c_dump_v3 provenance audit.** `git log experiments/phase_*/`
and search for the prompt-generation script. Decode the prompts.
Record finding in `journal/cdumpv3_provenance_2026-MM-DD.md`.
**If gibberish:** flag Phase α-ε oracle numbers for re-validation.
**If natural language:** annotate the journals with confirmation.

**C2. Journal current-state index.** Single short doc:
`journal/INDEX_2026-05-13.md`. For each major arc (substrate
claim, phase ζ, plan A red-team, plan B + red-team), one line
stating "current verdict" + pointer to the authoritative journal.
Annotate retracted claims with a banner pointing forward.

**C3. README / CONTRIBUTING resync.** Re-read with the Phase ζ
retraction in hand. Confirm scope language about substrate eviction
is accurate (it's research-mode, not production-default; harness
trend is inconclusive-positive). Update if drift exists.

**C4. Memory entry review.** Re-read all `feedback_*` and
`project_*` entries. Confirm each still reflects current best
understanding. Update or annotate any superseded by Phase ζ
retraction.

**C5. Token-ID audit.** `grep -rn "prompt[_-]tokens\|prompt_ids" experiments/`
for hard-coded token sequences. Decode each. Confirm natural
language.

## Key decisions and rationale

- **Why claim 2 over claim 1 first:** claim 2 (bridge) is the
  connective tissue between 1 (primitives) and 3 (substrate).
  Building it first lets claim 1's closure audit have meaning
  ("does the primitive set support the math the bridge produces?")
  and lets claim 3's tests have meaning beyond eviction ("does the
  path-graph metric serve the routings the bridge derives?").
  Claim 1 audit depends on claim 2 outputs; reverse order would
  measure nothing.

- **Why background the eviction battery:** the eviction trend is
  +6pp at N=20 with CI straddling zero. At N=50 the CI tightens.
  The expected value of the resolution is high-information (settles
  the territory), but the cycles to run it are mostly wall-clock,
  not cognitive. Background-friendly.

- **Why hygiene in parallel:** the hygiene items individually take
  ~30 minutes each, and several block trust in existing oracle
  numbers (c_dump_v3 provenance especially). Doing them as
  one-off side-quests during track A is cheap.

- **Why defer claim 1 closure audit:** the audit asks "do the
  current 6 primitives close over the math we need?" Without
  claim 2's bridge to derive concrete expressions, the audit has
  no concrete inputs — it would be hand-waved by me. Claim 2's
  output is the audit's input.

- **Why defer qsigdist tests:** production default is no eviction.
  Sigdist/qsigdist are research modes. Unit tests for research code
  is premature. The harness is the integration test for now.

## Success criteria

- [ ] Claim 2 bridge spec exists in `journal/claim2_bridge_spec.md`.
- [ ] Toy implementation runs end-to-end on at least one expression
      and produces a signature.
- [ ] First measurement loop reports a quantitative result on
      equivalence preservation.
- [ ] Eviction battery at N=50 either reaches significance or
      gets a written "inconclusive, parked" closeout.
- [ ] c_dump_v3 provenance is verified (decoded prompts in journal).
- [ ] Journal INDEX exists pointing at current-state-authoritative
      docs.
- [ ] README/CONTRIBUTING reflect Phase ζ retraction.
- [ ] Memory entries re-reviewed; no superseded claims left as-is.

## Anti-success criteria (what NOT to do)

- DO NOT start a new eviction experiment unless it's the N=50
  settling battery. Variations on eviction at N=20 are diminishing
  returns.
- DO NOT add unit tests for qsigdist while it's research-mode-only.
- DO NOT design claim 3 follow-on experiments before claim 2 has
  its first measurement loop. The vision wants 1 → 2 → 3 as a
  system, not three parallel corollaries.
- DO NOT delete old journals. Annotate retractions in place; the
  cognitive history is valuable.

## Loop-back triggers

- **Back to NODES** if claim 2's bridge spec turns out to have a
  fatal ambiguity (e.g., no deterministic derivation possible from
  expression to signature without additional constraints).
- **Back to REFLECT** if the eviction N=50 battery produces a
  result that's so positive (e.g., +12pp significant) that "claim
  3 corollary" reclassifies as "claim 3 territory test." The
  prioritization would shift.
- **Back to RAW** if the c_dump_v3 audit shows gibberish, since
  that retroactively invalidates several oracle results and the
  arc's status would need re-thinking from scratch.

## What this synthesis is NOT

This is not a complete project roadmap. It's a 1-2 week unit of
work focused on the highest-leverage foundational gap. The next
LMM cycle (post-claim-2-first-loop) should re-examine the gap list
and pick the next move.
