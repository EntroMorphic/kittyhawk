---
date: 2026-04-14
phase: SYNTHESIZE
topic: What mechanism makes rank-weighted k=5 voting recover ~9 cases per seed?
---

# Synthesize

## The mechanism

**Rank-weighted k=5 occupies the unique region of the (rule × k) search space where the weighting profile is steep enough to preserve correct-top-1 dominance AND the number of neighbors is large enough for ranks 2..k combined to outvote a wrong top-1 when they agree.** No other tested configuration satisfies both constraints:

- **Majority** at any k: uniform profile, no signal-dominance preservation.
- **Distance-weighted** at any k: profile is uniform-in-practice because Hamming distances in top-5 span a narrow range (~6% variation), so it doesn't discriminate.
- **Rank-k=3**: steep profile (weights 3:2:1, top-1 = 50% of total) but no counterweight; top-1 errors amplify.
- **Rank-k=5**: steep profile (weights 5:4:3:2:1, top-1 = 33% of total) AND ranks 2-5 can outvote top-1 when aligned. **Winning region.**

The mechanism is compositional — you can't reach it by optimizing rule and k independently. The surface isn't smooth: moving from majority-k=3 to rank-k=3 *hurts*; moving from majority-k=3 to rank-k=5 *helps*. Only a 2D search over (rule × k) reveals the winner.

## The recovery pattern

Rank-k=5 specifically recovers **shape A** failures:

```
Top-5 ranks = {correct, wrong, correct, wrong, wrong}
Rank-k=5 score: correct = 5 + 3 = 8;  wrong = 4 + 2 + 1 = 7.  Correct wins.
Majority k=5:     correct = 2 votes;   wrong = 3 votes.        Wrong wins.
Majority k=3:     (top-3) correct = 2; wrong = 1.              Correct wins.
```

It *loses* **shape C** failures in exchange:

```
Top-5 ranks = {wrong, correct, wrong, correct, correct}
Rank-k=5 score: wrong = 5 + 3 = 8;    correct = 4 + 2 + 1 = 7.  Wrong wins.
Majority k=5:     correct = 3; wrong = 2.                       Correct wins.
```

On MNIST, shape A is more common than shape C (because N_PROJ=2048 projections usually put the correct class at top-1; when failures occur, they're "correct top-1 crowded by wrong cluster" more often than "wrong top-1 with correct class only in a later cluster"). Net effect is +9 cases.

## What this means for future adaptation

### Predictive principle (provisional)

For failure-guided vote-rule tuning in the routing-LSH regime:

1. **Profile steepness matters**, but only up to a point. Uniform profiles (majority, distance-weighted-on-narrow-distances) don't discriminate; very-steep profiles at small k amplify top-1 errors. The sweet spot is "steep enough to weight top-1 heavily, not so steep at small k that top-1 is the only vote."
2. **k sets the hedge.** k must be large enough that ranks 2..k combined can outvote the top-1 slot. With rank weighting at k=5, 4+3+2+1 = 10 > 5; hedge exists. At k=3, 2+1 = 3 = top-1 weight; hedge is exactly zero (ties go to top-1 in tie-break).
3. **The winning (rule, k) pair is task-specific.** Rank-k=5 wins on MNIST. On a task with wider distance distributions or different failure-shape dominance, the winning pair would differ.

### Operational rule for future adaptation experiments

**Sweep (rule × k) as a 2D grid, not either axis alone.** The surface isn't smooth; gradients over one axis can mislead. A 3-rule × 2-k sweep took 90 seconds and produced the winning configuration. Larger sweeps (exponential weighting, k=7 / k=9) are cheap if this principle scales.

### What the trace should do differently

Future trace-based failure analysis should **sub-categorize by recoverable pattern**, not by magnitude of failure:

- "Correct top-1 outvoted by wrong cluster at ranks 2..5": recoverable by rank-weighted k=5.
- "Top-1 wrong, correct class forms own cluster at ranks 4..5": recoverable by larger k + lighter weighting.
- "Top-5 unanimously wrong-class": not recoverable by any vote rule. Needs projection/signature-level intervention.

Labeling these patterns at trace time would make recoverable-count predictions match reality, replacing my 3× overestimate with something accurate.

## What this does *not* claim

- The mechanism generalizes to every task. Shape A dominance is an MNIST + Hamming-LSH artifact; other tasks will look different.
- Rank-k=5 is globally optimal. Exponential weighting, higher k, mixed strategies — all untested.
- This is a deep insight into LSH. It's a narrow, specific observation about how vote rules interact with one failure distribution.

What it IS: a validated example of integer-arithmetic adaptation on the rebuilt substrate, with a decomposable mechanism that survived post-hoc analysis.

## Three cheap follow-ups enabled by this understanding

1. **Exponential weighting sweep.** Test whether weights 2^(k-i) at k=5 and k=7 beat rank weights. ~30-minute experiment. Predicts the "too-steep" hypothesis.

2. **Rank-weighted k=7.** Test whether more hedge helps further or flattens effective discrimination. ~30 minutes.

3. **Pattern-aware trace tool.** Sub-categorize NARROW MISS by top-5 label sequence. Produces accurate per-intervention recoverable counts. ~45 minutes.

Each is a discrete, self-contained experiment. Each uses integer statistics over the routing-surface outputs. None need gradients.

## What this cycle changed in my mental model

Before the cycle: "weighted voting helped a bit; distance-weighted at k=3 is a natural first try."

After: "The winning adaptation happens at a specific *interaction point* — (steep profile × sufficient k) — and navigating to it requires sweeping both axes, not reasoning about one at a time. The trace gives direction; the search reveals specifics. Magnitude predictions should be halved or quartered from what the trace suggests."

Before the cycle: "Rank-k=5 worked; probably rank-k=7 would work better."

After: "Rank-k=5 worked because it balances two forces; steeper profiles or more neighbors may cross a threshold in either direction. Untested in both directions; don't assume monotonicity."

Before the cycle: "Failure-guided adaptation is gradient descent without gradients."

After: "Failure-guided adaptation is *navigation* of a small discrete configuration space guided by trace-identified failure patterns. The adaptation isn't 'descend the gradient'; it's 'locate the interaction region in the configuration lattice.' That's structurally different — and fits the base-3 substrate naturally because all the coordinates are discrete."

## Success criteria for the cycle

- [x] Mechanism decomposed into testable structural claims (steepness + hedge).
- [x] Recovery pattern named (shape A) and loss pattern named (shape C).
- [x] Predictive principle derived (profile × k is 2D, not separable).
- [x] Followups identified with explicit cost estimates.
- [x] Honest about what the mechanism claims AND does not claim (task-specific, not general).

## Pointers

- Experiment that produced the data: `tools/mnist_routed_weighted.c`.
- Experiment writeup: `journal/weighted_voting_adaptation.md`.
- Inspectability source that motivated all of this: `journal/routed_inspectability_trace.md`.
- NORTH_STAR §Training: `docs/THESIS.md` and `journal/ternary_routing_helps_*.md` for the broader "how does adaptation work in base-3" question this cycle partially answered.
