---
date: 2026-04-24
scope: base3_benchmarks LMM cycle close-out
phase: CLOSE
---

# Close-out: base3_benchmarks cycle

## Cycle in one paragraph

This was a framing cycle, not an implementation cycle. After `routed_autodiff` closed with the finding "frozen-U selection-only routing collapses on multi-class," the next question wasn't "how do we fix the trainer" but "why have we been measuring the substrate on data that fails all three substrate-property criteria?" The RAW surveyed candidate directions without pre-filtering; NODES decomposed "base-3 native" into ternary-representable input / routing-load-bearing task / inspectability-credited evaluation; REFLECT named the category error (validating base-3 claims on base-2 canon); SYNTHESIZE committed to Go position evaluation as the primary direction with a custom synthetic diagnostic, gated by a half-day probe before further investment.

## What was decided

- **Primary benchmark direction**: ternary-state board-game position evaluation (Go first).
- **Diagnostic benchmark**: custom synthesized dataset with tunable routing load + tunable ternary completeness.
- **Regression suite**: MNIST / Fashion / CIFAR — explicitly demoted from north-star to regression-guard.
- **Deferred**: tabular (safer but weaker claim), NLP / extreme-classification / compositional (blocked by embedding/seq2seq infrastructure).

## What was not decided (deliberately)

- Whether to build a full Go trainer. Gated on the probe outcome.
- Whether to port backward kernels to NEON. Still deferred from `routed_autodiff`; reopen only when the benchmark direction is validated.
- Whether to revisit tabular. Only if Go probe fails.

## Carry-forward facts

- The last year of image-benchmark work remains meaningful as infrastructure and regression, not as a north star.
- The representation tax measured in `step_change` and the expert collapse documented in `routed_autodiff` are both explained retroactively by the benchmark mismatch: we were training routing on data that doesn't reward routing, then concluding routing doesn't work.
- "Routing is essential in base-3" is NORTH_STAR; testing this claim requires a benchmark where routing can specialize and input is already base-3. Go is that benchmark.

## Probe recipe (next step)

**Goal:** in half a day, determine whether raw ternary Go positions are Hamming-discriminable for either phase classification or next-move-wins evaluation.

**Steps:**
1. Acquire a public Go position dataset (KGS archives or Badukmovies SGF dumps), convert to per-move board states in `{empty, own, opponent}` labeling relative to current mover. Target ~50k positions distributed across opening / middle / endgame.
2. Pack each 19×19 state as 361 trits (or, for first probe, flatten + quantize to existing signature format). Label as phase (3-class) or next-move-outcome (binary).
3. Run `direct_lsh` against the ternary positions directly. No trainer, no MS4, no R4. Hamming distance, k ∈ {50, 100, 200}, Selective aggregation.
4. Record: phase accuracy, phase confusion matrix, next-move-wins accuracy, per-phase breakdown.

**Decision rule:**
- Phase id > 60% **or** next-move-wins > 55% → **probe green**, commit to `routed_go` LMM cycle.
- Phase id < 40% **and** next-move-wins < 52% → **probe red**, run tabular probe next; if that also fails, substrate-infrastructure review cycle.
- In between → inconclusive; look at what's happening per-phase and per-class before committing.

**Deliverable:** `journal/base3_go_probe.md` documenting the outcome.

## NORTH_STAR discipline maintained

- `§4` (scaffolding sanction) — probe tooling is explicitly scaffolding.
- `§12` (no binary float in compute) — unchanged; all probes use existing substrate.
- `§13` (training in consumer) — any future Go trainer lives in `/train_go` or equivalent, not in libm4t.
- Routing claim lives where it can be tested. Substrate chooses its data, not vice versa.

## Close

The substrate finally has a benchmark chosen for the *reasons the substrate exists*, rather than inherited from the dense-float lineage. The probe gates whether the choice is immediately reachable or whether we need one more infrastructure step first. Either way, CIFAR-gap-chasing is no longer the primary work.
