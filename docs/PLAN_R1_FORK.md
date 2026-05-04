---
status: P0 — owner directive 2026-05-04 (R1 fork experiment)
authority: owner directive — distinguishes F1/F2/F3 framings of R1 FAIL
scope: 3-day focused experiment to determine which framing of the R1 failure is correct
predecessor: journal/r1_path_forward_synthesize.md (LMM cycle on the path forward)
parent_plan: docs/PLAN_EXPRESSION_ROUTING_R2.md (the R2 plan, now superseded for sequencing by this fork)
---

# Plan — R1 Fork Experiment

## What this is

R1's PASS verdict was structurally weak; the 100/100 remediation FAILed two gates. The path-forward LMM cycle (`journal/r1_path_forward_*.md`) reduced the eight surface options (A–H) to **three structural framings of the failure**:

- **F1: "Wrong rule."** Signature richness was the right axis; the dual implementation is wrong. → Redesign.
- **F2: "Wrong axis."** The consumer needs MORE CELLS, not RICHER CELLS. → Revert + sig_dim sweep with sign-only.
- **F3: "Wrong layer."** The concerns are substrate-level, not consumer-level. The consumer can't deliver substrate-distinctness via signature richness. → Pivot to P1-1.

The R1 evidence we have does not distinguish these framings. Picking A-H without distinguishing F1/F2/F3 is choosing on intuition.

This plan is the **3-day experiment that distinguishes them empirically.**

## The experiment

Run BOTH signature rules (sign-only and dual) at sig_dim ∈ {16, 32, 64}, on curated arity-1 and arity-2 banks plus random-expression banks. Measure inter-class distance and partition-change rate. Apply pre-committed framing thresholds.

### Test inputs

- **sig_dim=16, arity-1:** existing curated set `{-30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30}`.
- **sig_dim=32, arity-1:** `{-30, -25, -20, -18, -15, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 22, 25, 27, 28, 30}` (32 values, symmetric structure preserved, finer near-zero).
- **sig_dim=64, arity-1:** further interpolation, 64 values spanning [-30, 30].
- **sig_dim=16, arity-2:** existing 4×4 grid over `{-10, -3, 3, 10}`.
- **sig_dim=32, arity-2:** 4×8 grid (x in `{-10, -3, 3, 10}`, y in `{-15, -8, -5, -1, 1, 5, 8, 15}`).
- **sig_dim=64, arity-2:** 8×8 grid over `{-15, -8, -5, -1, 1, 5, 8, 15}`.

### Measurements per (rule, sig_dim, arity)

1. **Curated bank inter-class distance:** min, mean, max.
2. **Random bank (100 random expressions, 3 seeds):** number of equivalence classes per seed, mean inter-class distance per seed.
3. **Partition-change rate:** for each (sig_dim, arity), compare BOTH rules' partitions on the same random expression set; report fraction of pairs whose same-class status flips between rules.

### Pre-committed framings

A framing wins iff its specific predictions hold at sig_dim=64. **Stated BEFORE running the experiment:**

| Framing | Wins iff | Implication |
|---------|----------|-------------|
| **F1 (wrong rule)** | dual at sig_dim=64 has arity-1 inter-class min ≥ sign-only at sig_dim=64 by ≥ 2 trits | R1 v2 is justified — write Option B plan |
| **F2 (wrong axis)** | sign-only at sig_dim=64 reaches arity-1 inter-class min ≥ 6 AND dual at sig_dim=64 doesn't add ≥ 2 over sign-only at same dim | Revert R1; resume R3/R2 with sign-only as primary — write Option A plan |
| **F3 (wrong layer)** | both rules at sig_dim=64 have arity-1 inter-class min < 6 AND neither shows ≥ 30% partition change rate from sig_dim=16 to sig_dim=64 | The consumer is signature-saturated; pivot to P1-1 — write Option F plan |

If results are **mixed** (e.g., F2 wins arity-1, F1 wins arity-2), per-arity rules (Option H) become the data-driven answer.

### Pass/fail/inconclusive

- **PASS (informative):** at least one framing wins clearly per its threshold.
- **WEAK (mixed):** different framings win for different arities; per-arity rules are the answer.
- **INCONCLUSIVE:** all three framings are within 2 trits of each other; need to extend sig_dim or rethink test inputs.

## What this plan deliberately does NOT do

- Does not commit to any of A–H prematurely. The fork is the precondition.
- Does not redesign R1 v2. That's the next cycle's job, contingent on F1.
- Does not begin R3 or R2 work. Those are downstream of the fork resolving.
- Does not retire R1 code. Per ship-with-FAIL discipline, R1 stays in the codebase regardless.
- Does not extend test inputs beyond sig_dim=64. If results are inconclusive at 64, the next cycle expands; this cycle does not.

## Substrate-discipline notes

- All measurements use existing kernels (`m4t_route_threshold_extract`, `m4t_route_threshold_extract_dual`, `m4t_popcount_dist`, `m4t_route_confidence_weighted_dist`).
- New code under `-Werror` with project standard flags.
- All gates pre-committed in this document before code runs.
- CHANGELOG entry lands with the work.

## Budget

- Test-input design + binary + analysis: ~3 days.
- Closeout doc: 0.5 day.
- Total: ~3.5 days.

## Action plan

1. Write this doc (in progress).
2. Write `gesh/bench/expr_routing_r1_fork.c` with all (rule, sig_dim, arity) combinations.
3. Update CMakeLists.
4. Build, run, apply framing thresholds.
5. Write closeout naming the winning framing.
6. The next plan (R1 v2 / R3 with sign-only / P1-2 / per-arity) is written based on the closeout's verdict.

## Independence

P1-1 (close primitives floor with exp/log) is independent of this experiment. Owner-authorized to begin in parallel.
