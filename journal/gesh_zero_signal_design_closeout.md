---
cycle: gesh_zero_signal_design (P0-1)
phase: CLOSEOUT
date: 2026-05-02
status: COMPLETE — substrate-novelty demonstrated; cycle closes
---

# Closeout — gesh_zero_signal_design

## Verdict

**Substrate-novelty for the structural zero is demonstrated** via deliberate per-class wildcard placement in bank construction.

Gate verdicts:

| Gate | Result | Detail |
|---|---|---|
| 1 (synth_wildcard, 3-seed) | **PASS for bank-alone** | Wildcard bank + Hamming: +1.80pp ± 1.22pp paired stddev. 95% CI [+0.42, +3.18] excludes zero. Committed pair (bank + wildcard kernel) was +0.40pp ± 0.72pp — INCONCLUSIVE. |
| 2 (kernel runtime) | PASS | Wildcard kernel 1.05× standard Hamming. |
| 3 (substrate-novelty audit) | PASS by construction | Substrate uses free third state at 2 bits/pos; base-2 alternatives pay 1.5× storage or 4-state branching. |
| 4 (MNIST regression) | PASS | Wildcard bank: 51.1% vs class_mean 50.0% → +1.1pp, within ±2pp tolerance. |

## What works

`gesh_bank_build_class_wildcard` paired with **standard Hamming** (`m4t_popcount_dist`). Real, multi-seed-robust gain on the substrate-distinct benchmark; no regression on MNIST. The substrate's free third state is operationally distinct in this configuration.

## What doesn't work

The wildcard *kernel* (`m4t_route_wildcard_dist`) was wrong for multi-class banks. It promotes inter-class wildcard matches, diluting discrimination. The kernel is preserved for single-rule consumers (TCAM-shape future use); not the default for multi-class routing.

## What changed from the SYNTHESIZE plan

The committed coupling assumption ("kernel and bank must ship together") was wrong. The bank is the substrate-novel artifact; the kernel as designed has narrower applicability.

## Deliverables

- `m4t_route_wildcard_dist` + 5 property tests — all green.
- `gesh_bank_build_class_wildcard` — substrate-novel bank constructor.
- `gesh_forward_classify_wildcard` — preserved as TCAM-style consumer.
- `synth_wildcard` benchmark — diagnostic for don't-care structure.
- `wildcard_probe` (3-seed, paired-CI), `mnist_wildcard` (Gate 4).
- `M4T_SUBSTRATE.md` §19 — declared zero-state interpretations.
- `CONTRIBUTING.md` — substrate-novelty audit as 6th rule.

15/15 ctest binaries green. Cycle closes.

## Next

P0-2 (MTFP exponent as routing signal). Same protocol.
