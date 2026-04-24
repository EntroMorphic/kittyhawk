---
date: 2026-04-22
scope: LMM cycle — next step-change for Glyph given the lr_scaffold + distance_function results
phase: SYNTHESIZE
---

# SYNTHESIZE: measure the signature Pareto before picking a step-change

## The reframe

Two cycles (lr_scaffold, distance_function) cleared the classifier and scorer layers — neither moves the CIFAR-10 ceiling. Both cycles implicitly assumed the bottleneck was downstream of the signature. The reflection surfaces the real gap in our knowledge: **we have never measured where the signature itself sits on the accuracy-vs-bits Pareto frontier.** Without that map, every downstream experiment is bounded by an unmeasured asymptote, and every cycle reports less than it should.

The next step-change is not a classifier or a distance. It is either:
- **More bits in the signature** (S1 multi-scale; S5 block summary; S6 MTFP retention), or
- **Different bits in the signature** (S2 pattern codebook; S4 per-region τ; S7 scaffolded learned encoder), or
- **Compound scoring over the existing signature** (S3 multi-distance ensemble — the one positive hint from the previous cycle).

Which of these is worth chasing depends on two measurements neither cycle has produced. Run those first.

## Decision

**Run two cheap gating measurements before committing to any S1–S7 step-change:**

**P1: Bit-budget Pareto on CIFAR-10.** Measure `direct_lsh --distance hamming` accuracy across total_dim ∈ {3072, 4500, 6000, 9024, 12000, 18000} on CIFAR-10. Adjust by scaling gradient channels or adding/removing scales. Record accuracy, oracle-over-union, and wall-time per dim budget. Goal: determine if the 46.63% ceiling is at asymptote or still climbing.

**P2: Multi-seed verification of CIFAR E3 Selective +0.56pp.** Rerun `direct_lsh --distance block_threshold` on CIFAR-10 with 3 different `--base_seed` quadruples. Measure Selective accuracy. If the mean lies in [46.90, 47.50] with σ < 0.2pp, the complementarity signal is real. If σ > 0.4pp or mean drops below 46.80, it's noise.

**Then branch by the two-bit result (P1_climbs, P2_real):**

| P1 | P2 | Branch |
|---|---|---|
| Climbs | Real | Run S1 (multi-scale) + S3 (ensemble) in parallel. Target: +3–5pp on CIFAR Selective, reaching ≥50%. |
| Climbs | Noise | Run S1 (multi-scale) primary; S4 (per-region τ) as a cheap secondary. Target: +2–4pp. |
| Plateau | Real | Run S2 (pattern codebook) or S5 (block summaries) primary — structural departures from direct quantization — and S3 alongside. Aim for +5–7pp via restructuring. |
| Plateau | Noise | **Thesis-calibration step:** run S7 (scaffold learned ternary encoder) to measure whether the substrate itself is bounded at ~47% on CIFAR-10 or only direct quantization is. Output distinguishes "substrate bound" from "encoding bound." Major project-level decision follows. |

## Success criteria

**Cycle-level (pre-measurement):**
- [ ] P1 produces a single plot/table mapping CIFAR-10 Selective accuracy against total_dim. Gate-passing evidence of "climbing" = accuracy strictly increases as dim increases from 3072 → 18000 by ≥2pp net. Gate-passing evidence of "plateau" = accuracy within ±0.5pp across the upper half of the range.
- [ ] P2 produces a 3-seed mean and standard deviation for CIFAR Selective under block_threshold. Real-signal gate defined in the Decision section.

**Cycle-level (post-measurement, branch-dependent):**
- Branch (Climbs, Real/Noise): measure S1 result within 2 weeks.
- Branch (Plateau, Real): measure S2 or S5 within 3 weeks.
- Branch (Plateau, Noise): propose and run S7 as a 4-week project; output is a thesis-calibration document, not a ship candidate.

**Thesis-level:**
- Regardless of S1–S7 outcome, the cycle must produce a statement of the form "direct-quantization + per-trit-distance on [these signatures] caps CIFAR-10 accuracy at approximately X%." X is the measured Pareto ceiling. This is a durable empirical finding.

## Implementation specification

### P1: bit-budget Pareto measurement

**Where:** direct_lsh, existing `--density` and `--gradients` flags. The channel-count axis is controlled by:
- total_dim ≈ 3072: intensity only (no gradients), density low (more zeros → effectively smaller discriminating budget).
- total_dim ≈ 6000: intensity + gradients at lower density.
- total_dim ≈ 9024: current CIFAR setting (intensity + gradients).
- total_dim > 9024: add multi-scale downsampled channels (S1 partial implementation) or increase gradient density. Simplest: add a 2× downsampled intensity channel (→ 3×16×16=768 extra trits) and a 2× downsampled gradient pair (→ ~1536 extra trits), for total_dim ≈ 11328.
- total_dim ≈ 18000: add further scales.

**Output:** per-setting table of LSH k=5-rw, Selective, oracle-over-union, and sweep wall-time. Plot accuracy vs total_dim.

**Cost:** each setting is ~10 min of compute on CIFAR-10 (current sweep takes 600s). Six settings = ~1 hour total. Implementation of extra-channel scales = ~2 hours.

### P2: multi-seed E3 verification

**Where:** direct_lsh with `--distance block_threshold` on CIFAR-10.

**How:** run three times with `--base_seed S S S S` for three distinct S values (e.g., 7, 13, 29). Record Selective accuracy each time. Compute mean and standard deviation.

**Cost:** ~30 minutes total (three runs at ~10 min each).

### Pre-declared gate (critical)

**Before running P1 and P2, commit in writing to the branching table above.** This prevents post-hoc rationalization. When P1 and P2 land, the branch is selected by the data, not by which S-experiment I feel most invested in.

## Handling the major tensions

- **T1 (S1 vs Pareto first):** Pareto measurement precedes S1 commitment. S1 only runs if P1 shows climbing.
- **T2 (S3 vs noise):** Multi-seed verification precedes S3 commitment.
- **T3 (substrate-native vs scaffold):** Resolved by making S7 a thesis-calibration experiment, not a ship candidate. It runs only in the (Plateau, Noise) branch.
- **T4 (ambitious target):** Target reframed from "beat SSTT" to "measure the Pareto and place the thesis." Reaching SSTT is still the S1/S2/S5 aim, but no longer the only success criterion.
- **T5 (parallel vs sequential):** P1 and P2 run in parallel (independent). S1 and S3 run in parallel if their branch fires.
- **T6 (data-selection axis):** Deferred to a follow-up cycle. Signature-layer Pareto must be mapped first.

## Quality check

- **Could someone else execute this?** Yes. The two pre-measurements have named flags, named datasets, quantitative gates, and a decision table. Follow-on experiments are branch-gated.
- **Does it address all major tensions?** Six tensions mapped. Deferrals named.
- **Is it simpler than the starting point?** The RAW listed seven S-experiments and worried about cost/benefit. The synthesis reduces to two cheap pre-measurements + a branching table. The S-experiments become outputs of the data, not inputs to the cycle.
- **Surprised?** Yes — entered expecting to pick one of S1–S7 and left with "don't pick; measure the Pareto, pre-commit the branch, and let data pick." Same pattern as the last two cycles where the reframe was more important than the experiment.

## Immediate next actions

1. **Implement P1 Pareto measurement.** Cost: ~2 hours (extra-channel scales) + ~1 hour (runs). Deliverable: Pareto table.
2. **Run P2 multi-seed verification.** Cost: ~30 minutes. Deliverable: mean ± σ.
3. **Apply branch table to select next experiment.** The (P1, P2) result uniquely determines which S-experiment to run next.
4. **Run selected S-experiment and report.** 2–4 weeks depending on branch.

## What this cycle produces regardless of outcome

- A bit-budget Pareto map for direct_lsh on CIFAR-10. Durable empirical artifact usable by every future cycle.
- A multi-seed verification (or falsification) of the block-distance complementarity signal. Closes T2 permanently.
- A pre-declared branching table that removes post-hoc rationalization pressure from the step-change selection.
- Regardless of which S-branch fires, the cycle produces a measurement of the substrate's CIFAR-10 ceiling under current preprocessing. That ceiling informs every forward decision.

## Thesis-level note

If P1 Plateau + P2 Noise lands, and S7 (scaffold) reaches SSTT or above on the same signature infrastructure, the thesis is in a specific corner: **base-3-native direct quantization is not the right encoding for CIFAR-10**, but the substrate can support other encodings. That's an encoding question, not a substrate question. If instead S7 also plateaus near 47%, the substrate has a CIFAR-10 bound that matters for future benchmark selection.

Either outcome is thesis-informative. Neither invalidates NORTH_STAR's base-3-geometric-fullness claim — they refine *where* that fullness cashes out and *where* it doesn't.
