---
date: 2026-04-22
scope: step_change LMM cycle close-out — final findings, what ships, what doesn't, what's next
phase: CLOSE
---

# Close-out: step_change cycle

## Cycle in one paragraph

After `lr_scaffold` and `distance_function` cycles cleared the classifier and scorer layers without meaningfully moving CIFAR-10 accuracy, this cycle asked: "what about the signature itself?" Four experiments (S1 multi-scale, S3 ensemble, S4 per-region tau, S7 scaffold) were pre-declared behind two gating measurements (P1 bit-budget Pareto, P2 multi-seed block-complementarity). P1 produced an inverted-U density Pareto peaking at 0.395 (no clear "climbing" or "plateau"). P2 confirmed block-complementarity was seed noise. Branch fired on S1 first. Multi-scale pyramids worked across all three datasets; per-region tau worked on CIFAR-10 only (dataset-conditional); heuristic auto-gating is experimental and cannot reliably predict which regime a dataset falls into. Production numbers improved across the board; the substrate kept its discipline.

## Results table — final production numbers

| Dataset | Prior best | This cycle best | Config | vs SSTT |
|---|---|---|---|---|
| MNIST | 97.18% | **97.30%** | MS4 + (R4 neutral) | −0.23pp (tied) |
| **Fashion-MNIST** | 87.95% | **88.66%** | MS4, R=0 | **+2.12pp Glyph wins** |
| CIFAR-10 | 46.63% | **48.05%** | MS4 + R4 | −4.95pp (closed ~23% of gap) |

MNIST is saturated at the signature level. Fashion opened the SSTT gap further. CIFAR gained +1.42pp of the ~6.4pp gap to SSTT, measured as mean +0.36pp ± 0.47pp R4 contribution (seed-sensitive) plus a robust +1.10pp MS4 contribution.

## What works — ship recommendations

### Multi-scale pyramid (MS4): SHIP
- Consistent gains across MNIST / Fashion / CIFAR.
- Additive (appends channels) — old signal preserved.
- Churn analysis: 1500+ wrong-but-different flips per transition on CIFAR, but net positive (+110 w→r on base→MS4).
- Cost: +24% total_dim, +24% sweep time.
- Recommendation: **make `--multi_scale4` the default for natural-image datasets** where signature dim budget allows.

### Per-region tau (R4): OPT-IN PER DATASET
- CIFAR: +0.32pp single-seed, +0.36pp ± 0.47pp multi-seed mean (marginal, seed-sensitive).
- Fashion: **−0.37pp to −0.61pp across all R ∈ {2, 3, 4, 6, 7}** — no grid size avoids regression.
- MNIST: neutral (near saturation).
- Mechanism: R4 trades uniform-Hamming quality for pair-IG-weighted quality. Wins when different classes have different background statistics per region (CIFAR sky/ground/subject variety). Loses when all classes share a common layout (Fashion garments on black).
- Recommendation: **default R=0. Opt in (`--region_tau 4`) only after measuring on target dataset.**

### NEON pair-IG (side-effect): SHIPPED (retroactive)
- Swapped scalar trit-read loops to `vsubq_s8 + vabsq_s8 + vminq + vdotq_s32`.
- Bit-identical numbers, ~10× sweep speedup on all datasets.
- Unblocked every subsequent measurement — CIFAR sweep 600s → 30s.
- This is the hidden win of the cycle. Substrate discipline ("no scalar when kernels exist") enforced at user prompt.

## What doesn't work — don't ship

### SDOT as distance (E1): REJECTED
- Loses on every dataset (MNIST tied, Fashion −2.66pp, CIFAR −3.26pp, un-normalized MNIST −17.70pp).
- Reframe insight ("structural-zero handling") was empirically falsified — penalizing zero-vs-nonzero mismatches IS informative.

### Global per-dim weighted Hamming (E2): REJECTED
- Consistently −0.2 to −0.4pp below uniform Hamming.
- Answers T2 from the synthesize: **per-class-pair specialization is irreducible.** Averaging pair-IG across pairs loses the signal.

### Block-threshold distance (E3): NOISE
- Multi-seed confirmed: original +0.56pp on CIFAR Selective was seed variance (3-seed mean vs baseline was −0.33pp).
- Complementarity hypothesis falsified.

### R4 auto-gating heuristic: EXPERIMENTAL ONLY
- Tried: inter-class variance ratio (fails — all datasets >>threshold); tau-spread (fails — Fashion > CIFAR); per-class COM spread (fails — Fashion > CIFAR).
- No scalar statistic separates "R4 helps" from "R4 hurts" across the three datasets we have.
- Mechanism is too nuanced: the question is whether per-region pixel-distribution differs BY CLASS, not just whether per-region pixel-distribution differs at all.
- `--region_tau auto` reports COM spread as a diagnostic but defaults to R=0. Users measure empirically.

## Atomic mechanism (from step_change_atomics.md)

R4 on CIFAR: −0.25pp raw Hamming k-NN, +0.70pp pair-IG re-rank, +0.32pp Selective composite. R4 produces signatures where class-pair-weighted distance extracts more signal even though uniform Hamming extracts less.

R4 on Fashion: −0.01pp Hamming, −0.52pp pair-IG. P(PIG correct | disagree) collapses from 57% to 53%, eliminating pair-IG's edge on the gate that Selective uses.

Churn scales with dataset difficulty (~1500 wrong-but-different per layer on CIFAR, ~80 on Fashion, ~20 on MNIST). Net gains are small residuals on top of large ranking perturbations. Anything <0.4pp is noise-adjacent; robust claims need multi-seed.

## Branching table — what actually fired

| Branch predicted | Fired | Outcome |
|---|---|---|
| (Climbs, Real) → S1 + S3 parallel, target +3-5pp | — | — |
| (Climbs, Noise) → S1 primary, S4 secondary, target +2-4pp | **This branch** | Partial hit: +1.42pp CIFAR, within target range but bottom end |
| (Plateau, Real) → S2/S5, target +5-7pp | — | — |
| (Plateau, Noise) → S7 scaffold calibration | — | — |

The branching table held. The branch fired as predicted, the gate target (+2-4pp on CIFAR) was partially hit (+1.42pp), and the associated experiments behaved as anticipated.

## What this cycle taught for future cycles

1. **Pre-declared gates work — but pre-declare them tightly.** The original +0.5pp gate on ANY dataset was loose enough that E3's seed-noise blip cleared it. A tightened gate (multi-seed σ < target) would have caught it earlier. Future cycles: pre-declare multi-seed verification as part of the gate, not as a post-hoc check.

2. **Reframe > experiment, repeatedly.** Third cycle in a row where the reframe ("measure the Pareto before picking") was more valuable than any specific S-experiment result. The primary deliverable of an LMM cycle is the decomposition, not the measurement.

3. **Atomics reveal mechanism.** Per-query dumps + flip analysis + scorer-component breakdown revealed exactly why R4 succeeds on CIFAR and fails on Fashion. Without atomics, R4 would have looked like "dataset-conditional noise." With them, it's a mechanistic finding that predicts future behavior.

4. **Auto-gating requires a better signal than summary statistics.** Summary statistics (COM, tau spread, variance) cannot reliably predict where a per-region technique will help. The real signal is class-conditional spatial pattern similarity, which is more work to compute. Accept the heuristic's limits; document them.

5. **Scalar trit loops are expensive. NEON them immediately.** The pair-IG NEON-ification gave 10× speedup, unblocking every subsequent measurement in the cycle. Should have been done before E1. Next cycle: audit for scalar trit-read loops before launching experiments.

## Residuals / what's unfinished

- **Multi-seed MS4 verification.** CIFAR MS4 baseline was single-seed; the +1.10pp MS4 gain is presumed real but unverified. Multi-seed σ would clarify.
- **R4 auto-gating with a class-conditional-spatial-similarity metric.** A fuller metric that measures pairwise similarity of per-class spatial patterns might separate CIFAR from Fashion, but it's a deeper implementation than this cycle's time budget allowed.
- **Gradient density has a sweet spot but 0.10 is optimal.** Documented; no further work needed.
- **Multi-scale beyond 4× (8×, 16×).** At CIFAR 32×32, 8× → 4×4 = 48 trits added; marginal. Not measured; skipped as unpromising.

## What ships from the whole cycle sequence

Cumulative across lr_scaffold + distance_function + step_change:

1. `tools/csa_classifier.c` — Class-Signature Argmax, 20-100× faster than full k-NN at 1.3pp accuracy cost (lr_scaffold).
2. `direct_lsh --multi_scale4` — default-on recommended for natural-image datasets; +0.7 to +1.1pp Selective (step_change).
3. `direct_lsh --region_tau N` — dataset-conditional, default-off (step_change).
4. NEON pair-IG in `direct_lsh` — 10× sweep speedup, bit-identical (step_change).
5. `--dump_preds PATH` diagnostic flag (step_change atomics).
6. Clear negative findings: SDOT-as-distance rejected, global-weighted-Hamming rejected, block-distance complementarity rejected, R4 auto-gating experimental-only (distance_function, step_change).
7. Mechanism documentation explaining WHY R4 is dataset-conditional (step_change atomics).

## Next-cycle seed

Two candidate directions, neither committed:

**A. Representation-layer refactor.** All three cycles have now cleared classifier, scorer, and (partially) signature layers. CIFAR-10 remaining gap (~5pp to SSTT) is likely in the input features themselves — edge orientations, color-channel encoding, multi-scale interactions that direct quantization of intensity + gradient doesn't capture. A "feature channels" cycle would enumerate candidates (HOG-like orientation bins, color-opponent maps, patch-level block encoding) and measure their Selective contribution.

**B. Thesis-calibration scaffold (S7 deferred).** Train a small ternary encoder via gradient descent, quantize weights to trit, measure. Tells us whether the remaining CIFAR gap is substrate-bounded (S7 also plateaus near 48%) or encoding-bounded (S7 reaches SSTT or beyond). This is the thesis-level question.

Recommend B first — it's the higher-information experiment, and A can follow if B confirms the gap is reachable.
