---
date: 2026-04-22
scope: LMM cycle — given the lr_scaffold + distance_function cycle results, what's the next step-change worth chasing?
phase: REFLECT
---

# REFLECT

## The core insight

**We've been searching for the lever in the wrong room.** Two cycles tested downstream-of-signature (classifier architecture, distance function). Both landed mostly-null, because they operate on a fixed signature whose information content is what it is. The remaining headroom — whatever it is — lives in the signature itself. But more importantly: **we don't actually know what the signature's discriminative ceiling is.** We've been comparing against pair-IG's 46.63% without measuring what a more-bits, more-structure, or more-signal signature could reach before the ceiling re-asserts itself.

The step-change question is not "which of S1-S7 wins." The step-change question is **"what does the signature-layer Pareto frontier look like, and where does the thesis sit on it?"** Until that's measured, every downstream experiment is bounded by an unknown, and every cycle output underreports what's actually being learned.

## Why this reframes everything

The reflection in distance_function landed on "SDOT semantics vs Hamming semantics" as the axis. That was a substrate-surface reading: *which of the primitives the substrate already has is the right distance?* It was falsified because all per-trit distances on this signature are near-equivalent. The deeper reading: **the signature is the binding constraint, and the substrate's variation-within-signature is too small to matter.**

Looking again at the evidence:
- MNIST: Hamming 97.23%, E1 SDOT 97.10%, E2 weighted 97.01%, E3 block 95.92%. Spread: 1.3pp.
- Fashion: Hamming 87.78%, E1 SDOT 85.12%, E2 weighted 87.56%, E3 block 87.69%. Spread: 2.7pp.
- CIFAR-10: Hamming 44.68%, E1 SDOT 41.42%, E2 weighted 44.32%, E3 block 44.61%. Spread: 3.3pp.

The spread among distances IS the signature-layer's statement of "how much of the information content does distance-choice move you among." It's small. The bit-budget hypothesis is the bigger lever: does *adding bits* (more channels, more dims) move the ceiling by more than *choosing bits* (which scorer)?

We do have one data point suggesting bits matter: adding gradients to the base intensity signature was the difference between ~40% and 44-47% on CIFAR-10 (per direct_lsh earlier measurements). That's +4-7pp from more channels. But we never measured whether adding MORE channels (S1's multi-scale) continues the trend or has hit the asymptote.

## Resolved tensions

**T1 (S1 cheap vs Pareto first):** Pareto first. If we add bits and accuracy climbs further, S1 is vindicated. If it plateaus, S1 won't save us. Cost is similar (few hours), information value is strictly higher for Pareto. Pareto subsumes the S1 decision.

**T2 (S3 ensemble vs noise check):** Run the multi-seed verification first. The entire premise of S3 depends on whether CIFAR E3 +0.56pp is a real axis. Cheap resolver — literally 3 reruns with different seeds — and it either confirms or refutes. Confirmation: S3 becomes worth building. Refutation: the "distance complementarity" axis closes cleanly.

**T3 (substrate-native vs scaffold):** the reflection sharpens this. S1-S6 are all within the same "direct ternary quantization + local distance" family — the same family the last two cycles tested. S7 (scaffold learned encoder) is outside that family. If the bit-budget Pareto plateaus and all S1-S6 experiments stay within 2pp of pair-IG, then the FAMILY is bounded, and S7 becomes informative NOT because it wins but because it distinguishes "family-bounded" from "approach-bounded." That's a thesis-level data point: does any ternary encoding beat direct quantization?

**T4 (realistic vs ambitious target):** reframed. The target is no longer "beat SSTT on CIFAR-10." The target is **"measure the ternary-signature Pareto frontier for CIFAR-10 and place the thesis on it."** Outcome: either (a) substrate-native experiments close the gap and the thesis is vindicated, (b) substrate-native experiments plateau below SSTT and a scaffold experiment tells us whether the substrate or the approach is bounded, or (c) some unexpected configuration leaps past SSTT and rewrites the roadmap. All three are informative.

**T5 (parallel vs sequential exploration):** clarified. Run the two cheap pre-experiments (bit-budget Pareto, multi-seed E3 verification) in parallel — they touch different parts of the tool and have no dependency. Then use their results to gate the larger S1-S6 selection.

**T6 (data-selection axis):** deferred, not dismissed. Training-set curation is an orthogonal lever that neither this cycle nor the previous two have touched. Parking it for a separate cycle; the signature-layer pass must come first because it's upstream of data selection's effectiveness.

## Hidden assumptions

- **Assumption: CIFAR-10 is the right target.** The project's thesis is about base-3-native primitives on Apple Silicon; CIFAR-10 is one benchmark. If CIFAR-10's class-boundaries require features that are fundamentally base-2-pixel-native (edge orientations, multi-scale textures defined in RGB intensity space), no amount of ternary re-encoding might close the gap. **The thesis may be better served by finding a benchmark that is *natively* base-3-structured (graph data, categorical features, structural equivalence problems) and showing dominance there.**

- **Assumption: "the signature has a ceiling" means direct quantization has a ceiling.** Direct quantization is ONE way to make a ternary signature. Block-level constructions (S2, S5) are different signatures, not different distances. If the ceiling is on direct quantization specifically, block-level constructions could break through.

- **Assumption: 18 measurements in the distance cycle are independent enough to draw MCS conclusions from.** They're not — many share preprocessing pipelines, share pair-IG downstream, share bucket filters. The +0.56pp CIFAR E3 Selective might be legitimately surprising-signal, or might be dependent on E3's specific filtering interaction with pair-IG. Multi-seed doesn't fully resolve this; a more careful replication design would.

## What I now understand

1. **The immediate next move is two cheap measurements, not an experiment.** Bit-budget Pareto (direct_lsh on CIFAR-10 at varying total_dim) and multi-seed E3 verification. Together ~4-6 hours. Information value: extremely high. They gate everything downstream.

2. **S1-S6 should not be picked before the pre-measurements.** Any picking now is pre-data commitment, which is what the last cycle already did and which contributed to E1 being the primary bet that failed.

3. **S7 (scaffold learned encoder) should be explicitly framed as a thesis-calibration experiment, not a ship candidate.** Its purpose is to measure whether the substrate's ceiling is below SSTT because the substrate is limited or because direct quantization is limited. If a scaffold encoder on the same hardware reaches ~53%, direct quantization is the ceiling, and base-3-native future work must either accept that or find a block-level direct encoding that matches. If a scaffold encoder also lands at ~47%, the substrate has a real bound and the thesis needs a different benchmark.

4. **The thesis is thesis-tested on CIFAR-10 but not thesis-defined by it.** NORTH_STAR explicitly says "MNIST is now effectively settled" and "a harder benchmark bed is still required to force a non-cooperative comparison." CIFAR-10 was that harder bed. If the substrate can't clear it, the cycle should output "consider whether CIFAR-10 is the right forcing function, or whether a base-3-native benchmark (graph embedding, categorical retrieval, structural protein classification) would be more thesis-appropriate."

5. **The most-valuable next cycle output is a decision framework, not an experiment.** Given the Pareto measurement and the multi-seed verification, there are three clean branches:
    - Pareto climbs + complementarity confirmed → S1 and S3 in parallel, aim for +3-5pp on CIFAR.
    - Pareto plateaus + complementarity confirmed → S2/S5 (block-level signatures) primary, S3 as cheap addition, aim for +5-7pp via structural departure from direct quantization.
    - Pareto plateaus + complementarity noise → S7 scaffold as calibration, then either find a new benchmark or commit to "substrate is bounded by direct quantization ceiling on CIFAR-10."

6. **The cycle's deliverable is the pre-measurement + the decision framework, not a chosen experiment.** Running S1-S7 without the pre-measurements is the same mistake the last cycle made.

## Open residuals

- **R1: naming the pre-measurements.** Bit-budget Pareto = "P1"; multi-seed E3 verification = "P2". These are not full LMM cycles; they're measurement gates informing this cycle's synthesis.

- **R2: what if both P1 and P2 are inconclusive?** (Pareto is noisy, complementarity verification is marginal.) Fallback decision: default to S3 (cheapest remaining action with preliminary positive signal) and run it as a low-stakes follow-up rather than a step-change.

- **R3: timeline.** P1 + P2 together ≈ one afternoon of compute. Branch selection = one evening. Chosen experiment from S1-S7 = several days to a week. Full cycle from "start P1" to "chosen experiment complete and measured" = realistic at 1-2 weeks.
