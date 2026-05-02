---
cycle: gesh_design
phase: NODES
date: 2026-05-01
scope: extract discrete points and tensions from RAW
---

# Nodes — gesh_design

## Node 1 — The architectural-unification claim is the load-bearing thing
"Attention-plus-FFN collapsed into routed retrieval from a frozen ternary store" is a structural claim about what attention IS. If true, it's the substrate's THESIS Part B demonstration. The three Gs are mechanism; the unification is the thesis.
**Why it matters:** Cycle scope should be sized to test the unification, not to ship three mechanisms.

## Node 2 — Three Gs as orthogonal vs as manifestations of one principle
The design treats Global, Geometric, Gradient as three independent fixes for three named LSH failure modes. Empirically they may correlate strongly (a learned multiscale projection might capture all three at once) or they may decompose cleanly (each ablation drop matters independently).
**Tension with Node 1:** the unification claim doesn't need orthogonal Gs to be true. It needs one working example.

## Node 3 — "Frozen bank" is the central architectural commitment, not an implementation detail
Most attention literature trains keys+values jointly with everything else. Gesh freezes the bank. The design treats this as a backdrop assumption; it's actually the primary structural choice that distinguishes Gesh from attention.
**Why it matters:** load-bearing for the unification claim. If the frozen-bank constraint forces Gesh into more complexity than joint-trained attention, the unification is *worse* on substrate-purity grounds.

## Node 4 — The benchmark is unspecified and load-bearing
Every other open question in §6 (bank size, dim_g, dim_l, k_g, k_l, refresh frequency, projection capacity) is conditional on what task Gesh is being asked to do. The design proposes MNIST as toy, but MNIST has almost no global structure for stage 1 to exercise.
**Tension with Node 1:** the unification-claim test requires a benchmark with hierarchical/compositional structure. MNIST gives stage 2 a free pass.

## Node 5 — Six provisional design choices in §6, stacked
Region assignment, threshold count, projection sharing, refresh frequency, bank dimensions, build-all-three-from-start. Each "provisionally" answered; together a lot of un-validated commitments.
**Why it matters:** same shape as the tier-3 plan's pre-red-team draft. Discipline pattern: convert provisionals to pre-committed gates with falsifiability tests.

## Node 6 — MTFP4 vs ternary projections inverts the discipline default
The design defaults to MTFP4 projections "unless ternary proves sufficient." Discipline says: try the simpler shape first, escalate by measurement.
**Resolution candidate:** flip the default; ternary first, MTFP4 only on measured failure.

## Node 7 — Kernel surface may be smaller than claimed
`signature_match` is `popcount_dist` (existing) + `topk_abs` (existing). That's a libglyph composition, not a new substrate primitive. `threshold_extract2` is genuinely new but only earns substrate residency if asymmetric thresholds beat symmetric ones empirically.
**Tension with Node 1:** the unification claim is a consumer-side architectural claim; it doesn't need new substrate primitives to test.

## Node 8 — PCA initialization carries a quiet assumption
PCA on training data assumes the data manifold is what queries route against. If the bank is a designed prototype set, the bank's geometry differs from the training-data geometry. The design defers this question without flagging it as load-bearing.
**Why it matters:** if PCA prior is misaligned with the bank, the gradient has to *fight* the prior. That changes the training dynamics.

## Node 9 — Three training mechanisms with unstudied interaction
STE through ternary signature, Gumbel-softmax over Hamming distances, periodic refresh of region signatures. Each has its own stability story; their interaction is novel. The design proposes building all three at once.
**Tension with Node 5:** stacked provisionals. Test mechanisms in isolation before combining.

## Node 10 — `apply_signed` as Gesh's accumulator may not exercise the cross-exp generalization
Gesh's tile-combination step is correctly an accumulator pattern. But if all selected tiles at a given query share a block_exp (likely, if they come from the same bank built from one training pass), then `apply_signed`'s same-block-exp degenerate case suffices. The cross-exp kernel's general form isn't exercised.
**Why it matters:** the design claims Gesh "validates the cross-exp kernel's existence." That's only true if Gesh's call pattern is genuinely cross-exp.

## Node 11 — "Build all three Gs from the start" vs incremental
The design's build plan ships all three Gs simultaneously. The §7 ablation framing ("if Geometric drops 2 points, defer Geometric") suggests an alternative: build the simplest Gesh that could work, measure where it fails, add the G that addresses the specific failure.
**Tension with Node 5:** stacked provisionals. The §7 framing IS the resolution; the build plan should respect it.

## Node 12 — "What kind of consumer is Gesh?"
Research probe (testing the unification claim) or production infrastructure (a routing layer for downstream tasks)? Different framings imply different validation plans, different correctness bars, different tolerance for partial success.
**Why it matters:** the cycle scope changes by an order of magnitude depending on the framing.

## Node 13 — The design is shaped by attention's surface area, not the task's
"What does attention have that LSH doesn't?" is the implicit framing question. Hierarchical structure, manifold awareness, gradient-based training — Gesh adds them. But the substrate-claim is "routing-first base-3 beats base-2 attention," not "ternary attention with three extra mechanisms." Designing against attention's surface inherits attention's design pressures even where they don't apply.
**Why it matters:** the design's mental model has the wrong target. The target should be the task; the constraint should be the substrate.

## Tensions summary

- **T1: Unification claim vs three-Gs mechanism (Nodes 1, 2)** — the unification doesn't require all three Gs.
- **T2: Frozen bank vs joint-training default (Node 3)** — Gesh's central commitment is unsurfaced.
- **T3: Benchmark choice vs everything else (Node 4)** — un-pickable answers to design questions until benchmark is committed.
- **T4: Six stacked provisionals vs discipline (Node 5)** — same pre-red-team pattern as before.
- **T5: MTFP4 default vs ternary-first discipline (Node 6)** — inverted default.
- **T6: Substrate territory vs libglyph composition (Node 7)** — over-claimed surface.
- **T7: PCA-on-training vs PCA-on-bank (Node 8)** — quiet assumption, load-bearing.
- **T8: Three training mechanisms vs simplicity (Node 9)** — stacked novelty.
- **T9: Cross-exp validation vs same-block reality (Node 10)** — claim may overstate consumer-pattern.
- **T10: All-three vs incremental (Node 11)** — build plan vs ablation framing contradict each other.
- **T11: Research probe vs production infra (Node 12)** — scope ambiguity.
- **T12: Designed-against-attention vs designed-against-task (Node 13)** — wrong-target risk.
