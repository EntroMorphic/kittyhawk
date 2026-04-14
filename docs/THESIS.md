---
title: Glyph Thesis — routing-first, hardware-up
status: open — consumer/thesis scope, not substrate scope
companion: m4t/docs/M4T_SUBSTRATE.md
---

# Glyph Thesis

This document owns the questions that sit *outside* M4T — what the substrate is in service of, and how we know the approach is working. M4T is routing-first by construction; this doc is where the routing thesis itself has to earn its keep.

The separation matters. M4T (the substrate) serves whatever consumer is built on it. This thesis doc is the consumer brief: which problems the substrate exists to beat, and on what terms.

---

## 1. The thesis in one sentence

Routing is a first-class primitive over dense computation. On Apple M-series silicon, a routing-native substrate (M4T) riding hardware shapes that already exist (TBL, masked-VCNT, SDOT) should match or beat dense-over-ternary-storage on problems where lattice geometry carries the signal — without bending hardware into shape.

## 2. What would falsify the thesis

A falsification-first framing keeps this honest. The thesis is falsified if:

1. On problems where lattice geometry plausibly matters, routing-native accuracy consistently underperforms dense-on-ternary-storage by margins that aren't explained by unoptimized code.
2. Routing-native achieves parity but only by importing dense-shaped primitives (matmul, layernorm, etc.) into the hot path — i.e., the routing story was a reskin.
3. Hardware measurements show the routing path doesn't actually ride the native instructions (SDOT, TBL, VCNT) at the intended utilization — meaning the "hardware-aligned" claim is aspirational rather than measured.

The current empirical state: on MNIST, routing-native (Trit Lattice LSH, 81.40%) underperforms dense-on-ternary-storage (97.61%) by 16 points. **This gap is the open empirical question.** It is the reason the benchmark bed matters (§4).

## 3. Which consumer is M4T being built for?

**Provisional primary consumer (as of 2026-04-14):** `tools/mnist_trit_lattice.c` — the Trit Lattice LSH tool. Pure routing: ternary projections, L1 distance, coarse-to-fine refinement over MNIST. Does not need nonlinearities. Uses `m4t_mtfp_ternary_matmul_bt` for the random ternary projection (Law #7 — ternary projections applied to MTFP data, NOT ternarized observations).

**Why this consumer provisionally.** It's the one routing-native workload kept out of archive. It exercises `m4t_ternary_matmul`, the routing-first surface of `m4t_route`, and the mantissa-layer vec primitives in `m4t_mtfp`. That's enough of the substrate surface to validate that first-light primitives compose correctly. It is NOT the end-game consumer — MNIST is a base-2-framed problem (see §4) and Trit Lattice LSH on MNIST is a diagnostic, not a thesis test.

**Primitive-surface rule while this is the provisional consumer.** New substrate primitives justify themselves by concrete demand from this consumer. "We'll probably need it" is not a justification. If the LSH tool doesn't call it, it doesn't land in M4T.

**Future/candidate consumers.** Listed here to keep the future scope explicit, NOT to justify building for them pre-emptively.

- **Routing transformer.** Multi-head routing instead of dense attention; k-of-T tile dispatch instead of dense FFN. Would need GELU/softmax LUTs pulled back from archive.
- **A base-3-native architecture not yet drawn.** A routing architecture purpose-built for the lattice thesis rather than adapted from a dense design. The end-game shape. Unknowable now (NORTH_STAR §5).

**When the consumer changes, this section is rewritten**, the primitive-surface rule is re-anchored, and the archive question (LUTs, wide MTFP39 paths, etc.) becomes actionable based on the new consumer's demand.

## 4. Benchmark bed

**OPEN (former §14.7 → here).** MNIST cannot separate the routing thesis from its null. Classical L1 k-NN on deskewed pixels was near 97% decades ago; "zero-float matches float on MNIST" demonstrates that MTFP storage is viable, not that routing beats dense. A harder bed is required.

Candidate benches, ordered by how much they force the thesis to earn its keep:

| Bench | Why it's a harder test | Why it's achievable |
|---|---|---|
| CIFAR-10 | Higher-dim, harder, non-trivial for classical methods | Small enough for small-compute research |
| Long-tailed classification (e.g. iNaturalist subset) | Class imbalance stresses routing decision quality specifically | Routing has a natural story here |
| Char-level text classification | LSH over n-gram signatures is already a good fit for ternary | Tiny models, fast iteration |
| Sparse-signal / one-shot tasks | Routing's "pick the right prototype" geometry matches the task | Benchmarks exist |

**Decision deferred** until the numeric core is rebuilt on the settled substrate answers. When the core is in place, the benchmark choice is the first experiment.

## 5. What "hardware-aligned" has to mean empirically

Claims of hardware alignment (in README, substrate spec, and this doc) need to be discharged by measurement once code exists. Specifically:

- **SDOT utilization.** The ternary matmul path should show close to 1 SDOT op/cycle on M-series big cores in the hot loop. Measured, not asserted.
- **TBL throughput.** Trit ops should measure at TBL-native rates.
- **Cache behavior.** The SoA mantissa/exponent layout should show clean prefetch on the mantissa stream and warm-in-L1 behavior on the exponent stream.
- **Text size.** The `.text` budget discipline (24 KB previously; target TBD post-rebuild) stays as a forcing function against feature creep.

If these don't measure out, the hardware-up story is aspirational and needs revision.

## 6. What the substrate does *not* promise

- That the thesis is correct. M4T is a hypothesis-testing instrument, not a hypothesis.
- That any specific benchmark will be beaten. The substrate serves; the consumer claims.
- That dense-on-ternary-storage is a worse architecture. The 97.61% MNIST result is real; the thesis claim is that routing-native is *more aligned with the hardware*, which is a different axis and has to be demonstrated on harder beds.

## 7. Relationship to M4T

- M4T (`m4t/docs/M4T_SUBSTRATE.md`) is the substrate. Routing-first by construction. Answers: *what primitives exist, what invariants they hold, what the hardware anchor is.*
- This doc is the thesis. Answers: *what the substrate is for, what consumer it serves, what would prove or disprove the claim, and what benchmark forces the claim to face its null.*

The two docs are deliberately separate. Conflating them was part of what got the previous era into trouble — substrate decisions got driven by ideological commitments that belonged one layer up.

## 8. Open items

- **C1.** Choose a consumer architecture (§3). Until chosen, `m4t_mtfp_nonlinear.c` + `m4t_mtfp_tables.c` remain in `archive/`.
- **B1.** Choose a benchmark bed (§4). Until chosen, "does routing beat dense" is an unanswered empirical question.
- **M1.** Discharge hardware-alignment claims with measurement (§5) once the substrate rebuild compiles and runs.

## 9. Traceability

- Separation from M4T substrate: LMM cycle on §14 (`journal/seven_open_decisions_{raw,nodes,reflect,synthesize}.md`, 2026-04-14). Former §14.6 (nonlinearity scope) → here as C1. Former §14.7 (benchmark bed) → here as B1.
