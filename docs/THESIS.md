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

**Current empirical state (updated 2026-04-15):** the routing-vs-dense gap on MNIST has been closed and inverted relative to the first measurement. Initially, routing-native (Trit Lattice LSH centroid, 81.40%) underperformed dense-on-ternary-storage (97.61%) by 16 points — this was the "open empirical question" for the first half of the rebuild. After the routing reframes (filter-ranker / information leverage / signature-as-address / multi-table composition), the routed production consumer reaches **97.24%** at N_PROJ=16 on deskewed MNIST — the first routed architecture in the project to exceed 97%. At M=64 the same consumer reaches 97.31%. At matched total signature bits, multi-table routed bucket LSH matches or slightly beats the pure-signature scaling curve (M=32 at 512 bits is +0.18 over pure N_PROJ=512 at 97.06%).

The dense-scaffolding era headline of 97.99% (Axis 2, `mnist_routed_knn`) was measured with routing primitives inside an O(N_train) dense outer loop — a compression win against dense L1, not a routing-architecture win. The Axis 5 / Axis 6 reframes replaced that dense shape with a signature-as-address bucket index + multi-table composition, which reaches similar accuracy at a fraction of the wall-time cost while honoring the routing contract end-to-end.

**What this changes for the thesis:** MNIST has saturated under routing. The gap between routing-native and dense-on-ternary-storage is now within ±1 accuracy point across the interesting configurations, and **routing wins on cost at matched accuracy** (multi-table bucket is ~2× faster than dense scan at matched bits). MNIST is now effectively settled as a thesis test — both architectures can reach ~97%, routing does so at lower cost. A harder benchmark bed is still required to force a non-cooperative comparison; see §4.

## 3. Which consumer is M4T being built for?

**Primary production consumer (as of 2026-04-15):** `tools/mnist_routed_bucket_multi.c` — the multi-table routed bucket LSH. Built on `libglyph`, which itself sits on `libm4t`. Uses the Trit Lattice signature as a *hash-table address* rather than as an operand: training prototypes are indexed by their packed-trit signatures into a sorted bucket table; query time is binary search + ternary multi-probe (O(1) amortized in N_train) followed by a routed summed-distance resolver over the candidate union. Zero dense scans at the application level.

**Companion consumer (single-table variant):** `tools/mnist_routed_bucket.c` — same signature-as-address architecture with M=1 table and an independent H2+H3+H4 resolver. Retained as the Axis 5 reference and as a simpler test of the library's single-table path.

**Why these consumers.** They exercise the full substrate surface — `m4t_ternary_matmul_bt`, `m4t_route_threshold_extract`, `m4t_popcount_dist`, the trit pack/unpack path — and they do so inside the routing-architecture shape the thesis demands (signature as address, not operand). They are the first Glyph consumers where wall-time cost is **independent of N_train** in the common case, which is what the hardware-up story has to mean in practice.

**Previously named provisional consumer (`mnist_trit_lattice.c`).** Retained in `tools/` as research scaffolding but no longer the thesis-bearing consumer. It uses the Trit Lattice as a centroid-based classifier, which reaches ~58-81% depending on configuration and is useful as an atomic probe of the centroid path, not as the production surface.

**Primitive-surface rule.** New substrate primitives justify themselves by concrete demand from the production consumers. "We'll probably need it" is not a justification. If `mnist_routed_bucket_multi` doesn't call it, it doesn't land in M4T or libglyph.

**Future/candidate consumers.** Listed here to keep the future scope explicit, NOT to justify building for them pre-emptively.

- **Multi-table fused-filter bucket.** Concatenate H1+H2 into 8-byte bucket keys to apply the information-leverage rule inside the bucket architecture. Requires `uint64_t` bucket keys in libglyph (currently named as a limitation in `src/glyph_bucket.h`). Expected to track the Axis 4d dense fused-filter result (88.44%) at single-table cost.
- **Routing transformer.** Multi-head routing instead of dense attention; k-of-T tile dispatch instead of dense FFN. Would need GELU/softmax LUTs pulled back from archive.
- **A base-3-native architecture not yet drawn.** A routing architecture purpose-built for the lattice thesis rather than adapted from a dense design. The end-game shape. Unknowable now (NORTH_STAR §5).

**When the consumer changes, this section is rewritten**, the primitive-surface rule is re-anchored, and the archive question (LUTs, wide MTFP39 paths, etc.) becomes actionable based on the new consumer's demand.

## 4. Benchmark bed

**STILL OPEN, but MNIST is now saturated under routing and the cooperative-task reading is explicit.**

Updated status (2026-04-15): MNIST has been driven to 97.24% under a pure routing architecture at N_PROJ=16 (multi-table bucket SUM), within 0.75 points of the best dense scaffolding result on the same substrate and matching the pure-signature scaling curve at equivalent total bits. Classical L1 k-NN on deskewed pixels is 97.16%. The three results cluster within 1 point of each other. **MNIST can no longer separate routing-native from dense-on-ternary-storage** — both approaches saturate around the same ceiling. Cost differs (routed bucket is ~2× faster than the equivalent dense scan at matched accuracy), but "routing beats dense" on accuracy is structurally unverifiable on this bed.

The Axis 4 / Axis 5 / Axis 6 empirical work settled this as a cooperative task: both architectures reach the MNIST ceiling. The thesis still needs a non-cooperative bed.

Candidate benches, ordered by how much they force the thesis to earn its keep:

| Bench | Why it's a harder test | Why it's achievable |
|---|---|---|
| CIFAR-10 | Higher-dim, harder, non-trivial for classical methods | Small enough for small-compute research; immediate next step |
| Long-tailed classification (e.g. iNaturalist subset) | Class imbalance stresses routing decision quality specifically | Routing has a natural story here |
| Char-level text classification | LSH over n-gram signatures is already a good fit for ternary | Tiny models, fast iteration |
| Sparse-signal / one-shot tasks | Routing's "pick the right prototype" geometry matches the task | Benchmarks exist |

**Next empirical step:** port `libglyph` + the routed bucket consumers to CIFAR-10. The dataset loader in `src/glyph_dataset.c` is currently MNIST-specific (IDX format); CIFAR-10 would need a new loader. Everything downstream (sig builder, bucket index, multi-probe, resolvers, CLI) is dataset-agnostic and should work without modification. The test is whether multi-table routed bucket reaches parity with pure-signature scaling on a harder bed, or whether the resolver gap widens — which would diagnose a real limit of random ternary projections on non-cooperative data.

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
- That dense-on-ternary-storage is a worse architecture. Both architectures now cluster near the MNIST ceiling (routing-native 97.24%, dense scaffolding 97.99%, classical dense-pixel k-NN 97.16%). The thesis claim is that routing-native is *more aligned with the hardware* — which on measured wall-time at matched accuracy is now empirically true on MNIST (routed bucket at M=32 runs ~2× faster than dense N_PROJ=512 scan at the same ~97% ceiling) — but the claim has to be re-earned on each new benchmark bed, not assumed from the MNIST result.

## 7. Relationship to M4T

- M4T (`m4t/docs/M4T_SUBSTRATE.md`) is the substrate. Routing-first by construction. Answers: *what primitives exist, what invariants they hold, what the hardware anchor is.*
- This doc is the thesis. Answers: *what the substrate is for, what consumer it serves, what would prove or disprove the claim, and what benchmark forces the claim to face its null.*

The two docs are deliberately separate. Conflating them was part of what got the previous era into trouble — substrate decisions got driven by ideological commitments that belonged one layer up.

## 8. Open items

- **C1.** ~~Choose a consumer architecture (§3).~~ **Resolved (2026-04-15):** the multi-table routed bucket (`mnist_routed_bucket_multi`) is the primary production consumer; the single-table variant (`mnist_routed_bucket`) is the Axis 5 reference. Both live on `libglyph` over `libm4t`. `m4t_mtfp_nonlinear.c` + `m4t_mtfp_tables.c` remain in `archive/` pending a routing transformer consumer.
- **B1.** Choose a benchmark bed beyond MNIST (§4). Still open. MNIST is now saturated under routing and no longer separates the thesis from its null. CIFAR-10 is the natural next step; requires a new dataset loader in `src/glyph_dataset.c` (the rest of libglyph is dataset-agnostic).
- **M1.** Discharge hardware-alignment claims with measurement (§5). Partially addressed: wall-time measurements show routed bucket M=32 running at ~1.92 ms/query vs dense N_PROJ=512 scan at ~4.0 ms/query at matched accuracy. SDOT / TBL / VCNT utilization-per-cycle measurements are still unrun.
- **A1.** (New.) Generalize libglyph's bucket index to `uint64_t` keys so the fused-filter variant (concatenated H1+H2 signatures) can be tested in the routed architecture without reintroducing dense scans.

## 9. Traceability

- Separation from M4T substrate: LMM cycle on §14 (`journal/seven_open_decisions_{raw,nodes,reflect,synthesize}.md`, 2026-04-14). Former §14.6 (nonlinearity scope) → here as C1. Former §14.7 (benchmark bed) → here as B1.
