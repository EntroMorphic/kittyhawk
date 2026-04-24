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

**Current empirical state (updated 2026-04-20):** the routing-vs-dense gap on MNIST has been closed and inverted relative to the first measurement. Initially, routing-native (Trit Lattice LSH centroid, 81.40%) underperformed dense-on-ternary-storage (97.61%) by 16 points. After the routing reframes (filter-ranker / information leverage / signature-as-address / multi-table composition / direct ternary quantization), the routed production consumer (`direct_lsh`) reaches **97.18%** on MNIST, **87.95%** on Fashion-MNIST (beating SSTT's 86.54%), and **46.63%** on CIFAR-10 via GSH selective scoring. The multi-table bucket path (`mnist_routed_bucket_multi`) reaches **97.24%** at N_PROJ=16 on deskewed MNIST. At matched total signature bits, multi-table routed bucket LSH matches or slightly beats the pure-signature scaling curve (M=32 at 512 bits is +0.18 over pure N_PROJ=512 at 97.06%).

The dense-scaffolding era headline of 97.99% (Axis 2, `mnist_routed_knn`) was measured with routing primitives inside an O(N_train) dense outer loop — a compression win against dense L1, not a routing-architecture win. The Axis 5 / Axis 6 reframes replaced that dense shape with a signature-as-address bucket index + multi-table composition, which reaches similar accuracy at a fraction of the wall-time cost while honoring the routing contract end-to-end.

**What this changes for the thesis:** MNIST has saturated under routing. The gap between routing-native and dense-on-ternary-storage is now within ±1 accuracy point across the interesting configurations, and **routing wins on cost at matched accuracy** (multi-table bucket is ~2× faster than dense scan at matched bits). MNIST is now effectively settled as a thesis test — both architectures can reach ~97%, routing does so at lower cost. A harder benchmark bed is still required to force a non-cooperative comparison; see §4.

**2026-04-23 → 04-24 — substrate extensions and benchmark pivot:**

1. **Routed autodiff MVP (`libtrain.a`)**: scalar forward/backward through `tlinear` and `rroute` primitives, hysteresis-aware re-quantization, 5-test suite. 2-class toy converges (96.50% mean, σ=2.79pp across 5 seeds). **The 10-class toy fails by 47pp** (routed 34% vs plain ternary linear 91% on the same data/trainer) — a structural finding: frozen-U selection-only routing causes MoE-style expert collapse. Fixing requires learned routing (soft/differentiable top-k), load-balancing loss, or per-tile specialization pressure. The MVP's artifacts (hysteresis requant, STE behavior monitor, principled-init derivations) are reusable by future trainers. See `journal/routed_autodiff_closeout.md` and `train/README.md`.

2. **Benchmark pivot**: `base3_benchmarks` LMM cycle demoted image-classification canon (MNIST/Fashion/CIFAR) from primary benchmark to regression-guard. "Base-3 native" decomposes into three criteria — ternary-representable input, routing-load-bearing task, inspectability-credited evaluation — which the image canon fails on all three. Primary direction shifted to ternary-state board-game position evaluation (Go first). See `journal/base3_benchmarks_closeout.md`.

3. **Substrate distance finding**: Go probe RED at first pass (raw Hamming phase-ID 40.40%, density-only 98.28%). Follow-up `substrate_distance_refinement` cycle identified that raw int8 trit Hamming has a **density-scaling bias** — `hamming_norm(a, b) = H · 1024 / (|a|₀ + |b|₀ + 1)` fixes it. Phase-ID lifts to 85.40% (position-split) / **88.40%** (game-split — red-team confirmed no within-game leakage effect). Raw Hamming also already achieved **413× random lift on same-game retrieval** (density-controlled task), showing the substrate was never structurally blind. **Red-team nuance**: the fix is mostly density-recovery, not a new structural axis; image-pipeline measurement (MNIST/Fashion/CIFAR under `hamming_norm`) is the decisive gate for whether this becomes a substrate primitive or stays a Go-specific refinement. See `journal/substrate_distance_refinement_closeout.md`.

**What this changes for the thesis:** the falsification framing in §2 remains intact. The MVP's expert-collapse finding is evidence that *frozen-gate* routing is insufficient for multi-class problems — not that *routing itself* is insufficient. The `hamming_norm` finding is evidence that substrate measurements to date have been partially a metric artifact; re-measuring the MNIST/Fashion/CIFAR numbers under `hamming_norm` is queued as a retrospective calibration. The "end-game unknowable" clause from NORTH_STAR continues to hold — it just got sharper: the next decisive measurement is whether `hamming_norm` helps or hurts on image canon, which will shape both the `routed_go` trainer priority and how we read the prior CIFAR representation-tax results.

## 3. Which consumer is M4T being built for?

**Primary production consumer (as of 2026-04-20):** `tools/direct_lsh.c` — direct ternary quantization with Hierarchical Trit Lattice LSH, GSH (Global Signature Hash) confidence, and pair-IG selective scoring. Built on `libglyph` over `libm4t`. Each trit represents a specific input dimension (pixel or gradient) — no random projections. Supports MNIST, Fashion-MNIST, and CIFAR-10 via `glyph_dataset_load_auto`. Production best across all three datasets.

**Companion consumer (multi-table bucket, random projection path):** `tools/mnist_routed_bucket_multi.c` — multi-table routed bucket LSH using random ternary projections and signature-as-address indexing. Retained as the Axis 6 reference and the first routed architecture to break 97% on MNIST. Uses the Trit Lattice signature as a hash-table address; query time is binary search + ternary multi-probe (O(1) amortized in N_train) followed by a routed summed-distance resolver.

**Companion consumer (single-table variant):** `tools/mnist_routed_bucket.c` — same signature-as-address architecture with M=1 table and an independent H2+H3+H4 resolver. Retained as the Axis 5 reference.

**Previously named provisional consumer (`mnist_trit_lattice.c`).** Retained in `tools/` as research scaffolding but no longer the thesis-bearing consumer. It uses the Trit Lattice as a centroid-based classifier, which reaches ~58-81% depending on configuration and is useful as an atomic probe of the centroid path, not as the production surface.

**Primitive-surface rule.** New substrate primitives justify themselves by concrete demand from the production consumers. "We'll probably need it" is not a justification. If `direct_lsh` doesn't call it, it doesn't land in M4T or libglyph.

**Future/candidate consumers.** Listed here to keep the future scope explicit, NOT to justify building for them pre-emptively.

- **Multi-table fused-filter bucket.** Concatenate H1+H2 into 8-byte bucket keys to apply the information-leverage rule inside the bucket architecture. Requires `uint64_t` bucket keys in libglyph (currently named as a limitation in `src/glyph_bucket.h`). Expected to track the Axis 4d dense fused-filter result (88.44%) at single-table cost.
- **Routing transformer.** Multi-head routing instead of dense attention; k-of-T tile dispatch instead of dense FFN. Would need GELU/softmax LUTs pulled back from archive.
- **A base-3-native architecture not yet drawn.** A routing architecture purpose-built for the lattice thesis rather than adapted from a dense design. The end-game shape. Unknowable now (NORTH_STAR §5).

**When the consumer changes, this section is rewritten**, the primitive-surface rule is re-anchored, and the archive question (LUTs, wide MTFP39 paths, etc.) becomes actionable based on the new consumer's demand.

## 4. Benchmark bed

**Partially resolved (updated 2026-04-20).** Three datasets now measured under the direct ternary quantization production consumer (`direct_lsh`):

| Dataset | Glyph | SSTT | Verdict |
|---|---|---|---|
| MNIST | 97.18% | 97.53% | Tied (within noise) |
| Fashion-MNIST | 87.95% | 86.54% | **Glyph wins** (+1.41pp) |
| CIFAR-10 | 46.63% | ~53% | Gap is 6.4pp (distance metric ceiling) |

MNIST is settled — both architectures saturate around the same ceiling; routing wins on cost. Fashion-MNIST is the first dataset where Glyph outperforms a published ternary baseline on accuracy. CIFAR-10 is the active frontier: the 6.4pp gap traces to per-trit Hamming distance vs. SSTT's pattern-level block scoring (see Axis 8 in FINDINGS.md).

Remaining candidate benches for harder tests:

| Bench | Why it's a harder test | Why it's achievable |
|---|---|---|
| Long-tailed classification (e.g. iNaturalist subset) | Class imbalance stresses routing decision quality specifically | Routing has a natural story here |
| Char-level text classification | LSH over n-gram signatures is already a good fit for ternary | Tiny models, fast iteration |
| Sparse-signal / one-shot tasks | Routing's "pick the right prototype" geometry matches the task | Benchmarks exist |

**Next empirical step:** close the CIFAR-10 distance-metric gap. The input representation (direct quantization + gradients) reaches 50.2% brute-force; the remaining gap is in the distance function. Pattern-level distance (block encoding with correlation-aware scoring) is the identified path forward.

## 5. What "hardware-aligned" has to mean empirically

Claims of hardware alignment (in README, substrate spec, and this doc) need to be discharged by measurement once code exists. Specifically:

- **SDOT utilization.** The ternary matmul path should show close to 1 SDOT op/cycle on M-series big cores in the hot loop. Measured, not asserted.
- **TBL throughput.** Trit ops should measure at TBL-native rates.
- **Cache behavior.** The SoA mantissa/exponent layout should show clean prefetch on the mantissa stream and warm-in-L1 behavior on the exponent stream.
- **Text size.** The `.text` budget discipline (24 KB previously; target TBD post-rebuild) stays as a forcing function against feature creep.

If these don't measure out, the hardware-up story is aspirational and needs revision.

**First measurements (2026-04-20, Apple M3, `m4t/tools/m4t_profile.c`):**

| Primitive | Config | Throughput | Notes |
|---|---|---|---|
| **SDOT** (`mtfp4_sdot_matmul_bt`) | M=1, K=512, N=16 | **60.3 Gops/s** | Sustained at 55-60 Gops/s across sizes; 8192 MACs in 136 ns |
| **SDOT** | M=16, K=256, N=32 | **54.1 Gops/s** | 131K MACs in 2.4 μs; scales linearly |
| **TBL** (`trit_mul`) | n=1024 trits | **41.6 Gtrits/s** | Flat throughput 38-42 Gtrits/s from 64 to 4096 trits |
| **VCNT** (`popcount_dist`) | N_PROJ=16 (4B) | **225 Mops/s, 4.4 ns** | Production hot path; scalar __builtin_popcount |
| **VCNT** | N_PROJ=64 (16B) | **406 Mops/s, 2.5 ns** | NEON loop entry point; fastest tier |
| **Masked-VCNT** (`trit_counts`) | n=1024 | **131 Gtrits/s** | Sustained reduction throughput |
| **MTFP19 matmul** (`ternary_matmul_bt`) | M=1, K=512, N=16 | **5.2 Gops/s** | TBL decode + scalar MAC; ~12× slower than SDOT path |

**What this means for the thesis:** The SDOT path sustains 55-60 Gops/s on M3 — confirming that the int8 ternary matmul rides SDOT at near-peak throughput. TBL-based trit ops sustain 40+ Gtrits/s, flat across vector lengths, confirming native-rate dispatch. The VCNT popcount_dist path at N_PROJ=16 (production) completes in 4.4 ns — the routing decision is essentially free relative to the data-fetch cost.

The MTFP19 matmul (5.2 Gops/s) is ~12× slower than the SDOT path because it uses scalar MAC with TBL trit decode. This confirms that the SDOT-native MTFP4 path is the correct choice for matmul-heavy workloads, and the MTFP19 path is appropriate only for setup-time operations.

Cache behavior and .text budget remain unmeasured (require Instruments profiling, not microbenchmark).

## 6. What the substrate does *not* promise

- That the thesis is correct. M4T is a hypothesis-testing instrument, not a hypothesis.
- That any specific benchmark will be beaten. The substrate serves; the consumer claims.
- That dense-on-ternary-storage is a worse architecture. Both architectures now cluster near the MNIST ceiling (routing-native 97.24%, dense scaffolding 97.99%, classical dense-pixel k-NN 97.16%). The thesis claim is that routing-native is *more aligned with the hardware* — which on measured wall-time at matched accuracy is now empirically true on MNIST (routed bucket at M=32 runs ~2× faster than dense N_PROJ=512 scan at the same ~97% ceiling) — but the claim has to be re-earned on each new benchmark bed, not assumed from the MNIST result.

## 7. Relationship to M4T

- M4T (`m4t/docs/M4T_SUBSTRATE.md`) is the substrate. Routing-first by construction. Answers: *what primitives exist, what invariants they hold, what the hardware anchor is.*
- This doc is the thesis. Answers: *what the substrate is for, what consumer it serves, what would prove or disprove the claim, and what benchmark forces the claim to face its null.*

The two docs are deliberately separate. Conflating them was part of what got the previous era into trouble — substrate decisions got driven by ideological commitments that belonged one layer up.

## 8. Open items

- **C1.** ~~Choose a consumer architecture (§3).~~ **Resolved (2026-04-20):** `direct_lsh` is the primary production consumer (direct ternary quantization + GSH selective scoring, all three datasets). The multi-table bucket (`mnist_routed_bucket_multi`) is retained as the Axis 6 reference. Both live on `libglyph` over `libm4t`.
- **B1.** ~~Choose a benchmark bed beyond MNIST (§4).~~ **Partially resolved (2026-04-20):** CIFAR-10 and Fashion-MNIST ported and measured. CIFAR-10 dataset loader (`glyph_dataset_load_cifar10`) and auto-detection (`glyph_dataset_load_auto`) implemented. Active frontier: closing the CIFAR-10 distance-metric gap (46.63% vs ~53% SSTT).
- **M1.** ~~Discharge hardware-alignment claims with measurement (§5).~~ **Partially resolved (2026-04-20):** `m4t/tools/m4t_profile.c` measures SDOT at 55-60 Gops/s, TBL trit ops at 40+ Gtrits/s, VCNT popcount_dist at 225-406 Mops/s on M3. All three instruction classes run at or near native throughput. Cache behavior and .text budget remain unmeasured (require Instruments).
- **A1.** (New.) Generalize libglyph's bucket index to `uint64_t` keys so the fused-filter variant (concatenated H1+H2 signatures) can be tested in the routed architecture without reintroducing dense scans.

## 9. Traceability

- Separation from M4T substrate: LMM cycle on §14 (`journal/seven_open_decisions_{raw,nodes,reflect,synthesize}.md`, 2026-04-14). Former §14.6 (nonlinearity scope) → here as C1. Former §14.7 (benchmark bed) → here as B1.
