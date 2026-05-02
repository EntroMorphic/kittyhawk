# Gesh — base-3 native routing layer

Lattice-native routing layer over a frozen ternary bank. Sits atop libm4t. The substrate's first measured consumer.

## Status

**Phase A.1 — forward pass + synthetic benchmark — ONLINE.** Bank construction, ternary projection, top-k tile retrieval, k-NN vote classification. Tests the pipeline end-to-end on a synthetic prototype-classification task. Hardened by red-team remediation (13 findings, 10 fixed, 3 deferred).

**Phase A.2 — lattice update training — ONLINE.** Coordinate-descent flips on projection trits to reduce loss. No STE. The lattice IS the geometry. **Measured across a multi-seed sig_dim sweep extended through sig_dim = 1024 (`docs/sweep_dims_results.md`, 5 seeds per cell, 12 dims): lattice update earns its complexity in the compression regime — gain plateaus at +8pp at sig_dim ∈ {16, 32}. Random ternary projection at sig_dim = D beats identity by +7pp robustly across seeds (hypothesis: implicit denoising; not mechanism-verified). Expansion saturates monotonically: at sig_dim = 1024 (16× input dim) random and trained converge to 98.6% ± 0.5pp with +0.0pp gain. The single-seed sweep's "+15pp peak" and "−2pp anomaly" were artifacts that didn't survive multi-seed averaging.**

**Phase B+ (gated)** — Global stage, MTFP4 escalation, etc. Each phase gated on a measured failure mode of the prior phase.

## Design lineage

- `Documents/GESH/GESH_DESIGN.md` — the original proposal (three Gs, multiscale, manifold-aware, gradient-trained).
- `journal/gesh_design_{raw,nodes,reflect,synthesize,closeout}.md` — LMM cycle that scoped Phase A:
  - Reframed against task demand (not attention's surface area).
  - Owner observation surfaced post-synthesize: **the lattice IS the geometry**. STE is a base-2 fix for a problem that doesn't exist in the lattice. Dropped from Phase A; replaced with lattice-update coordinate descent.
  - Phase A is stage-2 only with PCA-init ternary projections (no MTFP4) and hard top-k.

## What Phase A.1 ships

| File | Role |
|---|---|
| `bench/synth_proto.{h,c}` | Synthetic prototype-classification task generator. Class-prototype + per-trit noise, with informative-vs-noise dim split. Deterministic, closed-form. |
| `src/gesh_bank.{h,c}` | Frozen-bank construction (class-conditional ternary mean → bank tile per class). |
| `src/gesh_forward.{h,c}` | Forward pass: project query → top-k bank tiles by Hamming → k-NN vote classification. |
| `tests/test_synth_proto.c` | Verifies benchmark generator: class balance, label correctness, noise distribution. |
| `tests/test_gesh_bank.c` | Verifies bank construction: per-class tile, signature shape. |
| `tests/test_gesh_forward.c` | Verifies forward pass: untrained Hamming-NN baseline accuracy on synthetic task. |

## Synthetic prototype-classification benchmark

- **D = 64** input dimensions; first **K = 16** are class-informative, remaining 48 are noise.
- **C = 10** classes.
- For each class `c`, a true prototype `P_c ∈ {-1, 0, +1}^16` (random-balanced over informative dims; 0 over noise dims).
- Training/eval samples: `x_i = P_c + per_trit_noise(p=0.1)` over all D dims (informative dims get slight perturbation; noise dims get random ternary).
- Labels are `c`. Goal: classify `x_i → c`.

This task has a known solution (PCA on training data picks out the informative 16 dims). The Phase A measurement: does Hamming-NN over a frozen bank approach the ceiling? If yes, the pipeline works; lattice-update has a target to beat. If no, the pipeline is broken.

## Forward pass (Phase A.1)

```
input: x ∈ {-1, 0, +1}^D     (packed-trit signature)
parameters:
  R ∈ {-1, 0, +1}^{S × D}    (ternary projection, S < D, packed-trit)
  Bank tiles B ∈ {-1, 0, +1}^{T × S}  (T tiles, packed-trit)
  Tile labels c[T]            (which class each tile belongs to)

forward:
  s = ternary_project(R, x)   (S-dim ternary signature)
  d[t] = popcount_dist(s, B[t]) for t in [0, T)
  top_k = k_smallest_indices(d, k)
  vote = histogram(c[top_k])
  prediction = argmax(vote)
```

All integer arithmetic. No floats. Uses substrate primitives:
- `m4t_popcount_dist` (for Hamming).
- `m4t_pack_trits_1d` / `m4t_unpack_trits_1d` (between unpacked working buffers and packed tile signatures).

Plus two in-module helpers:
- Inline ternary sign-extract: per output trit, `(acc > 0) ? +1 : (acc < 0) ? -1 : 0` over the int32 dot-product accumulator. Functionally equivalent to `m4t_route_threshold_extract` with `tau = 0` but inline-friendly and per-cell rather than batch-shaped; promote to substrate if profile demands batch.
- Top-k-smallest insertion sort over T distances. O(T · top_k); fine for small top_k and moderate T (Phase A: T = C = 10).

Phase A.1 does not use `m4t_route_apply_signed` because there's no MTFP19 accumulation — classification produces a class label, not an MTFP19 vector. That changes when Phase B introduces stage 1 (region selection); `apply_signed` returns there.

## What this validates

Phase A.1's untrained Hamming-NN baseline on the synthetic task tells us:
- **Pipeline works:** the substrate composes into a working classifier. (Sanity check.)
- **Random-projection floor:** with random R, what accuracy does Gesh hit? This is the floor that lattice-update must beat in Phase A.2.
- **PCA-init floor:** with PCA-on-training-data init for R, what accuracy? This is the upper bound of what Phase A's *initialization* can give without training; the lattice-update gain is measured against it.

## What Phase A.2 added (now ONLINE)

Lattice-update coordinate descent on R, in `src/gesh_train.{h,c}`:

- **Per epoch:** sample a fresh training batch (with replacement); compute baseline classification error; then evaluate `n_flip_evals_per_epoch` random trit positions in R. For each position, try the two non-current ternary values; apply the flip that reduces error (or revert).
- **End-of-epoch:** rebuild bank from current R (the bank reflects the post-flip projection space).
- **Loss:** classification error count on the batch. Discrete; matches the discrete optimization shape.
- **Init:** `gesh_init_random_projection` writes random ±1 ternary R. PCA-init or variance-ranked init are deferred — random init turned out to be sufficient on the synthetic benchmark.

No STE. No shadow parameters. No Gumbel-softmax. The projection is ternary; the optimization walks the lattice directly.

**Measurement (prototype classification, D=64, K=16 informative + 48 noise, 10% per-trit noise, sig_dim=32, n_train=2000, n_test=500):**
- Random R baseline (untrained): **62%**
- Trained R (lattice update, 50 epochs × 200 flips, batch=128): **73%**
- Gain: **+11 percentage points** over random init
- Beats Phase A.1's identity-projection baseline (69%) using half the dims.

The substrate-claim probe at Phase A scope: routing-first base-3 with lattice-native training improves over random projection on a task with structure that random doesn't capture. The mechanism works.

## Open question being deferred

**PyTorch attention baseline for substrate-claim comparison.** Phase A's synthetic task isn't the substrate-claim demonstration — it's a probe of whether the mechanism works. The substrate-claim measurement ("routing-first base-3 matches base-2 attention") happens at Phase B+ on a richer benchmark. PyTorch comparison is scoped for that cycle, not this one.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

`gesh` builds as a static library (`libgesh.a`) linking against `libm4t.a`. Tests run alongside the m4t test suite.
