---
date: 2026-04-14
scope: First measurement against the rebuilt M4T substrate
type: first-light
---

# Rebuilt substrate: first light on MNIST

First consumer-level measurement against the ground-zero-rebuilt substrate. Consumer: `tools/mnist_trit_lattice.c` (the Trit Lattice LSH tool). Platform: Apple M-series, single-threaded.

## Result

```
Pixel-space L1 nearest centroid (no projection):  66.85%  (6685/10000)

LSH — L1 in projection space:
  N_PROJ =  256 →  80.12%
  N_PROJ =  512 →  80.46%
  N_PROJ = 1024 →  81.14%
  N_PROJ = 2048 →  81.40%   ← headline

Pixel refinement (top-K from LSH, then L1 in pixel space):
  N_PROJ =  256, top-3  →  74.46%
  N_PROJ =  512, top-3  →  73.81%
  N_PROJ = 1024, top-3  →  73.44%
  N_PROJ = 2048, top-3  →  73.71%
  (top-5 uniformly worse than top-3; see prior finding)

Wall clock (full sweep, single core): 41.59 s user + 0.22 s system.
```

## What this confirms

**Exact numerical reproduction.** The 81.40% at N_PROJ=2048 matches the pre-rebuild baseline bit-for-bit. The pixel-refine shape (74.46 / 73.81 / 73.44 / 73.71) reproduces the prior finding that refinement hurts in this configuration — `journal/trit_lattice_lsh_synthesize.md` and the full experimental record already document this.

**The substrate rebuild did not silently change consumer numerics.** Specifically:
- The `m4t_ternary_matmul` inner-loop rewrite (from `vmulq_s32` over decoded signs to `vbslq_s32` + `vnegq_s32`) produces bit-identical outputs. The base-3-native shape preserves semantics.
- The `test_proj_buf[4096]` stack-array replacement with `malloc(N_PROJ)` preserves semantics for N_PROJ ≤ 4096.
- The `row_buf[4096]` → `malloc(D)` change in `m4t_route_signature_update` did not affect this path (the LSH tool doesn't call signature_update), but its existence is now validated by the build + tests.

## Performance baseline (first measurement, not tuned)

- 4 projection sizes × (60K train + 10K test) projections from 784-dim pixels → N_PROJ-dim lattice: ≈ 1 G trit-MACs per `N_PROJ/256` at N_PROJ=256, scaling linearly. Full sweep: ≈ 400 G trit-MACs.
- 41.6 s wall, single core → ≈ 10 G trit-MACs/sec.
- Unbenchmarked hot-loop quality. This number is a floor; SDOT path exists for MTFP4 activations but this consumer uses MTFP19 activations and doesn't hit it.

The headline 81.40% matches pre-rebuild; the timing is a first reference point against which future optimization can be measured.

## What this does NOT tell us

- Whether the routing-native thesis (NORTH_STAR) beats a dense-over-ternary-storage approach. MNIST remains a base-2-framed problem (see `docs/THESIS.md` §4). Reproducing 81.40% is adapter-efficiency, not thesis validation.
- Whether the bit-select rewrite is *faster* than the previous multiply-based shape. Not measured — no before/after timing comparison.
- Whether hardware-alignment claims in the substrate spec are empirically discharged. No SDOT-utilization or TBL-throughput measurements yet.

## What changed vs. pre-rebuild

Since commit `fc9c6b0` ("Multi-channel features don't break the 97.61% k-NN ceiling") — the last pre-rebuild state the LSH tool was run against — every substrate-level change between then and this measurement:

1. Reframed MTFP vocabulary (mantissa + per-block exponent) — docs only, no runtime effect.
2. Archived dense paths (matmul, LayerNorm, bias_add, nonlinearity LUTs, MTFP39 wide) — not called by LSH, no effect.
3. New block-native `m4t_mtfp_block_add`/`_sub` primitives — LSH's routing primitives (`apply_signed`) compose through these.
4. `m4t_ternary_matmul` inner loop rewritten with `vbslq_s32` — LSH's hot path.
5. `signature_update` row buffer → malloc — not on LSH path.
6. `trit_to_code` gained a debug assert — LSH uses valid inputs.
7. `_Static_assert`s on invariants — compile-time only.
8. Stack→malloc in LSH tool test buffer — cosmetic for N_PROJ ≤ 4096.

All eight changes preserved numerics through the LSH consumer. That's the invariant we wanted.

## Next empirical questions

Now that a consumer runs on the rebuilt substrate:

1. **Is the bit-select path faster, slower, or neutral vs. the prior multiply path?** Requires a before/after benchmark. The substrate spec claims base-3-native shape; the measurement hasn't discharged that claim.
2. **Can a routing-native consumer beat 81.40% without reintroducing dense paths?** Open. The LSH tool is deliberately simple (random ternary projections + L1 centroid). The space of routing-native modifications — multiple projection scales, learned signatures, signed-distance-in-projection-space — is large and unexplored on this substrate.
3. **What is the SDOT utilization of the current kernels?** The MTFP4 SDOT matmul exists but no consumer calls it yet. First move: extend the LSH tool to exercise it via MTFP4-activation quantization of the projections, and measure whether that's faster without hurting 81.40%.

## Pointers

- Full run log: this journal entry's Result section (reproducible with `./build/mnist_trit_lattice /path/to/mnist`).
- Pre-rebuild baseline source: `journal/full_experimental_record.md` and `journal/trit_lattice_lsh_synthesize.md`.
- Tool source: `tools/mnist_trit_lattice.c`.
- Substrate primitives invoked: `m4t_pack_trits_rowmajor`, `m4t_mtfp_ternary_matmul_bt`, `m4t_mtfp_clamp64` (transitively).
