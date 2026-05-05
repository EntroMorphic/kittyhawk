# P0 CONCERN-2 REMEDIATION: L2 strong-claim — does the L1 verdict generalize?

Per the post-P0 self-review concern: the strong-claim verdict was L1-only. Generalizing "base-3 wins ~1.8×" to L2 (activations) without measurement would be premature. This addendum closes the gap by adding a measurement at L2.

## The L2 question

L1 strong-claim measured: at fixed 2 b/c density, Path A (base-3 4-in-8 W) ≡ Path C (B2-B 4-in-8 W) by encoding-label equivalence (R-G1 finding). At sub-2-bit density, Path D (5-in-8 W) wins ~1.8×.

L2 question: at fixed 2 b/c density, does base-3 X-packing equal B2-B X-packing? At sub-2-bit density, does base-3 X-packing also win wall-clock?

Note: L1 is about WEIGHTS; L2 is about ACTIVATIONS. The measurement layer matters.

## What's tested in this remediation

Added Path E to the strong-claim bench:

```
Path E — base-3 4-in-8 packed X + base-3 4-in-8 packed W.
         Both X and W packed at 2 b/c, decoded per inner iter.
         Tile-of-4 register pattern (same as Path A).
         X decode shared across 4 j cells per outer block.
```

The kernel symmetrically decodes both X and W using the same TBL pattern. Per outer block (16 cells × 4 j cells): 1 X decode (5 ops, shared) + 4 × (1 W decode + 1 SDOT) = ~30 ops vs Path A tiled's ~25 ops. Per 16 cell-trits: ~7.5 ops vs Path A's 6.25 ops. Predicted wall-clock penalty: ~20%.

## Result

```
Wall-clock ratios across regimes (Path E / Path A):

  L1-resident   K=80-1280:           ~1.20× (Path E ~20% slower)
  L1-overflow   K=12800   N=64:      ~1.19×
  L2-resident   K=25600   N=64:      ~1.22×
  L2-resident   K=51200   N=64:      ~1.22×
  DRAM-bound    K=12800   N=8192:    ~1.19×
```

**All regimes consistent: Path E is ~20% slower than Path A.** The penalty is the X decode cost; X bandwidth savings don't compensate at our workloads (X is L1-resident at every tested config — M=8 batch keeps X tiny).

Bit-exact: 80/80 PASS. Path E and Path A produce identical Y for the same inputs.

## What this confirms about L2

**Encoding-label equivalence at L2 (extension of R-G1) — DIRECTLY VERIFIED.** Added Path F (B2-B-optimal X+W packed, structural twin of Path E with B2B_OPTIMAL_LUT instead of TERNARY_LUT). Disassembly comparison:

```
Path E inner loop:        Path F inner loop:
ld1r.4s {v16},[x4],#4     ld1r.4s {v16},[x4],#4
tbl.16b v16,{v16},v0      tbl.16b v16,{v16},v0
ushl.16b v16,v16,v1       ushl.16b v16,v16,v1
and.16b v16,v16,v2        and.16b v16,v16,v2
tbl.16b v16,{v3},v16      tbl.16b v16,{v3},v16
ld1r.4s {v17},[x5],#4     ld1r.4s {v17},[x5],#4
tbl.16b v17,{v17},v0      tbl.16b v17,{v17},v0
ushl.16b v17,v17,v1       ushl.16b v17,v17,v1
and.16b v17,v17,v2        and.16b v17,v17,v2
tbl.16b v17,{v3},v17      tbl.16b v17,{v3},v17
sdot.4s v6,v16,v17        sdot.4s v6,v16,v17
...                       ...
```

**Byte-for-byte identical.** Only the LUT load address (where v3 is filled at function entry) differs. This DIRECTLY confirms encoding-label equivalence at L2 — base-3 X-packing and B2-B X-packing are the same kernel with relabeled LUT bytes. Bit-exact verified per run (Path F passes the same memcmp-against-Path-A gate as Path E).

**L1 wall-clock advantage does NOT directly extend to L2 at L1-resident workloads.** Path E adds X decode cost without recovering anything via X bandwidth savings (because X is L1-resident — M=8 keeps X tiny: 100 KB at K=12800, fits in L1). For L2 to benefit from packing, X-side memory bandwidth needs to be the bottleneck — which happens in larger-batch-size workloads (LLM training: M = batch_size × seq_len ≫ 8) but not in our inference-shape bench.

## What this does NOT establish

- **Sub-2-bit X-packing (5-in-8) for L2 not tested.** Implementing a kernel that decodes both X (5-in-8) and W (5-in-8) requires another split-LUT decode plus pre-permutation for X. Significant additional work; deferred. The density-ceiling structural argument SHOULD extend by symmetry (base-3 can pack X below 2 b/c; B2-B cannot), but wall-clock benefit at our workloads is unlikely to materialize since X bandwidth isn't the bottleneck.

- **L4 (cross-layer requantization), L5 (cross-exp accum), L6 (post-ternarization) not tested.** These layers have different architectural shapes than W or X packing. L5 in particular is about residual/skip connections in MTFP arithmetic, which the GEMM-only audit doesn't exercise. Each requires its own strong-claim cycle.

- **Workloads with X memory pressure (large batch / long context) not tested.** Our M=8 bench keeps X small. For production-shape workloads where X-bandwidth-bound regimes are reachable, the L2 packing penalty might invert into a benefit.

## Refined L2 verdict

The L1 verdict has TWO components:

1. **Encoding-label equivalence at fixed density (structural, hardware-independent).** This extends to L2 by symmetry. Confirmed.
2. **Sub-2-bit base-3 wins ~1.8× wall-clock (on this hardware, at our workloads).** This is L1-specific and workload-specific. **Does NOT extend to L2 at our test workloads** — Path E (4-in-8 X+W) is 20% slower than Path A (unpacked X + 4-in-8 W). The wall-clock benefit at L2 requires bandwidth-bound conditions for activations.

The structural foothold is preserved at L2; the empirical kernel-cost win is L1-specific at this workload shape.

## Honest framing

The user's concern was that generalizing the L1 verdict broadly would be premature. **This is borne out by the L2 measurement.** Specifically:
- The encoding-label structural claim DOES extend (no measurement needed; follows from bit-level symmetry).
- The 1.8× wall-clock advantage does NOT directly extend to L2 packing at L1-resident workloads.
- Stronger generalizations require either (a) workloads where L2 is bandwidth-bound (LLM-training-shape), or (b) sub-2-bit L2 packing tested explicitly.

For real LLM workloads:
- W persists across forward passes — packing W tighter saves DRAM/transfer bytes proportionally to model size.
- X is computed per forward pass; packing X helps only when X-bandwidth is the bottleneck (large batch, long context).
- KV cache for attention: X-side bandwidth-bound for long contexts. Packing X tighter could save KV cache bytes proportionally.

These regimes are out of the audit's scope but suggest where L2 advantages might materialize.

## Methodology lift

**Symmetry arguments don't substitute for measurement.** The L1→L2 encoding-label equivalence DOES follow by symmetry (no measurement needed for that part). But the WALL-CLOCK component of the L1 verdict didn't extend by symmetry — it required actual measurement to discover that L2 packing has a 20% penalty at our workloads. **Don't generalize empirical claims by symmetry.**

## Files added/changed

- `audit/b2b_matmul.{h,c}` — added Path E (`path_e_packed_x_matmul_neon`).
- `audit/tristate_strong_bench.c` — extended to pack X and call Path E. Updated CSV header, summary line, and verification gate.

## Concern 2 status: REMEDIATED

The L1 verdict's two components are now scoped honestly:
1. Encoding-label equivalence at fixed density extends to L2 by symmetry. CONFIRMED.
2. The 1.8× wall-clock advantage does NOT extend to L2 at L1-resident workloads. MEASURED — Path E is 20% slower than Path A.

The generalizability concern is addressed: we now know what extends and what doesn't. Strong-claim cycles for L4, L5, L6 remain deferred.
