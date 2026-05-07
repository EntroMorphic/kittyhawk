---
cycle: routed16 atomics
phase: profile + verdict
date: 2026-05-07
scope: where exactly does routed16 spend cycles? Per-NEON-op
       attribution, cache-tier sensitivity, and tested optimizations.
       Settles whether the sparsity-crossover gap can be closed by
       kernel-level optimization or only by algorithmic restructure.
companions: commit 8737959 (red-team remediation), commit c3c3c56
            (negative result on activation sparsity), benches in /tmp:
            bench_routed16_atomics.c, bench_routed16_cache.c,
            bench_routed16_optims.c.
---

# routed16 atomics — where the cycles go

## Method

Three benches:

1. **Atomic decomposition.** Built variant kernels V0..V4 that
   progressively turn on operations in the inner tile loop.
   V0 = walk only (touch tile metadata, accumulate scalar fields).
   V1 = +X load. V2 = +idx load. V3 = +gather. V4 = full kernel.
   Per-tile cost delta attributes ns to each operation class.

2. **Cache pressure.** Re-measured per-tile cost as N (and so
   total tiles, and so total metadata) shrinks from 9 MB toward
   sub-L1 sizes. If memory bandwidth bottlenecks the walk, per-tile
   cost should drop sharply at the L1d / L2 thresholds.

3. **Targeted optimizations.** Tested three candidates that would
   plausibly help if the bottleneck were memory or per-tile reduce
   latency: (a) `__builtin_prefetch` 4 tiles ahead, (b) vector
   accumulator that defers horizontal reduce until end of column,
   (c) both combined.

All variants bit-exact vs production routed16.

## Results

### Per-tile attribution (K=N=2560, 50% sparsity)

  V0_walk only           0.36 ns/tile   ≈ 1.15 cycles @ 3.2 GHz
  + X load (2× vld1q)   +0.13 ns
  + idx load (2× vld1q) +0.20 ns
  + gather (2× vqtbl2q) -0.04 ns   (variants don't isolate cleanly —
                                    V3 has shorter dep chain than V2)
  + reduce (2× vaddlvq) +0.07 ns
  Total V4 (full)        0.71 ns/tile   ≈ 2.27 cycles

The variants don't strictly isolate ops because adding ops also
changes dependency chains — V3 shrinks the EOR chain V2 introduced.
But the macro picture is clear:

**~50% of total kernel time is the walk; ~50% is NEON compute.**
Both halves are close to architectural floors.

### Cache pressure does not bind

  Working set    | per-tile cost
  ---------------+---------------
  L2-fitting (3MB) → 0.66 ns/tile
  L2 near-full (9MB) → 0.69 ns/tile
  DRAM (27MB)        → 0.66 ns/tile

Per-tile cost is essentially flat across L2 and DRAM. Apple Silicon's
hardware prefetcher handles the sequential tile array fine — we are
NOT memory-bandwidth-bound. The 0.36 ns/tile for V0 is dominated by
the data dependency chain (load tile pointer → load fields → 3
dependent scalar adds), not memory latency.

### Optimizations confirm the diagnosis

  variant              q_proj  gate/up  down_proj  zp=99%
  --------------------|-------|--------|----------|-------
  PREFETCH only         0.92x   0.95x    0.99x      0.92x
  VEC_ACC only          0.97x   0.98x    1.06x      0.87x
  PREFETCH + VEC_ACC    0.85x   0.90x    0.96x      0.74x

- Prefetch hurts everywhere by 1-8%. Issue-queue pressure with
  no benefit since the HW prefetcher already covers the access
  pattern.
- Vec-accumulator gives 1.06× on K=6912 only (long columns
  amortize periodic widening). Neutral or worse elsewhere.
- Combinations are worse than baseline.

No microarchitectural optimization closes the routed16-vs-dense gap
meaningfully on this representation.

## Why dense wins despite more ops on paper

Dense kernel per (80 trits × 4 output columns):
  - 5× vld1q_s8 for X — **shared across all 4 output columns**
  - 4× (1 W-byte load + 8 unpack ops + 5 vqtbl + 5 SDOT)
  - Total: ~76 NEON ops for 320 (output × trit) cells

Routed16 per (16 nonzeros × 1 output column):
  - 2× vld1q_s8 for X (32 bytes)
  - 2× vld1q_u8 for idx (32 bytes)
  - 2× vqtbl2q + 2× vaddlvq + 2 scalar adds
  - Total: ~10 ops for 16 (output × nonzero) cells

The decisive structural difference: **dense register-tiles 4 output
columns at once**, sharing the X load. Routed16's per-output sparse
pattern can't share — each column has its own tile sequence, so X
must be reloaded per column.

Per output cell:
  - Dense:    76 ops / 320 cells = 0.24 ops/cell
  - Routed16: 10 ops / 16 cells  = 0.63 ops/cell

Routed16 pays ~2.6× more NEON ops per output cell at moderate
sparsity. That's the structural floor. It only crosses dense when
sparsity is extreme enough that the dense path covers many more
positions than the routed path skips.

## What this teaches

The routed16 kernel sits at its representational architectural
limit. The cycle budget is consumed by:

  - Per-tile control flow at near-optimal cycles/tile.
  - NEON compute at near-optimal cycles/op.

The only path to meaningful speedup on this kernel is a different
representation, not microarchitectural tuning:

  1. **Group-wise tile scheduling.** Find tile patterns shared
     across multiple output columns; encode them once and process
     N output columns simultaneously (analogue of dense's register
     tiling). This is significant algorithmic work and may not
     align with arbitrary BitNet weight distributions.

  2. **Reduce tile count via wider tiles.** Move from 16-lane /
     32-trit-window tiles to 32-lane / 64-trit-window tiles using
     vqtbl4q. Doubles per-tile cost but halves tile count. Net:
     similar cycles, fewer per-tile control overhead. Unclear
     whether worth the complexity.

  3. **Bit-pack sign data.** 40-byte tile → 24-byte (5-bit indices
     + 1-bit signs). Saves ~40% memory but only marginal speedup
     since memory wasn't the bottleneck. Not worth it.

None of these is in scope. The honest disposition stands: routed16
is correct, near-optimal for its representation, and waits for an
operation whose sparsity exceeds 92% (no current operation in
BitNet does).

## Concrete numbers for the record

  Component             cycles/tile   ns/tile   % of total
  ---------------------+-------------+---------+-----------
  Tile walk (V0)             1.15      0.36       51%
  NEON compute (V4-V0)       1.12      0.35       49%
  Total                      2.27      0.71      100%

  Best-found optimization      1.06× on K=6912 only
                               0.97× to 0.99× on other shapes
                               No combination beats baseline.

The thing the bench DOESN'T measure but the architecture implies:
**dense's 0.24 ops/cell vs routed16's 0.63 ops/cell** — that
2.6× ops/cell ratio is the floor. Routed16 only wins when the
dense path's "ops per output cell" inflates due to high sparsity
(it covers many trits per output but few nonzeros), and that
crossover sits at 92-97% depending on shape.

## End of routed16 cycle

This closes the routed16 investigation:

  - Correctness: validated (33 tests, ASAN+UBSAN clean).
  - Crossover: characterized (92-97% by shape).
  - BitNet activation sparsity: measured (87.5% peak, never reaches
    crossover).
  - Atomic profile: characterized (50/50 control/compute, both
    near-optimal for representation).
  - Optimization headroom: ~5% best case (vec_acc on K=6912), zero
    elsewhere.

The kernel is ready. The use case is not. We know precisely where
each lives.
