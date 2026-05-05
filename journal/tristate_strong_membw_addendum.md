# ADDENDUM: memory-bandwidth-bound regime test (post-redteam)

Per `journal/tristate_strong_5in8_addendum.md` forward pointer + red-team `tristate_strong_membw_redteam.md` remediation. Tests whether the storage-vs-decode tradeoff inverts when W exceeds L1, L2, then DRAM cache levels.

**RED-TEAM REMEDIATION APPLIED:** original draft claimed "density advantage MANIFESTS as kernel-cost reduction." Post-remediation finding: the kernel-cost penalty NARROWS but does NOT crossover. The trajectory PLATEAUS at ~1.16-1.24× even at DRAM-bound regimes. This is a more honest verdict.

## Workload spec (post-remediation)

```
Cache regime          K       N       W size      Reps
─────────────────────────────────────────────────────
L1-resident           80-1280 64      < 20 KB      2000
L1-overflow           12800   64       200 KB       200
L2-resident (deeper)  25600   64       400 KB       100
L2-resident (deepest) 51200   64       800 KB        50
DRAM-bound (R-G2)     12800   8192    25.6 MB        3
```

Apple M-series L2 = 12-16 MB shared. K=12800, N=8192 → W = 25.6 MB **exceeds L2** → DRAM-bound regime.

**Methodology updates per red-team:**
- **R-G1:** `cache_flush()` walks a 32 MB buffer between kernel runs to evict prior kernel's W and X. Each kernel measures cold-cache (relative to prior kernel) + warm-cache reps.
- **R-G2:** added DRAM-bound config (N=8192, W=25.6 MB > L2) to test the actual extrapolated regime.
- **R-G3:** per-config standard deviation reported alongside mean.

## Results

```
                    L1-RESIDENT           L1-OVERFLOW        L2-RESIDENT          DRAM-BOUND
                    K=320 (warm)          K=12800/64         K=51200/64           K=12800/8192
                                          (W=200KB)          (W=800KB)            (W=25.6MB)
─────────────────────────────────────────────────────────────────────────────────────────
Path A (4-in-8)     1.00× (7.30ms ±0.0)   1.00× (50.8ms ±0.6) 1.00× (56.7ms ±1.8)  1.00× (102.0ms ±2.7)
Path B (B2-B hon)   1.50×                 1.08×              1.05×                1.09×
Path B-skip         2.01×                 1.18×              1.09×                1.17×
Path C (B2-B opt)   1.01×                 1.01×              1.02×                1.02×
Substrate (8 b/c)   0.77× (FASTER)        0.97×              0.99×                0.97×
Path D (5-in-8)     2.10× (15.4ms ±0.5)   1.24× (63.1ms ±0.5) 1.16× (65.7ms ±1.5)  1.24× (126.1ms ±0.9)
```

Verification: 80/80 bit-exact across all 5 audit kernels + substrate (60 from L1-resident multi-config + 15 from N=64 memory-bandwidth + 5 from DRAM-bound).

Per-config standard deviation (R-G3) is bounded: CV ≈ 1-3% across all configs.

## Critical finding: trajectory PLATEAUS, does not crossover

**At all tested memory regimes, Path D loses to Path A on wall-clock.**

```
L1-resident:        Path D 2.10× of A
L1-overflow:        Path D 1.24×
L2-resident:        Path D 1.16×    (asymptote)
DRAM-bound:         Path D 1.24×    (PLATEAU — does not continue narrowing)
```

The hypothesis from the first draft ("trajectory predicts DRAM crossover") is **NOT CONFIRMED**. The penalty narrowed from L1-resident to L2-resident, then plateaued. Pushing to DRAM-bound did not produce further narrowing.

**Why no crossover (post-hoc analysis):**

The arithmetic of the tradeoff:
- Path A reads 25.6 MB W per call (DRAM-bound config).
- Path D reads 20.5 MB W per call (1.25× denser).
- Bandwidth savings: 5.1 MB per call.
- At Apple Silicon's ~70-100 GB/s memory bandwidth: 5.1 MB / 100 GB/s = 0.05 ms saved per call.
- Per call wall-clock: ~34 ms (Path A).
- Bandwidth savings as % of total: 0.15%.

Path D's decode overhead (~12 vs 7 NEON ops per 16-cell, +71%) costs much more than the 0.15% bandwidth savings. **The decode penalty is the dominant cost regardless of cache regime.**

**Apple Silicon's unified memory architecture has very high bandwidth** (>200 GB/s on M2/M3), so memory pressure is rarely the bottleneck for typical workloads. The "memory-bandwidth-bound" regime that traditional CPU systems experience is much milder on M-series.

## Substrate's behavior also confirms the regime is not bandwidth-bound

Substrate uses 4× more bytes (unpacked 8 bits/cell vs Path A's 2 bits/cell). At DRAM-bound:
- W substrate = 102.4 MB (8 bits × 8192 × 12800)
- W Path A = 25.6 MB (2 bits × 8192 × 12800 / 4 — wait this is for the matmul, but substrate ACTUALLY reads unpacked 8-bit storage)

If memory bandwidth dominated, substrate should be ~4× slower than Path A. Actual: substrate is 0.97× of Path A — STILL SLIGHTLY FASTER.

This means even at W=25.6 MB exceeding L2, **memory bandwidth is not the rate-limiting factor on Apple Silicon**. The 4× more bytes substrate reads don't translate to 4× more time. The decode-free advantage of substrate's SDOT path roughly cancels the memory penalty.

## Refined verdict

Post-remediation, the strong claim's standing changes:

```
DENSITY CEILING (unchanged):
  Base-3 floor   = 1.585 bits/cell (theoretical) / 1.6 bits/cell (5-in-8)
  B2-B floor     = 2 bits/cell (sign + mask are independent)
  → BASE-3 HAS UNCONDITIONAL STRUCTURAL DENSITY ADVANTAGE.

KERNEL COST AT 2 BITS/CELL:
  Path A ≡ Path C (encoding-label equivalence).
  Path B-honest is a strawman (3 ops/block penalty).
  → Base-3 ties optimal B2-B; wins vs naive B2-B implementations.

KERNEL COST AT SUB-2 BITS/CELL:
  Path D pays ~1.16-1.24× wall-clock penalty across ALL memory regimes
  tested (L1-resident through DRAM-bound).
  Penalty does NOT crossover at DRAM-bound; PLATEAUS at ~1.2×.
  → Sub-2-bit base-3 packing is UNCONDITIONALLY SLOWER on
    Apple Silicon for this workload shape, despite density advantage.

WHEN BASE-3 SUB-2-BIT WOULD WIN (untested, hypothetical):
  - Hardware where memory bandwidth is the bottleneck (NOT M-series).
    Traditional CPU+RAM with much lower bandwidth-to-compute ratio.
  - Workloads where W is read once per matmul (no temporal locality).
  - Memory-cost-dominated metrics (e.g., RAM cost, transfer cost).
```

## What changed from the first draft

| Item | First draft verdict | Post-remediation verdict |
|------|---------------------|-------------------------|
| Path D at L1-resident | 1.95-2.10× (slower) | 2.10× (unchanged) |
| Path D at L2-resident | 1.16-1.24× (narrowing) | 1.16× (asymptote) |
| Path D at DRAM-bound | UNTESTED ("trajectory predicts crossover") | 1.24× (PLATEAU; no crossover) |
| Substrate at memory-bound | "advantage collapses" | 0.97× (still slightly faster, MBW not bottleneck) |
| Strong-claim framing | "density advantage manifests" | "density advantage is structural at the ceiling, but doesn't translate to kernel-cost win on Apple Silicon" |

## Methodology lifted

1. **Cache-flush between kernels is essential for memory-regime measurements.** Without it, kernel n+1 finds W warm from kernel n. The "memory-bandwidth-bound" framing requires actual cold-cache. R-G1's 32 MB flush is the right pattern.

2. **Apple Silicon's unified memory architecture is unusually generous on bandwidth.** Workloads that would be memory-bound on traditional CPUs are compute-bound on M-series. This affects the relevance of density-cost tradeoffs.

3. **Trajectory extrapolation is risky.** First-draft predicted DRAM crossover from L2-trajectory data. Actual DRAM measurement showed plateau, not crossover. Always test the actual destination, not extrapolate.

4. **SD reporting catches noise floor early.** R-G3's CV ≈ 1-3% confirms measurement reliability; without SD, we wouldn't know if 1.16× vs 1.24× narrowing is real or noise.

## Honest scope-of-claim

The strong claim on L1 weights:
- **Density ceiling structural advantage:** **CONFIRMED** (base-3 < 2 bits/cell achievable; B2-B floored).
- **Kernel cost advantage at any tested regime:** **NOT CONFIRMED** at sub-2-bit density. Path D pays a wall-clock penalty regardless of memory regime on Apple Silicon.
- **Encoding-label equivalence at 2 bits/cell:** **CONFIRMED** (Path A ≡ Path C).
- **Substrate's unpacked-SDOT preference:** **CONFIRMED** (substrate is at-or-faster than Path A in every tested regime, even when memory-bandwidth-pressured).

## Forward pointers (revised)

The kernel-cost direction at sub-2-bits/cell on Apple Silicon is now empirically settled: **base-3's density advantage does NOT manifest as kernel-cost win on this hardware**. Future work:

1. **Test on hardware where memory bandwidth is the bottleneck.** Older ARM chips, embedded systems, or scenarios where DRAM is genuinely slow relative to compute. The density advantage might manifest there.

2. **Consider density-as-storage-cost rather than density-as-throughput.** Memory cost (RAM size, transfer bytes) is a different metric than wall-clock. Base-3's density advantage manifests in storage and bandwidth bills, not in M-series wall-clock.

3. **Algebraic operations (balanced ternary arithmetic).** Out-of-scope here. Could be a genuine structural advantage independent of storage density.

4. **L2/L6 strong-claim cycles.** Apply the same comparative analysis to activation packing and post-ternarization. Likely same encoding-label-equivalence verdict at 2 bits/cell.

## Status

CLOSED. **Path D's density advantage does NOT crossover into kernel-cost advantage at any tested memory regime on Apple Silicon.** The first-draft "trajectory toward crossover" framing was an overclaim; remediated to "plateau at ~1.2× penalty across all regimes."

The strong claim's defensible foothold is the **density ceiling alone** — base-3 can pack below 2 bits/cell where B2-B cannot. Whether that matters for any specific use case depends on whether the use case is memory-cost-bound (yes) vs throughput-bound (no, on M-series).

Files updated:
```
audit/tristate_strong_bench.c — added cache_flush, SD computation, DRAM-bound config
audit/strong_results.csv      — 80 runs (60 small-K + 15 N=64 memory-bound + 5 DRAM-bound)
audit/strong_summary.txt      — extended summary with SD
journal/tristate_strong_membw_redteam.md — red-team analysis (3 critical, 3 high concerns)
journal/tristate_strong_membw_addendum.md — UPDATED with post-remediation verdict
```

20/20 ctest still PASS.

The red-team prevented the "trajectory crossover" overclaim from entering the project record. **Path D loses on Apple Silicon at every memory regime; the density-ceiling advantage is structural but doesn't pay off as wall-clock on this hardware.**
