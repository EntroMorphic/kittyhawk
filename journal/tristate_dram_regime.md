# DRAM-bound regime test (TD-9)

Closes TD-9 from `docs/TECHNICAL_DEBT.md`. Per `journal/tristate_strong_membw_addendum.md`.

## Question

Does sub-2-bit base-3's density advantage (Path D's 1.6 b/c vs Path A's 2.0 b/c) manifest as wall-clock crossover at TRUE DRAM-bound regimes (W substantially exceeds L2)? The membw addendum tested up to W = 25.6 MB and found a PLATEAU not a crossover. TD-9 extends the sweep into N=2048+ / K=large territory.

## Method

Compares Path A (4-in-8 packed W) vs Path D (5-in-8 packed W) across a W spectrum spanning L1-resident (0.02 MB) to far past DRAM band (200 MB). Cache-flush + warmup discipline mirrored from `tristate_strong_bench`:

- 64 MB flush buffer (exceeds M-series L2 = 12-16 MB)
- per-rep flush before each kernel call
- one warm rep before timing starts
- N_REPS scaled by config size to bound total runtime

Apple Silicon cache hierarchy (M-series, approximate):
- L1 data cache: 192 KB per P-core
- L2 cache (shared): 12-16 MB
- No discrete L3; system-level cache (SLC) ≈ 8-32 MB
- DRAM bandwidth: ~70-200 GB/s unified

## Results

| Config | W_A | W_D | reps | ms_A | ms_D | D/A |
|---|---|---|---|---|---|---|
| L1-resident | 0.02 MB | 0.02 MB | 100 | 0.016 | 0.010 | **0.625** |
| L2-resident | 0.20 MB | 0.16 MB | 50 | 0.145 | 0.083 | **0.571** |
| W ≈  3.2 MB near L2 | 3.12 MB | 2.50 MB | 20 | 2.239 | 1.241 | **0.554** |
| W ≈ 12.8 MB at L2 | 12.50 MB | 10.00 MB | 10 | 9.092 | 5.191 | **0.571** |
| W ≈ 25.6 MB past L2 | 25.00 MB | 20.00 MB | 5 | 18.438 | 10.741 | **0.583** |
| W ≈ 51.2 MB DRAM-bound | 50.00 MB | 40.00 MB | 3 | 37.394 | 22.844 | **0.611** |
| W ≈ 51.2 MB DRAM (alt) | 50.00 MB | 40.00 MB | 3 | 40.279 | 23.149 | **0.575** |
| W ≈102.4 MB deep DRAM | 100.00 MB | 80.00 MB | 2 | 80.733 | 47.180 | **0.584** |
| W ≈204.8 MB far past | 200.00 MB | 160.00 MB | 2 | 156.688 | 89.734 | **0.573** |

## Reading the trajectory

D/A spans 0.554 to 0.625 across the entire W range. Three observations:

1. **Path D wins consistently.** D/A < 1.0 at every config, including L1-resident. Path D is ~1.6-1.8× faster than Path A at every regime tested.
2. **D/A is roughly stable.** The ratio varies in a narrow band (0.55-0.63) without showing the monotone decrease that would indicate a true bandwidth-driven crossover.
3. **Slight upward drift at deep DRAM.** D/A rises slightly from 0.55 (peak advantage at L2-resident) to 0.58-0.61 at deep-DRAM configs. If anything, Path D's advantage SHRINKS marginally at deep DRAM, not grows.

## Verdict

**The PLATEAU finding from the membw addendum extends.** Path D's advantage over Path A is real and stable across the cache hierarchy, but it does NOT compound with the bandwidth bottleneck on Apple Silicon. The ~1.7× speedup is workload-independent — driven by SDOT amortization (5 SDOTs per outer block; per `journal/p0_concern1_mechanism.md`), not by Path D's 0.8× density advantage.

**Pre-committed gate:** D/A < 1.0 at any DRAM-bound config (W > 50 MB). PASSED — but for the wrong reason. Path D was already winning at L1; the DRAM regime didn't push the ratio LOWER, just maintained it.

**Why no DRAM-driven crossover on M-series:** Apple Silicon's unified memory bandwidth (~70-200 GB/s) is generous enough that even at W = 200 MB, the ~40 MB savings from Path D's tighter packing don't dominate the per-kernel cost. The decode work saved by SDOT amortization dominates throughout. Hardware with tighter bandwidth/compute ratio (older ARM, embedded, non-Apple Silicon) would likely show a different trajectory.

## Cumulative verdict (TD-9)

1. **Path D wins on Apple Silicon at every workload tested**, from L1 (0.02 MB) to far-DRAM (200 MB).
2. **The advantage is workload-independent.** D/A stays in [0.55, 0.63] across the full range; no bandwidth-driven compounding.
3. **The membw addendum's "plateau, not crossover" verdict extends.** TD-9's wider sweep confirms it.
4. **Crossover hardware-specific.** True DRAM-bound crossover (D/A getting LOWER as W grows) would likely require platforms with tighter bandwidth-to-compute ratio; this is not a substrate finding but a hardware-specific observation.

**TD-9 status: CLOSED.** Sub-2-bit base-3 density advantage manifests as a CONSTANT ~1.7× speedup on Apple Silicon, not as a regime-dependent crossover.

## Honest concerns

1. **N_REPS at deep-DRAM configs is small (2-3).** Per-call variance is bounded by the cache-flush discipline, but with so few reps, individual-config ratios should not be over-interpreted. The TREND across configs is the load-bearing finding.
2. **No system-level cache (SLC) explicit accounting.** The 8-32 MB SLC may absorb some "DRAM-bound" configs into a tighter-than-expected band. Distinguishing SLC vs DRAM bandwidth is hardware-specific and out of scope.
3. **Apple Silicon-specific.** As noted in the verdict — non-Apple ARM platforms with different bandwidth/compute ratios may show different trajectories. The strong-claim retrospective already flagged this caveat.
4. **Path A's tile pattern is the same as Path D's.** This is the apples-to-apples comparison verified by P0-3 (no tile-asymmetry confound).

## Cross-references

- Bench source: `audit/tristate_dram_regime.c`
- Membw addendum: `journal/tristate_strong_membw_addendum.md`
- Membw red-team: `journal/tristate_strong_membw_redteam.md` (R-G2 plateau finding)
- P0-Concern-1 mechanism: `journal/p0_concern1_mechanism.md` (SDOT amortization is the cause)
- Strong-claim retrospective: `journal/strong_claim_retrospective.md`
- TD entry: `docs/TECHNICAL_DEBT.md` TD-9 (now removed)
