# DRAM-bound regime test (TD-9)

Closes TD-9 from `docs/TECHNICAL_DEBT.md`. Per `journal/tristate_strong_membw_addendum.md`.

## v2 (REMEDIATED 2026-05-06) — supersedes v1

This file documents the v2 cycle. Per `journal/large_cycles_redteam_2026_05_06.md`:
- RC-4: v1's pre-committed gate ("D/A < 1.0 at any DRAM-bound config") was trivially met because Path D was already winning at L1. v2 tightens to require monotone improvement with W (deep-DRAM ≤ 0.8 × L1).
- RC-5: K=51200 is not a real ML workload. v2 marks K=25600 / K=51200 rows as sanity-check shapes; verdict is based on realistic-K (K ≤ 12800).
- RC-8: deep-DRAM reps doubled from 2-3 to 5-10.

## Question

Does sub-2-bit base-3's density advantage (Path D's 1.6 b/c vs Path A's 2.0 b/c) manifest as wall-clock crossover at TRUE DRAM-bound regimes (W substantially exceeds L2)?

## Method (v2)

Compares Path A (4-in-8 packed W) vs Path D (5-in-8 packed W) across a W spectrum, with cache-flush + warmup discipline. Realistic-K configs (K ≤ 12800) are the load-bearing measurement; K=25600 / K=51200 rows are sanity-check shapes.

Pre-committed gate (tightened): TRUE bandwidth-driven crossover requires `D/A at deep-DRAM (W ≥ 50 MB realistic K) ≤ 0.8 × D/A at L1-resident realistic K`.

## Results (v2)

### Realistic-K trajectory (load-bearing measurement)

| Config | reps | W_A | W_D | ms_A | ms_D | D/A |
|---|---|---|---|---|---|---|
| L1-resident | 200 | 0.02 MB | 0.02 MB | 0.015 | 0.009 | **0.615** |
| L2-resident | 100 | 0.20 MB | 0.16 MB | 0.146 | 0.082 | 0.563 |
| 3.2 MB near L2 | 40 | 3.12 MB | 2.50 MB | 2.304 | 1.271 | 0.552 |
| 12.8 MB at L2 | 20 | 12.50 MB | 10.00 MB | 9.772 | 5.648 | 0.578 |
| 25.6 MB past L2 | 10 | 25.00 MB | 20.00 MB | 19.423 | 11.393 | 0.587 |
| 51.2 MB DRAM | 10 | 50.00 MB | 40.00 MB | 39.037 | 23.644 | 0.606 |
| 102.4 MB deep DRAM | 5 | 100.00 MB | 80.00 MB | 78.804 | 49.727 | **0.631** |

### Sanity-check shapes (NOT load-bearing for verdict)

| Config | reps | W | D/A |
|---|---|---|---|
| 51.2 MB alt (K=25600) | 5 | 50.00 MB | 0.579 |
| 102.4 MB deep (K=25600) | 5 | 100.00 MB | 0.569 |
| 204.8 MB far past (K=51200) | 3 | 200.00 MB | 0.572 |

## Trajectory analysis

Realistic-K D/A:
- Minimum at 3.2 MB near L2: **0.552**
- L1-resident: 0.615
- Deep DRAM (102.4 MB): **0.631**

The ratio is U-shaped: drops slightly below L1 as we cross into L2 territory, then RISES as W grows. **Path D's advantage SHRINKS as W grows past L2, not grows.**

## Pre-committed gate evaluation

Tightened gate: deep-DRAM D/A (0.631) ≤ 0.8 × L1 D/A (0.615 × 0.8 = 0.492)?

**GATE FAILS.** 0.631 > 0.492 by a wide margin. There is NO bandwidth-driven crossover.

## Verdict (v2)

**The membw addendum's PLATEAU finding extends, with much stronger statistical support and a properly-set gate.**

Path D wins by ~1.6-1.8× across the entire W spectrum (D/A 0.55-0.63), but the advantage is workload-INDEPENDENT — driven by SDOT amortization (per `journal/p0_concern1_mechanism.md`), not by bandwidth savings from tighter packing.

If anything, Path D's advantage SHRINKS slightly at deep DRAM (0.55 → 0.63 ratio). Apple Silicon's unified memory bandwidth (~70-200 GB/s) is generous enough that the 0.8× density savings doesn't compound with the bandwidth bottleneck — decode work saved by SDOT amortization dominates throughout.

**TD-9 status: CLOSED.** Sub-2-bit base-3 advantage manifests as a CONSTANT ~1.7× speedup on Apple Silicon. NO regime-dependent DRAM-driven crossover. Hardware-specific (non-Apple ARM, embedded) may show different trajectories — out of scope.

## Sanity-check observations

K=25600 / K=51200 rows have D/A in [0.57, 0.58] — slightly *better* than realistic-K's deep DRAM (0.63). These shapes don't represent real ML workloads, but they do confirm that Path D's advantage stays in a narrow band even at extreme W. No support for crossover at synthetic shapes either.

## Honest concerns

1. **Apple Silicon-specific.** As noted, non-Apple ARM platforms with different bandwidth/compute ratios may show different trajectories. Strong-claim retrospective already flags this caveat.
2. **No system-level cache (SLC) explicit accounting.** M-series SLC ≈ 8-32 MB may absorb some "DRAM-bound" configs into a tighter-than-expected band; distinguishing SLC vs DRAM bandwidth is hardware-specific and out of scope.
3. **Path D's small-W advantage (0.55) at 3.2 MB is real but unexplained.** The U-shape (0.62 → 0.55 → 0.63) suggests there's a "sweet spot" cache regime where density savings briefly help. Likely an L1-eviction-pressure effect; not load-bearing for the headline verdict.
4. **Reps at deep-DRAM realistic-K (5) are still moderate.** Cross-rep variance bounded by cache-flush discipline, but not formally measured (e.g., no SD reported). Trend across configs is the load-bearing finding.

## Cross-references

- Bench source: `audit/tristate_dram_regime.c` (v2)
- Membw addendum: `journal/tristate_strong_membw_addendum.md`
- Membw red-team: `journal/tristate_strong_membw_redteam.md` (R-G2 plateau finding)
- P0-Concern-1 mechanism: `journal/p0_concern1_mechanism.md` (SDOT amortization)
- Strong-claim retrospective: `journal/strong_claim_retrospective.md`
- Red-team: `journal/large_cycles_redteam_2026_05_06.md` (RC-4, RC-5, RC-8)
- TD entry: `docs/TECHNICAL_DEBT.md` TD-9 (now removed)

## v1 archived

v1's pre-committed gate was met trivially because Path D was already winning at L1. v1's K=51200 row was weighted equally with realistic-K rows in the verdict. v2 tightens both. The headline (PLATEAU not crossover) is preserved with much stronger support.
