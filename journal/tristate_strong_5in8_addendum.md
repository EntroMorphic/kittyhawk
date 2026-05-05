# ADDENDUM: sub-2-bits/cell base-3 packing (Path D)

Per `journal/tristate_strong_redteam.md` forward pointer #1 ("test theoretical-optimal base-3 packing"). Adds Path D — base-3 5-trits-in-8-bits packing — to the strong-claim cycle to test whether base-3 has a STRUCTURAL density advantage where B2-B cannot follow.

## The structural argument

**Base-3 has a sub-2-bits/cell packing; B2-B does not.**

- Base-3: 5 trits in 8 bits = 1.6 bits/cell (close to log2(3) ≈ 1.585 theoretical optimum).
- B2-B (sign + mask): 1 sign bit + 1 mask bit per cell = 2 bits/cell minimum. **Cannot go below.** The two bits are functionally independent; you can't share information between cells.

This is the strong claim's structural foothold: there exists a density regime (1.58 ≤ density < 2 bits/cell) where base-3 packing is feasible and B2-B is not. The encoding-label-equivalence argument from R-G1 (which showed Path A ≡ Path C at 2 bits/cell) does NOT extend to sub-2-bit densities — at those densities, only base-3 is available.

## Empirical question

Does the kernel-cost overhead of decoding 5-in-8 packed base-3 amortize the 1.25× density advantage on practical workloads?

- For **L1-cache-resident workloads** (W < L1 cache): decode cost dominates; 5-in-8 likely loses.
- For **memory-bandwidth-bound workloads** (W exceeds L1): bandwidth savings might dominate; 5-in-8 might win.

This addendum tests the L1-resident regime explicitly. The memory-bandwidth regime is documented as untested.

## Kernel design

Path D — `base3_5in8_matmul_neon`:

**Encoding:** 5 trits per byte with 3^k positional encoding.
- trit_to_unsigned: -1 → 2, 0 → 0, +1 → 1
- byte = sum(u_i × 3^i) for i in [0, 5)
- Range: byte ∈ [0, 242]

**Inner loop processes 80 trits (16 packed bytes) per iteration:**
1. Load 16 packed bytes
2. Vectorized magic-multiply div-by-3 to extract 5 digits across all 16 lanes
3. TBL decode each digit's {0, 1, 2} → {0, +1, -1} via TRIT5_DECODE_LUT
4. Strided X gather via vqtbl4q (xc0..xc3) + vqtbl1q (xc4) + vbslq combine
5. 5 SDOT calls (one per digit position)

**NEON-only.** No scalar fallback. No scalar reference. Verification by NEON-vs-NEON cross-check against Path A and substrate kernels.

## Results

```
[overall] 60 runs | verify a==b:60/60 skip:60/60 opt:60/60 sub:60/60 5in8:60/60
[PASS] all four audit kernels + substrate cross-check bit-exact equivalent
```

All 60 runs bit-exact across all 5 audit kernels + substrate. Path D is verifiably correct.

## Op count and wall-clock (K=80, 320, 1280)

```
Per 16-cell equivalent NEON ops:
  Path A (base-3 4-in-8 packed):       7 ops
  Path C (B2-B 4-in-8 optimal):        7 ops (same kernel as A)
  Path B (B2-B 4-in-8 honest):        10 ops
  Path B-skip (B2-B 4-in-8 + skip):   13 ops
  Path D (base-3 5-in-8 packed):     ~12 ops  (~59 ops per 80-trit block)

Wall-clock ratios (vs Path A baseline, BitNet-typical headline regime
K=320, w_zero=0.60, a_zero=0.60):
  Path A:           7.30 ms (1.00×)
  Path C:           7.40 ms (1.01×)  TIE
  Path B:          10.93 ms (1.50×)
  Path B-skip:     14.69 ms (2.01×)
  Substrate:        5.65 ms (0.77×)  WIN (8 bits/cell, no decode)
  Path D (5-in-8): 13.95 ms (1.91×)  LOSS
```

Across all 12 configs, Path D ratio: 1.77×–1.96× slower than Path A.

## Density vs kernel cost: the tradeoff

| Path | bits/cell | NEON ops / 16-cell | Wall-clock vs A |
|------|-----------|---------------------|-----------------|
| Substrate (unpacked) | 8.0 | (varies) | 0.77× (faster) |
| Path A / Path C | 2.0 | 7 | 1.00× (baseline) |
| Path D (5-in-8) | 1.6 | ~12 | 1.91× (slower) |

Density and kernel cost trade off: **denser packing pays a per-byte decode penalty**. For L1-resident workloads, the decode penalty exceeds the bandwidth benefit, so denser ≠ faster.

The substrate's 8-bits/cell unpacked path is the FASTEST despite the worst density. Path A is the middle ground. Path D has the best density but the worst kernel cost.

## Verdict on sub-2-bits/cell base-3

**STRUCTURAL DENSITY ADVANTAGE: CONFIRMED.** Base-3 packs at 1.6 bits/cell; B2-B has no equivalent (structural floor at 2 bits/cell). This is a true structural advantage.

**KERNEL COST FOR L1-RESIDENT WORKLOADS: BASE-3 5-IN-8 LOSES.** The decode + strided X gather costs more in NEON ops than the per-byte savings. Path A (4-in-8) is the practical winner at L1-resident densities.

**MEMORY-BANDWIDTH-BOUND VERDICT: UNTESTED.** For workloads where W exceeds L1 cache (K > ~12000 at N=64), bandwidth savings might tip the verdict toward 5-in-8. Not tested in this cycle. Documented as future work.

## Where the strong claim now stands

After R-G1 (B2-B optimal collapse), R-G2 (external grounding), R-G3 (skip rate empirical), and the 5-in-8 addendum:

```
At 2 bits/cell density:
  Base-3 ≡ B2-B-optimal.        Encoding labels are aliases.
  Base-3 > B2-B-honest.         Real win against naive implementation.
  Substrate (8 bits) > base-3.  Storage-vs-decode tradeoff.

At sub-2-bits/cell density (1.6 bits/cell, only base-3):
  Base-3 5-in-8 > Path A on density (structural).
  Path A > Base-3 5-in-8 on kernel cost (L1-resident workloads).
  Memory-bandwidth regime untested.

Theoretical-optimal density (log2(3) ≈ 1.585 bits/cell):
  Base-3 floor.       Achievable.
  B2-B floor.         Not achievable.    ← STRUCTURAL ADVANTAGE
```

## Refined strong-claim summary

Vision claim 3 in operational form: **base-3 carries information that base-2 collapses, in a way structurally cheaper or more accurate than base-2's workaround.**

- "Information that base-2 collapses": the third state. CONFIRMED at the encoding level — base-3 has a native third state that B2-B must construct via sign + mask.
- "Structurally cheaper or more accurate": CONDITIONAL.
  - vs honest B2-B at 2 bits/cell: cheaper (3 ops/block).
  - vs optimal B2-B at 2 bits/cell: tie.
  - At sub-2-bit density: base-3 is the only option (B2-B cannot follow). **Cheaper than nothing.**
- "Workaround": B2-B's sign+mask machinery is a real workaround, but at fixed 2 bits/cell density it's optimizable to base-3 op count.

The strongest defensible claim: **base-3 has a STRUCTURAL DENSITY CEILING ADVANTAGE — it can pack below 2 bits/cell where B2-B cannot.** Whether that advantage manifests as kernel-cost savings depends on workload regime (memory-bound vs compute-bound).

## Honest framing

- The audit's L1 strong-claim cycle established the encoding-label equivalence at 2 bits/cell.
- This addendum establishes the density-ceiling advantage at sub-2 bits/cell.
- The kernel-cost direction at sub-2 bits/cell DEPENDS on workload regime; only L1-resident is tested.
- The most ML-relevant regime (large LLM weights, memory-bound) is untested. Future cycle.

## Files added

```
audit/b2b_matmul.h      — added Path D (base3_5in8_pack, base3_5in8_matmul_neon)
audit/b2b_matmul.c      — added Path D kernel + 5-in-8 packing helper
audit/tristate_strong_bench.c — extended with Path D + new K values (multiples of 80)
audit/strong_results.csv — re-run with 5 paths
audit/strong_summary.txt — per-config summary including Path D
```

K values changed to {80, 320, 1280} (multiples of both 16 and 80 for clean alignment).

## Forward pointer

**Memory-bandwidth-bound test:** repeat the bench at K ∈ {12800, 25600, 51200} where W exceeds L1 then L2. If the verdict tips toward 5-in-8 in those regimes, the strong claim's density advantage MANIFESTS as kernel-cost advantage in real LLM-scale workloads. If it doesn't, base-3's structural advantage is purely density-ceiling-bound, not throughput-bound.

This is the next-priority follow-on for the strong-claim cycle.

## Status

ADDENDUM CLOSED. Sub-2-bits/cell base-3 packing is feasible (Path D verified bit-exact) and demonstrates the structural density-ceiling advantage. Kernel cost at L1-resident densities favors 4-in-8 (Path A); the memory-bandwidth regime is the open question.

The strong claim's structural-advantage framing is now anchored at the density-ceiling layer. **Base-3 can go below 2 bits/cell; B2-B cannot.** That's the structural foothold.
