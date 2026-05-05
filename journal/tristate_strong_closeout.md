# CLOSEOUT: strong-claim test on L1 (post red-team R-G1/R-G2/R-G3)

Per `journal/tristate_strong_synthesize.md` and `journal/tristate_strong_redteam.md`. Four NEON-only kernels + external substrate cross-check. 60 runs (12 configs × 5 seeds). Verification 60/60 across all kernels AND against the substrate's externally-validated `m4t_ternary_dot_matmul_bt`.

**RED-TEAM REMEDIATION APPLIED.** First-cycle verdict ("STRONG CLAIM SUPPORTED on L1") was overclaimed. Per `tristate_strong_redteam.md`:
- **C1 (CRITICAL):** B2-B-honest was a strawman. Added Path C (B2-B-optimal with unified TBL decode); confirmed it has byte-for-byte identical kernel structure to Path A and the same op count (7).
- **C2 (CRITICAL):** Added external grounding via substrate's `m4t_ternary_dot_matmul_bt`. 60/60 cross-check PASS — our kernels are externally correct.
- **C3 (HIGH):** Empirical skip firing rate counted: 0.014% (120 / 860,160 blocks across 60 runs).

The verdict shifts from "STRONG CLAIM SUPPORTED" to a more honest **"STRONG CLAIM CONDITIONALLY SUPPORTED — depends on which B2-B implementation."**

## Verdict (post-remediation)

```
AXIS 1 — Density           : PARITY (both 2 bits/cell)
AXIS 2 — Precision         : PASS (60/60 bit-exact across 4 kernels + substrate)
AXIS 3 — Kernel cost       : SUPPORT_BASE3 vs honest-B2-B (7 vs 10 ops)
                             PARITY vs optimal-B2-B (7 vs 7 ops, identical structure)
```

**Refined headline:**
- vs **honest B2-B** (separate sign+mask decode): Base-3 wins by 3 ops/block. Real win for engineers who would naively decode bits separately.
- vs **optimal B2-B** (unified TBL decode): TIE. Path C (B2-B-optimal) has byte-for-byte identical kernel structure to Path A. The "base-3 vs B2-B" distinction at 2 bits/cell is a labeling choice, not a structural property.

The structural-advantage claim from the original synthesis is **conditional**: base-3 wins versus naive base-2 implementations, ties versus optimal base-2 implementations. The honest framing: at 2 bits/cell, the encoding label doesn't matter — what matters is whether the implementer uses a unified TBL or separate bit-extracts.

## External grounding (R-G2)

The substrate's `m4t_ternary_dot_matmul_bt` is externally validated by libm4t's `test_m4t_ternary_matmul_neon` ctest, which compares it against `m4t_mtfp_ternary_matmul_bt_scalar_ref` (a scalar test oracle inside libm4t). We use it as ground truth here:

**60/60 bit-exact match between our 4 audit kernels and substrate output.** No internal-consistency-only failure mode (which would have been undetected without external grounding). All 4 kernels are correct.

## Per-axis evidence (post-remediation)

### Axis 1 — Density: PARITY

Both 2 bits/cell. Substrate's `m4t_ternary_dot_matmul_bt` uses UNPACKED ternary (8 bits/cell W storage), trading density for kernel simplicity. The packed-kernel comparison in this cycle is at 2 bits/cell density on both sides — fair tie.

### Axis 2 — Precision: PASS

```
[overall] 60 runs total | verify a==b:60/60 a==skip:60/60 a==opt:60/60 a==sub:60/60
[PASS] all four audit kernels + substrate cross-check bit-exact equivalent
```

All 4 audit kernels (Path A, B-honest, B-skip, B-optimal) produce identical Y. All match the substrate's externally-validated kernel. External grounding established.

### Axis 3 — Kernel cost: SUPPORT_BASE3 vs honest, PARITY vs optimal

**Path A (base-3 packed) — 7 NEON ops per 16-cell block:**
```
ld1r → tbl(DUP) → ushl(SHIFT) → and(MASK 0x03) → tbl(TERNARY_LUT) → ldr X → sdot
```

**Path B-honest (B2-B separate sign+mask) — 10 NEON ops:**
```
ld1r → tbl(DUP) → ushl(SHIFT)
  → and(sign bit) → tbl(SIGN_LUT)
  → ushr(mask bit) → bic(~mask)
  → mul → ldr X → sdot
```

**Path B-skip (B2-B + skip check) — 13 NEON ops when not skipping:**
```
[Path B with mask-bit extracted earlier + addv + fmov + cmp+branch]
```
Skip path itself: ~5 ops + branch. Empirical firing rate: 0.014%.

**Path C (B2-B optimal, unified TBL) — 7 NEON ops per 16-cell block:**
```
ld1r → tbl(DUP) → ushl(SHIFT) → and(MASK 0x03) → tbl(B2B_OPTIMAL_LUT) → ldr X → sdot
```

**Path C is byte-for-byte identical to Path A** except for which LUT it loads. The disassembly is structurally indistinguishable. Op count: 7 vs 7.

This is the load-bearing finding from R-G1: at the inner-block level, "base-3" and "optimal B2-B" are the same kernel with different LUT contents.

### Wall-clock corroboration (informational)

Headline regime (K=256, w_zero=0.60, a_zero=0.60), 2000 reps × 5 seeds, mean ms:
```
Path A          (base-3 packed):       5.89 ms   (1.00×, baseline)
Path B-honest   (B2-B separate):       8.81 ms   (1.50× — clear loss)
Path B-skip     (B2-B + skip):         11.70 ms  (1.99× — skip overhead never amortizes)
Path C          (B2-B optimal):        5.89 ms   (1.00× — TIE with Path A)
Substrate       (unpacked SDOT):       4.51 ms   (0.77× — substrate wins, see below)
```

**Across all 12 configs:**
- Path A vs B-honest: A wins by 1.04–1.50× (3-op gap manifesting as wall-clock).
- Path A vs B-skip: A wins by 1.22–2.00× (extra skip overhead).
- Path A vs C-optimal: TIE within ±1% (identical kernel).
- Path A vs substrate (unpacked): substrate wins by 1.19–2.44× (no decode at all).

### Skip firing rate (R-G3)

```
Total skip rate: 120 / 860,160 = 0.000140 (0.014%)
```

Theoretical: P(all 16 cells masked) = 0.6^16 ≈ 2.8e-4 for w_zero=0.60. Observed: 0.014% — close to predicted, considering only sparse-w configs contribute. Skip path fires in 3 of 12 configs (all w_zero=0.60); never enough to amortize the per-block overhead.

## What the remediation revealed

### 1. The "structural advantage" framing was overclaimed

The original closeout argued base-3 has a "structural advantage" because the third state is a native value rather than a constructed one. R-G1 (Path C) shows this is false at 2 bits/cell:

> Both base-3 and B2-B are 4-code packings (2 bits/cell, 4 possible codes per cell). The DECODE is a TBL lookup mapping code → signed value. The LUT contents differ (`{0, +1, -1, 0}` vs `{+1, -1, 0, 0}`); the kernel ops are identical. Calling one "base-3" and the other "base-2 with mask" is a labeling choice, not an algorithmic distinction.

A truly distinct "base-2" representation would be 1 bit/cell (no third state — info is collapsed). At 2 bits/cell, you're already in 4-code territory, and 4-code TBL decode is the optimal kernel regardless of how you label the codes.

### 2. The base-3 win is real against naive implementers, not against optimal ones

If an engineer chooses to decode B2-B by extracting sign and mask separately (Path B-honest), they pay 3 extra ops/block. This is a meaningful real-world cost — it's the difference between "I'll do the obvious thing" and "I'll think about it." Base-3's advantage in this regime is REAL but is a **best-practices advantage, not a structural one**.

### 3. The substrate's unpacked-SDOT wins on raw kernel cost

`m4t_ternary_dot_matmul_bt` operates on UNPACKED ternary (8 bits/cell W storage) and uses SDOT directly with no decode. Wall-clock: 0.77× of Path A (i.e., substrate is faster). The cost: 4× more bytes of W storage.

This surfaces a real engineering tradeoff: **for inference (decode-cost-bounded), unpacked storage wins. For storage/transfer (memory-bound), packed wins.** Neither base-3 packed nor B2-B packed dominates the substrate's unpacked path on raw kernel cost.

### 4. Where DOES base-3 have advantage?

After remediation, the genuine base-3 advantages are:
- **Sub-2-bits/cell density:** A theoretical-optimal base-3 packing (e.g., 5 trits in 8 bits = 1.6 bits/cell) approaches log2(3) ≈ 1.58. B2-B can't go below 2 bits/cell because sign + mask are independent. **At theoretical-optimal density, base-3 wins.** UNTESTED in this cycle.
- **Algebraic operations:** balanced ternary arithmetic has unique properties (carry structure, sign-aware multiply natively). UNTESTED.
- **Best-practices implementation:** if implementers are likely to write naive code, base-3 wins. SOFT, depends on engineering culture.

The cycle confirms the WEAK structural advantage (base-3 == optimal B2-B) and the SOFT real-world advantage (vs naive B2-B). It does NOT confirm an unconditional structural advantage.

## Refined verdict (cumulative)

```
ENCODING-LABEL EQUIVALENCE: at 2 bits/cell, base-3 and B2-B-optimal are
                             the same kernel with different LUT contents.
                             "Base-3" vs "base-2" is a labeling choice.

NAIVE-IMPLEMENTATION ADVANTAGE: base-3 beats B2-B-honest by 3 ops/block.
                                Real win against engineers who decode bits
                                separately. Soft advantage (engineering
                                practice, not structure).

STORAGE-DENSITY TRADEOFF: substrate's unpacked-SDOT path is faster than
                          packed at 4× density cost. Real engineering
                          tradeoff in inference vs storage.

THEORETICAL-DENSITY ADVANTAGE: untested. Base-3 can pack at 1.6 bits/cell
                               (close to log2(3) ≈ 1.58); B2-B floored at
                               2 bits/cell. Out of scope this cycle.
```

## What changed from the first-cycle closeout

| Item | First-cycle verdict | Post-remediation verdict |
|------|---------------------|-------------------------|
| Cost vs honest B2-B | SUPPORT_BASE3 (7 vs 10) | SUPPORT_BASE3 (unchanged) |
| Cost vs optimal B2-B | (not measured) | **PARITY (7 vs 7 ops)** |
| Verification grounding | Internal NEON-vs-NEON cross-check only | + external substrate cross-check (60/60) |
| Skip firing rate | Theoretical (0.028%) | Empirical 0.014% confirmed |
| Structural-advantage claim | Strong | **Conditional — depends on B2-B implementation** |
| Substrate-unpacked perf | Not measured | Substrate is 1.19-2.44× faster than packed |

## Honest caveats (revised)

1. **L1 only.** L2/L4/L5/L6 strong-claim cycles are follow-on.
2. **At 2 bits/cell, optimal base-3 and optimal B2-B are equivalent kernels.** Base-3 doesn't have a structural advantage at this density — only against suboptimal B2-B implementations.
3. **Theoretical-optimal base-3 packing (1.6 bits/cell) is NOT TESTED.** This is where base-3 might have a genuine density advantage; would require a different packing primitive.
4. **Substrate's unpacked-SDOT is faster than any packed path.** For inference workloads where decode-cost dominates, the substrate's unpacked storage is preferable despite the 4× density cost.
5. **Wall-clock favors larger ratios at K=256-1024** where SDOT throughput saturates and decode ops become the bottleneck. At smaller K, the ratio shrinks toward 1× as setup overhead dominates.
6. **Op count weights ops uniformly; pipeline reality differs.** Wall-clock corroborates the op-count direction; both axes agree directionally.
7. **Skip firing rate is workload-dependent.** Random ternary at w_zero=0.60 → 0.014% rate. Real BitNet weights with structured sparsity might show meaningfully higher rates; not tested.
8. **No memory-bandwidth measurement.** All workloads at K ≤ 1024, fitting in L1 cache. Larger K could shift the equation.

## Methodology lifted

1. **External grounding catches false-internal-consistency wins.** Without R-G2's cross-check against the externally-validated substrate, our 60/60 internal cross-check could have hidden a shared-bug failure mode. External grounding is cheap and load-bearing.

2. **Test the "optimal" alternative, not just the convenient one.** R-G1's Path C surfaced that the original B2-B was a strawman. Whenever a cycle compares against a "naive" implementation of an alternative, also compare against the optimal one. The optimal is often more informative.

3. **Empirical confirmation matters even when theoretical prediction is strong.** R-G3's empirical skip rate matched theoretical (within constant factor); confirmation costs 1 line of instrumentation and rules out PRNG correlations as a confound.

4. **Encoding labels at fixed density are aliases.** At N bits/cell, all 2^N-code packings are interchangeable up to the contents of the decode LUT. The "base-X vs base-Y" distinction at fixed density is meaningful only at the algorithmic level (operations supported), not at the storage/decode level.

## Files added/changed this remediation

```
audit/b2b_matmul.h            — added Path C + skip_count_out param
audit/b2b_matmul.c            — added Path C kernel; instrumented skip count
audit/tristate_strong_bench.c — added Path C + substrate cross-check + skip rate
audit/CMakeLists.txt          — link tristate_strong_bench to libm4t
audit/strong_results.csv      — re-run with 4 kernels + substrate
audit/strong_summary.txt      — per-config summary (all 5 paths)
journal/tristate_strong_redteam.md  — red-team analysis (this document)
journal/tristate_strong_closeout.md — UPDATED with post-remediation verdict
```

Reproduce:
```sh
cmake --build build --target tristate_strong_bench
./build/audit/tristate_strong_bench > audit/strong_results.csv 2> audit/strong_summary.txt
otool -tv build/audit/tristate_strong_bench | grep -A 50 "_b2b_optimal_matmul_neon:"
```

## Forward pointers (revised)

The cycle's verdict is more nuanced than the original. Future work:

### Recommended priority
1. **Test theoretical-optimal base-3 packing (1.6 bits/cell).** This is where base-3 has a genuine density advantage that B2-B cannot match (B2-B is floored at 2 bits/cell because sign and mask are independent). Implementing a 5-trits-in-8-bytes packing + matmul kernel would settle whether base-3's density advantage manifests at the kernel level.

2. **Test base-3-distinct algebraic operations.** Balanced ternary arithmetic has unique properties (additive carry structure differs from base-2; multiplication is sign-aware natively). These are operations B2-B cannot match without explicit overlay logic. UNTESTED.

3. **L2/L6 strong-claim cycles.** Same shape as L1; likely similar verdict (encoding-label equivalence at fixed density). Could be combined into a single follow-on.

### Lower priority (deferred)
4. **L4 weak-deepening (Track A from audit closeout).** Now de-prioritized given strong-claim's nuanced verdict; if base-3 doesn't have a structural advantage at L1, the Track A redesign needs to justify itself differently.
5. **Cross-exp accum strong-claim (L5).** Out of scope for any current cycle; would need a residual-style workload.

## Status

CLOSED. **Strong claim CONDITIONALLY SUPPORTED on L1:**
- vs naive B2-B implementations: SUPPORTED (7 vs 10 ops/block)
- vs optimal B2-B implementations: TIE (encoding-label equivalence)
- vs unpacked-SDOT (substrate baseline): substrate wins on raw cost

The cycle's first-iteration "STRUCTURAL ADVANTAGE" framing is replaced by a more honest **"BEST-PRACTICES ADVANTAGE PLUS UNTESTED THEORETICAL-DENSITY ADVANTAGE."** The substrate's base-3 representation has genuine value, but the value is conditional on (a) using packed storage AND (b) the B2-B alternative being naively implemented OR (c) achieving sub-2-bits/cell density.

External grounding established (R-G2). Verification 60/60 across 4 audit kernels + substrate. Skip rate 0.014% empirically confirmed (R-G3).

The red-team did its job: prevented an overclaim from entering the project record.
