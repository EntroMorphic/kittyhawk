# RED-TEAM: production-shoring Items 1, 2, 3

Cold-eye review of the three production-shoring commits:
- Item 1 (5b4858f): register-tile libm4t matmul kernels
- Item 2 (842718b): 5-in-8 base-3 packing in libm4t
- Item 3 (9f8b12e): SDOT throughput tool to m4t/tools/

## Findings

### C1 — CRITICAL — Item 2's `#else scalar_ref(...)` is a no-scalar rule violation

The §20 kernel had:
```c
#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    /* NEON path */
#else
    /* "Defensive" call to test oracle as production fallback */
    m4t_ternary_5in8_matmul_bt_scalar_ref(Y, X, W_packed, M, K, N);
#endif
```

This is exactly the "fall back to scalar when X" pattern the project rule prohibits, regardless of whether the path is reachable at runtime. The rule is about CODE COMPLIANCE, not runtime behavior.

**Remediation:** replaced `#else m4t_ternary_5in8_matmul_bt_scalar_ref(...)` with `#error "m4t_ternary_5in8_matmul_bt requires NEON + ARM_FEATURE_DOTPROD; no scalar fallback per project rule."`

The change converts a runtime fallback into a compile-time hard failure on any platform without NEON+DOTPROD. Build still passes on the host (NEON+DOTPROD present). Any future port to non-NEON would fail loudly at build time, forcing the rule to be honored or explicitly amended.

### H1 — Item 1: tail correctness coverage

Concern: tile-by-4 leaves a 1-3 cell tail when N%4 != 0. Did existing tests catch tail bugs?

Investigation: `test_m4t_ternary_matmul.c` random-property tests use `N = rand_int(1, 4)` and `N = rand_int(1, 8)`, plus specific cases at N=1, 2, 3, 5. All (N%4) cases are exercised in CI. **H1 satisfied by existing tests** — no new test needed.

### H2 — Item 2: K%80 + N%4 strict alignment

Concern: real consumers may have arbitrary (K, N). The §20 kernel asserts strict alignment.

**Remediation:** documented explicitly in the kernel header. Current text:
> Strict alignment is intentional and matches the audit's verified shape. Real consumers with non-aligned (K, N) should pad to the next multiple of 80 / 4 (the trailing trits/cells contribute 0 since pack zero-pads). Future work: K%80 + N%4 tail handling for non-aligned shapes — would mirror Item 1's tile-with-tail pattern, deferred until a consumer demands it.

This is a known limitation, called out for consumers. Not blocking; deferred to consumer demand.

### H3 — Item 2: §20 wall-clock not directly measured in libm4t

Concern: the closeout cited audit Path D's 1.8× advantage, but didn't directly measure the libm4t §20 kernel's wall-clock.

**Remediation:** added `measure_5in8_matmul` to `bench_m4t_matmul_tile.c`. Direct libm4t measurements (M=8, N=64, K∈{1280, 12800, 51200}, min-of-5):

```
K       mtfp_ternary    ternary_dot    5in8_matmul
1280    0.045 ms        0.005 ms       0.008 ms       (5in8 1.5× slower than unpacked)
12800   0.450 ms        0.069 ms       0.079 ms       (5in8 1.14× slower)
51200   1.803 ms        0.277 ms       0.317 ms       (5in8 1.14× slower)
```

**Reframing required.** The audit's "1.8× advantage" was for Path D (5-in-8) vs Path A (audit's 4-in-8 packed kernel). In libm4t, the direct comparison is:
- ternary_dot: SDOT with UNPACKED W (8 bits/cell, no decode).
- §20 5in8_matmul: SDOT with 5-in-8 PACKED W (1.6 bits/cell, decode required).

Because libm4t has no 4-in-8 packed ternary-X kernel, the apples-to-apples audit comparison doesn't exist within libm4t. The §20 kernel's value proposition in libm4t is "5× tighter storage at 1.14-1.5× compute cost" — the storage-vs-decode tradeoff identified in the audit's earlier addenda.

For DRAM-bound real LLM workloads where model weights dominate memory bandwidth, the 5× storage savings pay off. For L1-resident inference (this bench), unpacked-SDOT wins.

The §20 kernel correctly enables the consumer's choice; it's not the universally-fastest path. Spec §20.4 already documents this:
> The 5-in-8 packing is NOT a replacement for 4-in-8. They coexist; 4-in-8 stays the default for SDOT-friendly paths, and 5-in-8 is available when storage density justifies the decode cost.

The same framing applies vs unpacked: 5-in-8 is chosen when storage matters more than per-call compute.

### M1 — Item 2 audit cross-check (G6) added late in the cycle

The audit cross-check was deferred to the very end. Verified bit-exact match (good outcome), but methodology-wise should have been a pre-build gate. **Lift for next cycle:** when porting validated audit code to libm4t, run cross-check IMMEDIATELY after first kernel build, not after all infrastructure (tests, CMakeLists, etc.) is in place.

### L1 — Item 3 had no closeout doc

Item 3 was housekeeping (file move) and proceeded with just a commit message. Per LMM discipline, every cycle should have a closeout. **Remediation:** writing minimal Item 3 closeout below.

### L2 — Item 3 tool's reference text contains audit/ paths

The moved `m4t/tools/sdot_pipeline_bench.c` prints reference output text mentioning `audit/strong_results.csv`. Files still exist; references valid. Not broken; minor stylistic inconsistency (tool in m4t/, references audit/). Leave alone.

## Severity classification

| ID | Concern | Severity | Action |
|---|---|---|---|
| C1 | Item 2 scalar_ref() called from production via #else | **CRITICAL** | Replaced with #error |
| H1 | Item 1 tail correctness coverage | HIGH | Verified existing tests cover it; no change needed |
| H2 | Item 2 strict alignment | HIGH | Documented as intentional + future work |
| H3 | Item 2 §20 wall-clock not directly measured | HIGH | Added direct measurement; reframed value prop |
| M1 | Item 2 audit cross-check added late | MEDIUM | Methodology lift for next cycle |
| L1 | Item 3 no closeout doc | LOW | Writing this combined doc |
| L2 | Item 3 tool references audit/ | LOW | No change |

## Item 3 closeout (was missing)

Per L1.

**Verdict: SHIPPED.** sdot_pipeline_bench moved from audit/ to m4t/tools/ following the existing tools convention (manually compiled per file headers, not in cmake build).

Verification:
- 21/21 ctest PASS (no regressions).
- Manual compile + run of moved tool succeeds (cc -O3 -mcpu=native ... → expected SDOT throughput numbers).
- audit/ targets (tristate_audit, tristate_strong_bench) still build cleanly without sdot_pipeline_bench.
- Spec §17 cross-reference for §13 updated: m4t/tools/ is now active, lists all four tools (gen_pow3_magic, bench_vmlal_throughput, bench_accum_baseline, sdot_pipeline_bench).

What this is NOT:
- Not a code change to the tool (file moved verbatim except header rewrite).
- Not a CI-wired-in target (intentional; matches existing m4t/tools/ convention).

## Cumulative production-shoring verdict

After red-team + remediation:

```
Item 1 (register-tile libm4t matmuls):
  - 2.0-3.9× wall-clock speedup (vmlal 2.0-2.5×, SDOT 2.5-3.9×).
  - Bit-exact preserved (21/21 ctest).
  - No project-rule violations.
  - VERDICT: SHIPPED.

Item 2 (5-in-8 base-3 in libm4t):
  - Spec amended (§20).
  - New API: pack/unpack + matmul + scalar_ref oracle.
  - 600 NEON-vs-scalar bit-exact samples + audit cross-check.
  - C1 violation FIXED: scalar_ref() no longer called from production
    via #else; replaced with #error.
  - H2: K%80 + N%4 alignment documented as intentional.
  - H3: §20 wall-clock directly measured in libm4t. 1.14-1.5× slower
    than unpacked-SDOT (storage-vs-decode tradeoff). Audit's "1.8×"
    was apples-to-apples vs another packed kernel that doesn't exist
    in libm4t; reframed value prop accordingly.
  - VERDICT: SHIPPED with reframed value prop.

Item 3 (SDOT tool to m4t/tools/):
  - File moved + header updated to project tools convention.
  - Spec §13 cross-reference updated.
  - VERDICT: SHIPPED.
```

The audit's measured strong-claim advantages remain accessible:
- Existing matmul kernels are 2-4× faster (Item 1, no choice required).
- Sub-2-bit dense packing is opt-in via Item 2 (storage-vs-decode tradeoff documented).
- Hardware characterization tooling consolidated (Item 3).

## Methodology lifts

1. **Even "defensive unreachable" `#else scalar` paths violate the no-scalar rule.** The rule is about code presence, not runtime reachability. Use `#error` to convert runtime fallbacks into compile-time guarantees.

2. **Apples-to-apples comparison must use the same axis.** Audit's Path D vs Path A was "5-in-8 packed vs 4-in-8 packed, both decoded." Libm4t's §20 vs ternary_dot is "5-in-8 packed vs 8-in-8 unpacked, decoded vs not." Different axes → different ratios. Don't carry one cycle's wall-clock numbers into a different comparison context without checking the axes line up.

3. **Audit cross-check is the right verification, but should run early.** When porting audit-validated code to libm4t, the cross-check is the cleanest gate. Run it immediately after first kernel build, not after all infrastructure is in place. Catches bugs while the implementation is still fresh in mind.
