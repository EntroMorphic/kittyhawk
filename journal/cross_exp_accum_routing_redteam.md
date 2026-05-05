# RED-TEAM: cross-exp accumulator routing cycle

Cold-eye review of `journal/cross_exp_accum_routing_closeout.md` and the productionized state. Nine A-G gates passed cleanly; this red-team examines whether they actually proved what they claimed and whether the framing is accurate. Of particular interest: did this cycle CORRECTLY apply the just-saved no-scalar rule, or did it leave a violation?

## Critical findings

### C1: Cross-exp saturation untested — only same-exp constructed

A-G4's saturation test (`test_saturation`) constructs two cases:
- Same-exp positive: running=MAX_VAL, addend=MAX_VAL, e_run==e_addend → sum=2×MAX_VAL → clamps. PASS.
- Same-exp negative: same with -MAX_VAL → -2×MAX_VAL → clamps. PASS.

But cross-exp saturation IS possible: aligned ≤ MAX_VAL/3 ≈ 1.94×10⁸ + other ≤ MAX_VAL = 5.81×10⁸ → sum ≤ 7.75×10⁸ > MAX_VAL = 5.81×10⁸. So the post-add clamp triggers in cross-exp configs too. **This wasn't constructed in the saturation-edge test.** The 1000 random configs probably hit it accidentally, but it wasn't deliberately probed.

**Risk:** medium. The math says NEON should produce the same SATURATED bit as scalar — both compute `sum != clamped`. But "deliberately probed" gives stronger evidence than "1000 random rolls of the dice happened to hit the boundary." A constructed cross-exp saturation case would tighten C1.

### C2: `m4t_mtfp_vec_accum_aligning_neon` redundant public API post-A-G7

Per the ternary MAC pattern (where `_vmlal` was removed at productionization), the prototype `_neon` wrapper should be removed once production dispatches to it. Currently it remains in the public API:
- `m4t_mtfp_vec_accum_aligning` (production, just wraps `_neon`)
- `m4t_mtfp_vec_accum_aligning_neon` (was prototype, now identical work)

Two public functions doing the same thing. Confusion risk for future maintainers; API surface bloat.

**Risk:** low. Cleanup, not correctness.

## High-severity findings

### H1: VIOLATION of just-saved no-scalar rule — same-exp + flags!=NULL goes to scalar

The new memory rule (`feedback_function_over_speed_no_scalar`, saved 2026-05-05): *"Production dispatchers are NEON-only. Don't include 'fall back to scalar when X' as a design option."*

**This cycle violated the rule it was supposed to be following.** The dispatcher (`m4t_mtfp_vec_accum_aligning_neon`) routes the same-exp branch entirely to `accum_aligning_scalar`. Inside `accum_aligning_scalar`, line 415-428:

```c
if (flags == NULL) {
    m4t_mtfp_vec_add_inplace(running, addend, n);  /* NEON via block_add */
    return;
}
for (int i = 0; i < n; i++) {  /* SCALAR fallback when flags != NULL */
    ...
}
```

When the same-exp branch is hit AND flags != NULL, production runs scalar code. This is exactly the "(a) drop on NEON path → scalar fallback when flags != NULL" pattern the rule explicitly forbids:

From `feedback_function_over_speed_no_scalar.md`:
> "Specifically the previous flag-tracking pattern from T2-C — 'fall back to scalar when flags != NULL' — is now disallowed for new work."

The cycle's CHANGELOG entry claimed: *"All lessons applied at cycle start, not as remediation."* That claim is **false** — the same-exp branch's scalar fallback was inherited from existing code (T2-C precedent) and not audited against the new rule.

**Risk:** high (rule violation). Same-exp + flags!=NULL is a real production code path; consumer using flag tracking on same-exp data will hit it.

### H2: REFLECT estimate (12-20×) was optimistic; actual 1.6-6.0×

REFLECT predicted ~12-20× speedup based on per-cell op count. Actual measured: 1.6× to 6.0× depending on shape. The closeout acknowledged this honestly, but the methodology gap remains: REFLECT didn't account for compiler auto-vectorization of the scalar baseline at higher delta values, OR for the per-lane flag bookkeeping cost on the NEON path.

The estimate was off by ~3× on the high end. For a future cycle's REFLECT, this means: **REFLECT estimates of NEON speedup over scalar should account for compiler auto-vectorization of the scalar baseline.** Currently, REFLECT pretends the scalar baseline is "the worst case" but compilers are smarter than that.

**Risk:** low for THIS cycle (number is honest in closeout). Risk for FUTURE cycles: REFLECT estimates may continue to overshoot. Worth lifting as methodology.

### H3: Bit-exact sample is 1030+ but matmul-style state space is large

Same shape as ternary MAC's red-team C1. The 1000 random configs sample (n, delta, with_flags, seed). Within each config, cell values are deterministic (gen_data via srand+rand). Coverage is finite but defensible.

Could be tightened with:
- More random configs (10000 instead of 1000) — cheap (test runs in ~1s)
- Saturation-edge construction for cross-exp (per C1)
- Or accept 1030+ as sufficient given the algebraic equivalence (NEON = scalar by construction; sample tests verify the implementation matches)

**Risk:** low. The test passes; confidence is reasonable. But not exhaustive.

## Medium findings

### M1: Disasm verification didn't check dispatcher inlining

A-G6 confirmed `accum_aligning_neon_block` emits expected NEON ops. It did NOT verify whether the public dispatcher `m4t_mtfp_vec_accum_aligning` inlines the helper or makes a function call. If the dispatcher does `bl _accum_aligning_neon_block` per call, there's per-call overhead.

**Risk:** low. Per-call overhead is small relative to bench results showing real speedup. Verifiable cheap.

### M2: Cross-cutting `#if !M4T_HAS_NEON` audit deferred

Same as A-G9's findings: 5-6 other locations have dead scalar fallback (block_add, block_sub, ternary_dot dispatch, vec_add_inplace tail, etc.). Per the new rule these should all be cleaned. Cycle scoped only to the cycle's own code.

**Risk:** medium for project consistency (the rule is supposed to be enforced project-wide). Real cleanup work outstanding.

### M3: ROUNDED reconstruction overflow proof verbal only

The closeout's saturation argument (A-G3 inline comment) bounds aligned*s ≤ MAX_VAL based on the magic-multiply property "quotient ≈ val/s, so quotient*s ≈ val ≤ MAX_VAL." Verbal but mathematically sound. No numerical edge case test that probes near-boundary aligned*s products.

**Risk:** very low. The math is right.

### M4: Speedup at n=4096 (1.7×) lower than n=64 (2.3×)

A-G6 results show LARGER n having LOWER speedup. Counter-intuitive: typically larger n amortizes setup. The fact that it's worse suggests cache misses, prefetcher effects, or memory bandwidth ceiling. Not characterized.

**Risk:** none for correctness. Perf characterization gap.

## Low findings

### L1: bench_accum_baseline.c not in CMakeLists

Same convention as gen_pow3_magic.c and bench_vmlal_throughput.c (project rule: tools/ files are standalone manual-build). Not a regression.

### L2: Closeout claim "all lessons applied at cycle start" is partially false

Per H1: the same-exp + flags!=NULL scalar fallback was inherited and NOT audited. The lesson was applied to the CYCLE'S NEW code (the cross-exp branch), but not to the existing same-exp branch the dispatcher delegates to. So "all lessons applied" is too strong; "all lessons applied to the new code" is accurate.

### L3: A-G2 baseline used different shapes than A-G6 perf bench

A-G2's shapes: n=64, n=4096, n=16 with various deltas.
A-G6's shapes: same n's but different setup/iter counts.
Direct comparison of A-G2 baseline to A-G6 production is approximate (same scalar function, different bench setup). The closeout doesn't directly compare them. Not necessarily a problem but not as crisp as it could be.

## Methodology issues this red-team surfaces

**1. The "no-scalar rule" needs to be applied at AUDIT time, not just at design time.** This cycle applied it to NEW code but inherited a violation in existing code that the dispatcher delegates to. Future cycles should include "audit existing functions touched by this cycle's dispatcher for scalar-fallback patterns" as a sub-step.

**2. REFLECT estimates of NEON speedup should account for compiler auto-vectorization.** The compiler can vectorize many scalar loops at -O3, narrowing the NEON-vs-scalar gap. REFLECT's "scalar baseline" should mean "the slowest realistic scalar implementation" not "the slowest theoretical scalar implementation." Apply: when estimating speedup, consider whether the compiler will auto-vectorize the scalar; if yes, the speedup is over the vectorized scalar, not the un-vectorized.

## What I'd want before declaring fully closed

In rough priority:

1. **H1 fix (no-scalar rule violation):** make the same-exp + flags!=NULL path NEON. Either reuse the cross-exp NEON pipeline (computes flags via NEON compare) restricted to delta=0, OR write a separate same-exp+flags NEON helper.
2. **C2 cleanup:** remove `m4t_mtfp_vec_accum_aligning_neon` from public API; production dispatcher inlines the body.
3. **C1 cross-exp saturation case:** construct one saturation-edge for the cross-exp branch.
4. **L2 closeout correction:** amend the "all lessons applied" claim to the accurate scoping.
5. **H2 methodology lift:** add to CONTRIBUTING throughput-microbench discipline checklist: "REFLECT estimates should bound the NEON-vs-scalar speedup by accounting for compiler auto-vectorization of the scalar baseline."

## Status

10 findings (2 critical, 3 high, 4 medium, 3 low). The C1 (untested cross-exp saturation) and H1 (no-scalar rule violation) are the load-bearing ones. The rest are completeness gaps or framing accuracy.

Estimated remediation: ~45 min for the load-bearing fixes (H1 NEON rewrite of same-exp + flags branch, C1 constructed test, C2 API cleanup) plus ~15 min for documentation amendments (L2, H2 methodology lift).

The cycle's productionization is functionally correct (bit-exact verified). The findings are about EVIDENCE COMPLETENESS (C1, H3) and RULE COMPLIANCE (H1) and API CLEANLINESS (C2). Honest standing: a cycle that almost-fully internalized the just-saved rule but leaked one place because the rule wasn't applied as an AUDIT to inherited code.
