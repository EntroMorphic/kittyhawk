---
cycle: harness switch to architecture-conformant kernels
phase: end-of-arc commit + bench
date: 2026-05-08
scope: switches BitNet inference and gesh consumers to call the
       routing-shaped kernels (_bt_route, _sdot_matmul_bt_route)
       built across V1-V4 of the pure-ternary audit. BitNet
       inference now operates on architecture-conformant compute
       paths from prompt embedding through LM-head logits.
companions: memory/feedback_pure_ternary_routed_architecture.md
            (the directive), journal/cumulative_bench_2026_05_08.md
            (pre-switch BitNet baseline), V1-V12 commits.
---

# Harness switched to architecture-conformant kernels

## What changed

### bitnet_harness.c — 7 call sites

  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  all switch from m4t_ternary_5in8_matmul_bt(...)
                  to m4t_ternary_5in8_matmul_bt_route(...)

  Replacement was a precise sed against the function-call pattern
  (paren-suffixed); 7 of 9 occurrences in the file matched (the
  other 2 are in comments referring to the kernel by name —
  documentation, not calls).

### m4t_ternary_dot_matmul_bt — internal switch (V5 fully closed)

  This wrapper delegates to mtfp4_sdot. Switched from
    m4t_mtfp4_sdot_matmul_bt(...)
                  to m4t_mtfp4_sdot_matmul_bt_route(...)

  gesh_project.c (3 call sites) inherits architecture conformance
  via this wrapper without any changes to gesh source.

## What is NOT changed

### Bench code keeps multiplicative calls (deliberate)

  /tmp/bench_*, m4t/audit/*, gesh/bench/denoise_probe.c — all
  deliberately call the multiplicative kernels for comparison
  measurements. These exist to MEASURE the substrate, not to BE
  the substrate. Leaving them alone is correct.

### Tests keep both kernels (deliberate)

  test_m4t_ternary_5in8_matmul, test_m4t_ternary_matmul_neon,
  test_m4t_ternary_5in8_xpacked, test_m4t_mtfp4 all explicitly
  test BOTH the multiplicative and route variants for bit-exact
  equivalence. That's their job.

## Bit-exactness verification

### Test suite: 29/29 pass

All m4t and gesh tests pass post-switch under release build.

### End-to-end on real BitNet weights

  Prompt: "The capital of France is" (5 tokens, 30 layers)
    PRE-switch:  argmax = 220, x[0..3] = [-297, 308, 133, 1379],
                 logits[0..3] = [393, 644, 397, 912]
    POST-switch: argmax = 220, x[0..3] = [-297, 308, 133, 1379],
                 logits[0..3] = [393, 644, 397, 912]
    DIFF:        bit-exact

  Two additional prompts ("Largest planet ...", "The quick brown
  fox ...") produce coherent argmax tokens and reasonable logit
  magnitudes (no crashes, no UB).

## Performance — measured 5-run end-to-end

  positions=32 user CPU, 5 fresh runs, σ ≈ 0.04s:

                          PRE-switch     POST-switch
  Per-token user CPU      31.1 ms        78.3 ms
  Slowdown                  1.00×          2.52×

  This matches the prediction (~2.5×) from the kernel-level
  bench in journal/route_matmul_kernel.md and the cumulative
  measurement in journal/cumulative_bench_2026_05_08.md.

  Steady-state user time at positions=32:
    2.52, 2.57, 2.57, 2.61, 2.62 s (mean 2.578 s)

  Per-token: (2.578 - 0.15_setup) / 31 = 78.3 ms

## Architecture status after this commit

The substrate's BitNet inference path is now architecture-conformant
end-to-end. Per memory/feedback_pure_ternary_routed_architecture.md:

  (1) Pure ternary           ✓ all matmul callers traverse
                                ternary-data, base-3-packed kernels
  (2) Routed                 ✓ all per-cell decisions dispatch on
                                trit value via mask+select, not
                                multiplication
  (3) Non-dense              ✓ zero-trit lanes contribute 0 by
                                routing (mask zeros), not by
                                multiply-by-zero arithmetic
  (4) No binary structures   ✓ no parallel binary indicator sets
                                (routed16 removed in V6); transient
                                instruction-level masks are
                                hardware-substrate primitives,
                                acceptable per the directive
  (5) No scalar ops          ✓ matmul kernels (V1-V4) have NEON
                                boundary-tile paths; V9-V11
                                element-wise scalar tails remain
                                under CONTRIBUTING.md's geometric
                                tail rule

## What this costs

  ~2.5× per-token wall-clock slowdown on BitNet inference.

  This cost is the architectural commitment, on this hardware,
  with this representation, in this implementation pass. There
  is residual optimization headroom (vaddlvq → vector accumulator,
  bit-pack signs, etc.) that could reduce the gap to ~2× or less.
  None of those optimizations are gating; they are future
  refinements.

## What the substrate is now

The substrate's matmul compute path is pure-ternary, operationally
routed, non-dense, no-binary-structures, no-scalar-ops. The math
the substrate computes is unchanged (bit-exact to the prior
implementation); the operations the hardware executes are
architecturally aligned with the project's foundation.

The "math as signatures via routing" claim from the project vision
is now substantively true at the operation level, not just the
mathematical level. Routing is what the substrate does, not just
what it represents.

## Audit final state

  V1-V8: closed via routing-shaped kernels + production switches
  V9-V11: filed (auxiliary element-wise scalar tails;
                  CONTRIBUTING.md geometric-tail rule permits;
                  not in BitNet's critical path)
  V12: closed (re-framing — "non-dense" is per-K-cell, not
                output-grid)

End of the pure-ternary architecture cycle.
