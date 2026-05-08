---
cycle: routing-shaped 5-in-8 matmul (V1 remediation)
phase: build + test + bench
date: 2026-05-08
scope: first kernel under the pure-ternary-routed architecture
       commitment. Replaces V1 (m4t_ternary_5in8_matmul_bt's
       multiplicative compute) with a dispatch-shaped sibling.
companions: memory/feedback_pure_ternary_routed_architecture.md
            (the directive), journal/pure_ternary_audit_2026_05_08.md
            (the violation map), journal/route_matmul_bench.txt
            (raw bench data).
---

# m4t_ternary_5in8_matmul_bt_route — first architecture-conformant kernel

## What changed

`m4t_ternary_5in8_matmul_bt_route` lands in `m4t_ternary_matmul.c`
as a sibling of the existing `m4t_ternary_5in8_matmul_bt`. Same I/O,
bit-exact output, structurally distinct compute.

Inner per-16-trit dispatch (no SDOT, no vmlal):

  1. Decode 16 packed trits → sign vector via vqtbl1q_s8 LUT
     (already a routing op — table lookup is dispatch).
  2. pos_mask = vceqq_s8(signs, +1)         [1 cycle]
  3. neg_mask = vceqq_s8(signs, -1)         [1 cycle]
  4. pos_sel  = vandq_s8(X, pos_mask)       [1 cycle]
  5. neg_sel  = vandq_s8(X, neg_mask)       [1 cycle]
  6. diff     = vsubq_s8(pos_sel, neg_sel)  [1 cycle]
  7. acc     += vaddlvq_s8(diff)            [reduction]

The masks are binary at the hardware-instruction level (a hardware
primitive beneath the architecture). The architecture-level operation
is "select X based on trit value." No multiplication is performed.
Lanes where the trit routes to 0 contribute 0 because the mask zeros
them — not because anything multiplies by 0.

## Architecture compliance

Per `memory/feedback_pure_ternary_routed_architecture.md`:

  (1) Pure ternary           — ✓ trit values stored 5-in-8 base-3,
                                decoded via vqtbl LUT to ±1/0
  (2) Routed                 — ✓ per-cell dispatch via mask+select
                                instead of multiply-by-trit
  (3) Non-dense              — ✓ lanes routing to 0 are masked to 0
                                (zero contribution by routing decision,
                                not by multiply-by-zero arithmetic)
  (4) No binary structures   — ✓ no parallel binary indicator sets;
                                masks exist only at instruction level
                                as transient hardware primitives
  (5) No scalar ops          — ✓ no scalar tail; K%80 boundary tile
                                handled via stack-local zero-padded W

This is the first kernel under the new architectural commitment that
satisfies all five conditions.

## Bit-exactness

Test extensions in `test_m4t_ternary_5in8_matmul.c`:
  - Aligned bit-exact across {K=80,160,320} × {M=1,4,8} × {N=4,16,64}
    (5 random samples each)
  - K%80 sweep: K = 160 + km for km ∈ {1..79}, M=1, N=64 (2 samples each)
  - K<80: K ∈ {1, 5, 17, 33, 79} × M ∈ {1, 4} × N ∈ {64, 5}
  - M>1 + K%80 ≠ 0: K=167 M=4 N=17 and K=287 M=8 N=5
  - K=0 explicit (verifies memset-zero behavior)

All pass bit-exact vs `m4t_ternary_5in8_matmul_bt_scalar_ref`.

## Bench (5 BitNet shapes + K%80 sweep, n=200, σ reported)

  Shape                   dense (ms)    route (ms)    dense/route
  ----------------------+-------------+-------------+--------------
  q/o_proj K=N=2560      0.082 ± 0.006 0.232 ± 0.017  0.351×
  k/v_proj K=2560 N=640  0.020 ± 0.001 0.055 ± 0.001  0.355×
  gate/up output         0.207 ± 0.006 0.590 ± 0.012  0.350×
  down_proj K=6912       0.210 ± 0.005 0.597 ± 0.009  0.351×
  K=2400 K%80=0          0.072 ± 0.002 0.206 ± 0.005  0.350×
  K=2479 K%80=79         0.075 ± 0.002 0.215 ± 0.006  0.349×
  K=2480 K%80=0          0.075 ± 0.003 0.213 ± 0.005  0.351×

**Route is ~2.85× slower than dense, consistently across every shape.**
The factor is shape-independent because the per-tile op-count delta
is constant (5 SDOTs replaced by 5 × (2 compares + 2 ands + 1 sub +
1 reduce) per 80-trit chunk, ~5× more NEON-issue ops).

## End-to-end implication for BitNet inference

If the harness switches from `_bt` to `_bt_route`:
  - BitLinear share per token: ~26.0 ms × 2.85 ≈ 74 ms
  - Total per-token user CPU: ~74 + ~5 (non-BitLinear) = ~79 ms
  - Was: 31 ms/token (per `journal/cumulative_bench_2026_05_08.md`)
  - **~2.5× per-token slowdown, end-to-end**

That's the cost of the architectural commitment, on this hardware,
on this representation. Bit-exact output; different compute.

## What this kernel does NOT do (yet)

- It still operates dense over the (i, j) output grid. Every output
  cell is computed regardless of any routing predicate at the
  output level. V12 in the audit (output-grid density) is not
  addressed by this kernel — that's a separate primitive.

- It does not exploit any structural sparsity in the trit data.
  The decoded sign vector has 0s for zero-trit lanes; the dispatch
  zeros them via mask but the cycle is still spent on the comparison
  and AND. Pure routing in the strictest sense would skip the
  comparison entirely for known-zero lanes — but that requires
  data-dependent control flow that NEON doesn't provide for free.

- It does not yet replace the dense kernel in the BitNet harness.
  The harness still calls `m4t_ternary_5in8_matmul_bt`. Switching
  is a deliberate next step requiring decision on whether to make
  route the default or keep both available with explicit selection.

## What's next

Two distinct directions:

1. **Switch the harness path**: replace `m4t_ternary_5in8_matmul_bt`
   calls in BitNet harness with `_bt_route`. Re-bench end-to-end.
   This makes BitNet inference architecturally conformant at the cost
   of ~2.5× wall time per token.

2. **Continue the audit remediation**: V2 (xpacked) is the same
   structural fix on the X-packed sibling. V3 (mtfp4_sdot) is the
   K%16 version. V4 (mtfp_ternary) needs a vmlal → mask+select
   refactor. None are in BitNet's hot path; none have BitLinear-shape
   urgency, but each closes a violation.

The user's directive was "100/100, methodically." Both paths above are
required to reach full conformance. (1) makes BitNet conformant in
practice; (2) makes the substrate API consistent.

## Cycle status

Kernel: built, bit-exact, ASAN-pending verify, benched. Production-
viable in the sense of correctness; production-deferred in the sense
of harness integration.

Next user decision: switch harness now (eats the 2.5× cost on
BitNet) or continue audit remediation across siblings (V2, V3, V4)
before the harness switch.
