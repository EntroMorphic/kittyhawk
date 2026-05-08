---
cycle: harness switch — red-team findings
phase: post-commit corrections
date: 2026-05-08
scope: red-team of commit 7b9b8ef ("Harness switch") finds three
       material issues. Two are corrections to claims; one is a
       new audit item (scalar code in the harness path that isn't
       part of any matmul kernel).
companions: journal/harness_route_switch.md (the commit being
            red-teamed), commit 7b9b8ef.
---

# Red-team of the harness switch

## Findings, ranked

### F1 [HIGH] "End-to-end architecture-conformant" overclaim

The previous journal stated:

> The substrate's BitNet inference path is architecture-conformant
> end-to-end.

This is **false**. The matmul-shape compute (V1-V8 closures + the
harness switch) is conformant. But the harness contains other
production code that is scalar:

  bitnet_harness.c:395-399 — residual-add scalar loop (per-layer,
                              per-token, in the per-token hot path)
  bitnet_harness.c:472-476 — second residual-add scalar loop
                              (per-layer, per-token)
  bitnet_harness.c:586-595 — bitnet_lm_head: scalar dot product
                              for top_n logits (one-shot)
  bitnet_harness.c:608-616 — bitnet_argmax_full_vocab: scalar dot
                              product over full vocabulary
                              (one-shot)

These are NOT geometric scalar tails (the geometric-tail rule per
CONTRIBUTING.md only permits sub-block remnants of NEON kernels).
They are entire scalar functions in production code. They violate
condition (5) "No scalar ops" of the pure-ternary directive.

They do NOT violate the routing or non-dense conditions, because
they operate on int32 mantissas (MTFP19), not on ternary trits.
There is no trit value to dispatch on.

**Corrected scope:** the BitNet **matmul-shape compute path** is
architecture-conformant. The auxiliary harness operations
(residual additions, lm_head dot product, argmax search) violate
condition (5) and are filed as a new audit item below.

### F2 [HIGH] New audit item filed

  V13 [audit candidate] Scalar production code in BitNet harness.
    Three sites in bitnet_harness.c (residual adds × 2, lm_head,
    argmax_full_vocab). All scalar. Per-token cost: ~150 µs from
    residuals alone (153,600 saturating adds × 2 cycles / 3.2 GHz).
    lm_head and argmax_full_vocab are one-shot per inference (or
    per generation step), not per-token-per-layer.

    Fix shape: NEON int32 vector ops:
      - vsubq_s32 + vmaxq_s32/vminq_s32 for saturating add
      - vmlal_s32 chains for the dot products (similar to existing
        ternary_dot_vmlal but with int32 × int32 instead of
        int8-trit × int32)

    Out of scope for this red-team (would expand the cycle's
    surface significantly). Filed in the audit's V13 slot.

### F3 [MEDIUM] Cross-session bench was approximate

The previous journal compared 31.1 ms/token (cross-session, from
journal/cumulative_bench_2026_05_08.md) to 78.3 ms/token (this
session) to claim 2.52× slowdown. Apple Silicon thermal/perf
state varies between sessions; that comparison was approximate.

**Corrected with paired in-session bench (5 runs each):**

  PRE-switch  user time @ positions=32: 1.10 ± 0.01 s
              per-token: 30.6 ms

  POST-switch user time @ positions=32: 2.57 ± 0.04 s
              per-token: 78.1 ms

  Slowdown:   78.1 / 30.6 = **2.55× per-token** (σ < 2%)

The slowdown is real and reproducible. Cross-session ~2.52× and
in-session 2.55× are within measurement noise.

### F4 [MEDIUM] Bit-exactness verified across 4 prompts (was 1)

The previous journal verified bit-exactness only on the "Capital
of France" prompt. Two other prompts were claimed to "produce
coherent outputs" without explicit pre-vs-post comparison.

**Corrected with cross-prompt verification:**

  Prompt                argmax  x[0..3]                   logits[0..3]
  --------------------- ------- ------------------------- ------------
  Capital of France     220     -297, 308, 133, 1379      393,644,397,912
  Largest planet        50789   -69, 203, -229, 498       217,422,145,118
  Quick brown fox       279     -204, -238, -183, 1527    861,961,553,547
  def fibonacci         471     266, 454, 176, 260        328,731,766,601

  ALL 4 prompts: PRE-switch output BIT-EXACT to POST-switch output.
  (`diff` of the captured logs is empty.)

The kernel-level bit-exact tests (V1-V4) plus this end-to-end
cross-prompt verification gives strong confidence the harness
switch is correct.

### F5 [LOW] Wrapper switch silently slows gesh consumers

V5's internal switch (m4t_ternary_dot_matmul_bt now wraps
_sdot_matmul_bt_route) makes gesh consumers (gesh_project, ~3
call sites) operate through the routing kernel without source
changes. They inherit conformance for free — but also inherit
the ~2× slowdown for free.

This is correct architecturally (gesh inherits the substrate's
architecture commitment) but a behavioral change that gesh's
test suite passes through. The change is documented in the
wrapper's docstring (m4t_ternary_matmul.c:343-353) but the
journal entry could have been more explicit about the
implication for gesh consumers.

Filed for documentation update; not a code-correctness issue.

## What the red-team did NOT change

- The matmul-shape compute is conformant per the directive.
- The harness switch IS bit-exact end-to-end (verified on 4 prompts).
- The 2.55× slowdown is real and measured.
- V1-V8 + V12 audit closures stand.

## Corrections to the previous journal

1. "End-to-end conformant" → "matmul-shape compute conformant"
   (residuals, lm_head, argmax remain scalar)
2. Cross-session 2.52× → paired in-session 2.55× (more rigorous)
3. Bit-exactness: 1 prompt → 4 prompts verified

## Audit state after this red-team

  V1-V8: closed (kernel-level remediation)
  V9-V11: filed (geometric-tail allowance per CONTRIBUTING.md)
  V12: closed (re-framing)
  V13: NEW — scalar production code in bitnet_harness.c
        (residual adds × 2, lm_head, argmax_full_vocab).
        Out of scope for the matmul audit; filed as separate
        cycle. ~150 µs/token impact for residuals; one-shot for
        lm_head/argmax.

## Disposition

The pure-ternary architecture commitment is satisfied for the
BitNet matmul-shape compute path. The harness has additional
scalar production code (V13) that is not subject to the routing
condition (no ternary involved) but does violate the no-scalar
condition. Addressing V13 requires writing or reusing NEON kernels
for int32 saturating add and int32 × int32 dot product.

For the BitNet inference path:
  - Per-token cost: 78.1 ms (vs 30.6 ms pre-switch, 2.55× slowdown)
  - All matmul cells dispatch on trit value
  - Output bit-exact to pre-switch
  - V13 represents ~150 µs/token of scalar overhead in residuals
    that would still be there even after V13 is fixed (since it's
    a constant per-cell cost), but at NEON throughput rather than
    scalar — likely ~10 µs/token instead of 150 µs.

The architecture commitment is substantially satisfied. The
remaining violation (V13) is bounded and addressable in a
follow-up cycle.
