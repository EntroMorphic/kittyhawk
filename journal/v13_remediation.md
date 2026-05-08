---
cycle: V13 — scalar production code in BitNet harness
phase: remediation + bit-exact verify + bench + commit
date: 2026-05-08
scope: closes V13 of pure-ternary architecture audit by NEON-ifying
       the four scalar production sites in bitnet_harness.c. Adds
       libm4t helper m4t_mtfp_vec_dot_i64 for the int32×int32→int64
       dot product reuse.
companions: journal/harness_switch_redteam.md (where V13 was filed),
            commit ef10c8d (the red-team that filed V13).
---

# V13 — scalar production code remediated

## What was scalar (pre-V13)

bitnet_harness.c had four production sites violating condition (5)
"No scalar ops":

  1. lines 395-399: residual-add scalar loop (per-token, per-layer)
  2. lines 472-476: second residual-add scalar loop (per-token, per-layer)
  3. lines 586-595: bitnet_lm_head — int32×int32 scalar dot product
  4. lines 608-616: bitnet_argmax_full_vocab — int32×int32 scalar dot,
                    over full 128k vocab

Sites 1+2 ran in the per-token-per-layer hot path. Sites 3+4 ran
once per inference (or per generation step).

None of these violate the routing/non-dense conditions (they
operate on int32 mantissas, no ternary trit to dispatch on), but
they ARE scalar in production code, violating condition (5).

## V13.A: residual adds → m4t_mtfp_vec_add_inplace

The substrate already had a NEON-only saturating-add primitive:

  void m4t_mtfp_vec_add_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n);

It uses m4t_mtfp_block_add (vaddq_s32 + vminq_s32 + vmaxq_s32) for
4-cell blocks. Saturation to ±MTFP19_MAX is built in.

The harness's scalar loops:

  for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) {
      int64_t v = (int64_t)s->residual[i] + (int64_t)s->x[i];
      s->x[i] = (int32_t)((v > 581130733) ? 581130733 :
                          (v < -581130733) ? -581130733 : v);
  }

581130733 == M4T_MTFP_MAX_VAL. Bit-exact equivalent to:

  m4t_mtfp_vec_add_inplace(s->x, s->residual, BITNET_HIDDEN_SIZE);

Both replaced. BITNET_HIDDEN_SIZE = 2560, n%4 = 0, so no scalar
tail fires within the helper. Pure NEON.

## V13.B: lm_head + argmax → new helper m4t_mtfp_vec_dot_i64

The substrate didn't have a generic int32×int32→int64 dot product.
Added a new helper:

  int64_t m4t_mtfp_vec_dot_i64(
      const m4t_mtfp_t* x, const m4t_mtfp_t* y, int n);
  /* + scalar_ref oracle for tests */

Implementation: vmlal_s32 chain over int32x4 chunks, two int64x2
accumulators (acc_lo, acc_hi for low/high halves). Boundary tile
handles n%4 != 0 via stack-local zero-padded 4-element buffers
(no scalar tail). Bound analysis documented: caller responsible
for n bound when both operands are near MTFP19_MAX.

bitnet_lm_head's per-row dot loop replaced:

  int64_t acc = m4t_mtfp_vec_dot_i64(x, row, BITNET_HIDDEN_SIZE);

bitnet_argmax_full_vocab's per-row dot loop replaced similarly.

## Tests added

test_m4t_mtfp.c gains 4 new test functions:

  test_vec_dot_i64_empty       — n=0 returns 0
  test_vec_dot_i64_aligned     — n%4=0 across n ∈ {4..2560}, NEON vs scalar_ref
  test_vec_dot_i64_unaligned   — n%4 ∈ {1,2,3} across n ∈ {1..2563},
                                  boundary tile fires, NEON vs scalar_ref
  test_vec_dot_i64_extreme     — high-magnitude inputs at n=2560

All pass. ASAN+UBSAN halt_on_error=0 clean across the full m4t
test suite (29/29).

## Bit-exactness — 4 prompts × 2 architectures

  Prompt           PRE-V13 argmax    POST-V13 argmax    diff
  ---------------- ----------------- ------------------ ----
  Capital France   220               220                0
  Largest planet   50789             50789              0
  Quick brown fox  279               279                0
  def fibonacci    471               471                0

Plus x[0..3] and logits[0..3] match across all 4 prompts.
End-to-end bit-exact.

## Performance — paired in-session bench

8 runs each variant, positions=32, user CPU:

  PRE-V13 (post-harness-switch): mean 2.683s, σ 0.009  → 81.7 ms/token
  POST-V13:                      mean 2.65s,  σ 0.022  → 80.6 ms/token
                                 (cool-start outlier excluded)

V13 is approximately neutral with slight improvement (~1 ms/token,
~1.4%). The improvement comes from argmax_full_vocab — the only
substantial dot product fired (128k × 2560 = 327M MACs / inference).

The residual replacement is not measurable because:
  - Per-token cost was ~150 µs scalar pre-V13
  - Function-call overhead for m4t_mtfp_vec_add_inplace is ~50 µs
    (1920 calls per inference × 30 ns/call avg)
  - Net is roughly neutral; compiler may have auto-vectorized the
    original scalar loops at -O3, narrowing the gap.

## Architecture state after V13

  (1) Pure ternary           ✓ (matmul path, V1-V8)
  (2) Routed                 ✓ (matmul path, V1-V8)
  (3) Non-dense              ✓ (matmul path, V1-V8)
  (4) No binary structures   ✓ (V6 routed16 removed)
  (5) No scalar ops          ✓ (V13 closes the harness's
                                 scalar production sites)

The substrate's complete production compute path is now
architecture-conformant. The geometric scalar tails permitted by
CONTRIBUTING.md (V9-V11) remain — those are sub-block remnants of
NEON kernels, not whole scalar functions.

## V13 audit closed

End of the pure-ternary architecture audit. All matmul-shape
violations addressed in kernels (V1-V8). All structural binary
indicators removed (V6). All harness-level scalar production
sites NEON-ified (V13). The "math as signatures via routing"
foundation is operationally true throughout the BitNet inference
path.

Cumulative cost (vs. multiplicative baseline pre-cycle):
  ~2.6× per-token wall time (78-81 ms vs 30.6 ms).
  Bit-exact across all tested prompts.

The architecture is the architecture. The substrate computes
ternary information through routing operations end-to-end, no
scalar code in production, no binary structures encoding ternary
state. End of arc.
