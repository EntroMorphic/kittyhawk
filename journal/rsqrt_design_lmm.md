---
cycle: rsqrt_design (compressed LMM)
phase: ALL (raw + nodes + reflect + synthesize, single doc)
date: 2026-05-06
scope: design the substrate's rsqrt primitive for RMSNorm. Per the
       bitnet_phase1 SYNTHESIZE work-unit 2. This is a compressed LMM
       cycle (per user authorization) — uncertainty contained enough
       that a single doc captures all four phases.
companions: bitnet_phase1_synthesize.md (work-unit 2); m4t_mtfp.h
---

# rsqrt design — compressed LMM

## RAW: what's actually uncertain

I've been hand-waving about "rsqrt for RMSNorm" for two days. Sitting down to design it surfaces real questions I haven't answered:

- **Representation question.** Substrate's MTFP19 numbers are `mantissa × 3^block_exp`. RMSNorm's `mean(x²) + ε` involves squaring (block exp doubles), summing (block exp same), adding ε (must match block exp), then rsqrt (block exp halves). I haven't worked out what the primitive's signature should be — whether it takes block exps as parameters or expects mantissa-only inputs.

- **Eps representation.** BitNet uses ε = 1e-5 (FP). Substrate has no FP at runtime. The "MTFP19 mantissa with block exp" representation can express 1e-5, but only as `(mantissa, block_exp)` pair. The primitive needs to accept ε in some defined form.

- **Algorithm choice.** Fixed-point rsqrt has at least three plausible implementations:
  1. Newton-Raphson with magic-number initial guess (Quake III-style, but for int).
  2. LUT-based (table lookup + linear/polynomial interpolation).
  3. Float64 evaluate + round-to-int (delegating to libm's sqrt — but libm is FP).

  Option 3 is forbidden in production (project rule). Option 1 is the standard choice for fixed-point but requires careful range bounds. Option 2 has bounded error but eats memory.

- **Output scale.** The rsqrt of a non-trivial input produces a number < 1 in real terms. To fit the result in int32, we need to scale up. By how much? If we choose poorly, we lose precision in the `gamma × x × rsqrt` final multiply.

- **Bit-exactness.** The project rule is NEON-vs-scalar_ref bit-exact verification. Newton-Raphson with limited iterations doesn't converge to a unique answer (depends on initial guess). To make NEON and scalar_ref bit-exact, both must use the SAME initial-guess + iteration count. That's doable but a real constraint.

## NODES: decisions to make

- **N1 — Primitive signature.** Mantissa-only? Or take block exps? Or use a "scale" parameter for the caller to specify?
- **N2 — eps form.** MTFP19 mantissa with implicit scale? (mantissa, block_exp) pair? Pre-scaled?
- **N3 — Algorithm.** Newton-Raphson vs LUT.
- **N4 — Output scale convention.** Fixed (e.g., output = round(2^31 / sqrt(input)))? Caller-specified?
- **N5 — Iteration count.** How many Newton-Raphson refinements? More = more precision, more cost.
- **N6 — Scope: bare rsqrt vs full RMSNorm.** The substrate has no rsqrt today. Should we ship just `rsqrt` (composable but caller assembles RMSNorm) or full `rmsnorm` (less flexible but matches BitNet's exact need)?

## REFLECT

**N6 first — scope.** Shipping just `rsqrt` keeps the primitive small and composable, but the caller still has to do the per-cell `γ · x · rsqrt` loop, which is its own NEON pipeline. Shipping `rmsnorm` matches the exact need but is less reusable. **Decision: ship both.** A bare `m4t_int32_rsqrt(x)` for the rsqrt-of-positive-integer operation, plus a `m4t_mtfp_rmsnorm(y, x, gamma, eps, n)` wrapper that uses it. The wrapper is what BitNet calls; the bare primitive is what other consumers might want.

**N1 — Signature.** For Phase 1, accept mantissa-only inputs. The block-exp semantics live in the caller (the harness handles per-tensor block exponents from the weights blob). RMSNorm's `mean(x²)` is computed in int64, accumulator is mantissa-squared units; the rsqrt operates on this and produces a mantissa-scale-back result. **Tradeoff:** the primitive is "use it correctly or get garbage" — caller is responsible for unit consistency.

**N2 — eps form.** Pass eps as `m4t_mtfp_t` in the same units as `mean(x²)`. The substrate doesn't help the caller pick this; it's documented as "small positive int that prevents division-by-zero at the mantissa-squared scale." Phase 1 fine.

**N3 — Algorithm.** Newton-Raphson with magic-number initial guess. Reasoning:
  - LUT eats memory and the input range (`mean(x²) + ε`) is unbounded above.
  - Newton-Raphson converges quadratically; 3-4 iterations from a good initial guess gives full int32 precision.
  - The "magic number" initial guess (per Quake III's `0x5f3759df` for floats, or analogous for ints) is well-studied.
  - For ints: take advantage of the integer's bit pattern. `rsqrt(x) ≈ 2^((31 - log2(x))/2)`. Compute `log2(x)` via `__builtin_clz` (count leading zeros), use it to pick initial guess.

**N4 — Output scale.** `m4t_int32_rsqrt(x)` returns `round(2^31 / sqrt(x))` for `x ∈ [1, INT32_MAX]`. Output range: [1, ~46341] for x ≥ 2^31; [46341, 2^31] for x ≤ 1. (`46341 ≈ sqrt(2^31)`.) The "2^31" scaling means the caller multiplies `gamma × x × rsqrt_result` and shifts right by 31 to recover the value in `gamma`'s scale.

**N5 — Iteration count.** 3 iterations from a good Quake-style initial guess. Empirically tested in the test_m4t_mtfp_rsqrt suite; if 3 isn't enough for bit-exactness, bump to 4. **For NEON-vs-scalar bit-exact**: scalar_ref uses the SAME initial-guess + 3 iterations; both produce identical output. This is the primitive's bit-exact guarantee, not "matches `1.0/sqrt(x)` from libm to ULP precision."

**Bit-exactness concern.** Newton-Raphson can land on different integers depending on rounding modes during intermediate steps. To make NEON and scalar identical, every step (including intermediate divides) must use the same round-to-zero (or round-to-even) convention. NEON's vshr_n_s32 rounds toward zero by default; we'll mirror that in scalar_ref. Test will catch any divergence.

## SYNTHESIZE: actionable plan

### Primitives to add to libm4t

```c
/* Integer rsqrt: dst = round(2^31 / sqrt(src)). For src ≥ 1.
 * Output range: [1, 46341] (since sqrt(2^31) ≈ 46341).
 * Newton-Raphson with Quake-style integer initial guess; 3 iterations.
 * Caller multiplies result by their input × scale, then >>31, to recover
 * the rsqrt-scaled result in their input's units.
 *
 * Special case: src ≤ 0 returns 0 (caller's responsibility to add ε).
 *
 * Bit-exact NEON vs scalar_ref guaranteed by identical rounding convention. */
m4t_mtfp_t m4t_int32_rsqrt(m4t_mtfp_t src);
m4t_mtfp_t m4t_int32_rsqrt_scalar_ref(m4t_mtfp_t src);

/* RMSNorm: y[i] = γ[i] · x[i] · rsqrt(mean(x²) + ε)
 * Mantissa-only operation. Caller manages block exponents.
 * Internally: int64 sum-of-squares accumulator (no overflow for realistic
 * BitNet inputs after A8 dequantization), call int32_rsqrt on the scaled
 * mean, multiply each cell by gamma and the rsqrt result.
 *
 * Saturating clamp on output (Case S; mantissa fits MTFP19).
 *
 * eps_mantissa is in same units as mean(x²) — caller picks based on
 * desired stability. For BitNet's ε=1e-5 with their typical activation
 * scale, this is something like 1 (small positive int).
 *
 * Per the no-scalar audit: NEON-only production. _scalar_ref test oracle. */
void m4t_mtfp_rmsnorm(
    m4t_mtfp_t* y, const m4t_mtfp_t* x, const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mantissa, int n);
void m4t_mtfp_rmsnorm_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* x, const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mantissa, int n);
```

### Implementation plan

1. **`m4t_int32_rsqrt`** — single-value rsqrt.
   - Initial guess via `__builtin_clz` (gets `log2(x)` cheaply).
   - 3 Newton-Raphson iterations.
   - Round-toward-zero arithmetic throughout (matches NEON `vshrq_n_s32` behavior in scalar_ref).
   - NEON path: same algorithm but using NEON intrinsics (operating on a single value via `vdupq` + `vget_lane`). Single-value rsqrt isn't naturally a SIMD op, but wrapping it in NEON intrinsics keeps the substrate consistent.

2. **`m4t_mtfp_rmsnorm`** — full RMSNorm wrapper.
   - Compute sum-of-squares in int64.
   - Mean: `sum / n` (integer divide).
   - Add eps.
   - Call rsqrt to get scaled rsqrt value.
   - Per-cell: `y[i] = γ[i] × x[i] × rsqrt_scaled` then `>>31` for scale recovery, with saturating clamp.
   - NEON: per-cell multiply via `vmulq_s32` chains + 64-bit intermediate via `vmull_s32` + clamp.

### Tests

- `test_m4t_rsqrt`:
  - Bit-exact NEON-vs-scalar_ref across 10K random inputs in [1, INT32_MAX].
  - Boundary: src=1, src=INT32_MAX, src=0 (returns 0).
  - Property: scalar_ref convergence (rsqrt²×src ≈ 2^62 within rounding).
- `test_m4t_rmsnorm`:
  - Bit-exact NEON-vs-scalar_ref across random γ, x, eps.
  - n boundary: n=1, n=4 (NEON minimum), n=2560 (BitNet's actual size).
  - Aliasing: y == x supported; y == γ asserted-against (or supported, TBD).
  - n=0 noop.

### Wiring into BitNet harness

After this work-unit lands, `bitnet_harness.c` replaces `bitnet_stub_rmsnorm` with `m4t_mtfp_rmsnorm` calls. The eps_mantissa value gets sized empirically — for skeleton run, `1` works; for real-data run, picked based on x's typical magnitude.

### Out of scope for work-unit 2

- Bit-exact match to BitNet's HF reference. The reference computes in bf16 + fp32; we compute in int. ε of difference is documented in work-unit 6's per-layer comparison, not gated here.
- Per-tensor block-exp arithmetic. The harness manages this; the primitive doesn't.
- Optimizing rsqrt's per-call cost. RMSNorm is called 120× per forward pass — significant — but Phase 1's gate is correctness, not throughput.

## Status

Cycle complete. Proceeding to implement per the synthesize plan.
