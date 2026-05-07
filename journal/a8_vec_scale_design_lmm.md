---
cycle: a8_vec_scale_design (compressed LMM)
phase: ALL (raw + nodes + reflect + synthesize)
date: 2026-05-05
scope: design the substrate's A8 quantize/dequantize + vector scale
       (Gap #5) primitives. Per work-unit 5 of bitnet_phase1_synthesize.
---

# A8 + vec_scale design — compressed LMM

## RAW: what's actually uncertain

- **A8 quantize input/output scale.** A8 = per-token absmax + int8.
  HF spec: y_int8 = round(x · 127 / absmax), clamped to [-127, 127].
  Substrate must produce identical output (bit-exact rounding).
- **Reciprocal use.** absmax is computed dynamically per call; can't
  precompute. Need pure-int (x · 127) / absmax with round-to-nearest.
  C's `/` truncates toward zero → use (num + half) / denom for round-half-away.
- **vec_scale signature.** BitLinear chain: y = matmul_out × α × absmax / 127.
  α is per-tensor scalar (per-projection). absmax is the activation scale.
  The combined scale is `(α × absmax) / 127`.
  Scaled by ratio of int64 numerator to int constant denominator.
- **NEON vs scalar.** Per-cell divide by absmax is scalar (no NEON int div
  intrinsic). Per cell-loop is scalar; documented per the cross-exp
  accum's degenerate-case precedent.
- **α block exponent.** convert_weights.py emits each α as
  `(mantissa, block_exp)`. For Phase 1, the mantissa lives in the
  blob; block_exp may or may not be 0. The scale apply must
  account for it: real_α = mantissa × 3^block_exp.
  - **Decision**: ship vec_scale as `y = x · num / den`. Caller
    composes (num, den) including any 3^block_exp factor. For Phase
    1 work-unit 5, apply the scale assuming block_exp = 0 if the
    blob doesn't have higher-precision metadata; address full block_exp
    semantics at work-unit 6 if needed.
  - **Aside**: 3^block_exp can be expressed as an integer multiplier
    when block_exp ≥ 0; for negative block_exp, it'd be a fractional
    multiplier (handled via larger den). Phase 1 simplification: assume
    α's block_exp ≥ 0 (almost always true for absmean of small bf16
    values).

## NODES

- **N1 — m4t_a8_quantize signature.** Returns the absmax (caller stores).
- **N2 — m4t_a8_dequantize signature.** Takes absmax explicitly.
- **N3 — vec_scale signature.** num/den ratio? Or num + shift?
- **N4 — Round-to-nearest convention.** Half-away-from-zero
  (matches torch.round in HF reference) vs banker's rounding.
- **N5 — Saturation.** A8 clamps to [-127, 127]. vec_scale clamps to
  ±M4T_MTFP_MAX_VAL.
- **N6 — α wiring into bitnet_layer_weights_t.** Add 7 alpha pointers
  per layer, plus the corresponding 7 block_exp ints.

## REFLECT

**N1 — A8 quantize.** Returns absmax as m4t_mtfp_t. Caller passes
this to dequant or composes with α for vec_scale.

**N2 — A8 dequantize.** y[i] = x_int8[i] · absmax / 127. Round half
away from zero. Saturating clamp to MTFP19.

**N3 — vec_scale signature.** y[i] = round(x[i] · num / den).
- num: int64. Composes the BitLinear scale: num = α_mantissa × absmax × (3^α_block_exp).
  For α_block_exp ≤ 18 (likely; α is small), 3^18 ≈ 3.9e8 fits int32, and
  α_mantissa × absmax × 3^bx fits int64. Documented constraint.
- den: int64. For BitLinear: den = 127.
- Round-half-away-from-zero, saturating to ±M4T_MTFP_MAX_VAL.

**N4 — Round-half-away-from-zero.** Matches HF's torch round semantics.
Stub already does this. Production matches.

**N5 — Saturation.** A8 → int8 clamp [-127, 127]. vec_scale → MTFP19
clamp via m4t_mtfp_clamp64.

**N6 — α wiring.** Extend bitnet_layer_weights_t:
```c
const m4t_mtfp_t* alpha_q;  /* scalar mantissa */
int alpha_q_block_exp;
... (7 projections × 2 fields)
```
Or as arrays:
```c
const m4t_mtfp_t* alpha[7];
int alpha_block_exp[7];
```
Latter is uglier but more compact. For Phase 1, separate fields per
projection (clearer at call sites).

Actually: the harness call sites are explicit per-projection, so name
each field. For block_exp: if α is always tiny enough to have
block_exp ≤ small_constant, we can store as a fixed int per
projection.

## SYNTHESIZE

### Primitives

```c
/* A8 quantize: returns per-tensor absmax. Output: int8 with
 * y_int8[i] = round(x[i] · 127 / absmax), saturating to [-127, 127].
 *
 * If input is all-zero, returns 0 absmax and zeros y (caller must
 * handle the all-zero case in dequant if it cares — here we treat
 * it as identity 0×anything = 0).
 *
 * Pure-int implementation: integer divide with half-away-from-zero
 * rounding. Per-cell scalar (NEON int divide isn't a thing). */
m4t_mtfp_t m4t_a8_quantize(int8_t* y, const m4t_mtfp_t* x, int n);
m4t_mtfp_t m4t_a8_quantize_scalar_ref(int8_t* y, const m4t_mtfp_t* x, int n);

/* A8 dequantize: y[i] = round(x_int8[i] · absmax / 127), MTFP19-clamped.
 * Used at the end of the BitLinear chain to recover MTFP19 mantissas.
 * For BitLinear: typically vec_scale composes the dequant + α multiply
 * in one step. */
void m4t_a8_dequantize(m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax, int n);
void m4t_a8_dequantize_scalar_ref(m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax, int n);

/* Vector scale: y[i] = round(x[i] · num / den), MTFP19-saturating.
 *
 * For BitLinear: num = α_mantissa × absmax × 3^α_block_exp,  den = 127.
 *
 * num and den are int64 to allow large composed scales. Caller
 * computes 3^α_block_exp as a separate int factor and folds into num.
 *
 * Per-cell __int128 multiply (x × num can hit 2^29.1 × 2^63 = 2^92.1,
 * exceeds int64). Per-cell scalar; NEON-vectorize deferred. */
void m4t_mtfp_vec_scale(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int64_t num, int64_t den, int n);
void m4t_mtfp_vec_scale_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int64_t num, int64_t den, int n);
```

### Tests

- **A8 round-trip**: quantize → dequantize → check |y - x| ≤
  absmax/127 (the inherent quantization error).
- **A8 boundary**: all-zero input, single-spike, MTFP19_MAX magnitude.
- **A8 vs FP scalar_ref**: bit-exact (since both use the same int
  divide convention).
- **vec_scale**: identity (num=den=1), scaling factor up/down,
  saturation at MTFP19_MAX boundary.

### Harness wiring (work-unit 5 scope)

- Add α pointer + block_exp fields to bitnet_layer_weights_t.
- Extend bitnet_weights.c loader to populate them.
- Replace bitnet_stub_a8_quantize/dequantize with substrate
  primitives.
- After each BitLinear matmul, apply m4t_mtfp_vec_scale with
  num = α_mantissa × absmax × 3^α_block_exp, den = 127.

### Out of scope

- α with non-trivial block_exp arithmetic (deferred to work-unit 6).
- BitLinear input-side quantization scale composition (handled by
  the per-call A8 quantize already).

## Status

Cycle complete. Proceeding to implementation.
