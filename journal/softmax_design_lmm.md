---
cycle: softmax_design (compressed LMM)
phase: ALL (raw + nodes + reflect + synthesize)
date: 2026-05-05
scope: design the substrate's softmax primitive for BitNet's attention.
       Per work-unit 4 of bitnet_phase1_synthesize.
companions: bitnet_phase1_synthesize.md (work-unit 4); bitnet_stubs.c
            (current FP stub).
---

# softmax design — compressed LMM

## RAW: what's actually uncertain

- **Input scale.** Attention scores arrive at *some* scale that depends
  on Q · K block exponents + 1/sqrt(head_dim) factor. The substrate
  primitive must take input in a defined form.
- **Reciprocal primitive.** Softmax is `exp(x − max) / Σ exp(x − max)`.
  The division requires `1/sum`. We have rsqrt (`1/sqrt`) but no
  reciprocal. New primitive needed.
- **exp LUT range.** exp(z) underflows to 0 for z < ~-20 (at scale
  2^30). LUT covers a finite range; values below get treated as 0.
- **Output scale.** Softmax probabilities ∈ [0, 1], summing to 1.
  Internal pipeline scale can be 2^30 or similar.
- **Wiring.** The harness's attention path is fully stubbed (scores =
  memset 0, output = memset 0). Softmax has no consumer yet — it's a
  primitive ship-only for Phase 1; wiring happens in work-unit 6 (full B).

## NODES

- **N1 — Input contract.** Take x[i] as int32, treated as real-valued
  natural-log argument. Caller is responsible for pre-scaling their
  scores. The primitive does NOT auto-rescale (would change the
  distribution).
- **N2 — Reciprocal primitive.** Add `m4t_int32_recip(int64_t src)`
  → `round(2^30 / src)` for src in some range. Newton-Raphson:
  `y_new = y · (2·Q − src · y) / Q` for fixed-point Q.
- **N3 — exp LUT structure.** LUT covers z ∈ [-LUT_RANGE, 0] sampled
  at LUT_RES points. exp(z) at scale 2^Q. Values below the range
  → 0 (underflow); values above 0 → assert fail (not allowed; caller
  should always pass z ≤ 0 after subtracting max).
- **N4 — Output scale.** y[i] = round(prob × 2^Q_OUT). Q_OUT = 30.
  Sum of y[i] ≈ 2^30.
- **N5 — Numerical stability.** Subtract max(x) before computing exp.
  Standard. No iterative refinement needed.

## REFLECT

**N1 — Input contract.**
The BitNet inference pipeline's attention scores will be at some
specific scale once we wire it (work-unit 6). For the softmax
primitive, we accept x[i] as int32 representing natural-log argument:
exp(x[i]) is the unnormalized weight. Caller pre-rescales their
QKᵀ × 1/sqrt(d) result to this form.

For a typical Llama-family attention:
- QKᵀ at scale Q^2_block_exp, range ~[-1e3, 1e3] in real terms.
- After / sqrt(head_dim) and shift to natural-log scale, values are
  typically in [-30, 30] for BitNet's distribution.

The substrate primitive expects the caller to deliver the int32
representation where 1 LSB = 1 natural-log unit. For Phase 1, this
is a documented contract; work-unit 6's score-prep step will produce
this form.

**N2 — Reciprocal primitive.**
NR for 1/x:
  y_{n+1} = y_n · (2 - x · y_n)
Fixed-point at scale Q = 2^30:
  y_{n+1} = y_n · (2·Q - x · y_n) / Q

Initial guess: y_0 = 2^30 / x_approx, where x_approx comes from the
bit pattern (analogous to rsqrt's clz trick).

For x in [1, INT32_MAX], output ∈ [1, 2^30]. Same shape as rsqrt.

Need: `m4t_int32_recip(m4t_mtfp_t src)` returning `round(2^30 / src)`.
Same structure as rsqrt — Newton-Raphson with __int128 intermediates,
3-5 iterations.

**N3 — exp LUT.**
LUT_RANGE = 30 (covers exp(-30) ≈ 1e-13 down to underflow).
LUT_RES = 4096 entries (granularity 30/4096 ≈ 0.0073). Linear
interpolation between entries doubles effective resolution.

LUT[k] = round(exp(-k · 30 / 4096) · 2^30) for k ∈ [0, 4095].
Built once at init via libm exp.

For input z ≤ 0:
  if (z ≤ -LUT_RANGE) return 0;
  index = -z · LUT_RES / LUT_RANGE
  (linear interp between LUT[index] and LUT[index+1])

**N4 — Output scale Q_OUT = 30.**
Each prob[i] ∈ [0, 2^30]. Sum ≈ 2^30 (within rounding).

When this softmax output is used downstream (multiplied by V):
  result = prob[i] · V[i] / 2^30
In Phase 1's wiring (unit 6), this looks like: int64 multiply →
>>30 → MTFP19 mantissa.

**N5 — Numerical stability.**
Standard subtract-max. Implementation:
  max = max(x)
  for each i:
    e[i] = exp_lut(x[i] - max)  [int32, scale 2^30]
  sum = Σ e[i]                   [int64, max ~ n × 2^30]
  inv_sum = m4t_int32_recip(sum / scale_to_int31) ... (with Q-format compensating)
  for each i:
    y[i] = (e[i] × inv_sum) >> 30

Sum can exceed int31 — for n=4096 and all e[i]=2^30, sum = 2^42. Need
to handle this as we did in rmsnorm (pre-shift before reciprocal,
compensate output).

## SYNTHESIZE

### Primitives

```c
/* Integer reciprocal: dst = round(2^30 / src) for src ∈ [1, INT32_MAX]. */
m4t_mtfp_t m4t_int32_recip(m4t_mtfp_t src);
m4t_mtfp_t m4t_int32_recip_scalar_ref(m4t_mtfp_t src);

/* Softmax: y[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x)).
 *
 * Input contract: x[i] is int32 representing natural-log units
 * (1 LSB = 1 nat). For BitNet attention scores, caller pre-rescales
 * their QKᵀ × 1/sqrt(d) result to this form.
 *
 * Output: y[i] ∈ [0, 2^30] with Σ y[i] ≈ 2^30 (probabilities at
 * scale 2^30).
 *
 * Implementation: subtract max, exp via LUT (init-time FP libm),
 * reciprocal of sum via m4t_int32_recip, per-cell multiply.
 *
 * n must be ≥ 1. For empty input, asserts. */
void m4t_mtfp_softmax(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n);
void m4t_mtfp_softmax_scalar_ref(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n);
```

Constants:
- M4T_SOFTMAX_LUT_RANGE = 30
- M4T_SOFTMAX_LUT_RES = 4096
- Output scale = 2^30

### Tests

- Recip: bit-exact prod-vs-scalar_ref tolerance, boundaries (1, INT32_MAX).
- Softmax:
  - Uniform input: y[i] = 2^30 / n for all i.
  - One-hot: y[max_idx] ≈ 2^30, others ≈ 0.
  - Tolerance vs FP scalar_ref across random inputs.
  - n=1 → y[0] = 2^30.
  - Sum-to-2^30 invariant within tolerance.

### Wiring

NOT wired into harness for Phase 1 work-unit 4 — attention path is
fully stubbed. Wiring happens at work-unit 6 (full B forward).

The bitnet_stub_softmax remains in bitnet_stubs.c temporarily; its
caller (memset 0 attn_output) doesn't actually flow values, so the
stub is dormant. Removed in unit 6.

## Status

Cycle complete. Proceeding to implementation.
