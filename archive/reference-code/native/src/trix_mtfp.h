/*
 * trix_mtfp.h — Multi-Trit Fixed Point arithmetic
 *
 * Native balanced ternary number representation and operations.
 * Every value is stored as an integer in balanced-ternary fixed point:
 *
 *   value = integer_repr / 3^radix
 *
 * With radix=10: resolution = 1/59049 ≈ 1.69e-5
 * With 21 trits: range ≈ ±88.3 (3^21-1)/(2*3^10)
 * With 32 trits: range ≈ ±12.2M — more than enough for all neural net values
 *
 * Storage: int32_t for the integer representation (balanced ternary scaled).
 * Operations: add/subtract are integer add/subtract. Multiply is integer
 * multiply with rescale. The ternary structure is in the VALUES, not the
 * storage format — we use int32 as a container for the balanced ternary
 * fixed-point number.
 *
 * The key insight: ternary weights {-1, 0, +1} have integer_repr values of
 * {-59049, 0, +59049} at radix=10, or simply {-1, 0, +1} at radix=0.
 * Multiplying by a ternary weight is just: add, subtract, or skip the
 * activation's integer_repr. NO MULTIPLY needed.
 *
 * For GELU and other nonlinearities: use MTFP lookup tables. Pre-compute
 * GELU(x) for all representable x values in the active range, store as
 * MTFP values. Forward pass is table lookup — zero arithmetic.
 */

#ifndef TRIX_MTFP_H
#define TRIX_MTFP_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Configuration ── */

#define MTFP_RADIX      10
#define MTFP_SCALE      59049       /* 3^10 */
#define MTFP_SCALE_F    59049.0f
#define MTFP_INV_SCALE  (1.0f / 59049.0f)
#define MTFP_MAX_VAL    ((int32_t)1073741823)  /* INT32_MAX / 2 — practical clamp limit */

/* ── Type ── */

/* An MTFP value is an int32_t. The real value is mtfp / MTFP_SCALE. */
typedef int32_t mtfp_t;

/* ── Conversion ── */

static inline mtfp_t mtfp_from_float(float x) {
    float scaled = x * MTFP_SCALE_F;
    int32_t rounded = (int32_t)lrintf(scaled);
    if (rounded > MTFP_MAX_VAL) rounded = MTFP_MAX_VAL;
    if (rounded < -MTFP_MAX_VAL) rounded = -MTFP_MAX_VAL;
    return rounded;
}

static inline float mtfp_to_float(mtfp_t v) {
    return (float)v * MTFP_INV_SCALE;
}

/* ── Arithmetic ── */

/* Add: exact in fixed point */
static inline mtfp_t mtfp_add(mtfp_t a, mtfp_t b) {
    return a + b;
}

/* Subtract: exact in fixed point */
static inline mtfp_t mtfp_sub(mtfp_t a, mtfp_t b) {
    return a - b;
}

/* Multiply: requires rescale by 1/MTFP_SCALE.
 * result = (a * b) / MTFP_SCALE
 * Use int64 to avoid overflow. */
static inline mtfp_t mtfp_mul(mtfp_t a, mtfp_t b) {
    int64_t prod = (int64_t)a * (int64_t)b;
    /* Round to nearest: add half-scale before divide */
    if (prod >= 0) prod += MTFP_SCALE / 2;
    else prod -= MTFP_SCALE / 2;
    return (mtfp_t)(prod / MTFP_SCALE);
}

/* Multiply by ternary weight {-1, 0, +1}: NO multiply, just sign select.
 * w must be -1, 0, or +1 as int8_t. */
static inline mtfp_t mtfp_mul_ternary(mtfp_t a, int8_t w) {
    /* Branchless: a * w where w is {-1, 0, 1} */
    return a * (int32_t)w;  /* compiler optimizes: 0→zero, 1→nop, -1→neg */
}

/* Negate */
static inline mtfp_t mtfp_neg(mtfp_t a) { return -a; }

/* Scale by output_scale (which is also MTFP) */
static inline mtfp_t mtfp_scale(mtfp_t a, mtfp_t scale) {
    return mtfp_mul(a, scale);
}

/* ── Batch operations ── */

/* Convert float array to MTFP */
void mtfp_from_float_batch(mtfp_t* dst, const float* src, int n);

/* Convert MTFP array to float */
void mtfp_to_float_batch(float* dst, const mtfp_t* src, int n);

/* Vector add: dst[i] = a[i] + b[i] */
void mtfp_vec_add(mtfp_t* dst, const mtfp_t* a, const mtfp_t* b, int n);

/* Vector add inplace: dst[i] += a[i] */
void mtfp_vec_add_inplace(mtfp_t* dst, const mtfp_t* a, int n);

/* Ternary matmul: Y[M,N] = X[M,K] @ W[N,K]^T
 * X is MTFP, W is ternary int8. Output is MTFP.
 * ZERO multiplies — only add/subtract of MTFP integers. */
void mtfp_ternary_matmul_bt(
    mtfp_t* Y, const mtfp_t* X, const int8_t* W,
    int M, int K, int N
);

/* Y[M,N] = X[M,K] @ W[K,N] — same, different layout */
void mtfp_ternary_matmul(
    mtfp_t* Y, const mtfp_t* X, const int8_t* W,
    int M, int K, int N
);

/* Bias add: x[i] += b[i % dim] where b is MTFP */
void mtfp_bias_add(mtfp_t* x, const mtfp_t* b, int batch, int dim);

/* Fan-in normalization: divide each element by isqrt(fan_in).
 * Apply ONLY before nonlinearities (GELU, softmax) to prevent saturation.
 * Do NOT apply to routing scores, W2 matmuls, or outputs with learned scale. */
void mtfp_fan_in_normalize(mtfp_t* x, int n, int fan_in);

/* GELU via lookup table. Pre-computed for the representable range.
 * Call mtfp_gelu_init() once at startup. */
void mtfp_gelu_init(void);
void mtfp_gelu(mtfp_t* dst, const mtfp_t* src, int n);

/* MTFP × MTFP matmul (not ternary — both operands are multi-trit).
 * Y[M,N] = X[M,K] @ W[K,N]
 * Each element requires mtfp_mul (int64 product + rescale).
 * This is slower than ternary matmul but eliminates float entirely. */
void mtfp_matmul(mtfp_t* Y, const mtfp_t* X, const mtfp_t* W, int M, int K, int N);
void mtfp_matmul_bt(mtfp_t* Y, const mtfp_t* X, const mtfp_t* W, int M, int K, int N);

/* MTFP softmax with causal masking.
 * Uses pre-computed exp lookup table. */
void mtfp_softmax_init(void);
void mtfp_softmax(mtfp_t* dst, const mtfp_t* src, int rows, int cols, int causal);

/* MTFP vector scale: dst[i] = src[i] * scale */
void mtfp_vec_scale(mtfp_t* dst, const mtfp_t* src, mtfp_t scale, int n);

/* LayerNorm in MTFP.
 * mean and variance computed in int64 for precision. */
void mtfp_layernorm(
    mtfp_t* dst, const mtfp_t* src,
    const mtfp_t* weight, const mtfp_t* bias,
    int rows, int cols
);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_MTFP_H */
