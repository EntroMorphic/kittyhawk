/*
 * m4t_mtfp.c — MTFP arithmetic core
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * All values are int32 cells holding balanced ternary fixed point with
 * scale 3^10 = 59049. Add/sub are exact; multiply widens to int64 and
 * rescales. Matmul and layernorm use int64 accumulators.
 */

#include "m4t_mtfp.h"
#include "m4t_internal.h"
#include <string.h>
#include <assert.h>

/* ── Vector arithmetic ─────────────────────────────────────────────────── */

void m4t_mtfp_vec_zero(m4t_mtfp_t* x, int n) {
    memset(x, 0, (size_t)n * sizeof(m4t_mtfp_t));
}

/* NEON saturation note: MAX_VAL = (3^19-1)/2 = 581,130,733. The sum of
 * two in-range cells is at most 1,162,261,466, which is less than
 * INT32_MAX (2,147,483,647) — no int32 wrap before we clamp. Plain
 * vaddq_s32 followed by vminq/vmaxq against splats of ±MAX_VAL is
 * sufficient and cheaper than vqaddq_s32 + clamp. */

void m4t_mtfp_vec_add(
    m4t_mtfp_t* dst, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n)
{
    int i = 0;
#if M4T_HAS_NEON
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    for (; i + 4 <= n; i += 4) {
        int32x4_t s = vaddq_s32(vld1q_s32(a + i), vld1q_s32(b + i));
        s = vminq_s32(s, vmax);
        s = vmaxq_s32(s, vmin);
        vst1q_s32(dst + i, s);
    }
#endif
    for (; i < n; i++) dst[i] = m4t_mtfp_add(a[i], b[i]);
}

void m4t_mtfp_vec_add_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n) {
    int i = 0;
#if M4T_HAS_NEON
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    for (; i + 4 <= n; i += 4) {
        int32x4_t s = vaddq_s32(vld1q_s32(dst + i), vld1q_s32(a + i));
        s = vminq_s32(s, vmax);
        s = vmaxq_s32(s, vmin);
        vst1q_s32(dst + i, s);
    }
#endif
    for (; i < n; i++) dst[i] = m4t_mtfp_add(dst[i], a[i]);
}

void m4t_mtfp_vec_sub_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n) {
    int i = 0;
#if M4T_HAS_NEON
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    for (; i + 4 <= n; i += 4) {
        int32x4_t s = vsubq_s32(vld1q_s32(dst + i), vld1q_s32(a + i));
        s = vminq_s32(s, vmax);
        s = vmaxq_s32(s, vmin);
        vst1q_s32(dst + i, s);
    }
#endif
    for (; i < n; i++) dst[i] = m4t_mtfp_sub(dst[i], a[i]);
}

void m4t_mtfp_vec_scale(
    m4t_mtfp_t* dst, const m4t_mtfp_t* src, m4t_mtfp_t scale, int n)
{
    /* Scalar path. A NEON widening multiply is possible with vmull_s32 +
     * rounding + vshrn_n_s64, but the scalar loop is correct and stays
     * within policy; vectorize after v0 lands. */
    for (int i = 0; i < n; i++) {
        dst[i] = m4t_mtfp_mul(src[i], scale);
    }
}

/* ── Dense MTFP × MTFP matmul ──────────────────────────────────────────── */

/* Accumulator uses __int128 to avoid overflow at large K.
 * With int64 accumulators, K > 27 with MAX_VAL cells would overflow.
 * With __int128, K_max ≈ 5e20 — effectively unlimited. */

static void matmul_row(
    m4t_mtfp_t* Y_row, const m4t_mtfp_t* X_row, const m4t_mtfp_t* W,
    int K, int N)
{
    for (int j = 0; j < N; j++) {
        __int128 acc = 0;
        for (int k = 0; k < K; k++) {
            acc += (__int128)X_row[k] * (__int128)W[(size_t)k * N + j];
        }
        if (acc >= 0) acc += M4T_MTFP_SCALE / 2;
        else          acc -= M4T_MTFP_SCALE / 2;
        Y_row[j] = m4t_mtfp_clamp64((int64_t)(acc / M4T_MTFP_SCALE));
    }
}

static void matmul_bt_row(
    m4t_mtfp_t* Y_row, const m4t_mtfp_t* X_row, const m4t_mtfp_t* W,
    int K, int N)
{
    for (int j = 0; j < N; j++) {
        const m4t_mtfp_t* wj = W + (size_t)j * K;
        __int128 acc = 0;
        for (int k = 0; k < K; k++) {
            acc += (__int128)X_row[k] * (__int128)wj[k];
        }
        if (acc >= 0) acc += M4T_MTFP_SCALE / 2;
        else          acc -= M4T_MTFP_SCALE / 2;
        Y_row[j] = m4t_mtfp_clamp64((int64_t)(acc / M4T_MTFP_SCALE));
    }
}

void m4t_mtfp_matmul(
    m4t_mtfp_t* Y, const m4t_mtfp_t* X, const m4t_mtfp_t* W,
    int M, int K, int N)
{
    assert(Y && X && W);
    assert(M >= 0 && K >= 0 && N >= 0);
    for (int i = 0; i < M; i++) {
        matmul_row(Y + (size_t)i * N, X + (size_t)i * K, W, K, N);
    }
}

void m4t_mtfp_matmul_bt(
    m4t_mtfp_t* Y, const m4t_mtfp_t* X, const m4t_mtfp_t* W,
    int M, int K, int N)
{
    assert(Y && X && W);
    assert(M >= 0 && K >= 0 && N >= 0);
    for (int i = 0; i < M; i++) {
        matmul_bt_row(Y + (size_t)i * N, X + (size_t)i * K, W, K, N);
    }
}

/* ── Bias / normalization ──────────────────────────────────────────────── */

void m4t_mtfp_bias_add(
    m4t_mtfp_t* x, const m4t_mtfp_t* b, int batch, int dim)
{
    for (int i = 0; i < batch; i++) {
        m4t_mtfp_vec_add_inplace(x + (size_t)i * dim, b, dim);
    }
}

void m4t_mtfp_fan_in_normalize(m4t_mtfp_t* x, int n, int fan_in) {
    int64_t norm = m4t_isqrt64((int64_t)fan_in);
    if (norm < 1) norm = 1;
    int64_t half = norm / 2;
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)x[i];
        x[i] = (m4t_mtfp_t)((v >= 0) ? (v + half) / norm : (v - half) / norm);
    }
}

/* ── Integer square root ───────────────────────────────────────────────── */

int64_t m4t_isqrt64(int64_t x) {
    if (x <= 0) return 0;
    /* Initial guess must be an OVERSHOOT so that Newton descends
     * monotonically. For x in [2^b, 2^(b+1)), sqrt(x) < 2^((b+2)/2);
     * this picks a power of two strictly above the true root. */
    int bits = 63 - __builtin_clzll((uint64_t)x);
    int64_t y = (int64_t)1 << ((bits + 2) / 2);
    /* Newton step: y' = (y + x/y) / 2. Bounded at 32 iterations; in
     * practice converges in ~6 for any int64 input. */
    for (int i = 0; i < 32; i++) {
        int64_t y_new = (y + x / y) / 2;
        if (y_new >= y) break;
        y = y_new;
    }
    return y;
}

int64_t m4t_mtfp_isqrt_inv(int64_t x) {
    if (x <= 0) return (int64_t)M4T_MTFP_SCALE;
    int64_t s = m4t_isqrt64(x);
    if (s <= 0) return (int64_t)M4T_MTFP_SCALE;
    int64_t S = (int64_t)M4T_MTFP_SCALE;
    return (S * S) / s;
}

/* ── LayerNorm (forward, inference only) ───────────────────────────────── */

static void layernorm_row(
    m4t_mtfp_t* yi, const m4t_mtfp_t* xi,
    const m4t_mtfp_t* weight, const m4t_mtfp_t* bias,
    int64_t eps_var, int cols)
{
    /* Mean. */
    int64_t sum = 0;
    for (int j = 0; j < cols; j++) sum += (int64_t)xi[j];
    int32_t mean = (int32_t)(sum / cols);

    /* Variance (in SCALE^2 units, held in int64). */
    int64_t var_sum = 0;
    for (int j = 0; j < cols; j++) {
        int64_t d = (int64_t)(xi[j] - mean);
        var_sum += d * d;
    }
    int64_t var = var_sum / cols;
    int64_t var_eps = var + eps_var;

    /* rstd in SCALE units. */
    int64_t rstd = m4t_mtfp_isqrt_inv(var_eps);
    int64_t S = (int64_t)M4T_MTFP_SCALE;

    for (int j = 0; j < cols; j++) {
        int64_t centered = (int64_t)(xi[j] - mean);
        int64_t norm = centered * rstd / S;  /* MTFP cell, unitless × SCALE. */
        /* Symmetric round-half-away-from-zero on the scale multiply.
         * Adding S/2 unconditionally biases negatives toward zero under
         * C's truncating integer divide. */
        int64_t num = norm * (int64_t)weight[j];
        int64_t scaled = (num >= 0)
            ? (num + (int64_t)(M4T_MTFP_SCALE / 2)) / S
            : (num - (int64_t)(M4T_MTFP_SCALE / 2)) / S;
        int64_t result = scaled + (int64_t)bias[j];
        yi[j] = m4t_mtfp_clamp64(result);
    }
}

void m4t_mtfp_layernorm(
    m4t_mtfp_t* dst, const m4t_mtfp_t* src,
    const m4t_mtfp_t* weight, const m4t_mtfp_t* bias,
    m4t_mtfp_t eps, int rows, int cols)
{
    if (rows <= 0 || cols <= 0) return;
    assert(dst && src && weight && bias);
    /* eps is an MTFP cell in real units. For variance comparison we need
     * eps squared in SCALE^2 units: (eps/S)^2 * S^2 = eps * eps. */
    int64_t eps_var = (int64_t)eps * (int64_t)eps;

    for (int i = 0; i < rows; i++) {
        layernorm_row(
            dst + (size_t)i * cols,
            src + (size_t)i * cols,
            weight, bias, eps_var, cols);
    }
}
