/*
 * trix_mtfp.c — Multi-Trit Fixed Point arithmetic
 *
 * Native balanced ternary operations. All values are int32 fixed-point
 * with scale factor 3^10 = 59049. Ternary weight multiply is add/sub/skip.
 * GELU is a pre-computed lookup table. Zero float arithmetic in the hot path.
 */

#include "trix_mtfp.h"
#include <stdlib.h>
#include <string.h>

#ifdef APPLE
#include <dispatch/dispatch.h>
#endif

/* Forward declaration — defined in LayerNorm section */
static int64_t isqrt64(int64_t x);

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TRIX_HAS_NEON 1
#else
#define TRIX_HAS_NEON 0
#endif

/* ══════════════════════════════════════════════════════════════════════
 * Batch conversions
 * ══════════════════════════════════════════════════════════════════════ */

void mtfp_from_float_batch(mtfp_t* dst, const float* src, int n) {
#if TRIX_HAS_NEON
    float32x4_t vscale = vdupq_n_f32(MTFP_SCALE_F);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vmulq_f32(vld1q_f32(src + i), vscale);
        int32x4_t vi = vcvtnq_s32_f32(vx);  /* round to nearest */
        vst1q_s32(dst + i, vi);
    }
    for (; i < n; i++) dst[i] = mtfp_from_float(src[i]);
#else
    for (int i = 0; i < n; i++) dst[i] = mtfp_from_float(src[i]);
#endif
}

void mtfp_to_float_batch(float* dst, const mtfp_t* src, int n) {
#if TRIX_HAS_NEON
    float32x4_t vinv = vdupq_n_f32(MTFP_INV_SCALE);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vcvtq_f32_s32(vld1q_s32(src + i));
        vst1q_f32(dst + i, vmulq_f32(vx, vinv));
    }
    for (; i < n; i++) dst[i] = mtfp_to_float(src[i]);
#else
    for (int i = 0; i < n; i++) dst[i] = mtfp_to_float(src[i]);
#endif
}

/* ══════════════════════════════════════════════════════════════════════
 * Vector operations
 * ══════════════════════════════════════════════════════════════════════ */

void mtfp_vec_add(mtfp_t* dst, const mtfp_t* a, const mtfp_t* b, int n) {
#if TRIX_HAS_NEON
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_s32(dst + i, vaddq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
    for (; i < n; i++) dst[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
#endif
}

void mtfp_vec_add_inplace(mtfp_t* dst, const mtfp_t* a, int n) {
#if TRIX_HAS_NEON
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_s32(dst + i, vaddq_s32(vld1q_s32(dst + i), vld1q_s32(a + i)));
    for (; i < n; i++) dst[i] += a[i];
#else
    for (int i = 0; i < n; i++) dst[i] += a[i];
#endif
}

/* ══════════════════════════════════════════════════════════════════════
 * Ternary matmul in MTFP: ZERO multiplies
 *
 * Y[M,N] = X[M,K] @ W[N,K]^T
 * X is mtfp_t (int32), W is int8 ternary. Output is mtfp_t.
 * Each element: add X where W=+1, subtract X where W=-1, skip where W=0.
 * ══════════════════════════════════════════════════════════════════════ */

void mtfp_ternary_matmul_bt(
    mtfp_t* Y, const mtfp_t* X, const int8_t* W,
    int M, int K, int N)
{
#if TRIX_HAS_NEON
#ifdef APPLE
    dispatch_apply((size_t)M, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
#else
    for (int i = 0; i < M; i++) {
#endif
        const mtfp_t* xi = X + i * K;
        for (int j = 0; j < N; j++) {
            const int8_t* wj = W + j * K;
            int32x4_t vacc = vdupq_n_s32(0);
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                int32x4_t vx = vld1q_s32(xi + k);
                int8_t w0 = wj[k], w1 = wj[k+1], w2 = wj[k+2], w3 = wj[k+3];
                int32x4_t vw = {w0, w1, w2, w3};
                uint32x4_t pos = vcgtq_s32(vw, vdupq_n_s32(0));
                uint32x4_t neg = vcltq_s32(vw, vdupq_n_s32(0));
                vacc = vaddq_s32(vacc, vandq_s32(vreinterpretq_s32_u32(pos), vx));
                vacc = vsubq_s32(vacc, vandq_s32(vreinterpretq_s32_u32(neg), vx));
            }
            int32_t sum = vaddvq_s32(vacc);
            for (; k < K; k++) {
                if (wj[k] == 1) sum += xi[k];
                else if (wj[k] == -1) sum -= xi[k];
            }
            Y[i * N + j] = sum;
        }
#ifdef APPLE
    });
#else
    }
#endif
#else
    for (int i = 0; i < M; i++) {
        const mtfp_t* xi = X + i * K;
        for (int j = 0; j < N; j++) {
            const int8_t* wj = W + j * K;
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                if (wj[k] == 1) sum += xi[k];
                else if (wj[k] == -1) sum -= xi[k];
            }
            Y[i * N + j] = sum;
        }
    }
#endif
}

void mtfp_ternary_matmul(
    mtfp_t* Y, const mtfp_t* X, const int8_t* W,
    int M, int K, int N)
{
    memset(Y, 0, (size_t)M * N * sizeof(mtfp_t));
#if TRIX_HAS_NEON
    for (int i = 0; i < M; i++) {
        const mtfp_t* xi = X + i * K;
        mtfp_t* yi = Y + i * N;
        for (int k = 0; k < K; k++) {
            int32_t xik = xi[k];
            if (xik == 0) continue;
            const int8_t* wk = W + k * N;
            int32x4_t vx = vdupq_n_s32(xik);
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                int8_t w0 = wk[j], w1 = wk[j+1], w2 = wk[j+2], w3 = wk[j+3];
                int32x4_t vw = {w0, w1, w2, w3};
                int32x4_t vy = vld1q_s32(yi + j);
                uint32x4_t pos = vcgtq_s32(vw, vdupq_n_s32(0));
                uint32x4_t neg = vcltq_s32(vw, vdupq_n_s32(0));
                vy = vaddq_s32(vy, vandq_s32(vreinterpretq_s32_u32(pos), vx));
                vy = vsubq_s32(vy, vandq_s32(vreinterpretq_s32_u32(neg), vx));
                vst1q_s32(yi + j, vy);
            }
            for (; j < N; j++) {
                if (wk[j] == 1) yi[j] += xik;
                else if (wk[j] == -1) yi[j] -= xik;
            }
        }
    }
#else
    for (int i = 0; i < M; i++) {
        const mtfp_t* xi = X + i * K;
        mtfp_t* yi = Y + i * N;
        for (int k = 0; k < K; k++) {
            int32_t xik = xi[k];
            const int8_t* wk = W + k * N;
            for (int j = 0; j < N; j++) {
                if (wk[j] == 1) yi[j] += xik;
                else if (wk[j] == -1) yi[j] -= xik;
            }
        }
    }
#endif
}

/* ══════════════════════════════════════════════════════════════════════
 * MTFP × MTFP matmul
 *
 * Each element: y[i,j] = sum_k( mtfp_mul(x[i,k], w[k,j]) )
 * mtfp_mul uses int64 product + rescale by 1/MTFP_SCALE.
 * Slower than ternary matmul (which is pure add/sub) but eliminates float.
 * ══════════════════════════════════════════════════════════════════════ */

void mtfp_matmul(mtfp_t* Y, const mtfp_t* X, const mtfp_t* W, int M, int K, int N) {
#ifdef APPLE
    dispatch_apply((size_t)M, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
#else
    for (int i = 0; i < M; i++) {
#endif
        const mtfp_t* xi = X + i * K;
        for (int j = 0; j < N; j++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int64_t)xi[k] * (int64_t)W[k * N + j];
            Y[i * N + j] = (mtfp_t)((acc + MTFP_SCALE / 2) / MTFP_SCALE);
        }
#ifdef APPLE
    });
#else
    }
#endif
}

void mtfp_matmul_bt(mtfp_t* Y, const mtfp_t* X, const mtfp_t* W, int M, int K, int N) {
#ifdef APPLE
    dispatch_apply((size_t)M, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
#else
    for (int i = 0; i < M; i++) {
#endif
        const mtfp_t* xi = X + i * K;
        for (int j = 0; j < N; j++) {
            const mtfp_t* wj = W + j * K;
            int64_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int64_t)xi[k] * (int64_t)wj[k];
            Y[i * N + j] = (mtfp_t)((acc + MTFP_SCALE / 2) / MTFP_SCALE);
        }
#ifdef APPLE
    });
#else
    }
#endif
}

/* ══════════════════════════════════════════════════════════════════════
 * MTFP vector scale
 * ══════════════════════════════════════════════════════════════════════ */

void mtfp_vec_scale(mtfp_t* dst, const mtfp_t* src, mtfp_t scale, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = mtfp_mul(src[i], scale);
}

/* ══════════════════════════════════════════════════════════════════════
 * MTFP Softmax
 *
 * For causal attention: each row r has active columns [0..r].
 * exp() via pre-computed lookup table. Division via integer divide.
 * ══════════════════════════════════════════════════════════════════════ */

/* exp table: maps mtfp_t input → mtfp_t output
 * Range: exp(-10) ≈ 4.5e-5 to exp(10) ≈ 22026
 * At MTFP scale: input range [-590490, +590490], output range [3, 1300000000]
 * We only need relative values (softmax normalizes), so we can use a smaller range. */
#define SOFTMAX_TABLE_HALF  354294  /* 6.0 * MTFP_SCALE — same as GELU */
#define SOFTMAX_TABLE_SIZE  (2 * SOFTMAX_TABLE_HALF + 1)

static mtfp_t* exp_table = NULL;

void mtfp_softmax_init(void) {
    if (exp_table) return;
    exp_table = malloc(SOFTMAX_TABLE_SIZE * sizeof(mtfp_t));
    for (int i = 0; i < SOFTMAX_TABLE_SIZE; i++) {
        int32_t repr = i - SOFTMAX_TABLE_HALF;
        float x = (float)repr * MTFP_INV_SCALE;
        float y = expf(x);
        exp_table[i] = mtfp_from_float(y);
    }
}

static mtfp_t mtfp_exp(mtfp_t v) {
    if (v >= SOFTMAX_TABLE_HALF) return mtfp_from_float(403.4f);  /* exp(6) ≈ 403 */
    if (v <= -SOFTMAX_TABLE_HALF) return 0;  /* exp(-6) ≈ 0 in MTFP resolution */
    return exp_table[v + SOFTMAX_TABLE_HALF];
}

void mtfp_softmax(mtfp_t* dst, const mtfp_t* src, int rows, int cols, int causal) {
    for (int r = 0; r < rows; r++) {
        const mtfp_t* si = src + r * cols;
        mtfp_t* di = dst + r * cols;
        int active = causal ? (r + 1) : cols;

        /* Find max for numerical stability */
        mtfp_t max_val = si[0];
        for (int c = 1; c < active; c++)
            if (si[c] > max_val) max_val = si[c];

        /* Compute exp(x - max) and sum */
        int64_t sum = 0;
        for (int c = 0; c < active; c++) {
            di[c] = mtfp_exp(si[c] - max_val);
            sum += (int64_t)di[c];
        }
        for (int c = active; c < cols; c++) di[c] = 0;

        /* Normalize: di[c] = di[c] / sum, keeping MTFP scale.
         * di[c] is in MTFP units. sum is in MTFP units.
         * Normalized value = di[c] / sum, which should be in [0, 1].
         * In MTFP: result = di[c] * MTFP_SCALE / sum */
        if (sum > 0) {
            for (int c = 0; c < active; c++)
                di[c] = (mtfp_t)(((int64_t)di[c] * MTFP_SCALE + sum / 2) / sum);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Bias add
 * ══════════════════════════════════════════════════════════════════════ */

void mtfp_fan_in_normalize(mtfp_t* x, int n, int fan_in) {
    int64_t norm = isqrt64((int64_t)fan_in);
    if (norm < 1) norm = 1;
    for (int i = 0; i < n; i++) x[i] = (mtfp_t)(x[i] / norm);
}

void mtfp_bias_add(mtfp_t* x, const mtfp_t* b, int batch, int dim) {
    for (int i = 0; i < batch; i++)
        mtfp_vec_add_inplace(x + i * dim, b, dim);
}

/* ══════════════════════════════════════════════════════════════════════
 * GELU lookup table
 *
 * Pre-compute GELU(x) for integer representations in the useful range.
 * Neural net activations typically live in [-6, 6]. At radix=10,
 * that's integer range [-354294, +354294]. We store GELU for this range.
 *
 * For values outside the range: GELU(x) ≈ x for large positive,
 * GELU(x) ≈ 0 for large negative.
 * ══════════════════════════════════════════════════════════════════════ */

#define GELU_TABLE_HALF  354294  /* 6.0 * MTFP_SCALE */
#define GELU_TABLE_SIZE  (2 * GELU_TABLE_HALF + 1)

static mtfp_t* gelu_table = NULL;

static float gelu_f32(float x) {
    float c = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

void mtfp_gelu_init(void) {
    if (gelu_table) return;
    gelu_table = malloc(GELU_TABLE_SIZE * sizeof(mtfp_t));
    for (int i = 0; i < GELU_TABLE_SIZE; i++) {
        int32_t repr = i - GELU_TABLE_HALF;
        float x = (float)repr * MTFP_INV_SCALE;
        float y = gelu_f32(x);
        gelu_table[i] = mtfp_from_float(y);
    }
}

void mtfp_gelu(mtfp_t* dst, const mtfp_t* src, int n) {
    for (int i = 0; i < n; i++) {
        int32_t v = src[i];
        if (v >= GELU_TABLE_HALF) {
            dst[i] = v;  /* GELU(x) ≈ x for large positive */
        } else if (v <= -GELU_TABLE_HALF) {
            dst[i] = 0;  /* GELU(x) ≈ 0 for large negative */
        } else {
            dst[i] = gelu_table[v + GELU_TABLE_HALF];
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * LayerNorm in MTFP — all integer arithmetic
 * ══════════════════════════════════════════════════════════════════════ */

/*
 * Integer square root via Newton-Raphson.
 * Returns floor(sqrt(x)) for x >= 0.
 */
static int64_t isqrt64(int64_t x) {
    if (x <= 0) return 0;
    /* Initial guess from bit count */
    int bits = 63 - __builtin_clzll((uint64_t)x);
    int64_t y = (int64_t)1 << ((bits + 1) / 2);
    /* Newton: y = (y + x/y) / 2 */
    for (int i = 0; i < 8; i++) {
        int64_t y_new = (y + x / y) / 2;
        if (y_new >= y) break;  /* converged */
        y = y_new;
    }
    return y;
}

/*
 * Integer inverse sqrt: returns S^2 / sqrt(x) where S = MTFP_SCALE.
 * x is in MTFP_SCALE^2 units (the variance).
 * Result is in MTFP_SCALE units (the rstd).
 */
static int64_t mtfp_isqrt_inv(int64_t x) {
    if (x <= 0) return (int64_t)MTFP_SCALE;
    int64_t s = isqrt64(x);    /* s ≈ sqrt(x), in MTFP_SCALE units */
    if (s <= 0) return (int64_t)MTFP_SCALE;
    int64_t S = (int64_t)MTFP_SCALE;
    return (S * S) / s;         /* S^2 / sqrt(x) = S / sqrt(x/S^2) */
}

void mtfp_layernorm(
    mtfp_t* dst, const mtfp_t* src,
    const mtfp_t* weight, const mtfp_t* bias,
    int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const mtfp_t* xi = src + r * cols;
        mtfp_t* yi = dst + r * cols;

        /* Mean: integer sum / cols */
        int64_t sum = 0;
        for (int j = 0; j < cols; j++) sum += xi[j];
        int32_t mean = (int32_t)(sum / cols);

        /* Variance: sum of squared deviations in int64 */
        int64_t var_sum = 0;
        for (int j = 0; j < cols; j++) {
            int64_t d = (int64_t)(xi[j] - mean);
            var_sum += d * d;
        }
        /* var = var_sum / cols, in MTFP_SCALE^2 units */
        int64_t var = var_sum / cols;

        /* eps in MTFP_SCALE^2 units: 1e-5 * 59049^2 = 34.87 ≈ 35 */
        int64_t eps_scaled = 35;
        int64_t var_eps = var + eps_scaled;

        /* rstd ≈ S / sqrt(var_eps) via integer Newton-Raphson — ZERO FLOAT */
        int64_t rstd = mtfp_isqrt_inv(var_eps);
        /* rstd is in MTFP_SCALE units: rstd ≈ S / sqrt(var / S^2) = S^2 / sqrt(var) / S ...
         * Actually rstd ≈ S / sqrt(var_eps) where var_eps is in S^2 units.
         * So rstd/S ≈ 1/sqrt(var_eps/S^2) = 1/sqrt(variance_float) = the actual rstd.
         *
         * normalized = (x - mean) * rstd / S = (x-mean) * (1/sqrt(var_f))
         * (x-mean) is in S units, so normalized is unitless in S units.
         * Then y = normalized * weight / S + bias */
        int64_t S = (int64_t)MTFP_SCALE;
        for (int j = 0; j < cols; j++) {
            int64_t centered = (int64_t)(xi[j] - mean);
            /* norm = centered * rstd / S — this is in MTFP units (unitless × S) */
            int64_t norm = centered * rstd / S;
            /* result = norm * weight / S + bias — weight is MTFP, so divide by S */
            int64_t result = (norm * (int64_t)weight[j] + S / 2) / S + (int64_t)bias[j];
            if (result > MTFP_MAX_VAL) result = MTFP_MAX_VAL;
            if (result < -MTFP_MAX_VAL) result = -MTFP_MAX_VAL;
            yi[j] = (mtfp_t)result;
        }
    }
}
