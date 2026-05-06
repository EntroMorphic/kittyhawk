/*
 * gesh/bitnet/bitnet_stubs.c — temporary scalar implementations of the
 * substrate primitives work-units 2-5 will provide. Per the file
 * header in bitnet_stubs.h: NOT production scalar fallbacks; harness-
 * side scaffolding only.
 *
 * Correctness is the only goal here. Performance is irrelevant — these
 * stubs run on a fixed test prompt during work-unit 1 to capture
 * empirical input ranges and shapes.
 */

#include "bitnet_stubs.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Saturating clamp helper — mirrors m4t_mtfp_clamp64 but inlined here
 * to keep the stubs zero-deps from libm4t. */
#define MTFP19_MAX  ((int32_t)581130733)
static inline int32_t clamp_to_mtfp19(int64_t v) {
    if (v >  (int64_t)MTFP19_MAX) return  MTFP19_MAX;
    if (v < -(int64_t)MTFP19_MAX) return -MTFP19_MAX;
    return (int32_t)v;
}

/* ── RMSNorm ─────────────────────────────────────────────────────── */

void bitnet_stub_rmsnorm(
    m4t_mtfp_t* y, const m4t_mtfp_t* x, const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mtfp19, int n)
{
    /* Per RC-1 (red-team 2026-05-06): use double accumulator to avoid
     * int64 overflow at MTFP19_MAX-magnitude inputs. n=2560 squares of
     * MTFP19_MAX produces ~8.7e20 — overflows int64 (max 9.2e18). FP
     * accumulator is fine in stubs (stubs ARE scaffolding, not
     * production paths). Production path (work-unit 2) will use a
     * scaled-sum or Welford accumulator instead. */
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double xi = (double)x[i];
        sum_sq += xi * xi;
    }
    double mean_sq = sum_sq / (double)n + (double)eps_mtfp19;
    double inv_rms = 1.0 / sqrt(mean_sq);
    /* y[i] = γ[i] · x[i] · inv_rms */
    for (int i = 0; i < n; i++) {
        double v = (double)gamma[i] * (double)x[i] * inv_rms;
        y[i] = clamp_to_mtfp19((int64_t)v);
    }
}

/* ── RoPE ────────────────────────────────────────────────────────── */

void bitnet_stub_rope_apply(
    m4t_mtfp_t* q, m4t_mtfp_t* k,
    int position,
    int num_q_heads, int num_kv_heads, int head_dim,
    double theta_base)
{
    /* RoPE pairs adjacent dims: (q[2i], q[2i+1]) → (q[2i]·cos − q[2i+1]·sin,
     *                                                q[2i]·sin + q[2i+1]·cos)
     * with θ_i = position / (theta_base ^ (2i / head_dim)) */
    int half = head_dim / 2;

    /* Apply to all Q heads. */
    for (int h = 0; h < num_q_heads; h++) {
        m4t_mtfp_t* qh = q + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            double freq = pow(theta_base, -2.0 * i / (double)head_dim);
            double angle = (double)position * freq;
            double c = cos(angle), s = sin(angle);
            double a = (double)qh[2*i];
            double b = (double)qh[2*i + 1];
            qh[2*i]     = clamp_to_mtfp19((int64_t)(a * c - b * s));
            qh[2*i + 1] = clamp_to_mtfp19((int64_t)(a * s + b * c));
        }
    }
    /* Apply to all K heads (same formula). */
    for (int h = 0; h < num_kv_heads; h++) {
        m4t_mtfp_t* kh = k + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            double freq = pow(theta_base, -2.0 * i / (double)head_dim);
            double angle = (double)position * freq;
            double c = cos(angle), s = sin(angle);
            double a = (double)kh[2*i];
            double b = (double)kh[2*i + 1];
            kh[2*i]     = clamp_to_mtfp19((int64_t)(a * c - b * s));
            kh[2*i + 1] = clamp_to_mtfp19((int64_t)(a * s + b * c));
        }
    }
}

/* ── Softmax ─────────────────────────────────────────────────────── */

void bitnet_stub_softmax(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n) {
    if (n <= 0) return;
    /* Find max for numerical stability. */
    m4t_mtfp_t mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    /* exp(x[i] - max) and sum. Use double for stability — production
     * path will use an exp LUT in MTFP19. */
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    if (!tmp) return;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double v = exp((double)(x[i] - mx));
        tmp[i] = v;
        sum += v;
    }
    /* Normalize. */
    for (int i = 0; i < n; i++) {
        double v = tmp[i] / sum * (double)MTFP19_MAX;  /* scale into MTFP19 range */
        y[i] = clamp_to_mtfp19((int64_t)v);
    }
    free(tmp);
}

/* ── A8 quantize / dequantize ────────────────────────────────────── */

m4t_mtfp_t bitnet_stub_a8_quantize(int8_t* y, const m4t_mtfp_t* x, int n) {
    /* Per RC-3 (red-team 2026-05-06): the returned value is the
     * per-token absmax, NOT the scale (= absmax/127). Dequant applies
     * the /127 explicitly. Renamed parameter at the dequant call site
     * to reflect this. */
    if (n <= 0) return 0;
    m4t_mtfp_t max_abs = 0;
    for (int i = 0; i < n; i++) {
        m4t_mtfp_t a = x[i] < 0 ? -x[i] : x[i];
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0) {
        memset(y, 0, (size_t)n);
        return 0;
    }
    /* y_int8 = round(x · 127 / max_abs) */
    for (int i = 0; i < n; i++) {
        double v = (double)x[i] * 127.0 / (double)max_abs;
        int32_t r = (int32_t)(v < 0 ? v - 0.5 : v + 0.5);
        if (r >  127) r =  127;
        if (r < -127) r = -127;
        y[i] = (int8_t)r;
    }
    return max_abs;
}

void bitnet_stub_a8_dequantize(
    m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax_mtfp19, int n)
{
    /* y[i] = x_int8[i] · absmax / 127 — RC-3 rename: parameter is the
     * absmax stored by stub_quantize, not the scale. */
    for (int i = 0; i < n; i++) {
        double v = (double)x[i] * (double)absmax_mtfp19 / 127.0;
        y[i] = clamp_to_mtfp19((int64_t)v);
    }
}

/* ── Element-wise ops ────────────────────────────────────────────── */

void bitnet_stub_relu2_inplace(m4t_mtfp_t* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] <= 0) {
            x[i] = 0;
        } else {
            int64_t sq = (int64_t)x[i] * (int64_t)x[i];
            x[i] = clamp_to_mtfp19(sq);
        }
    }
}

void bitnet_stub_elementwise_mul(
    m4t_mtfp_t* y, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n)
{
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)a[i] * (int64_t)b[i];
        y[i] = clamp_to_mtfp19(v);
    }
}
