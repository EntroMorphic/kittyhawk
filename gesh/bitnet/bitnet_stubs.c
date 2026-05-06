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
/* Removed: replaced by m4t_mtfp_rmsnorm (substrate primitive) at the
 * close of work-unit 2 of bitnet_phase1. */

/* ── RoPE ────────────────────────────────────────────────────────── */
/* Removed: replaced by m4t_mtfp_rope_apply (substrate primitive,
 * rotate_half convention) at the close of work-unit 3 of bitnet_phase1.
 * The previous stub used the WRONG (adjacent-pair) convention. */

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
