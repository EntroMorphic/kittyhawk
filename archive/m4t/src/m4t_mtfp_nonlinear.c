/*
 * m4t_mtfp_nonlinear.c — LUT-backed GELU, softmax, argmax for MTFP19
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The lookup tables are in m4t_mtfp_tables.c (auto-generated, ~9.8 MB).
 * This file provides the runtime functions that index into those tables.
 * Pure integer at runtime — the tables were computed offline from float.
 */

#include "m4t_mtfp.h"
#include <string.h>

/* ── External tables from m4t_mtfp_tables.c ────────────────────────────── */

#define M4T_LUT_TABLE_HALF  354294
#define M4T_LUT_TABLE_SIZE  708589

extern const m4t_mtfp_t m4t_gelu_table[M4T_LUT_TABLE_SIZE];
extern const m4t_mtfp_t m4t_exp_table[M4T_LUT_TABLE_SIZE];
extern const m4t_mtfp_t m4t_exp_clamp_hi;

/* ── GELU ──────────────────────────────────────────────────────────────── */

void m4t_mtfp_gelu(m4t_mtfp_t* dst, const m4t_mtfp_t* src, int n) {
    for (int i = 0; i < n; i++) {
        int32_t v = src[i];
        if (v >= M4T_LUT_TABLE_HALF) {
            dst[i] = v;  /* GELU(x) ≈ x for large positive */
        } else if (v <= -M4T_LUT_TABLE_HALF) {
            dst[i] = 0;  /* GELU(x) ≈ 0 for large negative */
        } else {
            dst[i] = m4t_gelu_table[v + M4T_LUT_TABLE_HALF];
        }
    }
}

/* ── Softmax ───────────────────────────────────────────────────────────── */

static m4t_mtfp_t mtfp_exp(m4t_mtfp_t v) {
    if (v >= M4T_LUT_TABLE_HALF) return m4t_exp_clamp_hi;
    if (v <= -M4T_LUT_TABLE_HALF) return 0;
    return m4t_exp_table[v + M4T_LUT_TABLE_HALF];
}

void m4t_mtfp_softmax(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                       int rows, int cols, int causal) {
    for (int r = 0; r < rows; r++) {
        const m4t_mtfp_t* si = src + (size_t)r * cols;
        m4t_mtfp_t* di = dst + (size_t)r * cols;
        int active = causal ? (r + 1) : cols;

        /* Max for numerical stability. */
        m4t_mtfp_t max_val = si[0];
        for (int c = 1; c < active; c++)
            if (si[c] > max_val) max_val = si[c];

        /* exp(x - max) and sum. */
        int64_t sum = 0;
        for (int c = 0; c < active; c++) {
            di[c] = mtfp_exp(si[c] - max_val);
            sum += (int64_t)di[c];
        }
        for (int c = active; c < cols; c++) di[c] = 0;

        /* Normalize: di[c] = di[c] * SCALE / sum. */
        if (sum > 0) {
            for (int c = 0; c < active; c++) {
                di[c] = (m4t_mtfp_t)(((int64_t)di[c] * M4T_MTFP_SCALE + sum / 2) / sum);
            }
        }
    }
}

/* ── Argmax ────────────────────────────────────────────────────────────── */

void m4t_mtfp_argmax(int* dst, const m4t_mtfp_t* src, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const m4t_mtfp_t* s = src + (size_t)r * cols;
        int best = 0;
        m4t_mtfp_t best_val = s[0];
        for (int c = 1; c < cols; c++) {
            if (s[c] > best_val) { best_val = s[c]; best = c; }
        }
        dst[r] = best;
    }
}
