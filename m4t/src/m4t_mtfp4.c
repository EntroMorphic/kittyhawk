/*
 * m4t_mtfp4.c — MTFP4 routing cell and SDOT ternary matmul
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The SDOT kernel is the crown jewel: vdotq_s32 processes 16 int8
 * multiply-accumulates per instruction. For ternary weights × MTFP4
 * activations, this is the fastest operation M4 silicon can do.
 */

#include "m4t_mtfp4.h"
#include "m4t_internal.h"

#include <assert.h>

/* ── SDOT ternary matmul ───────────────────────────────────────────────── */

void m4t_mtfp4_sdot_matmul_bt(
    m4t_mtfp4_t* Y,
    const m4t_mtfp4_t* X,
    const m4t_trit_t* W,
    int M, int K, int N)
{
    assert(Y && X && W);
    assert(M >= 0 && K >= 0 && N >= 0);

    for (int i = 0; i < M; i++) {
        const int8_t* xi = (const int8_t*)(X + (size_t)i * K);
        for (int j = 0; j < N; j++) {
            const int8_t* wj = (const int8_t*)(W + (size_t)j * K);
            int32_t acc = 0;
            int k = 0;

#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t vacc = vdupq_n_s32(0);
            for (; k + 16 <= K; k += 16) {
                int8x16_t va = vld1q_s8(xi + k);
                int8x16_t vw = vld1q_s8(wj + k);
                vacc = vdotq_s32(vacc, va, vw);
            }
            acc = vaddvq_s32(vacc);
#endif
            /* Scalar tail. */
            for (; k < K; k++) {
                acc += (int32_t)xi[k] * (int32_t)wj[k];
            }

            Y[i * N + j] = m4t_mtfp4_clamp(acc);
        }
    }
}

/* ── Conversion ────────────────────────────────────────────────────────── */

/* Scale ratio: MTFP19_SCALE / MTFP4_SCALE = 59049 / 9 = 6561. */
#define SCALE_RATIO 6561

void m4t_mtfp19_to_mtfp4(
    m4t_mtfp4_t* dst, const m4t_mtfp_t* src, int n)
{
    for (int i = 0; i < n; i++) {
        /* Rescale: divide by 6561, round half-away-from-zero. */
        int32_t v = src[i];
        int32_t q;
        if (v >= 0) q = (v + SCALE_RATIO / 2) / SCALE_RATIO;
        else        q = (v - SCALE_RATIO / 2) / SCALE_RATIO;
        dst[i] = m4t_mtfp4_clamp(q);
    }
}

void m4t_mtfp4_to_mtfp19(
    m4t_mtfp_t* dst, const m4t_mtfp4_t* src, int n)
{
    for (int i = 0; i < n; i++) {
        /* Rescale: multiply by 6561. Max output: 40 * 6561 = 262440,
         * well within MTFP19 MAX_VAL (581130733). No clamp needed. */
        dst[i] = (m4t_mtfp_t)src[i] * SCALE_RATIO;
    }
}
