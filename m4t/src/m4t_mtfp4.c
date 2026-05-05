/*
 * m4t_mtfp4.c — MTFP4 routing cell, SDOT ternary matmul, cell-width conversions
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The SDOT kernel is the substrate's hot path: vdotq_s32 processes 16
 * int8 multiply-accumulates per instruction. For ternary weights ×
 * MTFP4 activations, this is the fastest operation M4 silicon can do.
 * Output is MTFP19 (Case W per §8.4) — exact by construction.
 */

#include "m4t_mtfp4.h"
#include "m4t_internal.h"

#include <assert.h>

/* SCALE_RATIO = 3^(MTFP_RADIX - MTFP4_RADIX) = 3^(10 - 2) = 3^8 = 6561.
 * Odd by construction (powers of 3 are odd), which makes
 * round-to-nearest unambiguous in the narrowing conversion: ties at
 * rem == SCALE_RATIO/2 = 3280.5 are impossible for integer mantissas. */
#define SCALE_RATIO 6561

_Static_assert((SCALE_RATIO & 1) == 1,
               "SCALE_RATIO must be odd (round-to-nearest invariant)");
_Static_assert((int64_t)M4T_MTFP4_MAX_VAL * SCALE_RATIO <= (int64_t)M4T_MTFP_MAX_VAL,
               "MTFP4→MTFP19 widen must not overflow MTFP19 mantissa");

/* ── SDOT ternary matmul (Case W per §8.4) ──────────────────────────────── */

void m4t_mtfp4_sdot_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_mtfp4_t* X,
    const m4t_trit_t* W,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    /* Hard precondition: K must not exceed M4T_SDOT_K_MAX_EXACT, beyond
     * which the worst-case output |K · 40| exceeds MTFP19's documented
     * mantissa range and the substrate-invariant |cell| ≤ MAX_VAL is
     * violated. Compile-time-derived from the cell-type max values. */
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W)));
    /* Aliasing precondition: Y must not alias X or W. */
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W);
    /* Sample-based debug check: spot-check a few weights are valid trits.
     * Exhaustive validation is O(NK) per call, too expensive; sampling
     * catches "caller forgot to use ternary" without becoming a hot-path
     * tax. NOTE: under -DNDEBUG the asserts compile out. */
    if (N > 0 && K > 0) {
        assert(W[0] == -1 || W[0] == 0 || W[0] == 1);
        assert(W[(N - 1) * K + (K - 1)] == -1 || W[(N - 1) * K + (K - 1)] == 0 ||
               W[(N - 1) * K + (K - 1)] == 1);
    }

    /* Per journal/m4t_matmul_tile_synthesize.md: register-tile by 4 j cells.
     * 4 parallel SDOT accumulator chains pipeline better on M-series than
     * the prior single-acc chain (audit measured 1.8× wall-clock gain at
     * apples-to-apples comparison). N%4 tail handled by the original
     * single-j-cell NEON path (NEON, not scalar — geometric tail rule). */
    int j_tile_end = N - (N % 4);

    for (int i = 0; i < M; i++) {
        const int8_t* xi = (const int8_t*)(X + (size_t)i * K);

        /* Tiled body: 4 j cells per outer iter. */
        for (int j = 0; j < j_tile_end; j += 4) {
            const int8_t* wj0 = (const int8_t*)(W + (size_t)(j + 0) * K);
            const int8_t* wj1 = (const int8_t*)(W + (size_t)(j + 1) * K);
            const int8_t* wj2 = (const int8_t*)(W + (size_t)(j + 2) * K);
            const int8_t* wj3 = (const int8_t*)(W + (size_t)(j + 3) * K);

            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            int k = 0;

#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t vacc0 = vdupq_n_s32(0);
            int32x4_t vacc1 = vdupq_n_s32(0);
            int32x4_t vacc2 = vdupq_n_s32(0);
            int32x4_t vacc3 = vdupq_n_s32(0);
            for (; k + 16 <= K; k += 16) {
                int8x16_t va  = vld1q_s8(xi + k);             /* shared X */
                int8x16_t vw0 = vld1q_s8(wj0 + k);
                int8x16_t vw1 = vld1q_s8(wj1 + k);
                int8x16_t vw2 = vld1q_s8(wj2 + k);
                int8x16_t vw3 = vld1q_s8(wj3 + k);
                vacc0 = vdotq_s32(vacc0, va, vw0);
                vacc1 = vdotq_s32(vacc1, va, vw1);
                vacc2 = vdotq_s32(vacc2, va, vw2);
                vacc3 = vdotq_s32(vacc3, va, vw3);
            }
            acc0 = vaddvq_s32(vacc0);
            acc1 = vaddvq_s32(vacc1);
            acc2 = vaddvq_s32(vacc2);
            acc3 = vaddvq_s32(vacc3);
#endif
            /* Geometric scalar tail (k not multiple of 16). NEON loop
             * above handles whole 16-K blocks; this handles the 0-15
             * remainder per j cell. */
            for (; k < K; k++) {
                int32_t x_k = (int32_t)xi[k];
                acc0 += x_k * (int32_t)wj0[k];
                acc1 += x_k * (int32_t)wj1[k];
                acc2 += x_k * (int32_t)wj2[k];
                acc3 += x_k * (int32_t)wj3[k];
            }

            Y[i * N + j + 0] = (m4t_mtfp_t)acc0;
            Y[i * N + j + 1] = (m4t_mtfp_t)acc1;
            Y[i * N + j + 2] = (m4t_mtfp_t)acc2;
            Y[i * N + j + 3] = (m4t_mtfp_t)acc3;
        }

        /* N%4 tail: 1-3 remaining j cells, original single-j-cell NEON
         * path. Same kernel shape as pre-tile; geometric tail rule. */
        for (int j = j_tile_end; j < N; j++) {
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
            for (; k < K; k++) {
                acc += (int32_t)xi[k] * (int32_t)wj[k];
            }

            Y[i * N + j] = (m4t_mtfp_t)acc;
        }
    }
}

/* ── Cell-width conversion ──────────────────────────────────────────────── */

void m4t_mtfp19_to_mtfp4(
    m4t_mtfp4_t* dst, const m4t_mtfp_t* src,
    uint8_t* flags, int n)
{
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);

    for (int i = 0; i < n; i++) {
        int32_t v = src[i];
        /* Base-3 round-to-nearest-even divide by 6561. Odd divisor →
         * ties impossible. Same algorithm as m4t_pow3_round_div in
         * m4t_mtfp.c, inlined here to avoid linkage. */
        int32_t q = v / SCALE_RATIO;
        int32_t rem = v - q * SCALE_RATIO;
        int rounded = (rem != 0);
        if (rem > 0 && 2 * rem > SCALE_RATIO) q += 1;
        else if (rem < 0 && 2 * (-rem) > SCALE_RATIO) q -= 1;

        m4t_mtfp4_t out = m4t_mtfp4_clamp(q);
        int saturated = (q != (int32_t)out);

        if (flags) {
            uint8_t bits = 0;
            if (rounded)   bits |= M4T_FLAG_ROUNDED;
            if (saturated) bits |= M4T_FLAG_SATURATED;
            if (bits) m4t_flag_or(flags, i, bits);
        }
        dst[i] = out;
    }
}

void m4t_mtfp4_to_mtfp19(
    m4t_mtfp_t* dst, const m4t_mtfp4_t* src, int n)
{
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);

    /* Widen exactly: |src| ≤ 40, output ≤ 40·6561 = 262 440 < MTFP19_MAX.
     * Static-asserted at file top. */
    for (int i = 0; i < n; i++) {
        dst[i] = (m4t_mtfp_t)src[i] * SCALE_RATIO;
    }
}
