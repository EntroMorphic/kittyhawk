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

/* ── Cell-width conversion (default-block-exponent convention) ──────────
 *
 * Under the default-block-exponent convention (see m4t_types.h), MTFP4
 * blocks use `block_exp = -M4T_MTFP4_RADIX = -2` and MTFP19 blocks use
 * `block_exp = -M4T_MTFP_RADIX = -10`. Mapping a value across cell
 * widths without changing its real magnitude requires adjusting the
 * mantissa by the exponent offset (−2 − (−10) = +8 trits when widening,
 * −8 trits when narrowing). In default-convention integer terms that's
 * multiply/divide by 3^8 = 6561.
 *
 * mtfp4_to_mtfp19: Case W widen. Exact by construction — the widened
 * mantissa is at most 40 × 6561 = 262 440, well inside MTFP19.
 *
 * mtfp19_to_mtfp4: narrowing conversion. Inherently lossy; this function
 * rounds half-away-from-zero and then saturates. Callers that need
 * exactness must stay in MTFP19 or pre-scale so the mantissa fits.
 * (A spec-compliant Case R opt-in can land when a consumer drives it.)
 */

#define SCALE_RATIO 6561   /* 3^(MTFP_RADIX - MTFP4_RADIX) = 3^8 */

_Static_assert((int64_t)M4T_MTFP4_MAX_VAL * SCALE_RATIO <= (int64_t)M4T_MTFP_MAX_VAL,
               "MTFP4→MTFP19 widen must not overflow MTFP19 mantissa");

void m4t_mtfp19_to_mtfp4(
    m4t_mtfp4_t* dst, const m4t_mtfp_t* src, int n)
{
    for (int i = 0; i < n; i++) {
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
        dst[i] = (m4t_mtfp_t)src[i] * SCALE_RATIO;
    }
}
