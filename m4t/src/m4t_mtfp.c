/*
 * m4t_mtfp.c — MTFP19 mantissa-layer primitives, NEON where it helps.
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Minimum surface for the current kept consumers (m4t_route.c and
 * m4t_ternary_matmul.c). The MTFP19 block is 4 int32 cells = one
 * int32x4_t NEON register. Tail cells beyond a whole block are handled
 * scalar with identical saturation semantics.
 *
 * Saturation strategy: since M4T_MTFP_MAX_VAL = (3^19 - 1) / 2, any two
 * in-range mantissas sum to at most ±(3^19 - 1), which fits int32
 * without wrapping. So `vaddq_s32` / `vsubq_s32` followed by
 * vmin/vmax clamp is exact, no overflow to worry about.
 */

#include "m4t_mtfp.h"
#include "m4t_internal.h"

#include <string.h>

void m4t_mtfp_vec_zero(m4t_mtfp_t* dst, int n) {
    if (n <= 0) return;
    memset(dst, 0, (size_t)n * sizeof(m4t_mtfp_t));
}

void m4t_mtfp_vec_add_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n) {
    int i = 0;
#if M4T_HAS_NEON
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    for (; i + 4 <= n; i += 4) {
        int32x4_t d = vld1q_s32(dst + i);
        int32x4_t s = vld1q_s32(a   + i);
        int32x4_t r = vaddq_s32(d, s);
        r = vminq_s32(r, vmax);
        r = vmaxq_s32(r, vmin);
        vst1q_s32(dst + i, r);
    }
#endif
    for (; i < n; i++) {
        int64_t v = (int64_t)dst[i] + (int64_t)a[i];
        dst[i] = m4t_mtfp_clamp64(v);
    }
}

void m4t_mtfp_vec_sub_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n) {
    int i = 0;
#if M4T_HAS_NEON
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    for (; i + 4 <= n; i += 4) {
        int32x4_t d = vld1q_s32(dst + i);
        int32x4_t s = vld1q_s32(a   + i);
        int32x4_t r = vsubq_s32(d, s);
        r = vminq_s32(r, vmax);
        r = vmaxq_s32(r, vmin);
        vst1q_s32(dst + i, r);
    }
#endif
    for (; i < n; i++) {
        int64_t v = (int64_t)dst[i] - (int64_t)a[i];
        dst[i] = m4t_mtfp_clamp64(v);
    }
}
