/*
 * m4t_mtfp.c — MTFP19 mantissa-layer primitives, block-native.
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Block-native block_add / block_sub are the atomic substrate ops:
 * one 128-bit NEON vector load → add/sub → min/max clamp → store.
 * No loop, no tail, no branch. The vec variants loop over whole
 * blocks then handle the scalar tail (< 4 cells).
 *
 * Saturation strategy (§8.5 Case S): since M4T_MTFP_MAX_VAL =
 * (3^19 - 1)/2 = 581 130 733, two in-range mantissas sum to at most
 * 2·MAX_VAL ≈ 1.16·10⁹, which fits int32 (< 2.15·10⁹). So `vaddq_s32`
 * without saturation followed by min/max clamp is exact — the
 * compile-time assertion in m4t_mtfp.h guarantees this.
 */

#include "m4t_mtfp.h"
#include "m4t_internal.h"

#include <string.h>

/* ── Block-native ─────────────────────────────────────────────────────── */

void m4t_mtfp_block_add(
    m4t_mtfp_t dst[M4T_MTFP_CELLS_PER_BLOCK],
    const m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK])
{
#if M4T_HAS_NEON
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    int32x4_t d = vld1q_s32(dst);
    int32x4_t s = vld1q_s32(a);
    int32x4_t r = vaddq_s32(d, s);
    r = vminq_s32(r, vmax);
    r = vmaxq_s32(r, vmin);
    vst1q_s32(dst, r);
#else
    for (int i = 0; i < M4T_MTFP_CELLS_PER_BLOCK; i++) {
        dst[i] = m4t_mtfp_clamp64((int64_t)dst[i] + (int64_t)a[i]);
    }
#endif
}

void m4t_mtfp_block_sub(
    m4t_mtfp_t dst[M4T_MTFP_CELLS_PER_BLOCK],
    const m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK])
{
#if M4T_HAS_NEON
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    int32x4_t d = vld1q_s32(dst);
    int32x4_t s = vld1q_s32(a);
    int32x4_t r = vsubq_s32(d, s);
    r = vminq_s32(r, vmax);
    r = vmaxq_s32(r, vmin);
    vst1q_s32(dst, r);
#else
    for (int i = 0; i < M4T_MTFP_CELLS_PER_BLOCK; i++) {
        dst[i] = m4t_mtfp_clamp64((int64_t)dst[i] - (int64_t)a[i]);
    }
#endif
}

/* ── Vec-native (compositions) ────────────────────────────────────────── */

void m4t_mtfp_vec_zero(m4t_mtfp_t* dst, int n) {
    if (n <= 0) return;
    memset(dst, 0, (size_t)n * sizeof(m4t_mtfp_t));
}

void m4t_mtfp_vec_add_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n) {
    int i = 0;
    for (; i + M4T_MTFP_CELLS_PER_BLOCK <= n; i += M4T_MTFP_CELLS_PER_BLOCK) {
        m4t_mtfp_block_add(dst + i, a + i);
    }
    for (; i < n; i++) {
        dst[i] = m4t_mtfp_clamp64((int64_t)dst[i] + (int64_t)a[i]);
    }
}

void m4t_mtfp_vec_sub_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n) {
    int i = 0;
    for (; i + M4T_MTFP_CELLS_PER_BLOCK <= n; i += M4T_MTFP_CELLS_PER_BLOCK) {
        m4t_mtfp_block_sub(dst + i, a + i);
    }
    for (; i < n; i++) {
        dst[i] = m4t_mtfp_clamp64((int64_t)dst[i] - (int64_t)a[i]);
    }
}
