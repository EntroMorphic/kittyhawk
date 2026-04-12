/*
 * m4t_mtfp_w.c — MTFP39 wide-cell arithmetic (int64, 39 trits)
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Parallel to m4t_mtfp.c. Uses int64x2_t NEON (2 lanes) where m4t_mtfp.c
 * uses int32x4_t (4 lanes). Multiply and matmul use __int128 accumulators.
 */

#include "m4t_mtfp_w.h"
#include "m4t_trit_pack.h"
#include "m4t_internal.h"

#include <string.h>
#include <assert.h>

/* ── Vector arithmetic ─────────────────────────────────────────────────── */

void m4t_mtfp_w_vec_zero(m4t_mtfp_w_t* x, int n) {
    memset(x, 0, (size_t)n * sizeof(m4t_mtfp_w_t));
}

void m4t_mtfp_w_vec_add(
    m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* a, const m4t_mtfp_w_t* b, int n)
{
    int i = 0;
#if M4T_HAS_NEON
    const int64x2_t vmax = vdupq_n_s64( (int64_t)M4T_MTFPW_MAX_VAL);
    const int64x2_t vmin = vdupq_n_s64(-(int64_t)M4T_MTFPW_MAX_VAL);
    for (; i + 2 <= n; i += 2) {
        int64x2_t s = vaddq_s64(vld1q_s64(a + i), vld1q_s64(b + i));
        /* No vminq_s64 / vmaxq_s64 on aarch64 NEON — use compare + select. */
        uint64x2_t gt = vcgtq_s64(s, vmax);
        uint64x2_t lt = vcltq_s64(s, vmin);
        s = vbslq_s64(gt, vmax, s);
        s = vbslq_s64(lt, vmin, s);
        vst1q_s64(dst + i, s);
    }
#endif
    for (; i < n; i++) dst[i] = m4t_mtfp_w_add(a[i], b[i]);
}

void m4t_mtfp_w_vec_add_inplace(m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* a, int n) {
    int i = 0;
#if M4T_HAS_NEON
    const int64x2_t vmax = vdupq_n_s64( (int64_t)M4T_MTFPW_MAX_VAL);
    const int64x2_t vmin = vdupq_n_s64(-(int64_t)M4T_MTFPW_MAX_VAL);
    for (; i + 2 <= n; i += 2) {
        int64x2_t s = vaddq_s64(vld1q_s64(dst + i), vld1q_s64(a + i));
        uint64x2_t gt = vcgtq_s64(s, vmax);
        uint64x2_t lt = vcltq_s64(s, vmin);
        s = vbslq_s64(gt, vmax, s);
        s = vbslq_s64(lt, vmin, s);
        vst1q_s64(dst + i, s);
    }
#endif
    for (; i < n; i++) dst[i] = m4t_mtfp_w_add(dst[i], a[i]);
}

void m4t_mtfp_w_vec_sub_inplace(m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* a, int n) {
    int i = 0;
#if M4T_HAS_NEON
    const int64x2_t vmax = vdupq_n_s64( (int64_t)M4T_MTFPW_MAX_VAL);
    const int64x2_t vmin = vdupq_n_s64(-(int64_t)M4T_MTFPW_MAX_VAL);
    for (; i + 2 <= n; i += 2) {
        int64x2_t s = vsubq_s64(vld1q_s64(dst + i), vld1q_s64(a + i));
        uint64x2_t gt = vcgtq_s64(s, vmax);
        uint64x2_t lt = vcltq_s64(s, vmin);
        s = vbslq_s64(gt, vmax, s);
        s = vbslq_s64(lt, vmin, s);
        vst1q_s64(dst + i, s);
    }
#endif
    for (; i < n; i++) dst[i] = m4t_mtfp_w_sub(dst[i], a[i]);
}

void m4t_mtfp_w_vec_scale(
    m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* src, m4t_mtfp_w_t scale, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i] = m4t_mtfp_w_mul(src[i], scale);
    }
}

/* ── Dense MTFP39 × MTFP39 matmul ─────────────────────────────────────── */

void m4t_mtfp_w_matmul_bt(
    m4t_mtfp_w_t* Y, const m4t_mtfp_w_t* X, const m4t_mtfp_w_t* W,
    int M, int K, int N)
{
    assert(Y && X && W);
    assert(M >= 0 && K >= 0 && N >= 0);

    for (int i = 0; i < M; i++) {
        const m4t_mtfp_w_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const m4t_mtfp_w_t* wj = W + (size_t)j * K;
            __int128 acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (__int128)xi[k] * (__int128)wj[k];
            }
            if (acc >= 0) acc += M4T_MTFPW_SCALE / 2;
            else          acc -= M4T_MTFPW_SCALE / 2;
            Y[i * N + j] = m4t_mtfp_w_clamp128(acc / M4T_MTFPW_SCALE);
        }
    }
}

/* ── MTFP39 × packed-trit matmul ───────────────────────────────────────── */

void m4t_mtfp_w_ternary_matmul_bt(
    m4t_mtfp_w_t* Y, const m4t_mtfp_w_t* X, const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(Y && X && W_packed);
    assert(M >= 0 && K >= 0 && N >= 0);
    int Kp = M4T_TRIT_PACKED_BYTES(K);

    for (int i = 0; i < M; i++) {
        const m4t_mtfp_w_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_packed + (size_t)j * Kp;
            __int128 acc = 0;
            for (int k = 0; k < K; k++) {
                uint8_t code = (uint8_t)((wj[k >> 2] >> ((k & 3) * 2)) & 0x3u);
                if      (code == 0x01u) acc += (__int128)xi[k];
                else if (code == 0x02u) acc -= (__int128)xi[k];
            }
            Y[i * N + j] = m4t_mtfp_w_clamp128(acc);
        }
    }
}
