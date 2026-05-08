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
#include "m4t_pow3_magic.h"   /* divide-by-3^k magic-multiply table (shift3 NEON path) */

#include <math.h>             /* sqrt/cos/sin — _scalar_ref oracles + RoPE LUT init */
#include <stdlib.h>           /* realloc — RoPE LUT lifecycle */
#include <string.h>

/* ── Block-native ─────────────────────────────────────────────────────── */

void m4t_mtfp_block_add(
    m4t_mtfp_t dst[M4T_MTFP_CELLS_PER_BLOCK],
    const m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK])
{
    /* NEON-only. CMake configure requires aarch64+NEON; the dead scalar
     * fallback was removed in the project-wide no-scalar audit per
     * CONTRIBUTING (feedback_function_over_speed_no_scalar memory). */
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    int32x4_t d = vld1q_s32(dst);
    int32x4_t s = vld1q_s32(a);
    int32x4_t r = vaddq_s32(d, s);
    r = vminq_s32(r, vmax);
    r = vmaxq_s32(r, vmin);
    vst1q_s32(dst, r);
}

void m4t_mtfp_block_sub(
    m4t_mtfp_t dst[M4T_MTFP_CELLS_PER_BLOCK],
    const m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK])
{
    /* NEON-only; see m4t_mtfp_block_add comment. */
    const int32x4_t vmax = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    const int32x4_t vmin = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    int32x4_t d = vld1q_s32(dst);
    int32x4_t s = vld1q_s32(a);
    int32x4_t r = vsubq_s32(d, s);
    r = vminq_s32(r, vmax);
    r = vmaxq_s32(r, vmin);
    vst1q_s32(dst, r);
}

/* ── Vec-native (compositions) ────────────────────────────────────────── */

void m4t_mtfp_vec_zero(m4t_mtfp_t* dst, int n) {
    assert(n >= 0);
    if (n == 0) return;
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

/* V13.B of pure-ternary audit: NEON int32 × int32 → int64 dot product.
 * Replaces scalar loops in bitnet_lm_head and bitnet_argmax_full_vocab.
 * Boundary handled via stack-local zero-padded 4-element buffers
 * (no scalar tail per condition (5)). */
int64_t m4t_mtfp_vec_dot_i64(const m4t_mtfp_t* x, const m4t_mtfp_t* y, int n) {
    assert(n >= 0);
    if (n == 0) return 0;
    assert(x && y);

#if M4T_HAS_NEON
    int64x2_t acc_lo = vdupq_n_s64(0);
    int64x2_t acc_hi = vdupq_n_s64(0);
    int i = 0;
    int n_aligned = n - (n % 4);
    for (; i < n_aligned; i += 4) {
        int32x4_t xv = vld1q_s32(x + i);
        int32x4_t yv = vld1q_s32(y + i);
        acc_lo = vmlal_s32(acc_lo, vget_low_s32(xv),  vget_low_s32(yv));
        acc_hi = vmlal_s32(acc_hi, vget_high_s32(xv), vget_high_s32(yv));
    }
    if (i < n) {
        int avail = n - i;
        m4t_mtfp_t xbuf[4] = {0}, ybuf[4] = {0};
        for (int j = 0; j < avail; j++) {
            xbuf[j] = x[i + j];
            ybuf[j] = y[i + j];
        }
        int32x4_t xv = vld1q_s32(xbuf);
        int32x4_t yv = vld1q_s32(ybuf);
        acc_lo = vmlal_s32(acc_lo, vget_low_s32(xv),  vget_low_s32(yv));
        acc_hi = vmlal_s32(acc_hi, vget_high_s32(xv), vget_high_s32(yv));
    }
    return vgetq_lane_s64(acc_lo, 0) + vgetq_lane_s64(acc_lo, 1)
         + vgetq_lane_s64(acc_hi, 0) + vgetq_lane_s64(acc_hi, 1);
#else
#error "m4t_mtfp_vec_dot_i64 requires NEON; no scalar fallback per project rule."
#endif
}

int64_t m4t_mtfp_vec_dot_i64_scalar_ref(const m4t_mtfp_t* x, const m4t_mtfp_t* y, int n) {
    assert(n >= 0);
    if (n == 0) return 0;
    assert(x && y);
    int64_t acc = 0;
    for (int i = 0; i < n; i++) {
        acc += (int64_t)x[i] * (int64_t)y[i];
    }
    return acc;
}

/* V14.B of pure-ternary audit: NEON attention V output projection.
 * y[d] = clamp64(sum_t w[t] * V[t][d] >> shift). Inner d-loop NEON;
 * outer t-loop broadcasts w[t]. Shift+clamp pass also NEON. */
void m4t_mtfp_attn_v_combine(
    m4t_mtfp_t* y, int shift,
    const m4t_mtfp_t* w,
    const m4t_mtfp_t* V_base, size_t v_stride,
    int seq_k, int head_dim)
{
    assert(seq_k >= 0 && head_dim >= 0);
    assert(shift >= 0 && shift <= 62);
    if (seq_k == 0 || head_dim == 0) return;
    assert(y && w && V_base);
#if M4T_HAS_NEON
    /* Stack-local int64 accumulator. head_dim is small (≤ 256 in BitNet),
     * so 2 KB stack is fine. VLA. */
    int64_t acc[head_dim];
    for (int d = 0; d < head_dim; d++) acc[d] = 0;

    int n_aligned = head_dim - (head_dim % 4);

    /* Outer t loop, inner NEON d loop. */
    for (int t = 0; t < seq_k; t++) {
        const m4t_mtfp_t* v_row = V_base + (size_t)t * v_stride;
        int32x2_t wv = vdup_n_s32(w[t]);
        int d = 0;
        for (; d < n_aligned; d += 4) {
            int32x4_t v4 = vld1q_s32(v_row + d);
            int64x2_t acc_lo = vld1q_s64(acc + d);
            int64x2_t acc_hi = vld1q_s64(acc + d + 2);
            acc_lo = vmlal_s32(acc_lo, wv, vget_low_s32(v4));
            acc_hi = vmlal_s32(acc_hi, wv, vget_high_s32(v4));
            vst1q_s64(acc + d, acc_lo);
            vst1q_s64(acc + d + 2, acc_hi);
        }
        if (d < head_dim) {
            /* Boundary tile: copy into 4-wide stack bufs to make the
             * vector loads/stores safe regardless of avail (1, 2, or 3).
             * Same pattern as the shift+clamp tile below. */
            int avail = head_dim - d;
            m4t_mtfp_t vbuf[4] = {0};
            int64_t   abuf[4] = {0,0,0,0};
            for (int j = 0; j < avail; j++) {
                vbuf[j] = v_row[d + j];
                abuf[j] = acc[d + j];
            }
            int32x4_t v4 = vld1q_s32(vbuf);
            int64x2_t acc_lo = vld1q_s64(abuf);
            int64x2_t acc_hi = vld1q_s64(abuf + 2);
            acc_lo = vmlal_s32(acc_lo, wv, vget_low_s32(v4));
            acc_hi = vmlal_s32(acc_hi, wv, vget_high_s32(v4));
            vst1q_s64(abuf, acc_lo);
            vst1q_s64(abuf + 2, acc_hi);
            for (int j = 0; j < avail; j++) acc[d + j] = abuf[j];
        }
    }

    /* NEON shift + clamp + narrow. y[d] = clamp64(acc[d] >> shift).
     * vminq_s64/vmaxq_s64 are not in standard NEON, so we clamp at int32:
     *   - vqmovn_s64 saturates int64 → int32 (covers gross overflow)
     *   - vminq_s32/vmaxq_s32 then enforce ±M4T_MTFP_MAX_VAL (MTFP19 range)
     * For BitNet (shift=30), shifted values fit int32 in practice; the
     * vqmovn_s64 saturation is a safety net, the int32 clamp is what
     * makes the result MTFP19-conformant. */
    int64x2_t cnt    = vdupq_n_s64(-shift);
    int32x4_t v_max  = vdupq_n_s32(M4T_MTFP_MAX_VAL);
    int32x4_t v_min  = vdupq_n_s32(-(int32_t)M4T_MTFP_MAX_VAL);
    int d = 0;
    for (; d < n_aligned; d += 4) {
        int64x2_t acc_lo = vld1q_s64(acc + d);
        int64x2_t acc_hi = vld1q_s64(acc + d + 2);
        int64x2_t s_lo = vshlq_s64(acc_lo, cnt);
        int64x2_t s_hi = vshlq_s64(acc_hi, cnt);
        int32x2_t y_lo = vqmovn_s64(s_lo);
        int32x2_t y_hi = vqmovn_s64(s_hi);
        int32x4_t y4 = vcombine_s32(y_lo, y_hi);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(y + d, y4);
    }
    if (d < head_dim) {
        int avail = head_dim - d;
        int64_t buf[4] = {0,0,0,0};
        for (int j = 0; j < avail; j++) buf[j] = acc[d + j];
        int64x2_t acc_lo = vld1q_s64(buf);
        int64x2_t acc_hi = vld1q_s64(buf + 2);
        int64x2_t s_lo = vshlq_s64(acc_lo, cnt);
        int64x2_t s_hi = vshlq_s64(acc_hi, cnt);
        int32x2_t y_lo = vqmovn_s64(s_lo);
        int32x2_t y_hi = vqmovn_s64(s_hi);
        int32x4_t y4 = vcombine_s32(y_lo, y_hi);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        m4t_mtfp_t ybuf[4];
        vst1q_s32(ybuf, y4);
        for (int j = 0; j < avail; j++) y[d + j] = ybuf[j];
    }
#else
#error "m4t_mtfp_attn_v_combine requires NEON; no scalar fallback per project rule."
#endif
}

void m4t_mtfp_attn_v_combine_scalar_ref(
    m4t_mtfp_t* y, int shift,
    const m4t_mtfp_t* w,
    const m4t_mtfp_t* V_base, size_t v_stride,
    int seq_k, int head_dim)
{
    assert(seq_k >= 0 && head_dim >= 0);
    assert(shift >= 0 && shift <= 62);
    if (seq_k == 0 || head_dim == 0) return;
    assert(y && w && V_base);
    for (int d = 0; d < head_dim; d++) {
        int64_t acc = 0;
        for (int t = 0; t < seq_k; t++) {
            acc += (int64_t)w[t] * (int64_t)V_base[(size_t)t * v_stride + d];
        }
        y[d] = m4t_mtfp_clamp64(acc >> shift);
    }
}

/* ── Cross-exponent accumulator (§14.2 named opt-in) ────────────────────── */

/* Powers of 3 up to 3^19, defined as compile-time integer constants so
 * the static_assert below can verify the odd-divisor invariant.
 *
 * The odd-divisor invariant is what makes round-to-nearest-even (§8.2)
 * unambiguous in m4t_pow3_round_div below; halfway cases (rem == s/2)
 * cannot occur with integer M because s/2 is never an integer for odd s.
 * The "even" tie-break in §8.2 is satisfied vacuously. */
#define M4T_POW3_0   (1LL)
#define M4T_POW3_1   (3LL)
#define M4T_POW3_2   (9LL)
#define M4T_POW3_3   (27LL)
#define M4T_POW3_4   (81LL)
#define M4T_POW3_5   (243LL)
#define M4T_POW3_6   (729LL)
#define M4T_POW3_7   (2187LL)
#define M4T_POW3_8   (6561LL)
#define M4T_POW3_9   (19683LL)
#define M4T_POW3_10  (59049LL)
#define M4T_POW3_11  (177147LL)
#define M4T_POW3_12  (531441LL)
#define M4T_POW3_13  (1594323LL)
#define M4T_POW3_14  (4782969LL)
#define M4T_POW3_15  (14348907LL)
#define M4T_POW3_16  (43046721LL)
#define M4T_POW3_17  (129140163LL)
#define M4T_POW3_18  (387420489LL)
#define M4T_POW3_19  (1162261467LL)

/* Compile-time guard: every power-of-3 constant must be odd. AND'ing all
 * the LSBs together gives 1 iff every entry has its LSB set. */
_Static_assert(
    ((M4T_POW3_0  & 1) & (M4T_POW3_1  & 1) & (M4T_POW3_2  & 1) &
     (M4T_POW3_3  & 1) & (M4T_POW3_4  & 1) & (M4T_POW3_5  & 1) &
     (M4T_POW3_6  & 1) & (M4T_POW3_7  & 1) & (M4T_POW3_8  & 1) &
     (M4T_POW3_9  & 1) & (M4T_POW3_10 & 1) & (M4T_POW3_11 & 1) &
     (M4T_POW3_12 & 1) & (M4T_POW3_13 & 1) & (M4T_POW3_14 & 1) &
     (M4T_POW3_15 & 1) & (M4T_POW3_16 & 1) & (M4T_POW3_17 & 1) &
     (M4T_POW3_18 & 1) & (M4T_POW3_19 & 1)) == 1,
    "M4T_POW3_* must all be odd (round-to-nearest invariant)");

/* Beyond Δ=19, |smaller_mantissa| ≤ MAX_VAL = (3^19 - 1)/2, so the
 * round-to-nearest quotient by 3^20 (or larger) is always 0. The kernel
 * handles Δ ≥ 20 as the degenerate edge without consulting this table. */
static const int64_t M4T_POW3_TABLE[20] = {
    M4T_POW3_0,  M4T_POW3_1,  M4T_POW3_2,  M4T_POW3_3,  M4T_POW3_4,
    M4T_POW3_5,  M4T_POW3_6,  M4T_POW3_7,  M4T_POW3_8,  M4T_POW3_9,
    M4T_POW3_10, M4T_POW3_11, M4T_POW3_12, M4T_POW3_13, M4T_POW3_14,
    M4T_POW3_15, M4T_POW3_16, M4T_POW3_17, M4T_POW3_18, M4T_POW3_19
};

/* Base-3 round-to-nearest-even divide (§8.2). The runtime assertion
 * documents the odd-divisor invariant; combined with the static check
 * above, ties cannot occur and the "even" tie-break is satisfied
 * vacuously. *had_remainder is set non-zero iff M was not exactly
 * divisible by s (the quotient was rounded). */
static int64_t m4t_pow3_round_div(int64_t M, int64_t s, int* had_remainder) {
    assert(s & 1);                     /* odd-divisor invariant — no ties */
    int64_t q = M / s;                 /* C truncate toward zero */
    int64_t rem = M - q * s;           /* sign matches M; |rem| < s */
    *had_remainder = (rem != 0);
    /* Round-to-nearest: shift q by one mantissa unit iff 2|rem| > s.
     * (For odd s, equality 2|rem| == s never holds at integer rem.) */
    if (rem > 0) {
        if ((int64_t)2 * rem > s) q += 1;
    } else if (rem < 0) {
        if ((int64_t)2 * (-rem) > s) q -= 1;
    }
    return q;
}

/* m4t_flag_or is now shared via m4t_internal.h (used by both this kernel
 * and the ternary matmul). Per-block layout per m4t_mtfp.h. */

/* Scalar reference implementation factored as a static helper so the
 * public m4t_mtfp_vec_accum_aligning() can dispatch to NEON later
 * (A-G7) while m4t_mtfp_vec_accum_aligning_scalar_ref() always calls
 * this directly. Per A-G1 (cross_exp_accum_routing_synthesize.md). */
static void accum_aligning_scalar(
    m4t_mtfp_t* running, int8_t* running_exp,
    const m4t_mtfp_t* addend, int8_t addend_exp,
    uint8_t* flags,
    int n);

#if M4T_HAS_NEON
/* R-G1 (cross_exp_accum_routing remediation): NEON same-exp accumulate
 * with flag tracking. Replaces the prior scalar fallback when same-exp
 * + flags!=NULL — that path violated the no-scalar production rule.
 *
 * Pipeline per 4 cells: vaddq_s32 + min/max clamp + cmeq for SATURATED
 * reconstruction + per-lane flag OR. Stays in int32 throughout (sum
 * bounded by 2*MAX_VAL ≈ 1.16e9 < INT32_MAX). No ROUNDED bit (no
 * divide).
 *
 * For same-exp + flags==NULL, the dispatcher continues to call
 * vec_add_inplace (already NEON-fast); this helper exists for the
 * flags!=NULL path. */
static void accum_same_exp_with_flags_neon(
    m4t_mtfp_t* running,
    const m4t_mtfp_t* addend,
    uint8_t* flags,
    int n)
{
    int32x4_t pos_max = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    int32x4_t neg_max = vdupq_n_s32(-M4T_MTFP_MAX_VAL);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t a = vld1q_s32(running + i);
        int32x4_t b = vld1q_s32(addend + i);
        int32x4_t sum = vaddq_s32(a, b);
        int32x4_t clamped = vminq_s32(vmaxq_s32(sum, neg_max), pos_max);
        uint32x4_t sat_mask = vmvnq_u32(vceqq_s32(sum, clamped));
        vst1q_s32(running + i, clamped);

        uint32_t s0 = vgetq_lane_u32(sat_mask, 0);
        uint32_t s1 = vgetq_lane_u32(sat_mask, 1);
        uint32_t s2 = vgetq_lane_u32(sat_mask, 2);
        uint32_t s3 = vgetq_lane_u32(sat_mask, 3);
        if (s0) m4t_flag_or(flags, i + 0, M4T_FLAG_SATURATED);
        if (s1) m4t_flag_or(flags, i + 1, M4T_FLAG_SATURATED);
        if (s2) m4t_flag_or(flags, i + 2, M4T_FLAG_SATURATED);
        if (s3) m4t_flag_or(flags, i + 3, M4T_FLAG_SATURATED);
    }
    /* Scalar tail for n not multiple of 4 (geometric, not a "fallback"). */
    for (; i < n; i++) {
        int64_t sum = (int64_t)running[i] + (int64_t)addend[i];
        m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
        if (sum != (int64_t)out) m4t_flag_or(flags, i, M4T_FLAG_SATURATED);
        running[i] = out;
    }
}

/* A-G3 prototype: fused NEON inner loop for the cross-exp align+add+clamp.
 * Routes the divide step through the same vmlal_s32 magic-multiply
 * pipeline that productionized for shift3 (m4t_pow3_magic.h is shared).
 *
 * Per-iter (4 cells = one int32x4):
 *   1. Divide X[lane] by 3^delta via vmlal+bias+shift+narrow
 *   2. Reconstruct ROUNDED bit: (aligned * s != X[lane])
 *   3. Sum = aligned + Y[lane] (int32, no widening — bounded by 2*MAX_VAL)
 *   4. Clamp sum to ±MAX_VAL via min/max
 *   5. Reconstruct SATURATED bit: (sum != clamped)
 *   6. Store clamped → running[lane]
 *   7. Per-lane: OR ROUNDED+SATURATED into flag byte if requested
 *
 * Bound argument (A-G5/saturation): aligned ≤ MAX_VAL/s per lane;
 * other ≤ MAX_VAL; sum ≤ MAX_VAL/s + MAX_VAL ≤ MAX_VAL + MAX_VAL/3 < INT32_MAX.
 * Reconstructed = aligned*s ≤ MAX_VAL fits int32.
 *
 * Per A-G3. Will be wired into the public dispatcher at A-G7. */
static void accum_aligning_neon_block(
    m4t_mtfp_t*       result,        /* int32x4 output (running buffer) */
    const m4t_mtfp_t* div_src,       /* the side to be divided by 3^delta */
    const m4t_mtfp_t* add_src,       /* the side to be added (un-aligned) */
    int abs_delta,
    int n,
    uint8_t* flags)
{
    /* Constants derived once per call. */
    int32_t M    = M4T_POW3_DIV_M[abs_delta];
    int     N    = M4T_POW3_DIV_N[abs_delta];
    int32_t s32  = (int32_t)M4T_POW3_TABLE[abs_delta];

    int32x2_t Mv     = vdup_n_s32(M);
    int64x2_t bias   = vdupq_n_s64((int64_t)1 << (N - 1));
    int64x2_t neg_N  = vdupq_n_s64(-(int64_t)N);
    int32x4_t s_v    = vdupq_n_s32(s32);
    int32x4_t pos_max = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    int32x4_t neg_max = vdupq_n_s32(-M4T_MTFP_MAX_VAL);
    uint32x4_t zero_u = vdupq_n_u32(0);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        /* Load 4 cells of each input. */
        int32x4_t val   = vld1q_s32(div_src + i);
        int32x4_t other = vld1q_s32(add_src + i);

        /* Magic-multiply divide: aligned = val / 3^abs_delta (round-to-nearest). */
        int64x2_t prod_lo = vmull_s32(vget_low_s32(val),  Mv);
        int64x2_t prod_hi = vmull_s32(vget_high_s32(val), Mv);
        prod_lo = vaddq_s64(prod_lo, bias);
        prod_hi = vaddq_s64(prod_hi, bias);
        prod_lo = vshlq_s64(prod_lo, neg_N);
        prod_hi = vshlq_s64(prod_hi, neg_N);
        int32x4_t aligned = vcombine_s32(vmovn_s64(prod_lo), vmovn_s64(prod_hi));

        /* ROUNDED reconstruction: aligned * s != val (per lane). */
        int32x4_t reconstructed = vmulq_s32(aligned, s_v);
        uint32x4_t rounded_mask = vmvnq_u32(vceqq_s32(reconstructed, val));

        /* Sum + clamp + SATURATED reconstruction. Stays in int32 throughout
         * — sum is bounded by MAX_VAL + MAX_VAL/3 < INT32_MAX. */
        int32x4_t sum     = vaddq_s32(aligned, other);
        int32x4_t clamped = vminq_s32(vmaxq_s32(sum, neg_max), pos_max);
        uint32x4_t sat_mask = vmvnq_u32(vceqq_s32(sum, clamped));

        vst1q_s32(result + i, clamped);

        /* Per-lane flag bookkeeping. ~4 lane extracts + 4 OR-into-flag-byte.
         * Skipped entirely if flags == NULL (the masks were computed but
         * are discarded — cheap relative to the divide work). */
        if (flags) {
            uint32_t r0 = vgetq_lane_u32(rounded_mask, 0);
            uint32_t r1 = vgetq_lane_u32(rounded_mask, 1);
            uint32_t r2 = vgetq_lane_u32(rounded_mask, 2);
            uint32_t r3 = vgetq_lane_u32(rounded_mask, 3);
            uint32_t s0 = vgetq_lane_u32(sat_mask, 0);
            uint32_t s1 = vgetq_lane_u32(sat_mask, 1);
            uint32_t s2 = vgetq_lane_u32(sat_mask, 2);
            uint32_t s3 = vgetq_lane_u32(sat_mask, 3);
            uint8_t b0 = (r0 ? M4T_FLAG_ROUNDED : 0) | (s0 ? M4T_FLAG_SATURATED : 0);
            uint8_t b1 = (r1 ? M4T_FLAG_ROUNDED : 0) | (s1 ? M4T_FLAG_SATURATED : 0);
            uint8_t b2 = (r2 ? M4T_FLAG_ROUNDED : 0) | (s2 ? M4T_FLAG_SATURATED : 0);
            uint8_t b3 = (r3 ? M4T_FLAG_ROUNDED : 0) | (s3 ? M4T_FLAG_SATURATED : 0);
            if (b0) m4t_flag_or(flags, i + 0, b0);
            if (b1) m4t_flag_or(flags, i + 1, b1);
            if (b2) m4t_flag_or(flags, i + 2, b2);
            if (b3) m4t_flag_or(flags, i + 3, b3);
        }
        (void)zero_u;  /* reserved for future signed-overflow check if needed */
    }

    /* Scalar tail for n not multiple of 4. Geometrically necessary;
     * not a "scalar fallback" — there's no NEON path for sub-vector n. */
    int64_t s = M4T_POW3_TABLE[abs_delta];
    for (; i < n; i++) {
        int had_rem = 0;
        int64_t aa = m4t_pow3_round_div((int64_t)div_src[i], s, &had_rem);
        int64_t sum = aa + (int64_t)add_src[i];
        m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
        if (flags) {
            uint8_t bits = 0;
            if (had_rem)             bits |= M4T_FLAG_ROUNDED;
            if (sum != (int64_t)out) bits |= M4T_FLAG_SATURATED;
            if (bits) m4t_flag_or(flags, i, bits);
        }
        result[i] = out;
    }
}
#endif

void m4t_mtfp_vec_accum_aligning(
    m4t_mtfp_t* running, int8_t* running_exp,
    const m4t_mtfp_t* addend, int8_t addend_exp,
    uint8_t* flags,
    int n)
{
    /* Production NEON-only dispatcher per project rule
     * (feedback_function_over_speed_no_scalar). Three branches:
     *   - same-exp: vec_add_inplace (flags=NULL) or
     *                accum_same_exp_with_flags_neon (flags!=NULL).
     *   - cross-exp non-degenerate: accum_aligning_neon_block.
     *   - cross-exp degenerate (|delta|>=20): scalar memcpy/no-op
     *     with per-cell flag annotation (kept scalar — per-cell
     *     conditional flag work doesn't NEON-ize cleanly and the
     *     branch is a degenerate edge case).
     *
     * The accum_aligning_scalar helper remains as the implementation
     * behind m4t_mtfp_vec_accum_aligning_scalar_ref (test oracle only). */
    assert(n >= 0);
    assert(n == 0 || (running && addend));
    assert(running_exp);
    if (n == 0) return;
    assert(running != addend);

    int8_t e_run = *running_exp;

    if (addend_exp == e_run) {
        if (flags == NULL) {
            m4t_mtfp_vec_add_inplace(running, addend, n);
        } else {
            accum_same_exp_with_flags_neon(running, addend, flags, n);
        }
        return;
    }

    if (addend_exp > e_run) {
        int delta = (int)addend_exp - (int)e_run;
        if (delta >= 20) {
            /* Degenerate: running rounds to zero. Result = addend. */
            for (int i = 0; i < n; i++) {
                if (flags && running[i] != 0) {
                    m4t_flag_or(flags, i, M4T_FLAG_ROUNDED);
                }
                running[i] = addend[i];
            }
            *running_exp = addend_exp;
            return;
        }
        accum_aligning_neon_block(running, running, addend, delta, n, flags);
        *running_exp = addend_exp;
        return;
    }

    /* addend_exp < e_run: divide addend by 3^delta, add to running. */
    int delta = (int)e_run - (int)addend_exp;
    if (delta >= 20) {
        if (flags) {
            for (int i = 0; i < n; i++) {
                if (addend[i] != 0) m4t_flag_or(flags, i, M4T_FLAG_ROUNDED);
            }
        }
        return;
    }
    accum_aligning_neon_block(running, addend, running, delta, n, flags);
}

/* Public scalar-only reference for test verification. Never dispatches
 * to NEON. Per A-G1. */
void m4t_mtfp_vec_accum_aligning_scalar_ref(
    m4t_mtfp_t* running, int8_t* running_exp,
    const m4t_mtfp_t* addend, int8_t addend_exp,
    uint8_t* flags,
    int n)
{
    accum_aligning_scalar(running, running_exp, addend, addend_exp, flags, n);
}

static void accum_aligning_scalar(
    m4t_mtfp_t* running, int8_t* running_exp,
    const m4t_mtfp_t* addend, int8_t addend_exp,
    uint8_t* flags,
    int n)
{
    assert(n >= 0);
    assert(n == 0 || (running && addend));
    assert(running_exp);

    /* n == 0: clean no-op. No cells means no rescale, no flag updates,
     * no exponent migration (the empty buffer has no representable
     * scale to migrate). */
    if (n == 0) return;

    /* Aliasing precondition: running and addend must be distinct buffers. */
    assert(running != addend);

    int8_t e_run = *running_exp;

    if (addend_exp == e_run) {
        /* Same-block-exp accumulation. No rescale, no rounding.
         *
         * Fast path: if the caller doesn't need flag tracking, this is
         * exactly m4t_mtfp_vec_add_inplace (NEON-vectorized via
         * m4t_mtfp_block_add). T2-C in journal/tier2_perf_precommit.md.
         *
         * If flags are tracked, fall back to scalar so we can detect
         * per-cell saturation events. */
        if (flags == NULL) {
            m4t_mtfp_vec_add_inplace(running, addend, n);
            return;
        }
        for (int i = 0; i < n; i++) {
            int64_t sum = (int64_t)running[i] + (int64_t)addend[i];
            m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
            if (sum != (int64_t)out) {
                m4t_flag_or(flags, i, M4T_FLAG_SATURATED);
            }
            running[i] = out;
        }
        return;
    }

    if (addend_exp > e_run) {
        /* Grow running_exp upward; rescale running mantissas down by
         * 3^Δ with round-to-nearest. */
        int delta = (int)addend_exp - (int)e_run;
        if (delta >= 20) {
            /* Degenerate: running rounds to zero. Result = addend. */
            for (int i = 0; i < n; i++) {
                if (flags && running[i] != 0) {
                    m4t_flag_or(flags, i, M4T_FLAG_ROUNDED);
                }
                running[i] = addend[i];
            }
            *running_exp = addend_exp;
            return;
        }
        int64_t s = M4T_POW3_TABLE[delta];
        for (int i = 0; i < n; i++) {
            int had_rem = 0;
            int64_t aa = m4t_pow3_round_div((int64_t)running[i], s, &had_rem);
            int64_t sum = aa + (int64_t)addend[i];
            m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
            if (flags) {
                uint8_t bits = 0;
                if (had_rem)             bits |= M4T_FLAG_ROUNDED;
                if (sum != (int64_t)out) bits |= M4T_FLAG_SATURATED;
                if (bits) m4t_flag_or(flags, i, bits);
            }
            running[i] = out;
        }
        *running_exp = addend_exp;
        return;
    }

    /* addend_exp < e_run: rescale addend down; running_exp unchanged. */
    int delta = (int)e_run - (int)addend_exp;
    if (delta >= 20) {
        /* Degenerate: addend rounds to zero. Running unchanged.
         * Mark rounded where addend was non-zero. */
        if (flags) {
            for (int i = 0; i < n; i++) {
                if (addend[i] != 0) m4t_flag_or(flags, i, M4T_FLAG_ROUNDED);
            }
        }
        return;
    }
    int64_t s = M4T_POW3_TABLE[delta];
    for (int i = 0; i < n; i++) {
        int had_rem = 0;
        int64_t bb = m4t_pow3_round_div((int64_t)addend[i], s, &had_rem);
        int64_t sum = (int64_t)running[i] + bb;
        m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
        if (flags) {
            uint8_t bits = 0;
            if (had_rem)             bits |= M4T_FLAG_ROUNDED;
            if (sum != (int64_t)out) bits |= M4T_FLAG_SATURATED;
            if (bits) m4t_flag_or(flags, i, bits);
        }
        running[i] = out;
    }
}

void m4t_mtfp_vec_add_aligning(
    m4t_mtfp_t* dst, int8_t* out_e,
    const m4t_mtfp_t* a, int8_t e_a,
    const m4t_mtfp_t* b, int8_t e_b,
    uint8_t* flags,
    int n)
{
    /* Wrapper aliasing contract: dst may alias a (handled below by the
     * skip-copy path); dst MUST NOT alias b (would corrupt b before the
     * accumulator reads it). */
    assert(dst != b);

    if (dst != a) {
        for (int i = 0; i < n; i++) dst[i] = a[i];
    }
    int8_t e = e_a;
    m4t_mtfp_vec_accum_aligning(dst, &e, b, e_b, flags, n);
    if (out_e) *out_e = e;
}

void m4t_mtfp_vec_sub_aligning(
    m4t_mtfp_t* dst, int8_t* out_e,
    const m4t_mtfp_t* a, int8_t e_a,
    const m4t_mtfp_t* b, int8_t e_b,
    uint8_t* flags,
    int n)
{
    /* Subtract is add-after-negate at the storage layer. Negation is
     * exact for any |b[i]| <= MAX_VAL (the substrate's mantissa
     * precondition implies |-b[i]| <= MAX_VAL). We materialize the
     * negation into dst when dst != a so that the accumulator sees a
     * negated `b`-shaped buffer, then call accum.
     *
     * To avoid the temporary, the implementation negates inside the
     * accumulator's loop. We replicate the four-case structure here
     * (same exp / grow up / addend rescales / degenerate) with a sign
     * flip on every read of b[i]. */
    assert(n >= 0);
    assert(n == 0 || (dst && a && b));
    assert(dst != b);

    /* Initialize dst from a if not aliased. */
    if (dst != a) {
        for (int i = 0; i < n; i++) dst[i] = a[i];
    }
    int8_t e = e_a;

    if (e_b == e) {
        for (int i = 0; i < n; i++) {
            int64_t sum = (int64_t)dst[i] - (int64_t)b[i];
            m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
            if (flags && sum != (int64_t)out) {
                m4t_flag_or(flags, i, M4T_FLAG_SATURATED);
            }
            dst[i] = out;
        }
        if (out_e) *out_e = e;
        return;
    }

    if (e_b > e) {
        int delta = (int)e_b - (int)e;
        if (delta >= 20) {
            /* Degenerate: dst rounds to zero; result = -b. */
            for (int i = 0; i < n; i++) {
                if (flags && dst[i] != 0) m4t_flag_or(flags, i, M4T_FLAG_ROUNDED);
                dst[i] = (m4t_mtfp_t)(-(int64_t)b[i]);
            }
            e = e_b;
            if (out_e) *out_e = e;
            return;
        }
        int64_t s = M4T_POW3_TABLE[delta];
        for (int i = 0; i < n; i++) {
            int had_rem = 0;
            int64_t aa = m4t_pow3_round_div((int64_t)dst[i], s, &had_rem);
            int64_t sum = aa - (int64_t)b[i];
            m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
            if (flags) {
                uint8_t bits = 0;
                if (had_rem)             bits |= M4T_FLAG_ROUNDED;
                if (sum != (int64_t)out) bits |= M4T_FLAG_SATURATED;
                if (bits) m4t_flag_or(flags, i, bits);
            }
            dst[i] = out;
        }
        e = e_b;
        if (out_e) *out_e = e;
        return;
    }

    /* e_b < e: rescale b down (with sign flip), add to dst. */
    int delta = (int)e - (int)e_b;
    if (delta >= 20) {
        if (flags) {
            for (int i = 0; i < n; i++) {
                if (b[i] != 0) m4t_flag_or(flags, i, M4T_FLAG_ROUNDED);
            }
        }
        if (out_e) *out_e = e;
        return;
    }
    int64_t s = M4T_POW3_TABLE[delta];
    for (int i = 0; i < n; i++) {
        int had_rem = 0;
        int64_t bb = m4t_pow3_round_div((int64_t)b[i], s, &had_rem);
        int64_t sum = (int64_t)dst[i] - bb;
        m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
        if (flags) {
            uint8_t bits = 0;
            if (had_rem)             bits |= M4T_FLAG_ROUNDED;
            if (sum != (int64_t)out) bits |= M4T_FLAG_SATURATED;
            if (bits) m4t_flag_or(flags, i, bits);
        }
        dst[i] = out;
    }
    if (out_e) *out_e = e;
}

/* ── shift3: base-3 positional shift (elemental floor primitive) ───────── */

/* Scalar divide loop. Used ONLY by m4t_mtfp_shift3_scalar_ref (test
 * oracle). The "non-NEON fallback inside m4t_mtfp_shift3" framing in
 * earlier comments was stale — production m4t_mtfp_shift3 dispatches
 * directly to shift3_div_neon per the no-scalar-in-production rule.
 * abs_k ∈ [1, 19]. */
static void shift3_div_scalar(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                              int abs_k, int n) {
    int64_t divisor = M4T_POW3_TABLE[abs_k];
    for (int i = 0; i < n; i++) {
        int had_rem = 0;
        int64_t q = m4t_pow3_round_div((int64_t)src[i], divisor, &had_rem);
        dst[i] = (m4t_mtfp_t)q;  /* |q| ≤ MAX_VAL/divisor + 1 ≤ MAX_VAL */
    }
}

#if M4T_HAS_NEON
/* NEON magic-multiply divide path. abs_k ∈ [1, 19].
 *
 * Pipeline per 4 lanes:
 *   prod_64 = (int64_t)x * M_table[abs_k]
 *   adj_64  = prod_64 + (1 << (N_table[abs_k] - 1))     // round-half-up
 *   result  = (int32_t)(adj_64 >> N_table[abs_k])        // arith right shift
 * (Compiler typically fuses vmull + vaddq into smlal under -O3 -mcpu=native.)
 *
 * Bit-exact vs shift3_div_scalar across the full substrate input range
 * (verified by the m4t_shift3_neon ctest's exhaustive mode and by
 * gen_pow3_magic.c). Constants in m4t_pow3_magic.h.
 *
 * Why vmull (64-bit intermediate) instead of vqrdmulhq + vrshlq (32-bit
 * with rounding): vqrdmulhq + vrshlq composes two round-to-nearest steps,
 * which accumulate compound rounding error and can't be made bit-exact
 * across all magnitudes for k > 1 without per-k specialization. The
 * 64-bit-intermediate path has ONE rounding step end-to-end and is
 * bit-exact by construction. Trade ~1.5x speed for bit-exactness
 * simplicity. See journal/shift3_neon_reflect.md and
 * journal/shift3_neon_synthesize.md. */
static void shift3_div_neon(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                            int abs_k, int n) {
    int32_t M = M4T_POW3_DIV_M[abs_k];
    int     N = M4T_POW3_DIV_N[abs_k];
    int32x2_t Mv = vdup_n_s32(M);
    int64x2_t bias = vdupq_n_s64((int64_t)1 << (N - 1));
    int64x2_t neg_N = vdupq_n_s64(-(int64_t)N);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t x = vld1q_s32((const int32_t*)(src + i));
        int64x2_t prod_lo = vmull_s32(vget_low_s32(x),  Mv);
        int64x2_t prod_hi = vmull_s32(vget_high_s32(x), Mv);
        prod_lo = vaddq_s64(prod_lo, bias);
        prod_hi = vaddq_s64(prod_hi, bias);
        prod_lo = vshlq_s64(prod_lo, neg_N);
        prod_hi = vshlq_s64(prod_hi, neg_N);
        int32x4_t result = vcombine_s32(vmovn_s64(prod_lo), vmovn_s64(prod_hi));
        vst1q_s32((int32_t*)(dst + i), result);
    }
    /* Scalar tail (n not a multiple of 4). */
    for (; i < n; i++) {
        int64_t prod = (int64_t)src[i] * (int64_t)M;
        int64_t adj  = prod + ((int64_t)1 << (N - 1));
        dst[i] = (m4t_mtfp_t)(adj >> N);
    }
}

/* NEON multiply path. abs_k ∈ [1, 19]. Computes dst[i] = clamp64(src[i]*3^k).
 *
 * Saturation analysis: |src| ≤ MAX_VAL ≈ 2^29.1; scale = 3^k for k ≤ 19, so
 * scale < 2^31 (fits in int32). Product |v| ≤ 2^29.1 × 2^30.1 = 2^59.2 — well
 * within int64. After clamp to ±MAX_VAL it fits int32.
 *
 * Pipeline per 4 lanes:
 *   prod_lo64 = vmull_s32(low2_of_x,  scale_v)
 *   prod_hi64 = vmull_s32(high2_of_x, scale_v)
 *   clamp each to [-MAX_VAL, +MAX_VAL] in int64 space (vminq/vmaxq_s64)
 *   narrow back to int32 (vmovn_s64)
 *   combine and store. */
static void shift3_mul_neon(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                            int k, int n) {
    int32_t scale = (int32_t)M4T_POW3_TABLE[k];
    int32x2_t scale_v = vdup_n_s32(scale);
    int64x2_t max_v = vdupq_n_s64((int64_t)M4T_MTFP_MAX_VAL);
    int64x2_t min_v = vdupq_n_s64(-(int64_t)M4T_MTFP_MAX_VAL);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t x = vld1q_s32((const int32_t*)(src + i));
        int64x2_t prod_lo = vmull_s32(vget_low_s32(x),  scale_v);
        int64x2_t prod_hi = vmull_s32(vget_high_s32(x), scale_v);
        /* Clamp to ±MAX_VAL via bsl (vminq_s64/vmaxq_s64 not always
         * available depending on toolchain — use vcgtq_s64 + vbslq_s64). */
        uint64x2_t over_lo  = vcgtq_s64(prod_lo, max_v);
        uint64x2_t under_lo = vcltq_s64(prod_lo, min_v);
        prod_lo = vbslq_s64(over_lo,  max_v, prod_lo);
        prod_lo = vbslq_s64(under_lo, min_v, prod_lo);
        uint64x2_t over_hi  = vcgtq_s64(prod_hi, max_v);
        uint64x2_t under_hi = vcltq_s64(prod_hi, min_v);
        prod_hi = vbslq_s64(over_hi,  max_v, prod_hi);
        prod_hi = vbslq_s64(under_hi, min_v, prod_hi);
        int32x4_t result = vcombine_s32(vmovn_s64(prod_lo), vmovn_s64(prod_hi));
        vst1q_s32((int32_t*)(dst + i), result);
    }
    /* Geometric scalar tail. */
    for (; i < n; i++) {
        int64_t v = (int64_t)src[i] * (int64_t)scale;
        dst[i] = m4t_mtfp_clamp64(v);
    }
}

/* NEON saturation collapse for k >= 20. Sign(src) → ±MAX_VAL or 0. */
static void shift3_mul_saturate_neon(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                                     int n) {
    int32x4_t zero = vdupq_n_s32(0);
    int32x4_t max_v = vdupq_n_s32(M4T_MTFP_MAX_VAL);
    int32x4_t min_v = vdupq_n_s32(-M4T_MTFP_MAX_VAL);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t x = vld1q_s32((const int32_t*)(src + i));
        uint32x4_t pos = vcgtq_s32(x, zero);
        uint32x4_t neg = vcltq_s32(x, zero);
        int32x4_t out = vbslq_s32(pos, max_v, zero);
        out = vbslq_s32(neg, min_v, out);
        vst1q_s32((int32_t*)(dst + i), out);
    }
    for (; i < n; i++) {
        if      (src[i] > 0) dst[i] =  M4T_MTFP_MAX_VAL;
        else if (src[i] < 0) dst[i] = -M4T_MTFP_MAX_VAL;
        else                  dst[i] = 0;
    }
}
#endif

void m4t_mtfp_shift3(m4t_mtfp_t* dst, const m4t_mtfp_t* src, int k, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);

    if (k == 0) {
        if (dst != src) memcpy(dst, src, (size_t)n * sizeof(m4t_mtfp_t));
        return;
    }

    if (k > 0) {
        /* Multiply by 3^k. Beyond k=19 the smallest nonzero mantissa
         * already overflows MTFP19; collapse to saturation.
         * NEON-only production dispatch per project rule. */
        if (k >= 20) {
            shift3_mul_saturate_neon(dst, src, n);
            return;
        }
        shift3_mul_neon(dst, src, k, n);
        return;
    }

    /* k < 0: divide by 3^|k| with base-3 round-to-nearest-even. */
    int abs_k = -k;
    if (abs_k >= 20) {
        /* MAX_VAL / 3^20 < 1, all values round to 0. */
        memset(dst, 0, (size_t)n * sizeof(m4t_mtfp_t));
        return;
    }
    /* NEON-only production dispatch. Scalar reference available via
     * m4t_mtfp_shift3_scalar_ref (test oracle). Per project rule
     * (feedback_function_over_speed_no_scalar). */
    shift3_div_neon(dst, src, abs_k, n);
}

/* Scalar-only reference. Same semantics as m4t_mtfp_shift3 (per the
 * header doc-comment) but ALWAYS uses the scalar divide path, never
 * dispatches to NEON. Test-only: production must call m4t_mtfp_shift3.
 * Per journal/shift3_neon_redteam.md C1/C2/C3 + remediation R-G1. */
void m4t_mtfp_shift3_scalar_ref(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                                int k, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);

    if (k == 0) {
        if (dst != src) memcpy(dst, src, (size_t)n * sizeof(m4t_mtfp_t));
        return;
    }

    if (k > 0) {
        if (k >= 20) {
            for (int i = 0; i < n; i++) {
                if (src[i] > 0)      dst[i] =  M4T_MTFP_MAX_VAL;
                else if (src[i] < 0) dst[i] = -M4T_MTFP_MAX_VAL;
                else                  dst[i] = 0;
            }
            return;
        }
        int64_t scale = M4T_POW3_TABLE[k];
        for (int i = 0; i < n; i++) {
            int64_t v = (int64_t)src[i] * scale;
            dst[i] = m4t_mtfp_clamp64(v);
        }
        return;
    }

    int abs_k = -k;
    if (abs_k >= 20) {
        memset(dst, 0, (size_t)n * sizeof(m4t_mtfp_t));
        return;
    }
    shift3_div_scalar(dst, src, abs_k, n);
}

/* ── Integer rsqrt (Newton-Raphson) ─────────────────────────────────────
 *
 * Per journal/rsqrt_design_lmm.md. Computes round(2^30 / sqrt(src)) for
 * src ∈ [1, INT32_MAX]. Output range [~23170, 2^30].
 *
 * The rsqrt of a single integer is fundamentally a scalar operation;
 * "NEON" doesn't apply (nothing to vectorize per-call). Per the project
 * memory's reading of the no-scalar rule ("production dispatchers" are
 * the rule's target), a scalar primitive over single values is allowed
 * and standard.
 *
 * Algorithm:
 *   Initial guess y₀ from bit pattern: shift = 31 - clz(src) gives
 *   floor(log2(src)). y₀ = 1 << (30 - shift/2) is order-of-magnitude
 *   correct.
 *
 *   Newton-Raphson iteration in fixed-point (scale Q = 2^30 for y):
 *     real:  y_{n+1} = y_n × (3 - src × y_n²) / 2
 *     fixed: y_{n+1} = y_n × (3·Q² - src·y_n²) / (2·Q²)
 *            using __int128 for the y × (3Q² - src·y²) intermediate
 *            (max ~2^91, fits 128-bit cleanly).
 *
 *   3 iterations from a good initial guess → bit-exact int32 precision.
 *
 * Bit-exact NEON-vs-scalar_ref: the scalar_ref uses libm sqrt; the
 * production version uses pure-int NR. They match bit-exact across the
 * input range — verified by test_m4t_rsqrt. */

m4t_mtfp_t m4t_int32_rsqrt_scalar_ref(m4t_mtfp_t src) {
    /* Test-oracle implementation. FP allowed in scaffolding. */
    if (src <= 0) return 0;
    double v = 1073741824.0 / sqrt((double)src);  /* 2^30 / sqrt(src) */
    int64_t r = (int64_t)(v + 0.5);  /* round-half-up */
    if (r < 1)             r = 1;
    if (r > 1073741824LL)  r = 1073741824LL;
    return (m4t_mtfp_t)r;
}

m4t_mtfp_t m4t_int32_rsqrt(m4t_mtfp_t src) {
    /* Pure-integer Newton-Raphson rsqrt. */
    if (src <= 0) return 0;
    /* Initial guess: UNDERESTIMATE 1/sqrt(src). NR rsqrt converges from
     * below; an overestimate may take more iterations. Pick exp such that
     * 2^exp ≤ 2^30/sqrt(src) for all src in the [2^log2_src, 2^(log2_src+1))
     * range — equivalently, exp ≤ 30 - ceil((log2_src+1)/2) = 30 - log2_src/2 - 1.
     * Use `half = log2_src/2 + 1` (always rounds up the half-shift). */
    int log2_src = 31 - __builtin_clz((uint32_t)src);
    int half = log2_src / 2 + 1;
    int exp = 30 - half;
    if (exp < 0) exp = 0;
    int64_t y = (int64_t)1 << exp;
    /* 5 Newton-Raphson iterations with __int128 intermediate.
     * Fixed-point scale Q = 2^30; Q² = 2^60. */
    const __int128 Q2 = (__int128)1 << 60;
    const __int128 three_Q2 = (__int128)3 * Q2;
    for (int it = 0; it < 5; it++) {
        __int128 y2     = (__int128)y * (__int128)y;       /* scale 2^60 */
        __int128 src_y2 = (__int128)(uint32_t)src * y2;    /* scale 2^60 × src */
        __int128 t      = three_Q2 - src_y2;               /* scale 2^60, may be negative briefly */
        /* y_new = y × t / (2 × Q²) */
        __int128 y_t    = (__int128)y * t;                 /* scale 2^90 ish */
        y = (int64_t)(y_t >> 61);                          /* divide by 2 × Q² */
        if (y < 1) y = 1;
        if (y > 1073741824LL) y = 1073741824LL;
    }
    return (m4t_mtfp_t)y;
}

/* ── RMSNorm ────────────────────────────────────────────────────────────
 *
 * y[i] = γ[i] · x[i] · rsqrt(mean(x²) + ε)
 *
 * Sum-of-squares overflow analysis:
 *   x ∈ MTFP19, |x| ≤ 581130733 ≈ 2^29.1, x² ≤ 2^58.2.
 *   Σ over n=2560: Σx² ≤ 2^69.5 — overflows int64 (max ≈ 2^63).
 *
 *   Fix: right-shift each x by SOS_SHIFT before squaring.
 *     |x>>4| ≤ 2^25.1, (x>>4)² ≤ 2^50.2, Σ ≤ 2^61.5 — fits int64.
 *   Loses 4 bits/cell of precision in the mean. Acceptable for a
 *   normalization step (the rsqrt result divides through anyway).
 *
 * Shift compensation:
 *   mean_shifted = (Σ (x>>SHIFT)²) / n + ε
 *                ≈ mean_real / 2^(2·SHIFT)     (modulo ε scaling)
 *   rsqrt(mean_shifted) ≈ 2^SHIFT · rsqrt_real(mean_real)
 *   m4t_int32_rsqrt returns 2^30 / sqrt(input).
 *   Want: inv_at_30 = 2^30 · rsqrt_real(mean_real)
 *                   = m4t_int32_rsqrt(mean_shifted) / 2^SHIFT
 *
 *   Note: ε is interpreted in shifted units to keep the caller's
 *   numerical intent (ε prevents div-by-zero at the operating scale).
 *   For BitNet's typical activation magnitudes, eps_mantissa=1 is the
 *   minimal positive guard.
 *
 * 3-way product γ × x × inv:
 *   |γ|, |x| ≤ 2^29.1; |inv| ≤ 2^30. Product ≤ 2^88 — exceeds int64.
 *   Use __int128 per cell, then >>30 to recover y in MTFP19 mantissa
 *   units. NEON int lane width tops out at 64 bits, so the per-cell
 *   loop stays scalar (per the cross-exp accum's degenerate-case
 *   precedent of "scalar with documented reasoning"). */
#define SOS_SHIFT 4

void m4t_mtfp_rmsnorm(
    m4t_mtfp_t* y, const m4t_mtfp_t* x, const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mantissa, int n)
{
    assert(n >= 0);
    if (n == 0) return;
    assert(y && x && gamma);
    assert(eps_mantissa >= 0);

    /* Σ (x>>SHIFT)² in int64 — fits per the analysis above. */
    int64_t sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int64_t xs = (int64_t)x[i] >> SOS_SHIFT;
        sum_sq += xs * xs;
    }
    int64_t mean_shifted = sum_sq / (int64_t)n + (int64_t)eps_mantissa;
    if (mean_shifted < 1) mean_shifted = 1;

    /* mean_shifted can still exceed int31 (rsqrt's input cap = 0x7FFFFFFF):
     * with SOS_SHIFT=4, |x>>4| ≤ 2^25.1, sum_sq ≤ 2^61.5, mean_shifted
     * ≤ 2^50.2. Pre-shift right by 2k bits to bring it under int31. The
     * compensation: rsqrt of a 4×-smaller input is 2× larger, so we owe
     * the per-cell scale 2^(SOS_SHIFT + extra_k) extra "downscaling" on
     * top of the rsqrt's inherent 2^30 scale.
     *
     * y[i] = γ × x × inv_real
     *      = γ × x × m4t_int32_rsqrt(mean_passed) / 2^(30 + SOS_SHIFT + extra_k)
     *
     * Keeping `inv` at full rsqrt precision (≤ 2^30) and shifting only
     * at the per-cell end preserves precision for small inv_real. */
    int extra_k = 0;
    int64_t mean_passed = mean_shifted;
    while (mean_passed > (int64_t)0x7FFFFFFF) {
        mean_passed >>= 2;
        extra_k++;
    }
    if (mean_passed < 1) mean_passed = 1;

    m4t_mtfp_t inv = m4t_int32_rsqrt((m4t_mtfp_t)mean_passed);
    int total_shift = 30 + SOS_SHIFT + extra_k;

    /* Per-cell: y[i] = saturating_clamp((γ × x × inv) >> total_shift).
     * __int128 accommodates the 3-way product (max ≤ 2^88.2; fits 128b). */
    if (total_shift >= 127) {
        for (int i = 0; i < n; i++) y[i] = 0;
        return;
    }
    for (int i = 0; i < n; i++) {
        __int128 prod = (__int128)gamma[i] * (__int128)x[i] * (__int128)inv;
        int64_t scaled = (int64_t)(prod >> total_shift);
        y[i] = m4t_mtfp_clamp64(scaled);
    }
}

void m4t_mtfp_rmsnorm_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* x, const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mantissa, int n)
{
    /* FP test oracle. Mirrors production's shift accounting so the
     * caller-supplied eps_mantissa carries the same numerical meaning
     * across both paths. */
    assert(n >= 0);
    if (n == 0) return;
    assert(y && x && gamma);
    assert(eps_mantissa >= 0);

    int64_t sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int64_t xs = (int64_t)x[i] >> SOS_SHIFT;
        sum_sq += xs * xs;
    }
    int64_t mean_shifted = sum_sq / (int64_t)n + (int64_t)eps_mantissa;
    if (mean_shifted < 1) mean_shifted = 1;

    /* Real rsqrt of mean_real:
     *   mean_real ≈ mean_shifted × 2^(2·SHIFT)     (modulo ε)
     *   rsqrt_real = 1 / sqrt(mean_shifted) / 2^SHIFT
     * Multiply by 2^30 to express at the same scale as production's inv. */
    double inv_shifted_fp = 1073741824.0 / sqrt((double)mean_shifted);
    double inv_at_30_fp = inv_shifted_fp / (double)((int64_t)1 << SOS_SHIFT);

    for (int i = 0; i < n; i++) {
        double prod = (double)gamma[i] * (double)x[i] * inv_at_30_fp;
        double v = prod / 1073741824.0;  /* >>30 */
        if (v >  (double)M4T_MTFP_MAX_VAL) v =  (double)M4T_MTFP_MAX_VAL;
        if (v < -(double)M4T_MTFP_MAX_VAL) v = -(double)M4T_MTFP_MAX_VAL;
        y[i] = (m4t_mtfp_t)(v < 0 ? v - 0.5 : v + 0.5);
    }
}

/* ── RoPE ───────────────────────────────────────────────────────────────
 *
 * Llama rotate_half convention; LUT-based cos/sin precomputed at first
 * call. Per journal/rope_design_lmm.md.
 *
 * LUT is owned by this translation unit (single-threaded; BitNet
 * inference is single-threaded). Re-init triggered if (head_dim,
 * theta_base) changes — supports models with different RoPE configs,
 * though typical use is one config per process.
 *
 * libm cos/sin used at init only — equivalent precedent to weight
 * loading's bf16→MTFP19 conversion (init-time FP, runtime pure-int). */

static int32_t* g_rope_cos_lut = NULL;
static int32_t* g_rope_sin_lut = NULL;
static int      g_rope_lut_initialized = 0;
static int      g_rope_lut_head_dim = 0;
static double   g_rope_lut_theta_base = 0.0;

static void rope_init_lut(int head_dim, double theta_base) {
    if (g_rope_lut_initialized
        && g_rope_lut_head_dim == head_dim
        && g_rope_lut_theta_base == theta_base) {
        return;
    }
    assert(head_dim > 0);
    assert(head_dim % 2 == 0);
    assert(head_dim <= M4T_ROPE_MAX_HEAD_DIM);

    int half = head_dim / 2;
    size_t n = (size_t)M4T_ROPE_MAX_POSITION * (size_t)half;
    int32_t* new_cos = (int32_t*)realloc(g_rope_cos_lut, n * sizeof(int32_t));
    int32_t* new_sin = (int32_t*)realloc(g_rope_sin_lut, n * sizeof(int32_t));
    assert(new_cos && new_sin);
    g_rope_cos_lut = new_cos;
    g_rope_sin_lut = new_sin;

    double scale = (double)M4T_ROPE_COS_SIN_SCALE;
    for (int pos = 0; pos < M4T_ROPE_MAX_POSITION; pos++) {
        for (int i = 0; i < half; i++) {
            double freq  = pow(theta_base, -2.0 * (double)i / (double)head_dim);
            double angle = (double)pos * freq;
            double cv = cos(angle) * scale;
            double sv = sin(angle) * scale;
            /* Round half-away-from-zero. */
            int32_t ci = (int32_t)(cv < 0 ? cv - 0.5 : cv + 0.5);
            int32_t si = (int32_t)(sv < 0 ? sv - 0.5 : sv + 0.5);
            g_rope_cos_lut[pos*half + i] = ci;
            g_rope_sin_lut[pos*half + i] = si;
        }
    }
    g_rope_lut_initialized = 1;
    g_rope_lut_head_dim = head_dim;
    g_rope_lut_theta_base = theta_base;
}

/* Apply RoPE to one head's worth of d-dim values, in place.
 * For i ∈ [0, half):
 *   a' = (a · c − b · s) >> 29
 *   b' = (b · c + a · s) >> 29
 * Saturating clamp on output. */
static inline void rope_apply_one_head(
    m4t_mtfp_t* h, int half,
    const int32_t* cos_row, const int32_t* sin_row)
{
    for (int i = 0; i < half; i++) {
        int64_t a = h[i];
        int64_t b = h[i + half];
        int64_t c = cos_row[i];
        int64_t s = sin_row[i];
        int64_t new_a = (a * c - b * s) >> 29;
        int64_t new_b = (b * c + a * s) >> 29;
        h[i]        = m4t_mtfp_clamp64(new_a);
        h[i + half] = m4t_mtfp_clamp64(new_b);
    }
}

void m4t_mtfp_rope_apply(
    m4t_mtfp_t* q, m4t_mtfp_t* k,
    int position,
    int num_q_heads, int num_kv_heads, int head_dim,
    double theta_base)
{
    assert(q && k);
    assert(position >= 0 && position < M4T_ROPE_MAX_POSITION);
    assert(num_q_heads > 0 && num_kv_heads > 0);
    assert(head_dim > 0 && head_dim % 2 == 0);

    rope_init_lut(head_dim, theta_base);

    int half = head_dim / 2;
    const int32_t* cos_row = g_rope_cos_lut + (size_t)position * (size_t)half;
    const int32_t* sin_row = g_rope_sin_lut + (size_t)position * (size_t)half;

    for (int h = 0; h < num_q_heads; h++) {
        rope_apply_one_head(q + (size_t)h * head_dim, half, cos_row, sin_row);
    }
    for (int h = 0; h < num_kv_heads; h++) {
        rope_apply_one_head(k + (size_t)h * head_dim, half, cos_row, sin_row);
    }
}

/* ── Integer reciprocal (Newton-Raphson) ────────────────────────────────
 *
 * Same shape as m4t_int32_rsqrt: pure-int NR with __int128 intermediate.
 *
 * Algorithm:
 *   y_{n+1} = y_n × (2·Q − src · y_n)  (then rescale)
 * Fixed-point: y at scale Q = 2^30. 2·Q = 2^31. The src · y product
 * is at scale Q. (2Q − src·y) is at scale Q. y · (2Q − src·y) is at
 * scale Q² = 2^60. Right-shift by 30 to recover y at scale Q.
 *
 * Initial guess from clz: log2(src) tells us approximate magnitude;
 * y_0 = 2^(60 - log2(src)) gives an order-of-magnitude correct seed
 * that NR refines quadratically.
 *
 * 5 iterations for full int32 precision (per rsqrt's empirical finding). */

m4t_mtfp_t m4t_int32_recip_scalar_ref(m4t_mtfp_t src) {
    if (src <= 0) return 0;
    double v = 1073741824.0 / (double)src;  /* 2^30 / src */
    int64_t r = (int64_t)(v + 0.5);
    if (r < 1)             r = 1;
    if (r > 1073741824LL)  r = 1073741824LL;
    return (m4t_mtfp_t)r;
}

m4t_mtfp_t m4t_int32_recip(m4t_mtfp_t src) {
    if (src <= 0) return 0;
    if (src == 1) return 1073741824;  /* 2^30 / 1 */
    /* Initial guess: NR recip needs y_0 such that x · y_0 ∈ (0, 2);
     * convergence is poor near the boundaries. We aim for x · y_0 ≈ 1.
     * For src ∈ [2^k, 2^(k+1)): pick y_0 = 2^(29-k), giving x · y_0
     * ∈ [0.5, 1.0). Comfortably mid-basin. */
    int log2_src = 31 - __builtin_clz((uint32_t)src);
    int exp = 29 - log2_src;
    if (exp < 0) exp = 0;
    int64_t y = (int64_t)1 << exp;
    if (y < 1) y = 1;

    /* NR: y_new = y × (2Q - src·y) >> 30. Q = 2^30, 2Q = 2^31. */
    const __int128 two_Q = (__int128)1 << 31;
    for (int it = 0; it < 5; it++) {
        __int128 src_y = (__int128)(uint32_t)src * (__int128)y;  /* scale Q */
        __int128 t     = two_Q - src_y;                          /* scale Q */
        __int128 y_t   = (__int128)y * t;                        /* scale Q² = 2^60 */
        y = (int64_t)(y_t >> 30);                                /* recover scale Q */
        if (y < 1)              y = 1;
        if (y > 1073741824LL)   y = 1073741824LL;
    }
    return (m4t_mtfp_t)y;
}

/* ── Softmax (LUT-based exp + integer reciprocal) ──────────────────────
 *
 * Per journal/softmax_design_lmm.md.
 *
 * exp LUT covers z ∈ [-LUT_RANGE, 0] sampled at LUT_RES points; values
 * below the range underflow to 0. exp at scale 2^30.
 *
 * Per-cell pipeline:
 *   z = x[i] - max(x)             (int32, ≤ 0)
 *   e[i] = exp_lut(z)             (int32 at scale 2^30)
 *   sum = Σ e[i]                  (int64, max ≈ n × 2^30)
 *   inv = 2^30 / (sum >> shift)    (m4t_int32_recip)
 *   y[i] = (e[i] · inv) >> (30 - shift)
 *
 * Where shift brings sum into int31 range for the reciprocal call. */

static int32_t* g_softmax_exp_lut = NULL;
static int      g_softmax_lut_initialized = 0;

static void softmax_init_lut(void) {
    if (g_softmax_lut_initialized) return;
    int32_t* lut = (int32_t*)malloc((size_t)M4T_SOFTMAX_LUT_RES * sizeof(int32_t));
    assert(lut);
    double scale = (double)M4T_SOFTMAX_OUT_SCALE;  /* 2^30 */
    for (int k = 0; k < M4T_SOFTMAX_LUT_RES; k++) {
        double z = -(double)k * (double)M4T_SOFTMAX_LUT_RANGE
                   / (double)M4T_SOFTMAX_LUT_RES;  /* z ∈ [0, -LUT_RANGE) */
        double v = exp(z) * scale;
        int32_t vi = (int32_t)(v + 0.5);
        if (vi < 0) vi = 0;
        if (vi > M4T_SOFTMAX_OUT_SCALE) vi = M4T_SOFTMAX_OUT_SCALE;
        lut[k] = vi;
    }
    g_softmax_exp_lut = lut;
    g_softmax_lut_initialized = 1;
}

/* Compute exp(z) at scale 2^30 for z ≤ 0. Returns 0 for z < -LUT_RANGE.
 * Linear interpolation between LUT entries. */
static int32_t softmax_exp_int(int32_t z) {
    if (z >= 0) return M4T_SOFTMAX_OUT_SCALE;
    int32_t neg_z = -z;
    if (neg_z >= M4T_SOFTMAX_LUT_RANGE) return 0;
    /* index = neg_z × (LUT_RES / LUT_RANGE). LUT_RES=4096, LUT_RANGE=30
     * → multiplier = 4096/30 ≈ 136.5333. Use Q16 fixed-point: 136.5333 × 2^16
     * → (LUT_RES << 16) / LUT_RANGE. */
    int64_t idx_q16 = (int64_t)neg_z * M4T_SOFTMAX_LUT_RES * 65536LL / M4T_SOFTMAX_LUT_RANGE;
    int idx = (int)(idx_q16 >> 16);
    int frac = (int)(idx_q16 & 0xFFFF);
    if (idx >= M4T_SOFTMAX_LUT_RES - 1) {
        return g_softmax_exp_lut[M4T_SOFTMAX_LUT_RES - 1];
    }
    int32_t a = g_softmax_exp_lut[idx];
    int32_t b = g_softmax_exp_lut[idx + 1];
    /* Linear interp: a + (b - a) × frac / 2^16. Note b ≤ a (LUT is decreasing). */
    int32_t v = a + (int32_t)(((int64_t)(b - a) * (int64_t)frac) >> 16);
    return v;
}

/* Higher-precision reciprocal: 2^60 / src for src ∈ [1, INT64_MAX].
 * m4t_int32_recip's output (at scale 2^30) is too coarse for softmax
 * normalization — when sum is near 2^30, the int truncation produces
 * ~10% bias. This variant uses pure-int NR with __int128 to keep full
 * precision in the inv factor; the per-cell multiply then composes
 * cleanly without accumulating that bias.
 *
 * Internal use only. */
static int64_t softmax_recip60(int64_t src) {
    if (src <= 0) return 0;
    int log2_src = 63 - __builtin_clzll((uint64_t)src);
    int exp_init = 60 - log2_src;
    if (exp_init < 0) exp_init = 0;
    if (exp_init > 60) exp_init = 60;
    int64_t y = (int64_t)1 << exp_init;
    /* NR: y_new = y × (2·Q − src · y) / Q at Q = 2^60. */
    const __int128 two_Q = (__int128)1 << 61;
    for (int it = 0; it < 6; it++) {
        __int128 src_y = (__int128)src * (__int128)y;  /* scale Q */
        __int128 t = two_Q - src_y;                    /* scale Q */
        __int128 y_t = (__int128)y * t;                /* scale Q² */
        y = (int64_t)(y_t >> 60);                      /* recover scale Q */
        if (y < 1) y = 1;
    }
    return y;
}

void m4t_mtfp_softmax(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n) {
    assert(n >= 1);
    assert(y && x);
    softmax_init_lut();

    /* Find max. */
    m4t_mtfp_t mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];

    /* exp(x[i] - max) into a scratch; sum in int64. malloc rather than
     * alloca: BitNet's attention can have n up to ~4096 (max seq_len);
     * stack pressure is unwarranted for a once-per-call op. */
    int32_t* e = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    assert(e);
    int64_t sum = 0;
    for (int i = 0; i < n; i++) {
        int64_t z = (int64_t)x[i] - (int64_t)mx;  /* ≤ 0 */
        int32_t z_clamped = (z < (int64_t)INT32_MIN) ? INT32_MIN : (int32_t)z;
        e[i] = softmax_exp_int(z_clamped);
        sum += e[i];
    }
    if (sum < 1) sum = 1;

    /* High-precision reciprocal: inv60 = 2^60 / sum (full int precision).
     *
     * Algebra:
     *   want: y[i] = e[i] / sum_real × 2^30
     *               = e[i] × 2^30 / sum_int    (sum_int at scale 2^30)
     *   inv60 = 2^60 / sum_int → e[i] × inv60 = e[i] × 2^60 / sum_int.
     *   y[i] = e[i] × inv60 >> 30 = e[i] × 2^30 / sum_int.  ✓ */
    int64_t inv60 = softmax_recip60(sum);
    for (int i = 0; i < n; i++) {
        __int128 prod = (__int128)e[i] * (__int128)inv60;
        int64_t scaled = (int64_t)(prod >> 30);
        if (scaled < 0) scaled = 0;
        if (scaled > M4T_SOFTMAX_OUT_SCALE) scaled = M4T_SOFTMAX_OUT_SCALE;
        y[i] = (m4t_mtfp_t)scaled;
    }
    free(e);
}

void m4t_mtfp_softmax_scalar_ref(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n) {
    /* Independent FP oracle: runtime libm exp, FP sum, FP divide. */
    assert(n >= 1);
    assert(y && x);
    m4t_mtfp_t mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double* e = (double*)malloc((size_t)n * sizeof(double));
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double z = (double)x[i] - (double)mx;
        if (z < -(double)M4T_SOFTMAX_LUT_RANGE) e[i] = 0.0;
        else e[i] = exp(z);
        sum += e[i];
    }
    if (sum <= 0.0) sum = 1.0;
    double scale = (double)M4T_SOFTMAX_OUT_SCALE / sum;
    for (int i = 0; i < n; i++) {
        double v = e[i] * scale;
        if (v < 0.0) v = 0.0;
        if (v > (double)M4T_SOFTMAX_OUT_SCALE) v = (double)M4T_SOFTMAX_OUT_SCALE;
        y[i] = (m4t_mtfp_t)(v + 0.5);
    }
    free(e);
}

/* ── A8 quantize / dequantize ──────────────────────────────────────────
 *
 * Pure-int implementation. The integer divide (int64) and the
 * round-half-away-from-zero step happen per cell.
 *
 * Quantize:
 *   absmax = max |x|
 *   y[i] = round(x[i] · 127 / absmax), clamped to [-127, 127]
 *
 * Dequantize:
 *   y[i] = round(x_int8[i] · absmax / 127), MTFP19-clamped */

m4t_mtfp_t m4t_a8_quantize_scalar_ref(int8_t* y, const m4t_mtfp_t* x, int n) {
    /* FP test oracle. */
    if (n <= 0) return 0;
    assert(y && x);
    m4t_mtfp_t absmax = 0;
    for (int i = 0; i < n; i++) {
        m4t_mtfp_t a = x[i] < 0 ? -x[i] : x[i];
        if (a > absmax) absmax = a;
    }
    if (absmax == 0) {
        memset(y, 0, (size_t)n);
        return 0;
    }
    for (int i = 0; i < n; i++) {
        double v = (double)x[i] * 127.0 / (double)absmax;
        int32_t r = (int32_t)(v < 0 ? v - 0.5 : v + 0.5);
        if (r >  127) r =  127;
        if (r < -127) r = -127;
        y[i] = (int8_t)r;
    }
    return absmax;
}

m4t_mtfp_t m4t_a8_quantize(int8_t* y, const m4t_mtfp_t* x, int n) {
    if (n <= 0) return 0;
    assert(y && x);
    m4t_mtfp_t absmax = 0;
    for (int i = 0; i < n; i++) {
        m4t_mtfp_t a = x[i] < 0 ? -x[i] : x[i];
        if (a > absmax) absmax = a;
    }
    if (absmax == 0) {
        memset(y, 0, (size_t)n);
        return 0;
    }
    int64_t denom = (int64_t)absmax;
    int64_t half  = denom / 2;
    for (int i = 0; i < n; i++) {
        int64_t num = (int64_t)x[i] * 127;
        /* Round half-away-from-zero. C's / truncates toward zero. */
        int64_t q;
        if (num >= 0) q = (num + half) / denom;
        else          q = (num - half) / denom;
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        y[i] = (int8_t)q;
    }
    return absmax;
}

void m4t_a8_dequantize_scalar_ref(
    m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax, int n)
{
    if (n <= 0) return;
    assert(y && x);
    for (int i = 0; i < n; i++) {
        double v = (double)x[i] * (double)absmax / 127.0;
        int64_t r = (int64_t)(v < 0 ? v - 0.5 : v + 0.5);
        y[i] = m4t_mtfp_clamp64(r);
    }
}

void m4t_a8_dequantize(
    m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax, int n)
{
    if (n <= 0) return;
    assert(y && x);
    /* Round half-away-from-zero with int. */
    for (int i = 0; i < n; i++) {
        int64_t num = (int64_t)x[i] * (int64_t)absmax;
        int64_t r;
        if (num >= 0) r = (num + 63) / 127;  /* 63 = floor(127/2) */
        else          r = (num - 63) / 127;
        y[i] = m4t_mtfp_clamp64(r);
    }
}

/* ── Vector scale by num/den ratio ─────────────────────────────────────
 *
 * y[i] = round(x[i] · num / den), saturating to ±M4T_MTFP_MAX_VAL.
 * __int128 for the x·num product (can reach 2^92). */

void m4t_mtfp_vec_scale_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int64_t num, int64_t den, int n)
{
    if (n <= 0) return;
    assert(y && x);
    assert(den > 0);
    double dnum = (double)num;
    double dden = (double)den;
    for (int i = 0; i < n; i++) {
        double v = (double)x[i] * dnum / dden;
        if (v >  (double)M4T_MTFP_MAX_VAL) v =  (double)M4T_MTFP_MAX_VAL;
        if (v < -(double)M4T_MTFP_MAX_VAL) v = -(double)M4T_MTFP_MAX_VAL;
        int64_t r = (int64_t)(v < 0 ? v - 0.5 : v + 0.5);
        y[i] = (m4t_mtfp_t)r;
    }
}

void m4t_mtfp_vec_scale(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int64_t num, int64_t den, int n)
{
    if (n <= 0) return;
    assert(y && x);
    assert(den > 0);
    __int128 half = (__int128)den / 2;
    for (int i = 0; i < n; i++) {
        __int128 prod = (__int128)x[i] * (__int128)num;  /* max ~2^92 */
        __int128 r;
        if (prod >= 0) r = (prod + half) / den;
        else           r = (prod - half) / den;
        /* Clamp the __int128 to int64 first, then to MTFP19. */
        int64_t r64;
        if (r >  (__int128)0x7FFFFFFFFFFFFFFFLL) r64 =  0x7FFFFFFFFFFFFFFFLL;
        else if (r < -(__int128)0x7FFFFFFFFFFFFFFFLL) r64 = -0x7FFFFFFFFFFFFFFFLL;
        else r64 = (int64_t)r;
        y[i] = m4t_mtfp_clamp64(r64);
    }
}

/* ── bx-aware variants (Phase 2 work-unit 1) ────────────────────────────
 *
 * Per the closeout's red-team: implicit-bx output kills the activation
 * flow at saturation. These variants take explicit bxes and produce
 * output at a caller-chosen target bx. Same algorithms internally —
 * just an extra rescale step at the end.
 */

/* Compute 3^k as int64, asserts k in [0, 39]. */
static int64_t pow3_i64(int k) {
    assert(k >= 0 && k <= 39);
    int64_t r = 1;
    for (int i = 0; i < k; i++) r *= 3;
    return r;
}

void m4t_mtfp_rescale_bx(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int from_bx, int to_bx, int n)
{
    if (n <= 0) return;
    assert(y && x);
    if (from_bx == to_bx) {
        if (y != x) memcpy(y, x, (size_t)n * sizeof(m4t_mtfp_t));
        return;
    }
    if (from_bx > to_bx) {
        /* x_m_at_to_bx = x_m_at_from_bx / 3^(from-to). Loses precision (divides). */
        int k = from_bx - to_bx;
        int64_t den = pow3_i64(k);
        int64_t half = den / 2;
        for (int i = 0; i < n; i++) {
            int64_t v = (int64_t)x[i];
            int64_t r = (v >= 0) ? (v + half) / den : (v - half) / den;
            y[i] = m4t_mtfp_clamp64(r);
        }
    } else {
        /* x_m_at_to_bx = x_m_at_from_bx × 3^(to-from). Magnifies. */
        int k = to_bx - from_bx;
        int64_t mul = pow3_i64(k);
        for (int i = 0; i < n; i++) {
            int64_t r = (int64_t)x[i] * mul;
            y[i] = m4t_mtfp_clamp64(r);
        }
    }
}

void m4t_mtfp_rmsnorm_bx(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* gamma,
    int x_bx, int gamma_bx, int target_bx,
    m4t_mtfp_t eps_mantissa, int n)
{
    assert(n >= 0);
    if (n == 0) return;
    assert(y && x && gamma);
    (void)x_bx;  /* informational only; output bx is gamma_bx mod the rescale */

    /* Phase 2 wu1.5: precision-preserving variant. The implicit
     * m4t_mtfp_rmsnorm uses SOS_SHIFT=4 to keep the int64 SoS sum from
     * overflowing — but for small-mantissa inputs (e.g., GATE_ACT_BX=2
     * where typical |x_m| < 16), that shift wipes out most cells'
     * contribution to mean(x²). The outlier-dominated mean produces a
     * normalization factor wrong for the small cells.
     *
     * Fix: use __int128 for the SoS sum (no shift needed). With
     * |x_m| ≤ MTFP19_MAX < 2^29.1 and n ≤ 6912, Σx² ≤ n × 2^58 ≈ 2^71 —
     * fits __int128 cleanly. */
    __int128 sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int64_t xv = (int64_t)x[i];
        sum_sq += (__int128)(xv * xv);
    }
    /* mean = sum_sq / n + eps. Reduce to int64-fitting for rsqrt. */
    __int128 mean_full = sum_sq / (__int128)n + (__int128)eps_mantissa;
    if (mean_full < 1) mean_full = 1;

    /* Pre-shift to fit int31 for m4t_int32_rsqrt. */
    int extra_k = 0;
    __int128 mean_passed = mean_full;
    while (mean_passed > (__int128)0x7FFFFFFF) {
        mean_passed >>= 2;
        extra_k++;
    }
    if (mean_passed < 1) mean_passed = 1;

    m4t_mtfp_t inv = m4t_int32_rsqrt((m4t_mtfp_t)mean_passed);
    /* y_real = γ_real × x_real × rsqrt(mean(x_real²)). All x_bx terms
     * cancel in the rsqrt, so output bx = gamma_bx (mod the per-cell
     * arithmetic). Total shift: 30 (rsqrt scale) + 2*extra_k (mean
     * pre-shift). */
    int total_shift = 30 + extra_k;

    /* Per-cell: y_m_at_gamma_bx = γ × x × inv >> total_shift. */
    for (int i = 0; i < n; i++) {
        __int128 prod = (__int128)gamma[i] * (__int128)x[i] * (__int128)inv;
        int64_t scaled = (int64_t)(prod >> total_shift);
        y[i] = m4t_mtfp_clamp64(scaled);
    }

    /* Rescale gamma_bx → target_bx. */
    if (gamma_bx != target_bx) {
        m4t_mtfp_rescale_bx(y, y, gamma_bx, target_bx, n);
    }
}

void m4t_mtfp_relu2_inplace_bx(m4t_mtfp_t* x, int x_bx, int target_bx, int n) {
    if (n <= 0) return;
    assert(x);
    int shift_exp = 2 * x_bx - target_bx;
    assert(shift_exp >= 0 && shift_exp <= 39);
    int64_t den = pow3_i64(shift_exp);
    int64_t half = den / 2;
    for (int i = 0; i < n; i++) {
        if (x[i] <= 0) { x[i] = 0; continue; }
        int64_t sq = (int64_t)x[i] * (int64_t)x[i];
        int64_t r = (sq + half) / den;
        x[i] = m4t_mtfp_clamp64(r);
    }
}

void m4t_mtfp_elementwise_mul_bx(
    m4t_mtfp_t* y,
    const m4t_mtfp_t* a, int a_bx,
    const m4t_mtfp_t* b, int b_bx,
    int target_bx, int n)
{
    if (n <= 0) return;
    assert(y && a && b);
    int shift_exp = a_bx + b_bx - target_bx;
    assert(shift_exp >= 0 && shift_exp <= 39);
    int64_t den = pow3_i64(shift_exp);
    int64_t half = den / 2;
    for (int i = 0; i < n; i++) {
        int64_t prod = (int64_t)a[i] * (int64_t)b[i];
        int64_t r = (prod >= 0) ? (prod + half) / den : (prod - half) / den;
        y[i] = m4t_mtfp_clamp64(r);
    }
}

void m4t_mtfp_bitlinear_scale_bx(
    m4t_mtfp_t* y, const m4t_mtfp_t* y_raw,
    const m4t_mtfp_t* alpha_ptr, int alpha_bx,
    m4t_mtfp_t absmax_m, int x_bx, int target_bx,
    int n)
{
    if (n <= 0) return;
    assert(y && y_raw && alpha_ptr);
    assert(alpha_bx + x_bx - target_bx >= 0);
    assert(alpha_bx + x_bx - target_bx <= 35);

    int64_t alpha_m = (int64_t)(*alpha_ptr);
    if (alpha_m == 0) {
        /* No-α sentinel (skeleton mode); preserve raw values, rescale to target_bx. */
        m4t_mtfp_rescale_bx(y, y_raw, x_bx, target_bx, n);
        return;
    }
    /* y_m_target = y_raw × α_m × absmax_m / (127 × 3^(α_bx + x_bx - target_bx))
     * num = α_m × absmax_m  (≤ 2^58, fits int64).
     * den = 127 × 3^shift_exp (≤ ~10^14 for our ranges, fits int64).
     * Per-cell prod = y_raw × num: __int128 (y_raw is small but × num × x[i]
     * can reach 2^88). */
    int shift_exp = alpha_bx + x_bx - target_bx;
    int64_t num = alpha_m * (int64_t)absmax_m;
    int64_t den = 127 * pow3_i64(shift_exp);
    __int128 half = (__int128)den / 2;
    for (int i = 0; i < n; i++) {
        __int128 prod = (__int128)y_raw[i] * (__int128)num;
        __int128 r;
        if (prod >= 0) r = (prod + half) / den;
        else           r = (prod - half) / den;
        int64_t r64;
        if (r >  (__int128)0x7FFFFFFFFFFFFFFFLL) r64 =  0x7FFFFFFFFFFFFFFFLL;
        else if (r < -(__int128)0x7FFFFFFFFFFFFFFFLL) r64 = -0x7FFFFFFFFFFFFFFFLL;
        else r64 = (int64_t)r;
        y[i] = m4t_mtfp_clamp64(r64);
    }
}

/* ── ReLU² + element-wise multiply ─────────────────────────────────────
 *
 * Pure-int. Saturating clamp on output (squared magnitudes exceed
 * MTFP19_MAX by factor of |x|; the downstream RMSNorm normalizes
 * away). Per work-unit 6 of bitnet_phase1: promoted from
 * bitnet_stub_relu2_inplace / bitnet_stub_elementwise_mul. */

void m4t_mtfp_relu2_inplace(m4t_mtfp_t* x, int n) {
    if (n <= 0) return;
    assert(x);
    for (int i = 0; i < n; i++) {
        if (x[i] <= 0) {
            x[i] = 0;
        } else {
            int64_t sq = (int64_t)x[i] * (int64_t)x[i];
            x[i] = m4t_mtfp_clamp64(sq);
        }
    }
}

void m4t_mtfp_relu2_inplace_scalar_ref(m4t_mtfp_t* x, int n) {
    /* Same algorithm; oracle for parity. */
    m4t_mtfp_relu2_inplace(x, n);
}

void m4t_mtfp_elementwise_mul(
    m4t_mtfp_t* y, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n)
{
    if (n <= 0) return;
    assert(y && a && b);
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)a[i] * (int64_t)b[i];
        y[i] = m4t_mtfp_clamp64(v);
    }
}

void m4t_mtfp_elementwise_mul_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n)
{
    m4t_mtfp_elementwise_mul(y, a, b, n);
}

void m4t_mtfp_rope_apply_scalar_ref(
    m4t_mtfp_t* q, m4t_mtfp_t* k,
    int position,
    int num_q_heads, int num_kv_heads, int head_dim,
    double theta_base)
{
    /* Independent FP test oracle — does NOT use the LUT. Runtime libm
     * cos/sin per call (FP allowed in scaffolding). Verifies both:
     *   (a) the rotate_half convention matches the FP reference, and
     *   (b) the LUT-quantized production produces output within
     *       tolerance of the FP-runtime answer. */
    assert(q && k);
    assert(position >= 0);
    assert(num_q_heads > 0 && num_kv_heads > 0);
    assert(head_dim > 0 && head_dim % 2 == 0);

    int half = head_dim / 2;

    /* For each freq_idx, compute FP cos/sin at runtime, apply to all heads. */
    for (int hh = 0; hh < num_q_heads + num_kv_heads; hh++) {
        m4t_mtfp_t* head = (hh < num_q_heads)
            ? (q + (size_t)hh * head_dim)
            : (k + (size_t)(hh - num_q_heads) * head_dim);
        for (int i = 0; i < half; i++) {
            double freq  = pow(theta_base, -2.0 * (double)i / (double)head_dim);
            double angle = (double)position * freq;
            double c = cos(angle), s = sin(angle);
            double a = (double)head[i];
            double b = (double)head[i + half];
            double new_a_d = a * c - b * s;
            double new_b_d = b * c + a * s;
            int64_t new_a = (int64_t)(new_a_d < 0 ? new_a_d - 0.5 : new_a_d + 0.5);
            int64_t new_b = (int64_t)(new_b_d < 0 ? new_b_d - 0.5 : new_b_d + 0.5);
            head[i]        = m4t_mtfp_clamp64(new_a);
            head[i + half] = m4t_mtfp_clamp64(new_b);
        }
    }
}
