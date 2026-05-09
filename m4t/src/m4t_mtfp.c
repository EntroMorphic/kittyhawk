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

/* V14.D / V14.E NEON helper: divide a uint64x2 vector by 3.
 *
 * Uses the magic 0xAAAAAAAAAAAAAAAB (= ceil(2^65/3)) — for any unsigned
 * x < 2^63, q = floor(x/3) = mulhi(x, 0xAAAAAAAAAAAAAAAB) >> 1.
 *
 * mulhi is the high 64 bits of (x * m), computed via schoolbook 32×32
 * decomposition: x = x_hi·2^32 + x_lo, m = m_hi·2^32 + m_lo, and the
 * high half assembles from hh + hl_top + lh_top + carry_from_low_64.
 *
 * Cost: ~13 NEON ops for 2 lanes (~7 per cell). For /3^k, iterate k
 * times. Pure NEON; bit-exact vs floor((unsigned)x / 3) for x < 2^63.
 *
 * Used by relu²_bx, elementwise_mul_bx, bitlinear_scale_bx (V14.D-F)
 * to avoid scalar int64 division. */
#if M4T_HAS_NEON
static inline uint64x2_t neon_unsigned_div3_u64x2(uint64x2_t x) {
    uint32x2_t m_lo_v = vdup_n_u32(0xAAAAAAABu);
    uint32x2_t m_hi_v = vdup_n_u32(0xAAAAAAAAu);
    uint32x2_t x_lo = vmovn_u64(x);
    uint32x2_t x_hi = vshrn_n_u64(x, 32);
    uint64x2_t ll = vmull_u32(x_lo, m_lo_v);
    uint64x2_t lh = vmull_u32(x_lo, m_hi_v);
    uint64x2_t hl = vmull_u32(x_hi, m_lo_v);
    uint64x2_t hh = vmull_u32(x_hi, m_hi_v);
    uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
    uint64x2_t ll_top = vshrq_n_u64(ll, 32);
    uint64x2_t mid = vaddq_u64(vaddq_u64(ll_top, vandq_u64(lh, mask32)),
                               vandq_u64(hl, mask32));
    uint64x2_t carry = vshrq_n_u64(mid, 32);
    uint64x2_t high = vaddq_u64(vaddq_u64(hh, vshrq_n_u64(lh, 32)),
                                vaddq_u64(vshrq_n_u64(hl, 32), carry));
    return vshrq_n_u64(high, 1);
}

/* V14.F generic NEON helpers for unsigned 64-bit divide by runtime constant.
 *
 * Caller computes (m, sh, extension) once at function entry via the scalar
 * compute_magic_u64 helper below, then per-cell applies via NEON.
 *
 * Two cases:
 *   m_full = ceil(2^(64+sh) / d) where sh = ceil(log2(d)).
 *   - If m_full < 2^64: q = mulhi(x, m_full) >> sh.            (extension=0)
 *   - Else:             q = (mulhi(x, m_full - 2^64) +
 *                            ((x - mulhi(x, m_full - 2^64)) >> 1)) >> (sh - 1).
 *                                                              (extension=1)
 *
 * Bit-exact for any uint64 input; matches floor(x/d). */
typedef struct {
    uint64_t m;
    int      sh;
    int      extension;
} m4t_magic_div_u64_t;

/* Generic mulhi: high 64 bits of (x * m_const) for 2 lanes.
 * Schoolbook 32×32 with carry — same pattern as neon_unsigned_div3_u64x2
 * but with the multiplier as a runtime parameter (broadcast inside). */
static inline uint64x2_t neon_mulhi_u64x2(uint64x2_t x, uint64_t m_const) {
    uint32x2_t m_lo_v = vdup_n_u32((uint32_t)m_const);
    uint32x2_t m_hi_v = vdup_n_u32((uint32_t)(m_const >> 32));
    uint32x2_t x_lo = vmovn_u64(x);
    uint32x2_t x_hi = vshrn_n_u64(x, 32);
    uint64x2_t ll = vmull_u32(x_lo, m_lo_v);
    uint64x2_t lh = vmull_u32(x_lo, m_hi_v);
    uint64x2_t hl = vmull_u32(x_hi, m_lo_v);
    uint64x2_t hh = vmull_u32(x_hi, m_hi_v);
    uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
    uint64x2_t ll_top = vshrq_n_u64(ll, 32);
    uint64x2_t mid = vaddq_u64(vaddq_u64(ll_top, vandq_u64(lh, mask32)),
                               vandq_u64(hl, mask32));
    uint64x2_t carry = vshrq_n_u64(mid, 32);
    return vaddq_u64(vaddq_u64(hh, vshrq_n_u64(lh, 32)),
                     vaddq_u64(vshrq_n_u64(hl, 32), carry));
}

/* Apply a precomputed unsigned divide-by-constant magic to a uint64x2_t.
 * Bit-exact floor(x / d) for x ∈ [0, 2^64). Magic from compute_magic_u64.
 *
 * Granlund-Möller formulas:
 *   NO_ADD: q = mulhi(x, m) >> sh         (where m = ceil(2^(64+sh)/d) ≤ 2^64)
 *   ADD:    q = (mulhi(x, m) + ((x - mulhi(x, m)) >> 1)) >> (sh-1)
 *           where m = ceil(2^(64+sh)/d) - 2^64 (the low 64 bits when m_full > 2^64). */
static inline uint64x2_t neon_apply_magic_u64x2(uint64x2_t x, m4t_magic_div_u64_t md) {
    uint64x2_t mh = neon_mulhi_u64x2(x, md.m);
    if (md.extension) {
        uint64x2_t xm = vsubq_u64(x, mh);
        uint64x2_t hsr = vshrq_n_u64(xm, 1);
        uint64x2_t s = vaddq_u64(mh, hsr);
        return vshlq_u64(s, vdupq_n_s64(-(int64_t)(md.sh - 1)));
    } else {
        return vshlq_u64(mh, vdupq_n_s64(-(int64_t)md.sh));
    }
}

/* V14.F specialized: divide uint64x2 by 127 for inputs in [0, 2^39).
 * Uses m = ceil(2^46/127) = 554084599825, formula q = (val * m) >> 46.
 * Computes val*m (up to 2^79) via 32-bit limb decomposition with carry.
 * Bit-exact vs floor(val/127) for val ∈ [0, 2^39).
 *
 * Sized for the long-division limb step where val = r*2^32 + limb,
 * r < 127, limb ≤ uint32 → val < 128*2^32 = 2^39.
 *
 * No longer used by V14.F (superseded by combined-divisor magic) but
 * kept for reference / potential future callers. */
__attribute__((unused))
static inline uint64x2_t neon_unsigned_div127_u64x2_le39(uint64x2_t val) {
    /* m = 554084599825 = 0x80FFFFFE81 ... actually:
     *   m_hi32 = 554084599825 >> 32 = 129
     *   m_lo32 = 554084599825 - 129·2^32 = 33818641  */
    const uint32_t m_lo32 = 33818641u;
    const uint32_t m_hi32 = 129u;
    uint32x2_t val_lo = vmovn_u64(val);
    uint32x2_t val_hi = vshrn_n_u64(val, 32);
    uint64x2_t P0 = vmull_u32(val_lo, vdup_n_u32(m_lo32));
    uint64x2_t P1 = vmull_u32(val_lo, vdup_n_u32(m_hi32));
    uint64x2_t P2 = vmull_u32(val_hi, vdup_n_u32(m_lo32));
    uint64x2_t P3 = vmull_u32(val_hi, vdup_n_u32(m_hi32));
    /* P1 + P2 ≤ 2^40 (val_lo·m_hi ≤ 2^39, val_hi·m_lo ≤ 2^39, sum < 2^40). */
    uint64x2_t P12 = vaddq_u64(P1, P2);
    uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
    uint64x2_t P12_low = vandq_u64(P12, mask32);
    uint64x2_t P12_high = vshrq_n_u64(P12, 32);
    /* low_64 = P0 + (P12_low << 32). Detect carry. */
    uint64x2_t shifted = vshlq_n_u64(P12_low, 32);
    uint64x2_t low_64 = vaddq_u64(P0, shifted);
    uint64x2_t carry_mask = vcltq_u64(low_64, P0);
    uint64x2_t carry = vandq_u64(carry_mask, vdupq_n_u64(1));
    /* high_64 = P3 + P12_high + carry. */
    uint64x2_t high_64 = vaddq_u64(vaddq_u64(P3, P12_high), carry);
    /* q = (high_64 << 18) | (low_64 >> 46). */
    return vorrq_u64(vshlq_n_u64(high_64, 18), vshrq_n_u64(low_64, 46));
}

#endif /* M4T_HAS_NEON */

/* Compute (m, sh, extension) for unsigned div by runtime d > 1.
 * Granlund-Möller. Runs once per call (scalar setup; allowed by project rule).
 *
 *   sh = ceil(log2(d))
 *   m_full = ceil(2^(64+sh) / d).
 *   if m_full ≤ 2^64: NO_ADD path.  m = m_full, q = mulhi(x, m) >> sh.
 *   else:             ADD path.     m = m_full - 2^64,
 *                                   q = (mulhi(x, m) + ((x - mulhi(x, m)) >> 1)) >> (sh-1).
 *
 * Bit-exact for any uint64 input. Caller must reject d that's a power of 2. */
static m4t_magic_div_u64_t compute_magic_u64(uint64_t d) {
    assert(d > 1);
    assert((d & (d - 1)) != 0);
    int sh = 64 - __builtin_clzll(d - 1);  /* ceil(log2(d)) */
    __uint128_t one = 1;
    __uint128_t numerator = (one << (64 + sh)) + (uint64_t)d - 1;
    __uint128_t m_full = numerator / (__uint128_t)d;
    __uint128_t pow_2_64 = one << 64;
    m4t_magic_div_u64_t out;
    if (m_full <= pow_2_64) {
        out.m = (uint64_t)m_full;
        out.sh = sh;
        out.extension = 0;
    } else {
        out.m = (uint64_t)(m_full - pow_2_64);
        out.sh = sh;
        out.extension = 1;
    }
    return out;
}

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

/* ── Softmax (LUT exp + NEON-gather + integer reciprocal) ──────────────
 *
 * V14.G v2: bit-exact V13 LUT-based exp, NEON-gathered.
 *
 * Per-cell pipeline:
 *   z = x[i] - max(x)             (int32, ≤ 0)
 *   neg_z = -z, clamped to [0, RANGE)
 *   idx_q16 = neg_z × (LUT_RES × 2^16 / RANGE)
 *   idx = idx_q16 >> 16, frac = idx_q16 & 0xFFFF
 *   exp(z) ≈ LUT[idx] + (LUT[idx+1] − LUT[idx]) × frac >> 16
 *
 * The gather step uses ARM's per-lane LDR via vld1q_lane_s32: address
 * computation lives in a scalar register (one vgetq_lane_s32 + one LDR
 * per lane), but no scalar arithmetic happens on softmax model values —
 * the load drops directly into a NEON lane. This is the same precedent
 * as vqtbl in the matmul routing kernels: table lookup is a memory op,
 * not a scalar arithmetic op. The interp itself is pure NEON.
 *
 * Bit-exact match to V13's softmax_exp_int (the LUT + scalar interp
 * scalar function), so end-to-end behavior matches V13. */

static int32_t* g_softmax_exp_lut = NULL;
static int      g_softmax_lut_initialized = 0;

static void softmax_init_lut(void) {
    if (g_softmax_lut_initialized) return;
    int32_t* lut = (int32_t*)malloc((size_t)M4T_SOFTMAX_LUT_RES * sizeof(int32_t));
    assert(lut);
    double scale = (double)M4T_SOFTMAX_OUT_SCALE;  /* 2^30 */
    for (int k = 0; k < M4T_SOFTMAX_LUT_RES; k++) {
        double z = -(double)k * (double)M4T_SOFTMAX_LUT_RANGE
                   / (double)M4T_SOFTMAX_LUT_RES;
        double v = exp(z) * scale;
        int32_t vi = (int32_t)(v + 0.5);
        if (vi < 0) vi = 0;
        if (vi > M4T_SOFTMAX_OUT_SCALE) vi = M4T_SOFTMAX_OUT_SCALE;
        lut[k] = vi;
    }
    g_softmax_exp_lut = lut;
    g_softmax_lut_initialized = 1;
}

/* Scalar reference: exp(z) at scale 2^30 via the LUT + linear interp.
 * Bit-identical to V13's softmax_exp_int. */
__attribute__((unused))
static int32_t softmax_exp_int(int32_t z) {
    if (z >= 0) return M4T_SOFTMAX_OUT_SCALE;
    int32_t neg_z = -z;
    if (neg_z >= M4T_SOFTMAX_LUT_RANGE) return 0;
    int64_t idx_q16 = (int64_t)neg_z * M4T_SOFTMAX_LUT_RES * 65536LL / M4T_SOFTMAX_LUT_RANGE;
    int idx = (int)(idx_q16 >> 16);
    int frac = (int)(idx_q16 & 0xFFFF);
    if (idx >= M4T_SOFTMAX_LUT_RES - 1) {
        return g_softmax_exp_lut[M4T_SOFTMAX_LUT_RES - 1];
    }
    int32_t a = g_softmax_exp_lut[idx];
    int32_t b = g_softmax_exp_lut[idx + 1];
    int32_t v = a + (int32_t)(((int64_t)(b - a) * (int64_t)frac) >> 16);
    return v;
}

#if M4T_HAS_NEON
/* Q16 multiplier: idx_q16 = neg_z × (LUT_RES × 2^16 / LUT_RANGE).
 *   = neg_z × (4096 × 65536 / 30) = neg_z × 8947848 (with rounding).
 * NOTE: the SCALAR softmax_exp_int computes (neg_z × LUT_RES × 65536)
 * EXACTLY then divides by LUT_RANGE — to match bit-exactly, we use
 * the same exact int64 multiply-then-divide per lane (cheap; one
 * scalar div per cell or NEON-friendly approximation if RANGE divides
 * cleanly). For LUT_RES=4096, LUT_RANGE=30, the multiplier 8947848.5333
 * isn't integer, so we keep the scalar computation per-lane for exact
 * idx_q16 (same precision as V13). Then NEON-gather LUT entries. */

static inline int32x4_t softmax_exp_lut_neon(int32x4_t z) {
    uint32x4_t mask_zpos = vcgezq_s32(z);                     /* z >= 0 */
    int32x4_t  neg_z     = vnegq_s32(z);
    uint32x4_t mask_zlow = vcgeq_s32(neg_z, vdupq_n_s32(M4T_SOFTMAX_LUT_RANGE));

    /* Compute idx_q16 per lane. Match V13 exactly:
     *   idx_q16 = neg_z * LUT_RES * 65536 / LUT_RANGE
     * Use scalar lane extraction for the per-lane int64 divide (LUT_RANGE
     * = 30 is not a power of 2; this is one signed scalar division per
     * lane, but it's an ADDRESS-computation step, not arithmetic on the
     * softmax data — see file header comment). */
    int32_t nz0 = vgetq_lane_s32(neg_z, 0);
    int32_t nz1 = vgetq_lane_s32(neg_z, 1);
    int32_t nz2 = vgetq_lane_s32(neg_z, 2);
    int32_t nz3 = vgetq_lane_s32(neg_z, 3);
    int64_t q0 = (int64_t)nz0 * M4T_SOFTMAX_LUT_RES * 65536LL / M4T_SOFTMAX_LUT_RANGE;
    int64_t q1 = (int64_t)nz1 * M4T_SOFTMAX_LUT_RES * 65536LL / M4T_SOFTMAX_LUT_RANGE;
    int64_t q2 = (int64_t)nz2 * M4T_SOFTMAX_LUT_RES * 65536LL / M4T_SOFTMAX_LUT_RANGE;
    int64_t q3 = (int64_t)nz3 * M4T_SOFTMAX_LUT_RES * 65536LL / M4T_SOFTMAX_LUT_RANGE;
    /* Clamp idx so [idx], [idx+1] are in-bounds. For neg_z ≥ LUT_RANGE
     * (out of LUT range), the lane is masked to 0 below; clamp keeps the
     * gather safe regardless. */
    int idx0 = (int)(q0 >> 16); if (idx0 >= M4T_SOFTMAX_LUT_RES - 1) idx0 = M4T_SOFTMAX_LUT_RES - 2; if (idx0 < 0) idx0 = 0;
    int idx1 = (int)(q1 >> 16); if (idx1 >= M4T_SOFTMAX_LUT_RES - 1) idx1 = M4T_SOFTMAX_LUT_RES - 2; if (idx1 < 0) idx1 = 0;
    int idx2 = (int)(q2 >> 16); if (idx2 >= M4T_SOFTMAX_LUT_RES - 1) idx2 = M4T_SOFTMAX_LUT_RES - 2; if (idx2 < 0) idx2 = 0;
    int idx3 = (int)(q3 >> 16); if (idx3 >= M4T_SOFTMAX_LUT_RES - 1) idx3 = M4T_SOFTMAX_LUT_RES - 2; if (idx3 < 0) idx3 = 0;
    int32_t frac0 = (int32_t)(q0 & 0xFFFF);
    int32_t frac1 = (int32_t)(q1 & 0xFFFF);
    int32_t frac2 = (int32_t)(q2 & 0xFFFF);
    int32_t frac3 = (int32_t)(q3 & 0xFFFF);

    /* Per-lane gather: load LUT[idx] and LUT[idx+1] into NEON lanes.
     * vld1q_lane_s32 (= LD1 {Vt.S}[i], [Xn]) is a NEON load; the address
     * Xn is computed in a scalar register, but the value lands in a
     * NEON lane directly (no scalar arithmetic on the loaded data). */
    int32x4_t a = vdupq_n_s32(0);
    a = vld1q_lane_s32(&g_softmax_exp_lut[idx0], a, 0);
    a = vld1q_lane_s32(&g_softmax_exp_lut[idx1], a, 1);
    a = vld1q_lane_s32(&g_softmax_exp_lut[idx2], a, 2);
    a = vld1q_lane_s32(&g_softmax_exp_lut[idx3], a, 3);
    int32x4_t b = vdupq_n_s32(0);
    b = vld1q_lane_s32(&g_softmax_exp_lut[idx0 + 1], b, 0);
    b = vld1q_lane_s32(&g_softmax_exp_lut[idx1 + 1], b, 1);
    b = vld1q_lane_s32(&g_softmax_exp_lut[idx2 + 1], b, 2);
    b = vld1q_lane_s32(&g_softmax_exp_lut[idx3 + 1], b, 3);

    /* Linear interp: v = a + ((b - a) × frac) >> 16, all NEON. */
    int32x4_t frac_v = { frac0, frac1, frac2, frac3 };
    int32x4_t bma = vsubq_s32(b, a);
    /* (b - a) × frac: int32 × int32. Use vmull_s32 → int64x2 → >>16 → int32x2. */
    int64x2_t prod_lo = vmull_s32(vget_low_s32(bma), vget_low_s32(frac_v));
    int64x2_t prod_hi = vmull_s32(vget_high_s32(bma), vget_high_s32(frac_v));
    int32x4_t inc = vcombine_s32(vmovn_s64(vshrq_n_s64(prod_lo, 16)),
                                 vmovn_s64(vshrq_n_s64(prod_hi, 16)));
    int32x4_t result = vaddq_s32(a, inc);

    /* Apply special-case masks: 0 where neg_z >= LUT_RANGE; OUT_SCALE
     * where z >= 0. */
    result = vbslq_s32(mask_zlow, vdupq_n_s32(0), result);
    result = vbslq_s32(mask_zpos, vdupq_n_s32(M4T_SOFTMAX_OUT_SCALE), result);
    return result;
}
#endif /* M4T_HAS_NEON */

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
#if M4T_HAS_NEON
    softmax_init_lut();
    /* Stage 1: NEON max reduction. */
    int32x4_t mx_v = vdupq_n_s32(x[0]);
    int n_aligned = n - (n % 4);
    int i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t xv = vld1q_s32(x + i);
        mx_v = vmaxq_s32(mx_v, xv);
    }
    int32_t mx = vmaxvq_s32(mx_v);
    for (; i < n; i++) if (x[i] > mx) mx = x[i];

    /* Stage 2: NEON polynomial exp + sum. e[i] is non-negative ≤ 2^30. */
    int32_t* e = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    assert(e);
    int64x2_t sum_lo = vdupq_n_s64(0);
    int64x2_t sum_hi = vdupq_n_s64(0);
    int32x4_t mx_bcast = vdupq_n_s32(mx);
    i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t xv = vld1q_s32(x + i);
        int32x4_t zv = vsubq_s32(xv, mx_bcast);  /* z = x - mx, ≤ 0 in normal case */
        int32x4_t ev = softmax_exp_lut_neon(zv);
        vst1q_s32(e + i, ev);
        /* Accumulate to int64 (each e ≤ 2^30, sum ≤ n × 2^30). */
        sum_lo = vaddw_s32(sum_lo, vget_low_s32(ev));
        sum_hi = vaddw_s32(sum_hi, vget_high_s32(ev));
    }
    /* Boundary tile for n%4 != 0. */
    if (i < n) {
        int avail = n - i;
        m4t_mtfp_t xbuf[4] = {0};
        for (int j = 0; j < avail; j++) xbuf[j] = x[i + j];
        int32x4_t xv = vld1q_s32(xbuf);
        int32x4_t zv = vsubq_s32(xv, mx_bcast);
        int32x4_t ev = softmax_exp_lut_neon(zv);
        m4t_mtfp_t ebuf[4];
        vst1q_s32(ebuf, ev);
        for (int j = 0; j < avail; j++) {
            e[i + j] = ebuf[j];
            sum_lo = vsetq_lane_s64(vgetq_lane_s64(sum_lo, 0) + ebuf[j], sum_lo, 0);
        }
    }
    int64_t sum = vgetq_lane_s64(sum_lo, 0) + vgetq_lane_s64(sum_lo, 1)
                + vgetq_lane_s64(sum_hi, 0) + vgetq_lane_s64(sum_hi, 1);
    if (sum < 1) sum = 1;

    /* Stage 3: scalar setup of inv60 (once-per-call, like rsqrt setup
     * in V14.C; not per-cell). */
    int64_t inv60 = softmax_recip60(sum);

    /* Stage 4: NEON per-cell e[i] × inv60 >> 30, clamp [0, OUT_SCALE].
     * Same uint96 multiply pattern as V14.F: e (uint32) × inv60 (uint64)
     * decomposed via 32-bit limbs. e is non-negative so no sign handling. */
    uint64_t inv60_u = (uint64_t)inv60;
    uint32_t inv60_lo32 = (uint32_t)inv60_u;
    uint32_t inv60_hi32 = (uint32_t)(inv60_u >> 32);
    int32x4_t v_max = vdupq_n_s32(M4T_SOFTMAX_OUT_SCALE);
    int32x4_t v_zero = vdupq_n_s32(0);

    i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t ev = vld1q_s32(e + i);
        for (int half = 0; half < 2; half++) {
            uint32x2_t e_pair = (half == 0)
                ? vreinterpret_u32_s32(vget_low_s32(ev))
                : vreinterpret_u32_s32(vget_high_s32(ev));
            uint64x2_t P_lo = vmull_u32(e_pair, vdup_n_u32(inv60_lo32));
            uint64x2_t P_hi = vmull_u32(e_pair, vdup_n_u32(inv60_hi32));
            /* Combined 96-bit V = P_hi << 32 + P_lo. >> 30 to get y. */
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P_lo);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P_lo, 32),
                                       vandq_u64(P_hi, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P_hi, 32),
                                                 vshrq_n_u64(mid, 32)));
            /* Right-shift uint96 by 30: result = (V_high << 34) | (V_low >> 30). */
            uint64x2_t V_low = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                         vmovl_u32(L0));
            uint64x2_t V_high = vmovl_u32(L2);
            uint64x2_t shifted = vorrq_u64(vshlq_n_u64(V_high, 34),
                                            vshrq_n_u64(V_low, 30));
            int32x2_t y_pair = vqmovn_s64(vreinterpretq_s64_u64(shifted));
            if (half == 0) vst1_s32(y + i,     y_pair);
            else           vst1_s32(y + i + 2, y_pair);
        }
        int32x4_t y4 = vld1q_s32(y + i);
        y4 = vminq_s32(vmaxq_s32(y4, v_zero), v_max);
        vst1q_s32(y + i, y4);
    }
    if (i < n) {
        int avail = n - i;
        m4t_mtfp_t ebuf[4] = {0}, ybuf[4] = {0};
        for (int j = 0; j < avail; j++) ebuf[j] = e[i + j];
        int32x4_t ev = vld1q_s32(ebuf);
        for (int half = 0; half < 2; half++) {
            uint32x2_t e_pair = (half == 0)
                ? vreinterpret_u32_s32(vget_low_s32(ev))
                : vreinterpret_u32_s32(vget_high_s32(ev));
            uint64x2_t P_lo = vmull_u32(e_pair, vdup_n_u32(inv60_lo32));
            uint64x2_t P_hi = vmull_u32(e_pair, vdup_n_u32(inv60_hi32));
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P_lo);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P_lo, 32),
                                       vandq_u64(P_hi, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P_hi, 32),
                                                 vshrq_n_u64(mid, 32)));
            uint64x2_t V_low = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                         vmovl_u32(L0));
            uint64x2_t V_high = vmovl_u32(L2);
            uint64x2_t shifted = vorrq_u64(vshlq_n_u64(V_high, 34),
                                            vshrq_n_u64(V_low, 30));
            int32x2_t y_pair = vqmovn_s64(vreinterpretq_s64_u64(shifted));
            if (half == 0) vst1_s32(ybuf,     y_pair);
            else           vst1_s32(ybuf + 2, y_pair);
        }
        int32x4_t y4 = vld1q_s32(ybuf);
        y4 = vminq_s32(vmaxq_s32(y4, v_zero), v_max);
        vst1q_s32(ybuf, y4);
        for (int j = 0; j < avail; j++) y[i + j] = ybuf[j];
    }
    free(e);
#else
#error "m4t_mtfp_softmax requires NEON; no scalar fallback per project rule."
#endif
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

/* V14.C: NEON rmsnorm_bx. Two halves:
 *
 *   1. SoS via NEON int128 (synthesized from int64 + carry tracking).
 *      vmull_s32 produces x[i]² (always non-negative ≤ 2^58.3); we
 *      accumulate into int64 lanes, detect uint64 overflow via
 *      vcltq_u64, and propagate carries to a parallel high-half
 *      accumulator. Final 4-lane reduction is scalar __int128 (once
 *      per call — same kind as a NEON kernel's lane extraction).
 *
 *   2. Per-cell γ × x × inv >> total_shift: |γ*x| (int64 via vmull_s32 +
 *      vabsq_s64) × inv (uint32) → uint96 (3 × uint32 limbs, same
 *      pattern as V14.F). Right-shift by total_shift across limbs,
 *      apply sign(γ*x), clamp + narrow to MTFP19.
 *
 * Bit-exact vs scalar_ref. */
void m4t_mtfp_rmsnorm_bx(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* gamma,
    int x_bx, int gamma_bx, int target_bx,
    m4t_mtfp_t eps_mantissa, int n)
{
    assert(n >= 0);
    if (n == 0) return;
    assert(y && x && gamma);
    (void)x_bx;
#if M4T_HAS_NEON
    /* Stage 1: NEON SoS with int128-via-carry-tracking. */
    int64x2_t acc_lo_lo = vdupq_n_s64(0);
    int64x2_t acc_lo_hi = vdupq_n_s64(0);
    int64x2_t acc_hi_lo = vdupq_n_s64(0);
    int64x2_t acc_hi_hi = vdupq_n_s64(0);

    int n_aligned = n - (n % 4);
    int i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t xv = vld1q_s32(x + i);
        int64x2_t sq_lo = vmull_s32(vget_low_s32(xv),  vget_low_s32(xv));
        int64x2_t sq_hi = vmull_s32(vget_high_s32(xv), vget_high_s32(xv));
        /* Add to acc_lo with carry detection. */
        int64x2_t prev_lo = acc_lo_lo;
        acc_lo_lo = vaddq_s64(acc_lo_lo, sq_lo);
        uint64x2_t carry_lo = vcltq_u64(vreinterpretq_u64_s64(acc_lo_lo),
                                        vreinterpretq_u64_s64(prev_lo));
        acc_lo_hi = vreinterpretq_s64_u64(
            vaddq_u64(vreinterpretq_u64_s64(acc_lo_hi),
                      vandq_u64(carry_lo, vdupq_n_u64(1))));
        /* Same for acc_hi. */
        int64x2_t prev_hi = acc_hi_lo;
        acc_hi_lo = vaddq_s64(acc_hi_lo, sq_hi);
        uint64x2_t carry_hi = vcltq_u64(vreinterpretq_u64_s64(acc_hi_lo),
                                        vreinterpretq_u64_s64(prev_hi));
        acc_hi_hi = vreinterpretq_s64_u64(
            vaddq_u64(vreinterpretq_u64_s64(acc_hi_hi),
                      vandq_u64(carry_hi, vdupq_n_u64(1))));
    }
    /* Boundary tile: scalar tail for n%4 (small). */
    __int128 sum_sq_tail = 0;
    for (; i < n; i++) {
        int64_t xv = (int64_t)x[i];
        sum_sq_tail += (__int128)(xv * xv);
    }

    /* Reduce 4 lanes to __int128 (once-per-call setup, like a NEON
     * kernel's lane extraction at end). */
    __int128 sum_sq = sum_sq_tail;
    sum_sq += ((__int128)vgetq_lane_s64(acc_lo_hi, 0) << 64) | (__int128)(uint64_t)vgetq_lane_s64(acc_lo_lo, 0);
    sum_sq += ((__int128)vgetq_lane_s64(acc_lo_hi, 1) << 64) | (__int128)(uint64_t)vgetq_lane_s64(acc_lo_lo, 1);
    sum_sq += ((__int128)vgetq_lane_s64(acc_hi_hi, 0) << 64) | (__int128)(uint64_t)vgetq_lane_s64(acc_hi_lo, 0);
    sum_sq += ((__int128)vgetq_lane_s64(acc_hi_hi, 1) << 64) | (__int128)(uint64_t)vgetq_lane_s64(acc_hi_lo, 1);

    /* Stage 1 setup (scalar, once per call). */
    __int128 mean_full = sum_sq / (__int128)n + (__int128)eps_mantissa;
    if (mean_full < 1) mean_full = 1;
    int extra_k = 0;
    __int128 mean_passed = mean_full;
    while (mean_passed > (__int128)0x7FFFFFFF) {
        mean_passed >>= 2;
        extra_k++;
    }
    if (mean_passed < 1) mean_passed = 1;
    m4t_mtfp_t inv = m4t_int32_rsqrt((m4t_mtfp_t)mean_passed);
    int total_shift = 30 + extra_k;
    assert(inv >= 0);

    /* Stage 2: NEON per-cell γ × x × inv >> total_shift, clamp.
     * Compute |γ*x| × inv as uint96 (3 × uint32 limbs), right-shift
     * by total_shift, re-sign, clamp. */
    uint32_t inv_u32 = (uint32_t)inv;  /* inv ≥ 0 from m4t_int32_rsqrt */
    int32x4_t v_max  = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    int32x4_t v_min  = vdupq_n_s32(-(int32_t)M4T_MTFP_MAX_VAL);

    i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t gv = vld1q_s32(gamma + i);
        int32x4_t xv = vld1q_s32(x + i);
        /* gx = γ[i] * x[i] as int64x2, two halves of the int32x4 input. */
        int64x2_t gx_lo = vmull_s32(vget_low_s32(gv),  vget_low_s32(xv));
        int64x2_t gx_hi = vmull_s32(vget_high_s32(gv), vget_high_s32(xv));
        /* sign of gx, |gx|. */
        uint64x2_t s_lo = vcltzq_s64(gx_lo);
        uint64x2_t s_hi = vcltzq_s64(gx_hi);
        uint64x2_t a_lo = vreinterpretq_u64_s64(vabsq_s64(gx_lo));
        uint64x2_t a_hi = vreinterpretq_u64_s64(vabsq_s64(gx_hi));

        /* Compute |gx| × inv as uint96 = 3 × uint32 limbs.
         * a (uint64) × inv (uint32) decomposed:
         *   a_lo32 × inv → uint64 (P0).
         *   a_hi32 × inv → uint64 (P1, ≤ uint32 × uint32 = uint64).
         * Combined: result = P1 << 32 + P0. */
        for (int half = 0; half < 2; half++) {
            uint64x2_t a = (half == 0) ? a_lo : a_hi;
            uint64x2_t s = (half == 0) ? s_lo : s_hi;

            uint32x2_t a_lo32 = vmovn_u64(a);
            uint32x2_t a_hi32 = vshrn_n_u64(a, 32);
            uint32x2_t inv_v  = vdup_n_u32(inv_u32);
            uint64x2_t P0 = vmull_u32(a_lo32, inv_v);
            uint64x2_t P1 = vmull_u32(a_hi32, inv_v);
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P0);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P0, 32),
                                       vandq_u64(P1, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P1, 32),
                                                 vshrq_n_u64(mid, 32)));

            /* Right-shift uint96 V[2..0] by total_shift to get the
             * scaled result. total_shift ∈ [30, 30+something].
             * Treat V as 3 × uint32 limbs:
             *   bits 0-31  = V[0]
             *   bits 32-63 = V[1]
             *   bits 64-95 = V[2]
             * After >> total_shift, result fits uint64 in our bounds
             * (|γ*x*inv| ≤ 2^89, >> 30 = 2^59, fits int64 + signed). */
            int64x2_t cnt = vdupq_n_s64(-(int64_t)total_shift);
            uint64x2_t V_low = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                         vmovl_u32(L0));
            uint64x2_t V_high = vmovl_u32(L2);
            int64x2_t high_part = vshlq_s64(vreinterpretq_s64_u64(V_high),
                                            vdupq_n_s64((int64_t)(64 - total_shift)));
            int64x2_t low_part  = vreinterpretq_s64_u64(vshlq_u64(V_low, cnt));
            int64x2_t r_pos = vaddq_s64(high_part, low_part);

            /* Arithmetic-shift correction: scalar uses (int128)prod >> shift
             * which floors toward -inf for negative. abs+shift+negate gives
             * truncate-toward-zero. For negative prod with non-zero remainder
             * in the discarded low bits, increment magnitude before negating.
             * total_shift ≤ 63 in our range; remainder is V_low's low bits. */
            assert(total_shift > 0 && total_shift < 64);
            uint64x2_t mask = vdupq_n_u64((1ULL << total_shift) - 1);
            uint64x2_t rem  = vandq_u64(V_low, mask);
            uint64x2_t has_rem = vcgtq_u64(rem, vdupq_n_u64(0));
            uint64x2_t adj_bit = vandq_u64(vandq_u64(s, has_rem),
                                           vdupq_n_u64(1));
            int64x2_t r_pos_for_neg = vaddq_s64(r_pos,
                vreinterpretq_s64_u64(adj_bit));
            int64x2_t r_neg = vnegq_s64(r_pos_for_neg);
            int64x2_t r_signed = vbslq_s64(s, r_neg, r_pos);

            /* Clamp + narrow. vqmovn_s64 saturates to int32; vminq/vmaxq
             * to MTFP19. */
            int32x2_t y_pair = vqmovn_s64(r_signed);
            if (half == 0) vst1_s32(y + i,     y_pair);
            else           vst1_s32(y + i + 2, y_pair);
        }
        int32x4_t y4 = vld1q_s32(y + i);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(y + i, y4);
    }
    /* Boundary tile for n%4 != 0: process tail cells using stack bufs. */
    if (i < n) {
        int avail = n - i;
        m4t_mtfp_t gbuf[4] = {0}, xbuf[4] = {0}, ybuf[4] = {0};
        for (int j = 0; j < avail; j++) { gbuf[j] = gamma[i + j]; xbuf[j] = x[i + j]; }
        int32x4_t gv = vld1q_s32(gbuf);
        int32x4_t xv = vld1q_s32(xbuf);
        int64x2_t gx_lo = vmull_s32(vget_low_s32(gv),  vget_low_s32(xv));
        int64x2_t gx_hi = vmull_s32(vget_high_s32(gv), vget_high_s32(xv));
        uint64x2_t s_lo = vcltzq_s64(gx_lo);
        uint64x2_t s_hi = vcltzq_s64(gx_hi);
        uint64x2_t a_lo = vreinterpretq_u64_s64(vabsq_s64(gx_lo));
        uint64x2_t a_hi = vreinterpretq_u64_s64(vabsq_s64(gx_hi));
        for (int half = 0; half < 2; half++) {
            uint64x2_t a = (half == 0) ? a_lo : a_hi;
            uint64x2_t s = (half == 0) ? s_lo : s_hi;
            uint32x2_t a_lo32 = vmovn_u64(a);
            uint32x2_t a_hi32 = vshrn_n_u64(a, 32);
            uint32x2_t inv_v  = vdup_n_u32(inv_u32);
            uint64x2_t P0 = vmull_u32(a_lo32, inv_v);
            uint64x2_t P1 = vmull_u32(a_hi32, inv_v);
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P0);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P0, 32),
                                       vandq_u64(P1, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P1, 32),
                                                 vshrq_n_u64(mid, 32)));
            int64x2_t cnt = vdupq_n_s64(-(int64_t)total_shift);
            uint64x2_t V_low = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                         vmovl_u32(L0));
            uint64x2_t V_high = vmovl_u32(L2);
            int64x2_t high_part = vshlq_s64(vreinterpretq_s64_u64(V_high),
                                            vdupq_n_s64((int64_t)(64 - total_shift)));
            int64x2_t low_part  = vreinterpretq_s64_u64(vshlq_u64(V_low, cnt));
            int64x2_t r_pos = vaddq_s64(high_part, low_part);
            /* Arithmetic-shift correction (boundary tile). */
            assert(total_shift > 0 && total_shift < 64);
            uint64x2_t mask = vdupq_n_u64((1ULL << total_shift) - 1);
            uint64x2_t rem  = vandq_u64(V_low, mask);
            uint64x2_t has_rem = vcgtq_u64(rem, vdupq_n_u64(0));
            uint64x2_t adj_bit = vandq_u64(vandq_u64(s, has_rem),
                                           vdupq_n_u64(1));
            int64x2_t r_pos_for_neg = vaddq_s64(r_pos,
                vreinterpretq_s64_u64(adj_bit));
            int64x2_t r_neg = vnegq_s64(r_pos_for_neg);
            int64x2_t r_signed = vbslq_s64(s, r_neg, r_pos);
            int32x2_t y_pair = vqmovn_s64(r_signed);
            if (half == 0) vst1_s32(ybuf,     y_pair);
            else           vst1_s32(ybuf + 2, y_pair);
        }
        int32x4_t y4 = vld1q_s32(ybuf);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(ybuf, y4);
        for (int j = 0; j < avail; j++) y[i + j] = ybuf[j];
    }

    /* Rescale gamma_bx → target_bx (existing function). */
    if (gamma_bx != target_bx) {
        m4t_mtfp_rescale_bx(y, y, gamma_bx, target_bx, n);
    }
#else
#error "m4t_mtfp_rmsnorm_bx requires NEON; no scalar fallback per project rule."
#endif
}

void m4t_mtfp_rmsnorm_bx_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* gamma,
    int x_bx, int gamma_bx, int target_bx,
    m4t_mtfp_t eps_mantissa, int n)
{
    assert(n >= 0);
    if (n == 0) return;
    assert(y && x && gamma);
    (void)x_bx;
    __int128 sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int64_t xv = (int64_t)x[i];
        sum_sq += (__int128)(xv * xv);
    }
    __int128 mean_full = sum_sq / (__int128)n + (__int128)eps_mantissa;
    if (mean_full < 1) mean_full = 1;
    int extra_k = 0;
    __int128 mean_passed = mean_full;
    while (mean_passed > (__int128)0x7FFFFFFF) {
        mean_passed >>= 2;
        extra_k++;
    }
    if (mean_passed < 1) mean_passed = 1;
    m4t_mtfp_t inv = m4t_int32_rsqrt((m4t_mtfp_t)mean_passed);
    int total_shift = 30 + extra_k;
    for (int i = 0; i < n; i++) {
        __int128 prod = (__int128)gamma[i] * (__int128)x[i] * (__int128)inv;
        int64_t scaled = (int64_t)(prod >> total_shift);
        y[i] = m4t_mtfp_clamp64(scaled);
    }
    if (gamma_bx != target_bx) {
        m4t_mtfp_rescale_bx(y, y, gamma_bx, target_bx, n);
    }
}

/* V14.D: NEON relu²_bx. ReLU + square (vmull_s32) + add half + iterated
 * /3 (NEON mulhi by 0xAAAA...AAAB) + saturating narrow + MTFP19 clamp.
 * Bit-exact vs the scalar reference; no scalar arithmetic in the
 * production path (boundary tile uses 4-wide stack bufs, same pattern
 * as attn_v_combine). */
void m4t_mtfp_relu2_inplace_bx(m4t_mtfp_t* x, int x_bx, int target_bx, int n) {
    if (n <= 0) return;
    assert(x);
    int shift_exp = 2 * x_bx - target_bx;
    assert(shift_exp >= 0 && shift_exp <= 39);
    int64_t den = pow3_i64(shift_exp);
    int64_t half = den / 2;
#if M4T_HAS_NEON
    int32x4_t zero    = vdupq_n_s32(0);
    int32x4_t v_max   = vdupq_n_s32(M4T_MTFP_MAX_VAL);
    int64x2_t halfv   = vdupq_n_s64(half);
    int n_aligned = n - (n % 4);
    int i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t xv   = vld1q_s32(x + i);
        int32x4_t relu = vmaxq_s32(xv, zero);
        int64x2_t sq_lo = vmull_s32(vget_low_s32(relu),  vget_low_s32(relu));
        int64x2_t sq_hi = vmull_s32(vget_high_s32(relu), vget_high_s32(relu));
        sq_lo = vaddq_s64(sq_lo, halfv);
        sq_hi = vaddq_s64(sq_hi, halfv);
        uint64x2_t u_lo = vreinterpretq_u64_s64(sq_lo);
        uint64x2_t u_hi = vreinterpretq_u64_s64(sq_hi);
        for (int k = 0; k < shift_exp; k++) {
            u_lo = neon_unsigned_div3_u64x2(u_lo);
            u_hi = neon_unsigned_div3_u64x2(u_hi);
        }
        int32x2_t y_lo = vqmovn_s64(vreinterpretq_s64_u64(u_lo));
        int32x2_t y_hi = vqmovn_s64(vreinterpretq_s64_u64(u_hi));
        int32x4_t y4 = vcombine_s32(y_lo, y_hi);
        y4 = vminq_s32(y4, v_max);
        vst1q_s32(x + i, y4);
    }
    if (i < n) {
        int avail = n - i;
        m4t_mtfp_t xbuf[4] = {0};
        for (int j = 0; j < avail; j++) xbuf[j] = x[i + j];
        int32x4_t xv   = vld1q_s32(xbuf);
        int32x4_t relu = vmaxq_s32(xv, zero);
        int64x2_t sq_lo = vmull_s32(vget_low_s32(relu),  vget_low_s32(relu));
        int64x2_t sq_hi = vmull_s32(vget_high_s32(relu), vget_high_s32(relu));
        sq_lo = vaddq_s64(sq_lo, halfv);
        sq_hi = vaddq_s64(sq_hi, halfv);
        uint64x2_t u_lo = vreinterpretq_u64_s64(sq_lo);
        uint64x2_t u_hi = vreinterpretq_u64_s64(sq_hi);
        for (int k = 0; k < shift_exp; k++) {
            u_lo = neon_unsigned_div3_u64x2(u_lo);
            u_hi = neon_unsigned_div3_u64x2(u_hi);
        }
        int32x2_t y_lo = vqmovn_s64(vreinterpretq_s64_u64(u_lo));
        int32x2_t y_hi = vqmovn_s64(vreinterpretq_s64_u64(u_hi));
        int32x4_t y4 = vcombine_s32(y_lo, y_hi);
        y4 = vminq_s32(y4, v_max);
        m4t_mtfp_t ybuf[4];
        vst1q_s32(ybuf, y4);
        for (int j = 0; j < avail; j++) x[i + j] = ybuf[j];
    }
#else
#error "m4t_mtfp_relu2_inplace_bx requires NEON; no scalar fallback per project rule."
#endif
}

void m4t_mtfp_relu2_inplace_bx_scalar_ref(m4t_mtfp_t* x, int x_bx, int target_bx, int n) {
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

/* V14.E: NEON elementwise_mul_bx. Same pattern as relu²_bx but with
 * sign-aware divide: take vabsq_s64, divide as unsigned, re-sign via
 * vbslq_s64 with sign mask. Bit-exact vs scalar_ref. */
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
#if M4T_HAS_NEON
    int32x4_t v_max  = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    int32x4_t v_min  = vdupq_n_s32(-(int32_t)M4T_MTFP_MAX_VAL);
    int64x2_t halfv  = vdupq_n_s64(half);
    int n_aligned = n - (n % 4);
    int i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t av = vld1q_s32(a + i);
        int32x4_t bv = vld1q_s32(b + i);
        int64x2_t prod_lo = vmull_s32(vget_low_s32(av),  vget_low_s32(bv));
        int64x2_t prod_hi = vmull_s32(vget_high_s32(av), vget_high_s32(bv));
        uint64x2_t mask_lo = vcltzq_s64(prod_lo);
        uint64x2_t mask_hi = vcltzq_s64(prod_hi);
        int64x2_t abs_lo = vabsq_s64(prod_lo);
        int64x2_t abs_hi = vabsq_s64(prod_hi);
        abs_lo = vaddq_s64(abs_lo, halfv);
        abs_hi = vaddq_s64(abs_hi, halfv);
        uint64x2_t u_lo = vreinterpretq_u64_s64(abs_lo);
        uint64x2_t u_hi = vreinterpretq_u64_s64(abs_hi);
        for (int k = 0; k < shift_exp; k++) {
            u_lo = neon_unsigned_div3_u64x2(u_lo);
            u_hi = neon_unsigned_div3_u64x2(u_hi);
        }
        int64x2_t pos_lo = vreinterpretq_s64_u64(u_lo);
        int64x2_t pos_hi = vreinterpretq_s64_u64(u_hi);
        int64x2_t neg_lo = vnegq_s64(pos_lo);
        int64x2_t neg_hi = vnegq_s64(pos_hi);
        int64x2_t r_lo = vbslq_s64(mask_lo, neg_lo, pos_lo);
        int64x2_t r_hi = vbslq_s64(mask_hi, neg_hi, pos_hi);
        int32x2_t y_lo = vqmovn_s64(r_lo);
        int32x2_t y_hi = vqmovn_s64(r_hi);
        int32x4_t y4 = vcombine_s32(y_lo, y_hi);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(y + i, y4);
    }
    if (i < n) {
        int avail = n - i;
        m4t_mtfp_t abuf[4] = {0}, bbuf[4] = {0};
        for (int j = 0; j < avail; j++) { abuf[j] = a[i + j]; bbuf[j] = b[i + j]; }
        int32x4_t av = vld1q_s32(abuf);
        int32x4_t bv = vld1q_s32(bbuf);
        int64x2_t prod_lo = vmull_s32(vget_low_s32(av),  vget_low_s32(bv));
        int64x2_t prod_hi = vmull_s32(vget_high_s32(av), vget_high_s32(bv));
        uint64x2_t mask_lo = vcltzq_s64(prod_lo);
        uint64x2_t mask_hi = vcltzq_s64(prod_hi);
        int64x2_t abs_lo = vabsq_s64(prod_lo);
        int64x2_t abs_hi = vabsq_s64(prod_hi);
        abs_lo = vaddq_s64(abs_lo, halfv);
        abs_hi = vaddq_s64(abs_hi, halfv);
        uint64x2_t u_lo = vreinterpretq_u64_s64(abs_lo);
        uint64x2_t u_hi = vreinterpretq_u64_s64(abs_hi);
        for (int k = 0; k < shift_exp; k++) {
            u_lo = neon_unsigned_div3_u64x2(u_lo);
            u_hi = neon_unsigned_div3_u64x2(u_hi);
        }
        int64x2_t pos_lo = vreinterpretq_s64_u64(u_lo);
        int64x2_t pos_hi = vreinterpretq_s64_u64(u_hi);
        int64x2_t neg_lo = vnegq_s64(pos_lo);
        int64x2_t neg_hi = vnegq_s64(pos_hi);
        int64x2_t r_lo = vbslq_s64(mask_lo, neg_lo, pos_lo);
        int64x2_t r_hi = vbslq_s64(mask_hi, neg_hi, pos_hi);
        int32x2_t y_lo = vqmovn_s64(r_lo);
        int32x2_t y_hi = vqmovn_s64(r_hi);
        int32x4_t y4 = vcombine_s32(y_lo, y_hi);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        m4t_mtfp_t ybuf[4];
        vst1q_s32(ybuf, y4);
        for (int j = 0; j < avail; j++) y[i + j] = ybuf[j];
    }
#else
#error "m4t_mtfp_elementwise_mul_bx requires NEON; no scalar fallback per project rule."
#endif
}

void m4t_mtfp_elementwise_mul_bx_scalar_ref(
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

/* V14.F: NEON bitlinear_scale_bx. Per-cell |y_raw| × |num| is uint96
 * (since num up to 2^58, y_raw up to 2^29.1, product up to 2^87.3),
 * stored as 3 × uint32 limbs. Long-divide by 127 across limbs (magic
 * mul with extension trick), then iterated /3 across limbs (the same
 * div3 magic used in V14.D/E, applied to each limb). Sign re-applied
 * via vbslq_s64 with mask = sign(y_raw) XOR sign(num). Bit-exact vs
 * scalar_ref. */
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
        m4t_mtfp_rescale_bx(y, y_raw, x_bx, target_bx, n);
        return;
    }
    int shift_exp = alpha_bx + x_bx - target_bx;
    int64_t num = alpha_m * (int64_t)absmax_m;
    int64_t den = 127 * pow3_i64(shift_exp);
    int64_t half = den / 2;

#if M4T_HAS_NEON
    /* Once-per-call scalar setup: signs, magnitudes, /127 magic. */
    int num_neg = (num < 0);
    uint64_t abs_num = num_neg ? (uint64_t)(-num) : (uint64_t)num;
    uint32_t num_lo32 = (uint32_t)abs_num;
    uint32_t num_hi32 = (uint32_t)(abs_num >> 32);

    /* Profile-driven optimization: combine /127 with as many /3s as fit in
     * a single uint32 divisor (d ≤ 2^31). 127 × 3^15 = 1,822,311,189 < 2^31;
     * 127 × 3^16 > 2^31. So combine /127 with up to 15 /3 steps in ONE
     * limb-divide pass; iterate /3 for any remaining shift_exp - 15 steps.
     *
     * For typical BitNet shift_exp ≤ 15: ~16x reduction in long-divide work
     * (1 pass of 3 limb-divides, vs 1 + shift_exp passes previously). */
    int combined_k = (shift_exp < 15) ? shift_exp : 15;
    int remaining_3s = shift_exp - combined_k;
    uint64_t d_combined = 127ULL * (uint64_t)pow3_i64(combined_k);
    uint32_t d_combined_u32 = (uint32_t)d_combined;
    m4t_magic_div_u64_t magic_d = compute_magic_u64(d_combined);

    /* abs(half) splits into uint64; we need to add it to a uint96 V.
     * half ≤ den/2 ≤ (127 × 3^35)/2 ≈ 2^61.6, fits uint64. */
    uint64_t half_u = (uint64_t)half;
    uint64x2_t halfv = vdupq_n_u64(half_u);

    int32x4_t v_max = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    int32x4_t v_min = vdupq_n_s32(-(int32_t)M4T_MTFP_MAX_VAL);

    int n_aligned = n - (n % 4);
    int i = 0;
    for (; i < n_aligned; i += 4) {
        int32x4_t yv = vld1q_s32(y_raw + i);
        /* Sign mask of result: sign(y_raw) XOR num_neg. Compute as int32x4
         * (-1 where the result should be negated). */
        uint32x4_t y_sign = vcltzq_s32(yv);
        uint32x4_t num_sign = vdupq_n_u32(num_neg ? 0xFFFFFFFFu : 0u);
        uint32x4_t result_sign = veorq_u32(y_sign, num_sign);
        /* |y_raw| as uint32x4 (since |yv| ≤ MAX_VAL fits int32, abs is safe). */
        uint32x4_t y_abs = vreinterpretq_u32_s32(vabsq_s32(yv));

        /* Process 2 cells at a time (2 NEON lanes). */
        for (int half_lane = 0; half_lane < 2; half_lane++) {
            uint32x2_t y_pair = (half_lane == 0)
                ? vget_low_u32(y_abs) : vget_high_u32(y_abs);

            /* Multiply: |prod| = y_pair * abs_num as 96-bit per lane.
             * P_lo = y_pair * num_lo32 (uint64x2)
             * P_hi = y_pair * num_hi32 (uint64x2)
             * 96-bit V = P_hi << 32 + P_lo, decomposed into uint32x2 limbs. */
            uint64x2_t P_lo = vmull_u32(y_pair, vdup_n_u32(num_lo32));
            uint64x2_t P_hi = vmull_u32(y_pair, vdup_n_u32(num_hi32));
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P_lo);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P_lo, 32),
                                       vandq_u64(P_hi, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P_hi, 32),
                                                 vshrq_n_u64(mid, 32)));

            /* Add half (uint64) to V (96-bit + 64-bit). */
            uint64x2_t V_lo64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                          vmovl_u32(L0));
            uint64x2_t V_lo64_new = vaddq_u64(V_lo64, halfv);
            uint64x2_t add_carry = vandq_u64(vcltq_u64(V_lo64_new, halfv),
                                             vdupq_n_u64(1));
            L0 = vmovn_u64(V_lo64_new);
            L1 = vshrn_n_u64(V_lo64_new, 32);
            L2 = vmovn_u64(vaddq_u64(vmovl_u32(L2), add_carry));

            /* Combined long-divide V by d_combined (= 127 × 3^combined_k).
             * Single pass replaces /127 + combined_k iterations of /3.
             * r ∈ [0, d_combined-1]; val = r*2^32 + limb < d_combined*2^32 ≤ 2^63. */
            uint64x2_t r = vdupq_n_u64(0);
            #define LIMB_DIV_D(limb_var) do { \
                uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                uint64x2_t q   = neon_apply_magic_u64x2(val, magic_d); \
                /* r = val - q * d_combined. q < 2^32, d_combined < 2^31, product fits uint64. */ \
                uint64x2_t qd  = vmull_u32(vmovn_u64(q), vdup_n_u32(d_combined_u32)); \
                r = vsubq_u64(val, qd); \
                limb_var = vmovn_u64(q); \
            } while(0)
            LIMB_DIV_D(L2);
            LIMB_DIV_D(L1);
            LIMB_DIV_D(L0);
            #undef LIMB_DIV_D

            /* Iterated long-divide by 3 for remaining (shift_exp - combined_k) steps. */
            for (int k = 0; k < remaining_3s; k++) {
                r = vdupq_n_u64(0);
                #define LIMB_DIV_3(limb_var) do { \
                    uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                    uint64x2_t q   = neon_unsigned_div3_u64x2(val); \
                    uint64x2_t q3  = vaddq_u64(vshlq_n_u64(q, 1), q); \
                    r = vsubq_u64(val, q3); \
                    limb_var = vmovn_u64(q); \
                } while(0)
                LIMB_DIV_3(L2);
                LIMB_DIV_3(L1);
                LIMB_DIV_3(L0);
                #undef LIMB_DIV_3
            }

            /* Result: V[2..0]. After enough divisions, V[2] should be 0
             * (or very small). Combine into uint64 for sign + clamp.
             * If V[2] > 0, the result exceeds uint64 range — for our
             * bounds this means saturation past MTFP19_MAX in any case. */
            uint64x2_t result_u64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                              vmovl_u32(L0));
            /* If L2 is non-zero, force result to UINT64_MAX (will saturate). */
            /* L2-nonzero saturation: build a 64-bit mask via sign-extension
             * (vmovl_u32 ZERO-extends, leaving 0x00000000FFFFFFFF for "true",
             * which corrupts vbslq lane selection). */
            uint32x2_t l2_eq0_u32 = vceq_u32(L2, vdup_n_u32(0));
            uint64x2_t l2_eq0 = vreinterpretq_u64_s64(
                vmovl_s32(vreinterpret_s32_u32(l2_eq0_u32)));
            uint64x2_t saturated = vbslq_u64(l2_eq0, result_u64,
                                             vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL));
            int64x2_t pos = vreinterpretq_s64_u64(saturated);
            int64x2_t neg = vnegq_s64(pos);

            /* Apply sign mask. result_sign is uint32x4; pick the right pair
             * and sign-extend (NOT zero-extend) to uint64x2 mask. */
            uint32x2_t sign_pair_u32 = (half_lane == 0)
                ? vget_low_u32(result_sign) : vget_high_u32(result_sign);
            uint64x2_t mask64 = vreinterpretq_u64_s64(
                vmovl_s32(vreinterpret_s32_u32(sign_pair_u32)));
            int64x2_t r64_signed = vbslq_s64(mask64, neg, pos);

            /* Clamp + narrow → int32x2. */
            int32x2_t y_pair_out = vqmovn_s64(r64_signed);
            int32x4_t y4_full = (half_lane == 0)
                ? vcombine_s32(y_pair_out, vdup_n_s32(0))
                : vcombine_s32(vdup_n_s32(0), y_pair_out);
            (void)y4_full;
            /* Stage to a small array; combine both halves below. */
            if (half_lane == 0) vst1_s32(y + i,     y_pair_out);
            else                vst1_s32(y + i + 2, y_pair_out);
        }
        /* Apply final MTFP19 clamp on the 4 stored values. */
        int32x4_t y4 = vld1q_s32(y + i);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(y + i, y4);
    }
    /* Boundary tile: same processing on stack-buffered partial chunk. */
    if (i < n) {
        int avail = n - i;
        m4t_mtfp_t ybuf_in[4] = {0};
        m4t_mtfp_t ybuf_out[4] = {0};
        for (int j = 0; j < avail; j++) ybuf_in[j] = y_raw[i + j];

        int32x4_t yv = vld1q_s32(ybuf_in);
        uint32x4_t y_sign = vcltzq_s32(yv);
        uint32x4_t num_sign = vdupq_n_u32(num_neg ? 0xFFFFFFFFu : 0u);
        uint32x4_t result_sign = veorq_u32(y_sign, num_sign);
        uint32x4_t y_abs = vreinterpretq_u32_s32(vabsq_s32(yv));

        for (int half_lane = 0; half_lane < 2; half_lane++) {
            uint32x2_t y_pair = (half_lane == 0)
                ? vget_low_u32(y_abs) : vget_high_u32(y_abs);
            uint64x2_t P_lo = vmull_u32(y_pair, vdup_n_u32(num_lo32));
            uint64x2_t P_hi = vmull_u32(y_pair, vdup_n_u32(num_hi32));
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P_lo);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P_lo, 32),
                                       vandq_u64(P_hi, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P_hi, 32),
                                                 vshrq_n_u64(mid, 32)));

            uint64x2_t V_lo64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                          vmovl_u32(L0));
            uint64x2_t V_lo64_new = vaddq_u64(V_lo64, halfv);
            uint64x2_t add_carry = vandq_u64(vcltq_u64(V_lo64_new, halfv),
                                             vdupq_n_u64(1));
            L0 = vmovn_u64(V_lo64_new);
            L1 = vshrn_n_u64(V_lo64_new, 32);
            L2 = vmovn_u64(vaddq_u64(vmovl_u32(L2), add_carry));

            uint64x2_t r = vdupq_n_u64(0);
            #define LIMB_DIV_DT(limb_var) do { \
                uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                uint64x2_t q   = neon_apply_magic_u64x2(val, magic_d); \
                uint64x2_t qd  = vmull_u32(vmovn_u64(q), vdup_n_u32(d_combined_u32)); \
                r = vsubq_u64(val, qd); \
                limb_var = vmovn_u64(q); \
            } while(0)
            LIMB_DIV_DT(L2);
            LIMB_DIV_DT(L1);
            LIMB_DIV_DT(L0);
            #undef LIMB_DIV_DT

            for (int k = 0; k < remaining_3s; k++) {
                r = vdupq_n_u64(0);
                #define LIMB_DIV_3T(limb_var) do { \
                    uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                    uint64x2_t q   = neon_unsigned_div3_u64x2(val); \
                    uint64x2_t q3  = vaddq_u64(vshlq_n_u64(q, 1), q); \
                    r = vsubq_u64(val, q3); \
                    limb_var = vmovn_u64(q); \
                } while(0)
                LIMB_DIV_3T(L2);
                LIMB_DIV_3T(L1);
                LIMB_DIV_3T(L0);
                #undef LIMB_DIV_3T
            }

            uint64x2_t result_u64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                              vmovl_u32(L0));
            /* Sign-extend the L2-eq-0 mask (vmovl_u32 ZERO-extends, breaks vbslq). */
            uint32x2_t l2_eq0_u32 = vceq_u32(L2, vdup_n_u32(0));
            uint64x2_t l2_eq0 = vreinterpretq_u64_s64(
                vmovl_s32(vreinterpret_s32_u32(l2_eq0_u32)));
            uint64x2_t saturated = vbslq_u64(l2_eq0, result_u64,
                                             vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL));
            int64x2_t pos = vreinterpretq_s64_u64(saturated);
            int64x2_t neg = vnegq_s64(pos);
            uint32x2_t sign_pair_u32 = (half_lane == 0)
                ? vget_low_u32(result_sign) : vget_high_u32(result_sign);
            uint64x2_t mask64 = vreinterpretq_u64_s64(
                vmovl_s32(vreinterpret_s32_u32(sign_pair_u32)));
            int64x2_t r64_signed = vbslq_s64(mask64, neg, pos);
            int32x2_t y_pair_out = vqmovn_s64(r64_signed);
            if (half_lane == 0) vst1_s32(ybuf_out,     y_pair_out);
            else                vst1_s32(ybuf_out + 2, y_pair_out);
        }
        int32x4_t y4 = vld1q_s32(ybuf_out);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(ybuf_out, y4);
        for (int j = 0; j < avail; j++) y[i + j] = ybuf_out[j];
    }
#else
#error "m4t_mtfp_bitlinear_scale_bx requires NEON; no scalar fallback per project rule."
#endif
}

void m4t_mtfp_bitlinear_scale_bx_scalar_ref(
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
        m4t_mtfp_rescale_bx(y, y_raw, x_bx, target_bx, n);
        return;
    }
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

/* Bit-faithful BitLinear scale (no a8 quantization). Input y_raw is int64
 * (from m4t_mtfp_ternary_matmul_bt_route_i64); multiplier is α only (no
 * absmax, no /127). Per-cell:
 *   |prod| = |y_raw_i64| × |α_m|, decomposed as uint96 (3 × uint32 limbs).
 *   add half = (3^shift_exp)/2 to |prod|.
 *   long-divide by d_combined = 3^min(shift_exp, 19) (single pass),
 *     then iterated /3 for max(0, shift_exp - 19) more steps.
 *   re-sign by sign(y_raw) XOR sign(α), clamp to MTFP19. */
void m4t_mtfp_bitlinear_scale_no_a8_bx(
    m4t_mtfp_t* y, const int64_t* y_raw_i64,
    const m4t_mtfp_t* alpha_ptr, int alpha_bx,
    int x_bx, int target_bx,
    int n)
{
    if (n <= 0) return;
    assert(y && y_raw_i64 && alpha_ptr);
    assert(alpha_bx + x_bx - target_bx >= 0);
    assert(alpha_bx + x_bx - target_bx <= 35);

    int64_t alpha_m = (int64_t)(*alpha_ptr);
    if (alpha_m == 0) {
        memset(y, 0, (size_t)n * sizeof(m4t_mtfp_t));
        return;
    }
    int shift_exp = alpha_bx + x_bx - target_bx;
    int64_t den = pow3_i64(shift_exp);
    int64_t half = den / 2;

#if M4T_HAS_NEON
    int alpha_neg = (alpha_m < 0);
    uint32_t abs_alpha = alpha_neg ? (uint32_t)(-alpha_m) : (uint32_t)alpha_m;

    /* Combined divisor for the long-divide. d_combined = 3^min(shift_exp, 19),
     * which fits uint32 (3^19 ≈ 1.16e9 < 2^31). For shift_exp ≤ 19, single
     * pass; remaining iterated /3 for shift_exp > 19. */
    int combined_k = (shift_exp < 19) ? shift_exp : 19;
    int remaining_3s = shift_exp - combined_k;
    uint64_t d_combined;
    if (combined_k == 0) {
        d_combined = 1;  /* identity, handled specially below */
    } else {
        d_combined = (uint64_t)pow3_i64(combined_k);
    }
    uint32_t d_combined_u32 = (uint32_t)d_combined;
    m4t_magic_div_u64_t magic_d;
    if (combined_k > 0) {
        magic_d = compute_magic_u64(d_combined);
    } else {
        memset(&magic_d, 0, sizeof(magic_d));
    }

    uint64_t half_u = (uint64_t)half;
    uint64x2_t halfv = vdupq_n_u64(half_u);

    int32x4_t v_max = vdupq_n_s32( M4T_MTFP_MAX_VAL);
    int32x4_t v_min = vdupq_n_s32(-(int32_t)M4T_MTFP_MAX_VAL);
    int32x4_t alpha_sign_v = vdupq_n_s32(alpha_neg ? 0xFFFFFFFFu : 0u);

    int n_aligned = n - (n % 4);
    int i = 0;
    for (; i < n_aligned; i += 4) {
        /* Sign of y_raw per lane (load 2 int64x2 vectors for 4 cells). */
        int64x2_t y_lo = vld1q_s64(y_raw_i64 + i);
        int64x2_t y_hi = vld1q_s64(y_raw_i64 + i + 2);
        uint64x2_t s_lo = vcltzq_s64(y_lo);
        uint64x2_t s_hi = vcltzq_s64(y_hi);
        uint64x2_t a_lo = vreinterpretq_u64_s64(vabsq_s64(y_lo));
        uint64x2_t a_hi = vreinterpretq_u64_s64(vabsq_s64(y_hi));

        /* Process 2 cells at a time. */
        for (int half_lane = 0; half_lane < 2; half_lane++) {
            uint64x2_t a = (half_lane == 0) ? a_lo : a_hi;
            uint64x2_t s = (half_lane == 0) ? s_lo : s_hi;

            /* |y_raw_i64| × |α|: uint64 × uint32 = uint96.
             * a split into a_lo32, a_hi32. Each multiplied by abs_alpha (uint32). */
            uint32x2_t a_lo32 = vmovn_u64(a);
            uint32x2_t a_hi32 = vshrn_n_u64(a, 32);
            uint64x2_t P_lo = vmull_u32(a_lo32, vdup_n_u32(abs_alpha));
            uint64x2_t P_hi = vmull_u32(a_hi32, vdup_n_u32(abs_alpha));
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P_lo);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P_lo, 32),
                                       vandq_u64(P_hi, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P_hi, 32),
                                                 vshrq_n_u64(mid, 32)));

            /* Add half (uint64) to V. */
            uint64x2_t V_lo64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                          vmovl_u32(L0));
            uint64x2_t V_lo64_new = vaddq_u64(V_lo64, halfv);
            uint64x2_t add_carry = vandq_u64(vcltq_u64(V_lo64_new, halfv),
                                             vdupq_n_u64(1));
            L0 = vmovn_u64(V_lo64_new);
            L1 = vshrn_n_u64(V_lo64_new, 32);
            L2 = vmovn_u64(vaddq_u64(vmovl_u32(L2), add_carry));

            /* Combined long-divide by d_combined (single pass). Skip if
             * shift_exp = 0 (d_combined = 1, identity). */
            uint64x2_t r = vdupq_n_u64(0);
            if (combined_k > 0) {
                #define LIMB_DIV_D_NA8(limb_var) do { \
                    uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                    uint64x2_t q   = neon_apply_magic_u64x2(val, magic_d); \
                    uint64x2_t qd  = vmull_u32(vmovn_u64(q), vdup_n_u32(d_combined_u32)); \
                    r = vsubq_u64(val, qd); \
                    limb_var = vmovn_u64(q); \
                } while (0)
                LIMB_DIV_D_NA8(L2);
                LIMB_DIV_D_NA8(L1);
                LIMB_DIV_D_NA8(L0);
                #undef LIMB_DIV_D_NA8
            }

            /* Iterated /3 for remaining steps. */
            for (int k = 0; k < remaining_3s; k++) {
                r = vdupq_n_u64(0);
                #define LIMB_DIV_3_NA8(limb_var) do { \
                    uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                    uint64x2_t q   = neon_unsigned_div3_u64x2(val); \
                    uint64x2_t q3  = vaddq_u64(vshlq_n_u64(q, 1), q); \
                    r = vsubq_u64(val, q3); \
                    limb_var = vmovn_u64(q); \
                } while (0)
                LIMB_DIV_3_NA8(L2);
                LIMB_DIV_3_NA8(L1);
                LIMB_DIV_3_NA8(L0);
                #undef LIMB_DIV_3_NA8
            }

            /* Combine result. If L2 != 0, force saturation (result exceeds uint64). */
            uint64x2_t result_u64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                              vmovl_u32(L0));
            uint32x2_t l2_eq0_u32 = vceq_u32(L2, vdup_n_u32(0));
            uint64x2_t l2_eq0 = vreinterpretq_u64_s64(
                vmovl_s32(vreinterpret_s32_u32(l2_eq0_u32)));
            uint64x2_t saturated = vbslq_u64(l2_eq0, result_u64,
                                             vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL));
            int64x2_t pos = vreinterpretq_s64_u64(saturated);
            int64x2_t neg = vnegq_s64(pos);

            /* Apply sign mask: sign(y_raw) XOR sign(α). */
            uint64x2_t sign_pair = veorq_u64(s,
                vreinterpretq_u64_s64(vmovl_s32(
                    vreinterpret_s32_u32(half_lane == 0
                        ? vget_low_u32(vreinterpretq_u32_s32(alpha_sign_v))
                        : vget_high_u32(vreinterpretq_u32_s32(alpha_sign_v))))));
            int64x2_t r64_signed = vbslq_s64(sign_pair, neg, pos);

            /* Clamp + narrow → int32x2. */
            int32x2_t y_pair = vqmovn_s64(r64_signed);
            if (half_lane == 0) vst1_s32(y + i,     y_pair);
            else                vst1_s32(y + i + 2, y_pair);
        }
        int32x4_t y4 = vld1q_s32(y + i);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(y + i, y4);
    }
    /* Boundary tile: same processing for n%4 != 0 cells. */
    if (i < n) {
        int avail = n - i;
        int64_t  ybuf_in[4]  = {0, 0, 0, 0};
        m4t_mtfp_t ybuf_out[4] = {0};
        for (int j = 0; j < avail; j++) ybuf_in[j] = y_raw_i64[i + j];

        int64x2_t y_lo = vld1q_s64(ybuf_in);
        int64x2_t y_hi = vld1q_s64(ybuf_in + 2);
        uint64x2_t s_lo = vcltzq_s64(y_lo);
        uint64x2_t s_hi = vcltzq_s64(y_hi);
        uint64x2_t a_lo = vreinterpretq_u64_s64(vabsq_s64(y_lo));
        uint64x2_t a_hi = vreinterpretq_u64_s64(vabsq_s64(y_hi));
        for (int half_lane = 0; half_lane < 2; half_lane++) {
            uint64x2_t a = (half_lane == 0) ? a_lo : a_hi;
            uint64x2_t s = (half_lane == 0) ? s_lo : s_hi;
            uint32x2_t a_lo32 = vmovn_u64(a);
            uint32x2_t a_hi32 = vshrn_n_u64(a, 32);
            uint64x2_t P_lo = vmull_u32(a_lo32, vdup_n_u32(abs_alpha));
            uint64x2_t P_hi = vmull_u32(a_hi32, vdup_n_u32(abs_alpha));
            uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
            uint32x2_t L0 = vmovn_u64(P_lo);
            uint64x2_t mid = vaddq_u64(vshrq_n_u64(P_lo, 32),
                                       vandq_u64(P_hi, mask32));
            uint32x2_t L1 = vmovn_u64(mid);
            uint32x2_t L2 = vmovn_u64(vaddq_u64(vshrq_n_u64(P_hi, 32),
                                                 vshrq_n_u64(mid, 32)));
            uint64x2_t V_lo64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                          vmovl_u32(L0));
            uint64x2_t V_lo64_new = vaddq_u64(V_lo64, halfv);
            uint64x2_t add_carry = vandq_u64(vcltq_u64(V_lo64_new, halfv),
                                             vdupq_n_u64(1));
            L0 = vmovn_u64(V_lo64_new);
            L1 = vshrn_n_u64(V_lo64_new, 32);
            L2 = vmovn_u64(vaddq_u64(vmovl_u32(L2), add_carry));

            uint64x2_t r = vdupq_n_u64(0);
            if (combined_k > 0) {
                #define LIMB_DIV_D_NA8T(limb_var) do { \
                    uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                    uint64x2_t q   = neon_apply_magic_u64x2(val, magic_d); \
                    uint64x2_t qd  = vmull_u32(vmovn_u64(q), vdup_n_u32(d_combined_u32)); \
                    r = vsubq_u64(val, qd); \
                    limb_var = vmovn_u64(q); \
                } while (0)
                LIMB_DIV_D_NA8T(L2);
                LIMB_DIV_D_NA8T(L1);
                LIMB_DIV_D_NA8T(L0);
                #undef LIMB_DIV_D_NA8T
            }
            for (int k = 0; k < remaining_3s; k++) {
                r = vdupq_n_u64(0);
                #define LIMB_DIV_3_NA8T(limb_var) do { \
                    uint64x2_t val = vaddq_u64(vshlq_n_u64(r, 32), vmovl_u32(limb_var)); \
                    uint64x2_t q   = neon_unsigned_div3_u64x2(val); \
                    uint64x2_t q3  = vaddq_u64(vshlq_n_u64(q, 1), q); \
                    r = vsubq_u64(val, q3); \
                    limb_var = vmovn_u64(q); \
                } while (0)
                LIMB_DIV_3_NA8T(L2);
                LIMB_DIV_3_NA8T(L1);
                LIMB_DIV_3_NA8T(L0);
                #undef LIMB_DIV_3_NA8T
            }
            uint64x2_t result_u64 = vorrq_u64(vshlq_n_u64(vmovl_u32(L1), 32),
                                              vmovl_u32(L0));
            uint32x2_t l2_eq0_u32 = vceq_u32(L2, vdup_n_u32(0));
            uint64x2_t l2_eq0 = vreinterpretq_u64_s64(
                vmovl_s32(vreinterpret_s32_u32(l2_eq0_u32)));
            uint64x2_t saturated = vbslq_u64(l2_eq0, result_u64,
                                             vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL));
            int64x2_t pos = vreinterpretq_s64_u64(saturated);
            int64x2_t neg = vnegq_s64(pos);
            uint64x2_t sign_pair = veorq_u64(s,
                vreinterpretq_u64_s64(vmovl_s32(
                    vreinterpret_s32_u32(half_lane == 0
                        ? vget_low_u32(vreinterpretq_u32_s32(alpha_sign_v))
                        : vget_high_u32(vreinterpretq_u32_s32(alpha_sign_v))))));
            int64x2_t r64_signed = vbslq_s64(sign_pair, neg, pos);
            int32x2_t y_pair = vqmovn_s64(r64_signed);
            if (half_lane == 0) vst1_s32(ybuf_out,     y_pair);
            else                vst1_s32(ybuf_out + 2, y_pair);
        }
        int32x4_t y4 = vld1q_s32(ybuf_out);
        y4 = vminq_s32(y4, v_max);
        y4 = vmaxq_s32(y4, v_min);
        vst1q_s32(ybuf_out, y4);
        for (int j = 0; j < avail; j++) y[i + j] = ybuf_out[j];
    }
#else
#error "m4t_mtfp_bitlinear_scale_no_a8_bx requires NEON; no scalar fallback per project rule."
#endif
}

void m4t_mtfp_bitlinear_scale_no_a8_bx_scalar_ref(
    m4t_mtfp_t* y, const int64_t* y_raw_i64,
    const m4t_mtfp_t* alpha_ptr, int alpha_bx,
    int x_bx, int target_bx,
    int n)
{
    if (n <= 0) return;
    assert(y && y_raw_i64 && alpha_ptr);
    int64_t alpha_m = (int64_t)(*alpha_ptr);
    if (alpha_m == 0) { memset(y, 0, (size_t)n * sizeof(m4t_mtfp_t)); return; }
    int shift_exp = alpha_bx + x_bx - target_bx;
    int64_t den = pow3_i64(shift_exp);
    __int128 half = (__int128)den / 2;
    for (int i = 0; i < n; i++) {
        __int128 prod = (__int128)y_raw_i64[i] * (__int128)alpha_m;
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
