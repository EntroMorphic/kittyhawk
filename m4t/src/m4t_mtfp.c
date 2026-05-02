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

/* Set the SATURATED and/or ROUNDED bit for cell `i` in the per-block
 * flags array. Layout per m4t_mtfp.h: bits 0-1 cell 0, bits 2-3 cell 1,
 * etc. within each byte; one byte per 4-cell MTFP19 block. */
static inline void m4t_flag_or(uint8_t* flags, int i, uint8_t event_bits) {
    int block = i / M4T_MTFP_CELLS_PER_BLOCK;
    int slot  = i % M4T_MTFP_CELLS_PER_BLOCK;
    flags[block] |= (uint8_t)(event_bits << (slot * 2));
}

void m4t_mtfp_vec_accum_aligning(
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
        /* Same-block-exp accumulation. No rescale, no rounding. */
        for (int i = 0; i < n; i++) {
            int64_t sum = (int64_t)running[i] + (int64_t)addend[i];
            m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
            if (flags && sum != (int64_t)out) {
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
