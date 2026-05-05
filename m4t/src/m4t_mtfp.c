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

/* Scalar divide loop. Used by m4t_mtfp_shift3_scalar_ref directly and as
 * the non-NEON fallback inside m4t_mtfp_shift3. abs_k ∈ [1, 19]. */
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
         * already overflows MTFP19; collapse to saturation. */
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
