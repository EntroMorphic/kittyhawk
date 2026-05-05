/*
 * audit/b2b_matmul.c — strong-claim comparison kernels (NEON only).
 *
 * Three packed-W matmul kernels for the L1 strong-claim test. All three
 * NEON-only with no scalar fallback. K must be a multiple of 16.
 *
 * Decode patterns mirror the substrate's m4t_ternary_matmul vmlal pipeline
 * (DUP/SHIFT/MASK to expand 4 packed bytes into 16 codes).
 */

#include "b2b_matmul.h"

#include <arm_neon.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/* ── Common decode constants (mirror substrate m4t_ternary_matmul.c) ───── */

static const uint8_t DUP_IDX[16] __attribute__((aligned(16))) = {
    0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3
};
static const uint8_t SHIFT_LANE[16] __attribute__((aligned(16))) = {
    0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6
};

/* Ternary code → signed value LUT (matches substrate's M4T_TRIT_DECODE_LUT).
 * 0b00→0, 0b01→+1, 0b10→-1, 0b11→0. Pattern repeats 4× for vqtbl1q_s8. */
static const int8_t TERNARY_LUT[16] __attribute__((aligned(16))) = {
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
};

/* B2-B sign-bit → ±1 LUT. Indexed by sign_bit ∈ {0, 1}.
 * sign=0 → +1, sign=1 → -1. */
static const int8_t B2B_SIGN_LUT[16] __attribute__((aligned(16))) = {
     1, -1,  1, -1,
     1, -1,  1, -1,
     1, -1,  1, -1,
     1, -1,  1, -1,
};

/* 5-in-8 trit-decode LUT (Path D). Indexed by digit value ∈ {0, 1, 2}:
 *   0 → 0
 *   1 → +1
 *   2 → -1
 * Byte storage convention: trit_to_unsigned: -1 → 2, 0 → 0, +1 → 1.
 * Pattern repeats 4× for vqtbl1q_s8 against any 16-byte digit vector
 * (lanes hold values in [0, 2]; lane 3+ is unused in our codepath). */
static const int8_t TRIT5_DECODE_LUT[16] __attribute__((aligned(16))) = {
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
};

/* B2-B unified value LUT (Path C). Indexed by 2-bit B2-B code:
 *   0b00 (mask=0, sign=0) → +1
 *   0b01 (mask=0, sign=1) → -1
 *   0b10 (mask=1, sign=0) → 0
 *   0b11 (mask=1, sign=1) → 0
 * Pattern repeats 4× for vqtbl1q_s8. Per red-team C1: this LUT collapses
 * the "honest" sign+mask decode into a single TBL lookup, equivalent to
 * Path A in op shape. */
static const int8_t B2B_OPTIMAL_LUT[16] __attribute__((aligned(16))) = {
     1, -1,  0,  0,
     1, -1,  0,  0,
     1, -1,  0,  0,
     1, -1,  0,  0,
};

/* ── Pack utilities ────────────────────────────────────────────────────── */

void base3_pack(uint8_t* dst, const int8_t* src, int n) {
    int nb = (n + 3) / 4;
    memset(dst, 0, (size_t)nb);
    for (int i = 0; i < n; i++) {
        int8_t t = src[i];
        uint8_t code;
        if      (t ==  1) code = 0x01u;
        else if (t == -1) code = 0x02u;
        else              code = 0x00u;
        dst[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

void base3_5in8_pack(uint8_t* dst, const int8_t* src, int n) {
    /* 5 trits per byte. dst has ceil(n / 5) bytes. */
    int nb = (n + 4) / 5;
    memset(dst, 0, (size_t)nb);
    for (int i = 0; i < n; i++) {
        int8_t t = src[i];
        uint8_t u;
        if      (t ==  1) u = 1;
        else if (t == -1) u = 2;
        else              u = 0;  /* t == 0 */
        /* Position within byte. Power of 3 for digit. */
        int byte_idx  = i / 5;
        int digit_pos = i % 5;
        static const uint8_t POW3[5] = { 1, 3, 9, 27, 81 };
        dst[byte_idx] = (uint8_t)(dst[byte_idx] + u * POW3[digit_pos]);
    }
}

void b2b_pack(uint8_t* dst, const int8_t* src, int n) {
    int nb = (n + 3) / 4;
    memset(dst, 0, (size_t)nb);
    for (int i = 0; i < n; i++) {
        int8_t t = src[i];
        uint8_t code;
        if      (t ==  1) code = 0x00u;  /* mask=0, sign=0 */
        else if (t == -1) code = 0x01u;  /* mask=0, sign=1 */
        else              code = 0x02u;  /* mask=1, sign=0 → 0 */
        dst[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

/* ── Path A: base-3 packed via SDOT ────────────────────────────────────── */

__attribute__((noinline))
void base3_packed_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K % 16 == 0);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_packed)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_packed);

    int Kp = (K + 3) / 4;

    const uint8x16_t dup_idx = vld1q_u8(DUP_IDX);
    const int8x16_t  shift_s = vreinterpretq_s8_u8(vld1q_u8(SHIFT_LANE));
    const uint8x16_t mask_03 = vdupq_n_u8(0x03u);
    const int8x16_t  lut     = vld1q_s8(TERNARY_LUT);

    for (int i = 0; i < M; i++) {
        const int8_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_packed + (size_t)j * Kp;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < K; k += 16) {
                /* Inner-block decode + SDOT. ~7 NEON ops. */
                uint32_t w32;
                memcpy(&w32, wj + (k >> 2), 4);
                uint8x16_t packed  = vreinterpretq_u8_u32(vdupq_n_u32(w32));
                uint8x16_t dup     = vqtbl1q_u8(packed, dup_idx);
                uint8x16_t shifted = vshlq_u8(dup, vnegq_s8(shift_s));
                uint8x16_t codes   = vandq_u8(shifted, mask_03);
                int8x16_t  w_vec   = vqtbl1q_s8(lut, codes);

                int8x16_t  x_vec   = vld1q_s8(xi + k);
                acc = vdotq_s32(acc, x_vec, w_vec);
            }

            Y[(size_t)i * N + j] = vaddvq_s32(acc);
        }
    }
}

/* ── Path B: B2-B honest separate sign + mask decode ───────────────────── */

__attribute__((noinline))
void b2b_honest_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_b2b,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K % 16 == 0);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_b2b)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_b2b);

    int Kp = (K + 3) / 4;

    const uint8x16_t dup_idx  = vld1q_u8(DUP_IDX);
    const int8x16_t  shift_s  = vreinterpretq_s8_u8(vld1q_u8(SHIFT_LANE));
    const uint8x16_t mask_03  = vdupq_n_u8(0x03u);
    const uint8x16_t mask_01  = vdupq_n_u8(0x01u);
    const int8x16_t  sign_lut = vld1q_s8(B2B_SIGN_LUT);

    for (int i = 0; i < M; i++) {
        const int8_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_b2b + (size_t)j * Kp;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < K; k += 16) {
                /* Inner-block decode (separate sign + mask) + SDOT. ~11 ops. */
                uint32_t w32;
                memcpy(&w32, wj + (k >> 2), 4);
                uint8x16_t packed  = vreinterpretq_u8_u32(vdupq_n_u32(w32));
                uint8x16_t dup     = vqtbl1q_u8(packed, dup_idx);
                uint8x16_t shifted = vshlq_u8(dup, vnegq_s8(shift_s));
                uint8x16_t codes   = vandq_u8(shifted, mask_03);

                /* Extract sign bit (LSB) and mask bit (bit 1) separately. */
                uint8x16_t sign_b  = vandq_u8(codes, mask_01);
                uint8x16_t mask_b  = vshrq_n_u8(codes, 1);  /* {0, 1} */

                /* Decode sign bit to ±1 via TBL. */
                int8x16_t  sign_v  = vqtbl1q_s8(sign_lut, sign_b);

                /* Compute multiplier: 1 - mask_b ∈ {0, 1}. */
                int8x16_t  mult    = vsubq_s8(vdupq_n_s8(1),
                                              vreinterpretq_s8_u8(mask_b));

                /* Apply: w = sign_v * mult. Where masked, w=0; else ±1. */
                int8x16_t  w_vec   = vmulq_s8(sign_v, mult);

                int8x16_t  x_vec   = vld1q_s8(xi + k);
                acc = vdotq_s32(acc, x_vec, w_vec);
            }

            Y[(size_t)i * N + j] = vaddvq_s32(acc);
        }
    }
}

/* ── Path B': B2-B with all-masked-block skip ──────────────────────────── */

__attribute__((noinline))
void b2b_skip_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_b2b,
    int M, int K, int N,
    int* skip_count_out)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K % 16 == 0);
    if (M == 0 || N == 0) {
        if (skip_count_out) *skip_count_out = 0;
        return;
    }
    assert(Y && (K == 0 || (X && W_b2b)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_b2b);

    int Kp = (K + 3) / 4;
    int skip_count = 0;

    const uint8x16_t dup_idx  = vld1q_u8(DUP_IDX);
    const int8x16_t  shift_s  = vreinterpretq_s8_u8(vld1q_u8(SHIFT_LANE));
    const uint8x16_t mask_03  = vdupq_n_u8(0x03u);
    const uint8x16_t mask_01  = vdupq_n_u8(0x01u);
    const int8x16_t  sign_lut = vld1q_s8(B2B_SIGN_LUT);

    for (int i = 0; i < M; i++) {
        const int8_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_b2b + (size_t)j * Kp;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < K; k += 16) {
                /* Block decode head: extract codes → mask bits. */
                uint32_t w32;
                memcpy(&w32, wj + (k >> 2), 4);
                uint8x16_t packed  = vreinterpretq_u8_u32(vdupq_n_u32(w32));
                uint8x16_t dup     = vqtbl1q_u8(packed, dup_idx);
                uint8x16_t shifted = vshlq_u8(dup, vnegq_s8(shift_s));
                uint8x16_t codes   = vandq_u8(shifted, mask_03);
                uint8x16_t mask_b  = vshrq_n_u8(codes, 1);

                /* Skip check: if all 16 mask bits are 1, every cell is
                 * masked (contributes 0). Sum mask_b lanes — if equals
                 * 16, skip the rest of the block. */
                if (vaddvq_u8(mask_b) == 16) {
                    skip_count++;
                    continue;
                }

                /* Otherwise: full decode + SDOT (same as honest path). */
                uint8x16_t sign_b = vandq_u8(codes, mask_01);
                int8x16_t  sign_v = vqtbl1q_s8(sign_lut, sign_b);
                int8x16_t  mult   = vsubq_s8(vdupq_n_s8(1),
                                             vreinterpretq_s8_u8(mask_b));
                int8x16_t  w_vec  = vmulq_s8(sign_v, mult);

                int8x16_t  x_vec  = vld1q_s8(xi + k);
                acc = vdotq_s32(acc, x_vec, w_vec);
            }

            Y[(size_t)i * N + j] = vaddvq_s32(acc);
        }
    }

    if (skip_count_out) *skip_count_out = skip_count;
}

/* ── Path D: base-3 5-trits-in-8-bits packing (sub-2-bits/cell) ────────── */

__attribute__((noinline))
void base3_5in8_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_packed_5in8,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K % 80 == 0);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_packed_5in8)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_packed_5in8);

    int Kp = K / 5;  /* bytes per row */
    int K5 = K / 5;  /* length of each strided X array */

    const int8x16_t  lut       = vld1q_s8(TRIT5_DECODE_LUT);
    const uint8x16_t three_v   = vdupq_n_u8(3);

    /* P0-1 optimization: pre-permute X[i, :] into 5 strided arrays once per
     * row i. The inner-loop SDOT then uses contiguous vld1q_s8 against
     * X_strided[d] instead of the prior vqtbl4q + vqtbl1q + vbslq gather.
     * Permutation cost: K bytes/row (amortized across N j-iterations) =
     * O(1/N) ≈ 1.6% overhead at N=64. */
    int8_t* X_strided = (int8_t*)malloc((size_t)K);
    if (!X_strided) return;  /* defensive; K is small so this rarely fails */
    int8_t* X_d[5];
    for (int d = 0; d < 5; d++) {
        X_d[d] = X_strided + (size_t)d * K5;
    }

    for (int i = 0; i < M; i++) {
        const int8_t* xi = X + (size_t)i * K;

        /* Permute this row of X into 5 stride-aligned arrays. */
        for (int n = 0; n < K5; n++) {
            X_d[0][n] = xi[5 * n + 0];
            X_d[1][n] = xi[5 * n + 1];
            X_d[2][n] = xi[5 * n + 2];
            X_d[3][n] = xi[5 * n + 3];
            X_d[4][n] = xi[5 * n + 4];
        }

        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_packed_5in8 + (size_t)j * Kp;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < K; k += 80) {
                /* Load 16 packed bytes (= 80 trits). */
                uint8x16_t bytes = vld1q_u8(wj + k / 5);

                /* Decode 5 digits via vectorized magic-multiply div-by-3.
                 * Digits 0..3 each: q = b/3, m = b - 3q, lookup m. Then b←q.
                 * Digit 4: just lookup the final quotient (b ∈ [0, 2]). */
                int8x16_t d0, d1, d2, d3, d4;
                uint8x16_t b = bytes;

                /* Digit 0 */
                {
                    uint16x8_t lo = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 171), 9);
                    uint16x8_t hi = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 171), 9);
                    uint8x16_t q  = vcombine_u8(vmovn_u16(lo), vmovn_u16(hi));
                    uint8x16_t m  = vsubq_u8(b, vmulq_u8(q, three_v));
                    d0 = vqtbl1q_s8(lut, m);
                    b  = q;
                }
                /* Digit 1 */
                {
                    uint16x8_t lo = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 171), 9);
                    uint16x8_t hi = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 171), 9);
                    uint8x16_t q  = vcombine_u8(vmovn_u16(lo), vmovn_u16(hi));
                    uint8x16_t m  = vsubq_u8(b, vmulq_u8(q, three_v));
                    d1 = vqtbl1q_s8(lut, m);
                    b  = q;
                }
                /* Digit 2 */
                {
                    uint16x8_t lo = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 171), 9);
                    uint16x8_t hi = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 171), 9);
                    uint8x16_t q  = vcombine_u8(vmovn_u16(lo), vmovn_u16(hi));
                    uint8x16_t m  = vsubq_u8(b, vmulq_u8(q, three_v));
                    d2 = vqtbl1q_s8(lut, m);
                    b  = q;
                }
                /* Digit 3 */
                {
                    uint16x8_t lo = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 171), 9);
                    uint16x8_t hi = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 171), 9);
                    uint8x16_t q  = vcombine_u8(vmovn_u16(lo), vmovn_u16(hi));
                    uint8x16_t m  = vsubq_u8(b, vmulq_u8(q, three_v));
                    d3 = vqtbl1q_s8(lut, m);
                    b  = q;
                }
                /* Digit 4: b is now in [0, 2]; direct lookup. */
                d4 = vqtbl1q_s8(lut, b);

                /* P0-1 optimization: direct loads from pre-permuted X.
                 * Each X_d[digit] is contiguous; lane i holds X[5*i + digit]
                 * for the current outer-block's k. */
                int x_idx = k / 5;
                int8x16_t xv0 = vld1q_s8(X_d[0] + x_idx);
                int8x16_t xv1 = vld1q_s8(X_d[1] + x_idx);
                int8x16_t xv2 = vld1q_s8(X_d[2] + x_idx);
                int8x16_t xv3 = vld1q_s8(X_d[3] + x_idx);
                int8x16_t xv4 = vld1q_s8(X_d[4] + x_idx);

                /* SDOT each digit. */
                acc = vdotq_s32(acc, xv0, d0);
                acc = vdotq_s32(acc, xv1, d1);
                acc = vdotq_s32(acc, xv2, d2);
                acc = vdotq_s32(acc, xv3, d3);
                acc = vdotq_s32(acc, xv4, d4);
            }

            Y[(size_t)i * N + j] = vaddvq_s32(acc);
        }
    }

    free(X_strided);
}

/* ── Path C: B2-B optimal (unified TBL decode) ─────────────────────────── */

__attribute__((noinline))
void b2b_optimal_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_b2b,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K % 16 == 0);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_b2b)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_b2b);

    int Kp = (K + 3) / 4;

    const uint8x16_t dup_idx = vld1q_u8(DUP_IDX);
    const int8x16_t  shift_s = vreinterpretq_s8_u8(vld1q_u8(SHIFT_LANE));
    const uint8x16_t mask_03 = vdupq_n_u8(0x03u);
    const int8x16_t  lut     = vld1q_s8(B2B_OPTIMAL_LUT);

    for (int i = 0; i < M; i++) {
        const int8_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_b2b + (size_t)j * Kp;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < K; k += 16) {
                /* Inner-block decode + SDOT. Identical structure to Path A;
                 * only the LUT contents differ. ~7 NEON ops. */
                uint32_t w32;
                memcpy(&w32, wj + (k >> 2), 4);
                uint8x16_t packed  = vreinterpretq_u8_u32(vdupq_n_u32(w32));
                uint8x16_t dup     = vqtbl1q_u8(packed, dup_idx);
                uint8x16_t shifted = vshlq_u8(dup, vnegq_s8(shift_s));
                uint8x16_t codes   = vandq_u8(shifted, mask_03);
                int8x16_t  w_vec   = vqtbl1q_s8(lut, codes);

                int8x16_t  x_vec   = vld1q_s8(xi + k);
                acc = vdotq_s32(acc, x_vec, w_vec);
            }

            Y[(size_t)i * N + j] = vaddvq_s32(acc);
        }
    }
}
