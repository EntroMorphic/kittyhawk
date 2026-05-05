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

/* P0-2: split-LUT decode tables for the 5-in-8 kernel.
 *
 * Old approach: 4 sequential div-by-3 magic-multiplies extract digits 0..4.
 * New approach: 1 div-by-9 splits byte b into (high = b/9, low = b%9), then
 * direct LUT lookups extract all 5 digits in parallel.
 *
 * Encoding (matches base3_5in8_pack):
 *   trit_to_unsigned: -1 → 2, 0 → 0, +1 → 1
 *   byte = u0 + 3*u1 + 9*u2 + 27*u3 + 81*u4
 *
 * Per byte b ∈ [0, 242]:
 *   high = b / 9   (range [0, 26], encodes digits 4, 3, 2)
 *   low  = b % 9   (range [0, 8],  encodes digits 1, 0)
 *
 * Then:
 *   u0 = low % 3        u1 = low / 3
 *   u2 = high % 3       u3 = (high / 3) % 3       u4 = high / 9
 *
 * trit_value(u) = {0 → 0, 1 → +1, 2 → -1}.
 *
 * 5 LUTs: 3 for high (vqtbl4q with 27 valid entries) and 2 for low (vqtbl1q
 * with 9 valid entries). All 16-byte aligned for NEON loads. */

/* LUT_LOW_DIGIT_d[low] = trit value for digit d, indexed by low ∈ [0, 8]. */
static const int8_t LUT_LOW_DIGIT0[16] __attribute__((aligned(16))) = {
    /* low % 3:  0,+1,-1, 0,+1,-1, 0,+1,-1, (rest auto-zero) */
     0,  1, -1,  0,  1, -1,  0,  1, -1,
};
static const int8_t LUT_LOW_DIGIT1[16] __attribute__((aligned(16))) = {
    /* low / 3:  0, 0, 0,+1,+1,+1,-1,-1,-1, (rest auto-zero) */
     0,  0,  0,  1,  1,  1, -1, -1, -1,
};

/* LUT_HIGH_DIGIT_d[high] = trit value for digit d, indexed by high ∈ [0, 26].
 * Stored as 32 bytes (2×16) for vqtbl2q_s8. Bytes >= 27 are unused (high<27). */
static const int8_t LUT_HIGH_DIGIT2[32] __attribute__((aligned(16))) = {
    /* high % 3, repeating every 3: */
     0,  1, -1,  0,  1, -1,  0,  1, -1,    /* high 0..8 */
     0,  1, -1,  0,  1, -1,  0,  1, -1,    /* high 9..17 */
     0,  1, -1,  0,  1, -1,  0,  1, -1,    /* high 18..26 */
    /* remaining 37 entries auto-zero per C standard. */
};
static const int8_t LUT_HIGH_DIGIT3[32] __attribute__((aligned(16))) = {
    /* (high / 3) % 3: */
     0,  0,  0,  1,  1,  1, -1, -1, -1,    /* high 0..8 */
     0,  0,  0,  1,  1,  1, -1, -1, -1,    /* high 9..17 */
     0,  0,  0,  1,  1,  1, -1, -1, -1,    /* high 18..26 */
    /* remaining 37 entries auto-zero per C standard. */
};
static const int8_t LUT_HIGH_DIGIT4[32] __attribute__((aligned(16))) = {
    /* high / 9: */
     0,  0,  0,  0,  0,  0,  0,  0,  0,    /* high 0..8 → 0 */
     1,  1,  1,  1,  1,  1,  1,  1,  1,    /* high 9..17 → +1 */
    -1, -1, -1, -1, -1, -1, -1, -1, -1,    /* high 18..26 → -1 */
    /* remaining 5 entries auto-zero per C standard. */
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

    const uint8x16_t nine_v = vdupq_n_u8(9);

    /* P0-2: split-LUT decode constants. Hoisted out of the inner loop. */
    const int8x16_t lut_d0 = vld1q_s8(LUT_LOW_DIGIT0);
    const int8x16_t lut_d1 = vld1q_s8(LUT_LOW_DIGIT1);
    const int8x16x2_t lut_d2 = { {
        vld1q_s8(LUT_HIGH_DIGIT2 +  0),
        vld1q_s8(LUT_HIGH_DIGIT2 + 16),
    } };
    const int8x16x2_t lut_d3 = { {
        vld1q_s8(LUT_HIGH_DIGIT3 +  0),
        vld1q_s8(LUT_HIGH_DIGIT3 + 16),
    } };
    const int8x16x2_t lut_d4 = { {
        vld1q_s8(LUT_HIGH_DIGIT4 +  0),
        vld1q_s8(LUT_HIGH_DIGIT4 + 16),
    } };

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
                uint8x16_t b = vld1q_u8(wj + k / 5);

                /* P0-2: split-LUT decode.
                 * high = b / 9 (magic-mul: (b * 57) >> 9), range [0, 26].
                 * low  = b - 9 * high, range [0, 8].
                 * 5 digits extracted via direct LUT lookups. */
                uint16x8_t lo16 = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 57), 9);
                uint16x8_t hi16 = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 57), 9);
                uint8x16_t high = vcombine_u8(vmovn_u16(lo16), vmovn_u16(hi16));
                uint8x16_t low  = vsubq_u8(b, vmulq_u8(high, nine_v));

                int8x16_t d0 = vqtbl1q_s8(lut_d0, low);
                int8x16_t d1 = vqtbl1q_s8(lut_d1, low);
                int8x16_t d2 = vqtbl2q_s8(lut_d2, high);
                int8x16_t d3 = vqtbl2q_s8(lut_d3, high);
                int8x16_t d4 = vqtbl2q_s8(lut_d4, high);

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
