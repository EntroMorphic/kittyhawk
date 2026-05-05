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
}
