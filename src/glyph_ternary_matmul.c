/*
 * glyph_ternary_matmul.c — MTFP × packed-trit matmul
 *
 * GLYPH IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 *   Y[M,N] = X[M,K] @ W^T     where W is [N,K] packed ternary
 *
 * Activations X are glyph_mtfp_t (int32 MTFP cells).
 * Weights W_packed are 2-bit packed trits, LSB-first, Kp = (K+3)/4 bytes/row.
 * Output Y is glyph_mtfp_t.
 *
 * Inner product per (i, j): decode K trits from W_packed[j], conditionally
 * add or subtract X[i, k] into an int64 accumulator. Zero multiplies.
 *
 * The trit-decode idiom (vld1q_u8 → shift/mask → vqtbl1q_s8) is borrowed
 * from trix-z's trix_ternary_matvec_i8. The accumulator pattern is new:
 * trix-z has no kernel that combines packed-trit weights with MTFP
 * activations.
 *
 * Bound analysis: |X[k]| ≤ GLYPH_MTFP_MAX_VAL ≈ 1.07e9. Summing K ≤ 2^33
 * contributions fits in int64 (2^63 ≈ 9.2e18). Plain int64 accumulators
 * are safe; no periodic narrowing required.
 */

#include "glyph_ternary_matmul.h"
#include "glyph_trit_pack.h"
#include "glyph_mtfp.h"
#include "glyph_internal.h"
#include <string.h>
#include <assert.h>

/* ── Inner product for a single output cell ───────────────────────────── */

static int64_t ternary_dot(
    const glyph_mtfp_t* xi,    /* [K] MTFP activations */
    const uint8_t* wj,         /* [Kp] packed-trit weights */
    int K)
{
    int64_t acc = 0;
    int k = 0;

#if GLYPH_HAS_NEON
    /* Process 16 trits (= 4 packed bytes) per iteration. For each block,
     * decode the 16 trit codes into sign bytes, widen signedly to int32,
     * multiply against 16 loaded MTFP cells, then widen-accumulate into
     * int64 lanes.
     *
     * Sign extension note: we use vmovl_s8 / vmovl_s16 so that decoded -1
     * stays as -1 when widened. Unsigned widening would turn -1 into 255.
     */
    static const uint8_t DUP_IDX[16] = {
        0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3
    };
    static const uint8_t SHIFT_LANE[16] = {
        0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6
    };
    const uint8x16_t dup_idx = vld1q_u8(DUP_IDX);
    const int8x16_t  shift_s = vreinterpretq_s8_u8(vld1q_u8(SHIFT_LANE));
    const uint8x16_t mask_03 = vdupq_n_u8(0x03u);
    const int8x16_t  lut_sign = vld1q_s8(GLYPH_TRIT_DECODE_LUT);

    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);

    while (k + 16 <= K) {
        /* Load 4 packed bytes (16 trits) from W row j. */
        uint32_t w32;
        memcpy(&w32, wj + (k >> 2), 4);
        uint8x16_t packed = vreinterpretq_u8_u32(vdupq_n_u32(w32));

        /* Duplicate each byte 4× → [b0 b0 b0 b0 | b1 b1 b1 b1 | ...]. */
        uint8x16_t dup = vqtbl1q_u8(packed, dup_idx);

        /* Per-lane right shift by {0,2,4,6,0,2,4,6,...} then mask to 2 bits. */
        uint8x16_t shifted = vshlq_u8(dup,
            vnegq_s8(shift_s));  /* vshlq_u8 uses signed shift count; negative = right shift */
        uint8x16_t codes = vandq_u8(shifted, mask_03);

        /* Decode 16 codes → 16 signed trit bytes {-1, 0, +1}. */
        int8x16_t signs = vqtbl1q_s8(lut_sign, codes);

        /* Widen signs: int8 → int16 → int32 in 4 lanes of 4. */
        int16x8_t s16_lo = vmovl_s8(vget_low_s8(signs));
        int16x8_t s16_hi = vmovl_s8(vget_high_s8(signs));
        int32x4_t s0 = vmovl_s16(vget_low_s16(s16_lo));
        int32x4_t s1 = vmovl_s16(vget_high_s16(s16_lo));
        int32x4_t s2 = vmovl_s16(vget_low_s16(s16_hi));
        int32x4_t s3 = vmovl_s16(vget_high_s16(s16_hi));

        /* Load 16 MTFP activations. */
        int32x4_t a0 = vld1q_s32(xi + k);
        int32x4_t a1 = vld1q_s32(xi + k + 4);
        int32x4_t a2 = vld1q_s32(xi + k + 8);
        int32x4_t a3 = vld1q_s32(xi + k + 12);

        /* Signed multiply: lanes with sign=0 become 0, ±1 select ±a. */
        int32x4_t p0 = vmulq_s32(a0, s0);
        int32x4_t p1 = vmulq_s32(a1, s1);
        int32x4_t p2 = vmulq_s32(a2, s2);
        int32x4_t p3 = vmulq_s32(a3, s3);

        /* Widen and accumulate into int64 lanes. */
        acc0 = vaddw_s32(acc0, vget_low_s32(p0));
        acc1 = vaddw_s32(acc1, vget_high_s32(p0));
        acc0 = vaddw_s32(acc0, vget_low_s32(p1));
        acc1 = vaddw_s32(acc1, vget_high_s32(p1));
        acc0 = vaddw_s32(acc0, vget_low_s32(p2));
        acc1 = vaddw_s32(acc1, vget_high_s32(p2));
        acc0 = vaddw_s32(acc0, vget_low_s32(p3));
        acc1 = vaddw_s32(acc1, vget_high_s32(p3));

        k += 16;
    }
    acc = vgetq_lane_s64(acc0, 0) + vgetq_lane_s64(acc0, 1)
        + vgetq_lane_s64(acc1, 0) + vgetq_lane_s64(acc1, 1);
#endif

    /* Scalar tail for the last < 16 trits (and entire loop on non-NEON). */
    for (; k < K; k++) {
        uint8_t code = (uint8_t)((wj[k >> 2] >> ((k & 3) * 2)) & 0x3u);
        if      (code == 0x01u) acc += (int64_t)xi[k];
        else if (code == 0x02u) acc -= (int64_t)xi[k];
    }
    return acc;
}

/* ── Row driver ────────────────────────────────────────────────────────── */

static void ternary_matmul_bt_row(
    glyph_mtfp_t* Y_row, const glyph_mtfp_t* X_row, const uint8_t* W_packed,
    int K, int N)
{
    int Kp = GLYPH_TRIT_PACKED_BYTES(K);
    for (int j = 0; j < N; j++) {
        int64_t acc = ternary_dot(X_row, W_packed + (size_t)j * Kp, K);
        Y_row[j] = glyph_mtfp_clamp64(acc);
    }
}

/* ── Public entry ──────────────────────────────────────────────────────── */

void glyph_mtfp_ternary_matmul_bt(
    glyph_mtfp_t* Y, const glyph_mtfp_t* X, const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(Y && X && W_packed);
    assert(M >= 0 && K >= 0 && N >= 0);
#if GLYPH_HAS_DISPATCH
    if (M >= GLYPH_SERIAL_ROW_THRESHOLD) {
        dispatch_apply((size_t)M,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t i) {
                ternary_matmul_bt_row(
                    Y + (size_t)i * N,
                    X + (size_t)i * K,
                    W_packed, K, N);
            });
        return;
    }
#endif
    for (int i = 0; i < M; i++) {
        ternary_matmul_bt_row(
            Y + (size_t)i * N,
            X + (size_t)i * K,
            W_packed, K, N);
    }
}
