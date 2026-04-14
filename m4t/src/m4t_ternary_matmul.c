/*
 * m4t_ternary_matmul.c — MTFP × packed-trit matmul
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 *   Y[M,N] = X[M,K] @ W^T     where W is [N,K] packed ternary
 *
 * Activations X are m4t_mtfp_t (int32 MTFP cells).
 * Weights W_packed are 2-bit packed trits, LSB-first, Kp = (K+3)/4 bytes/row.
 * Output Y is m4t_mtfp_t.
 *
 * Inner product per (i, j): decode K trits from W_packed[j], conditionally
 * add or subtract X[i, k] into an int64 accumulator. Zero multiplies.
 *
 * The trit-decode idiom (vld1q_u8 → shift/mask → vqtbl1q_s8) is borrowed
 * from trix-z's trix_ternary_matvec_i8. The accumulator pattern is new:
 * trix-z has no kernel that combines packed-trit weights with MTFP
 * activations.
 *
 * Bound analysis: |X[k]| ≤ M4T_MTFP_MAX_VAL = 581,130,733 (MTFP19).
 * Summing K contributions: for K ≤ 15.9e9 the total fits in int64
 * (9.2e18 / 581e6 ≈ 15.9e9). Plain int64 accumulators are safe; no
 * periodic narrowing required.
 */

#include "m4t_ternary_matmul.h"
#include "m4t_trit_pack.h"
#include "m4t_mtfp.h"
#include "m4t_internal.h"
#include <string.h>
#include <assert.h>

/* ── Inner product for a single output cell ───────────────────────────── */

static int64_t ternary_dot(
    const m4t_mtfp_t* xi,    /* [K] MTFP activations */
    const uint8_t* wj,         /* [Kp] packed-trit weights */
    int K)
{
    int64_t acc = 0;
    int k = 0;

#if M4T_HAS_NEON
    /* Process 16 trits (= 4 packed bytes) per iteration. For each block,
     * decode the 16 trit codes into sign bytes, then apply them to 16
     * loaded MTFP activations WITHOUT a multiply. Hardware-native shape:
     *   trit ==  0 → contribute 0    (bit-select zero)
     *   trit == +1 → contribute +a   (pass activation)
     *   trit == -1 → contribute -a   (vnegq_s32 activation)
     *
     * Multiplying by a sign in {-1, 0, +1} is a base-2 shortcut through a
     * general-purpose opcode. The base-3-native expression is a mask and a
     * conditional negate — which is what TBL + bit-select compute directly.
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
    const int8x16_t  lut_sign = vld1q_s8(M4T_TRIT_DECODE_LUT);

    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);

    while (k + 16 <= K) {
        /* Load 4 packed bytes (16 trits) from W row j. */
        uint32_t w32;
        memcpy(&w32, wj + (k >> 2), 4);
        uint8x16_t packed = vreinterpretq_u8_u32(vdupq_n_u32(w32));

        /* Duplicate each byte 4× → [b0 b0 b0 b0 | b1 b1 b1 b1 | ...]. */
        uint8x16_t dup = vqtbl1q_u8(packed, dup_idx);

        /* Per-lane right shift to extract each 2-bit trit code from its byte.
         * vshlq_u8 with a negated shift count performs per-lane right shift. */
        uint8x16_t shifted = vshlq_u8(dup, vnegq_s8(shift_s));
        uint8x16_t codes = vandq_u8(shifted, mask_03);

        /* Decode 16 codes → 16 signed trit bytes {-1, 0, +1}. */
        int8x16_t signs = vqtbl1q_s8(lut_sign, codes);

        /* Build two masks from the sign bytes, widened to int32 lanes:
         *   nonzero_mask[i] = (signs[i] != 0) ? 0xFFFFFFFF : 0
         *   neg_mask[i]     = (signs[i] < 0)  ? 0xFFFFFFFF : 0
         * Widening preserves the mask semantics because sign-extending a
         * byte-wide all-ones is still all-ones in the wider lane. */
        int8x16_t zero8   = vdupq_n_s8(0);
        uint8x16_t nz8    = vmvnq_u8(vceqq_s8(signs, zero8));   /* 0xFF where sign != 0 */
        uint8x16_t neg8   = vcltq_s8(signs, zero8);             /* 0xFF where sign <  0 */

        /* Widen both masks from 16×u8 to 4×4×u32 via two pairwise widens. */
        uint16x8_t nz16_lo = vmovl_u8(vget_low_u8(nz8));
        uint16x8_t nz16_hi = vmovl_u8(vget_high_u8(nz8));
        uint32x4_t nz0 = vmovl_u16(vget_low_u16(nz16_lo));
        uint32x4_t nz1 = vmovl_u16(vget_high_u16(nz16_lo));
        uint32x4_t nz2 = vmovl_u16(vget_low_u16(nz16_hi));
        uint32x4_t nz3 = vmovl_u16(vget_high_u16(nz16_hi));
        /* vmovl_u8 zero-extends — make the masks all-ones where set by
         * turning 0x01 → 0xFFFFFFFF via compare-not-equal-to-zero. */
        uint32x4_t zero32 = vdupq_n_u32(0);
        nz0 = vmvnq_u32(vceqq_u32(nz0, zero32));
        nz1 = vmvnq_u32(vceqq_u32(nz1, zero32));
        nz2 = vmvnq_u32(vceqq_u32(nz2, zero32));
        nz3 = vmvnq_u32(vceqq_u32(nz3, zero32));

        uint16x8_t ng16_lo = vmovl_u8(vget_low_u8(neg8));
        uint16x8_t ng16_hi = vmovl_u8(vget_high_u8(neg8));
        uint32x4_t ng0 = vmovl_u16(vget_low_u16(ng16_lo));
        uint32x4_t ng1 = vmovl_u16(vget_high_u16(ng16_lo));
        uint32x4_t ng2 = vmovl_u16(vget_low_u16(ng16_hi));
        uint32x4_t ng3 = vmovl_u16(vget_high_u16(ng16_hi));
        ng0 = vmvnq_u32(vceqq_u32(ng0, zero32));
        ng1 = vmvnq_u32(vceqq_u32(ng1, zero32));
        ng2 = vmvnq_u32(vceqq_u32(ng2, zero32));
        ng3 = vmvnq_u32(vceqq_u32(ng3, zero32));

        /* Load 16 MTFP activations. */
        int32x4_t a0 = vld1q_s32(xi + k);
        int32x4_t a1 = vld1q_s32(xi + k + 4);
        int32x4_t a2 = vld1q_s32(xi + k + 8);
        int32x4_t a3 = vld1q_s32(xi + k + 12);

        /* Conditional negate where neg_mask is set. */
        int32x4_t aneg0 = vnegq_s32(a0);
        int32x4_t aneg1 = vnegq_s32(a1);
        int32x4_t aneg2 = vnegq_s32(a2);
        int32x4_t aneg3 = vnegq_s32(a3);
        int32x4_t s0 = vbslq_s32(ng0, aneg0, a0);
        int32x4_t s1 = vbslq_s32(ng1, aneg1, a1);
        int32x4_t s2 = vbslq_s32(ng2, aneg2, a2);
        int32x4_t s3 = vbslq_s32(ng3, aneg3, a3);

        /* Zero out lanes where the trit is 0 (bit-select against all-zeros). */
        int32x4_t zero_v = vdupq_n_s32(0);
        int32x4_t p0 = vbslq_s32(nz0, s0, zero_v);
        int32x4_t p1 = vbslq_s32(nz1, s1, zero_v);
        int32x4_t p2 = vbslq_s32(nz2, s2, zero_v);
        int32x4_t p3 = vbslq_s32(nz3, s3, zero_v);

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
    m4t_mtfp_t* Y_row, const m4t_mtfp_t* X_row, const uint8_t* W_packed,
    int K, int N)
{
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    for (int j = 0; j < N; j++) {
        int64_t acc = ternary_dot(X_row, W_packed + (size_t)j * Kp, K);
        Y_row[j] = m4t_mtfp_clamp64(acc);
    }
}

/* ── Public entry ──────────────────────────────────────────────────────── */

void m4t_mtfp_ternary_matmul_bt(
    m4t_mtfp_t* Y, const m4t_mtfp_t* X, const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(Y && X && W_packed);
    assert(M >= 0 && K >= 0 && N >= 0);
    for (int i = 0; i < M; i++) {
        ternary_matmul_bt_row(
            Y + (size_t)i * N,
            X + (size_t)i * K,
            W_packed, K, N);
    }
}
