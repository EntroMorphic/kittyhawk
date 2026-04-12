/*
 * m4t_trit_reducers.c — masked-VCNT ternary reductions
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The key insight: in our 2-bit trit encoding (0=zero, 01=+1, 10=-1),
 * the even bits (0x55 mask) are set iff the trit is +1, and the odd bits
 * (0xAA mask) are set iff the trit is -1. VCNT on the masked byte counts
 * trits of that polarity. No decode step needed.
 *
 * Per 16-byte (64-trit) block:
 *   - 2 loads (same data, or reuse register)
 *   - 2 ANDs (even mask, odd mask)
 *   - 2 VCNTs
 *   - 2 pairwise widen chains (u8 → u16 → u32)
 *   - 2 accumulates into u32x4
 * Total: ~14 NEON instructions per 64 trits.
 */

#include "m4t_trit_reducers.h"
#include "m4t_internal.h"

void m4t_trit_counts(
    const uint8_t* packed, int n_trits,
    int64_t* out_pos, int64_t* out_neg)
{
    int n_bytes = M4T_TRIT_PACKED_BYTES(n_trits);
    int64_t pos = 0, neg = 0;
    int i = 0;

#if M4T_HAS_NEON
    uint8x16_t mask_even = vdupq_n_u8(0x55);  /* bit 0 of each pair: +1 */
    uint8x16_t mask_odd  = vdupq_n_u8(0xAA);  /* bit 1 of each pair: -1 */
    uint32x4_t acc_pos = vdupq_n_u32(0);
    uint32x4_t acc_neg = vdupq_n_u32(0);

    for (; i + 16 <= n_bytes; i += 16) {
        uint8x16_t v = vld1q_u8(packed + i);

        /* Count +1 trits: popcount of even bits. */
        uint8x16_t pos_bits = vandq_u8(v, mask_even);
        uint16x8_t p16 = vpaddlq_u8(vcntq_u8(pos_bits));
        acc_pos = vaddq_u32(acc_pos, vpaddlq_u16(p16));

        /* Count -1 trits: popcount of odd bits. */
        uint8x16_t neg_bits = vandq_u8(v, mask_odd);
        uint16x8_t n16 = vpaddlq_u8(vcntq_u8(neg_bits));
        acc_neg = vaddq_u32(acc_neg, vpaddlq_u16(n16));
    }

    pos = (int64_t)vaddvq_u32(acc_pos);
    neg = (int64_t)vaddvq_u32(acc_neg);
#endif

    /* Scalar tail. */
    for (; i < n_bytes; i++) {
        uint8_t b = packed[i];
        /* Count +1: bits 0, 2, 4, 6 (even positions). */
        uint8_t p = b & 0x55u;
        while (p) { pos++; p &= (uint8_t)(p - 1u); }
        /* Count -1: bits 1, 3, 5, 7 (odd positions). */
        uint8_t n = b & 0xAAu;
        while (n) { neg++; n &= (uint8_t)(n - 1u); }
    }

    /* Handle padding trits in the last byte: if n_trits is not a multiple
     * of 4, the last byte has (4 - n_trits%4) padding trits in the high
     * bits. These are zero-coded (0b00) by the pack functions, so they
     * contribute 0 to both pos and neg counts. No correction needed. */

    *out_pos = pos;
    *out_neg = neg;
}

int64_t m4t_trit_signed_sum(const uint8_t* packed, int n_trits) {
    int64_t pos, neg;
    m4t_trit_counts(packed, n_trits, &pos, &neg);
    return pos - neg;
}

int64_t m4t_trit_sparsity(const uint8_t* packed, int n_trits) {
    int64_t pos, neg;
    m4t_trit_counts(packed, n_trits, &pos, &neg);
    return pos + neg;
}
