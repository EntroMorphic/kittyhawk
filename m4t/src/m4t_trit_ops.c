/*
 * m4t_trit_ops.c — TBL-based binary trit operations
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Each binary op uses a 16-byte LUT indexed by (a_code << 2) | b_code,
 * where codes are the 2-bit packed-trit encoding (0=zero, 1=+1, 2=-1).
 * The LUT outputs are ALSO 2-bit codes, so the result packs directly
 * into the output byte with no re-encoding step.
 *
 * NEON kernel: processes 16 packed bytes (64 trits) per iteration.
 * For each of 4 trit positions within each byte, extract codes from a
 * and b, combine into a 4-bit pair index, TBL lookup, shift to position,
 * OR together. ~28 instructions per 64 trits.
 *
 * neg uses a bit-swap trick instead of TBL: in the 2-bit encoding,
 * negation is just swapping bit 0 and bit 1 of each pair.
 */

#include "m4t_trit_ops.h"
#include "m4t_internal.h"

/* ── LUTs ──────────────────────────────────────────────────────────────────
 *
 * Indexed by (a_code << 2) | b_code. Output is the result's 2-bit code.
 * Code mapping: 0 → trit 0, 1 → trit +1, 2 → trit -1, 3 → reserved.
 *
 * Derived from m4t/tools/m4t_trit_golden.c truth tables, then converted
 * from signed trit values to 2-bit codes.
 */

static const uint8_t LUT_MUL[16] __attribute__((aligned(16))) = {
    0, 0, 0, 0,   0, 1, 2, 0,   0, 2, 1, 0,   0, 0, 0, 0
};
static const uint8_t LUT_SAT_ADD[16] __attribute__((aligned(16))) = {
    0, 1, 2, 0,   1, 1, 0, 1,   2, 0, 2, 2,   0, 1, 2, 0
};
static const uint8_t LUT_MAX[16] __attribute__((aligned(16))) = {
    0, 1, 0, 0,   1, 1, 1, 1,   0, 1, 2, 0,   0, 1, 0, 0
};
static const uint8_t LUT_MIN[16] __attribute__((aligned(16))) = {
    0, 0, 2, 0,   0, 1, 2, 0,   2, 2, 2, 2,   0, 0, 2, 0
};
static const uint8_t LUT_EQ[16] __attribute__((aligned(16))) = {
    1, 0, 0, 1,   0, 1, 0, 0,   0, 0, 1, 0,   1, 0, 0, 1
};

/* ── Shared kernel ─────────────────────────────────────────────────────── */

static void trit_binary_op(
    uint8_t* dst, const uint8_t* a, const uint8_t* b,
    int n_trits, const uint8_t* lut_data)
{
    int n_bytes = M4T_TRIT_PACKED_BYTES(n_trits);
    int i = 0;

#if M4T_HAS_NEON
    uint8x16_t lut   = vld1q_u8(lut_data);
    uint8x16_t mask2 = vdupq_n_u8(0x03);

    for (; i + 16 <= n_bytes; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);

        /* Position 0: bits 0-1 of each byte. */
        uint8x16_t a0 = vandq_u8(va, mask2);
        uint8x16_t b0 = vandq_u8(vb, mask2);
        uint8x16_t r0 = vqtbl1q_u8(lut, vorrq_u8(vshlq_n_u8(a0, 2), b0));

        /* Position 1: bits 2-3. */
        uint8x16_t a1 = vandq_u8(vshrq_n_u8(va, 2), mask2);
        uint8x16_t b1 = vandq_u8(vshrq_n_u8(vb, 2), mask2);
        uint8x16_t r1 = vshlq_n_u8(
            vqtbl1q_u8(lut, vorrq_u8(vshlq_n_u8(a1, 2), b1)), 2);

        /* Position 2: bits 4-5. */
        uint8x16_t a2 = vandq_u8(vshrq_n_u8(va, 4), mask2);
        uint8x16_t b2 = vandq_u8(vshrq_n_u8(vb, 4), mask2);
        uint8x16_t r2 = vshlq_n_u8(
            vqtbl1q_u8(lut, vorrq_u8(vshlq_n_u8(a2, 2), b2)), 4);

        /* Position 3: bits 6-7. */
        uint8x16_t a3 = vshrq_n_u8(va, 6);
        uint8x16_t b3 = vshrq_n_u8(vb, 6);
        uint8x16_t r3 = vshlq_n_u8(
            vqtbl1q_u8(lut, vorrq_u8(vshlq_n_u8(a3, 2), b3)), 6);

        vst1q_u8(dst + i,
            vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3)));
    }
#endif

    /* Scalar tail. */
    for (; i < n_bytes; i++) {
        uint8_t ba = a[i], bb = b[i], out = 0;
        for (int t = 0; t < 4; t++) {
            uint8_t ac = (ba >> (t * 2)) & 0x03u;
            uint8_t bc = (bb >> (t * 2)) & 0x03u;
            out |= lut_data[(ac << 2) | bc] << (t * 2);
        }
        dst[i] = out;
    }
}

/* ── Public entry points ───────────────────────────────────────────────── */

void m4t_trit_mul(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits) {
    trit_binary_op(dst, a, b, n_trits, LUT_MUL);
}

void m4t_trit_sat_add(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits) {
    trit_binary_op(dst, a, b, n_trits, LUT_SAT_ADD);
}

void m4t_trit_max(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits) {
    trit_binary_op(dst, a, b, n_trits, LUT_MAX);
}

void m4t_trit_min(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits) {
    trit_binary_op(dst, a, b, n_trits, LUT_MIN);
}

void m4t_trit_eq(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits) {
    trit_binary_op(dst, a, b, n_trits, LUT_EQ);
}

void m4t_trit_neg(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits) {
    (void)b;
    int n_bytes = M4T_TRIT_PACKED_BYTES(n_trits);
    int i = 0;

#if M4T_HAS_NEON
    /* Negation in our 2-bit encoding is swapping bit 0 and bit 1 of each
     * pair: code 1 (+1) ↔ code 2 (-1), code 0 (zero) stays 0.
     * ((x & 0x55) << 1) | ((x & 0xAA) >> 1) */
    uint8x16_t even = vdupq_n_u8(0x55);
    uint8x16_t odd  = vdupq_n_u8(0xAA);
    for (; i + 16 <= n_bytes; i += 16) {
        uint8x16_t v = vld1q_u8(a + i);
        vst1q_u8(dst + i, vorrq_u8(
            vshlq_n_u8(vandq_u8(v, even), 1),
            vshrq_n_u8(vandq_u8(v, odd),  1)));
    }
#endif

    for (; i < n_bytes; i++) {
        dst[i] = (uint8_t)(((a[i] & 0x55u) << 1) | ((a[i] & 0xAAu) >> 1));
    }
}
