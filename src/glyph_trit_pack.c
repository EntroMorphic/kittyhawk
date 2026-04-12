/*
 * glyph_trit_pack.c — trit packing, unpacking, popcount distance
 *
 * GLYPH IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Pure container-level operations. No MTFP, no float.
 */

#include "glyph_trit_pack.h"
#include "glyph_internal.h"
#include <string.h>

/* Decode LUT: maps 2-bit trit codes to signed trit values.
 * Pattern repeats 4× so the same LUT drives vqtbl1q_s8 against any 16-byte
 * register of codes in [0, 3]. */
const int8_t GLYPH_TRIT_DECODE_LUT[16] __attribute__((aligned(16))) = {
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
};

/* ── Pack / unpack ─────────────────────────────────────────────────────── */

static inline uint8_t trit_to_code(glyph_trit_t t) {
    /* {-1, 0, +1} → {0b10, 0b00, 0b01}.
     * Out-of-range inputs are undefined; we defensively return 0. */
    return (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
}

static inline glyph_trit_t code_to_trit(uint8_t code) {
    /* 0b00 → 0, 0b01 → +1, 0b10 → -1, 0b11 → 0 (reserved). */
    return (code == 0x01u) ? 1 : (code == 0x02u) ? -1 : 0;
}

void glyph_pack_trits_1d(uint8_t* dst, const glyph_trit_t* src, int n) {
    int nb = GLYPH_TRIT_PACKED_BYTES(n);
    memset(dst, 0, (size_t)nb);
    for (int i = 0; i < n; i++) {
        uint8_t code = trit_to_code(src[i]);
        dst[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

void glyph_unpack_trits_1d(glyph_trit_t* dst, const uint8_t* src, int n) {
    for (int i = 0; i < n; i++) {
        uint8_t code = (uint8_t)((src[i >> 2] >> ((i & 3) * 2)) & 0x3u);
        dst[i] = code_to_trit(code);
    }
}

void glyph_pack_trits_rowmajor(
    uint8_t* dst, const glyph_trit_t* src, int M, int K)
{
    int Kp = GLYPH_TRIT_PACKED_BYTES(K);
    memset(dst, 0, (size_t)M * (size_t)Kp);
    for (int m = 0; m < M; m++) {
        glyph_pack_trits_1d(dst + (size_t)m * Kp, src + (size_t)m * K, K);
    }
}

void glyph_unpack_trits_rowmajor(
    glyph_trit_t* dst, const uint8_t* src, int M, int K)
{
    int Kp = GLYPH_TRIT_PACKED_BYTES(K);
    for (int m = 0; m < M; m++) {
        glyph_unpack_trits_1d(dst + (size_t)m * K, src + (size_t)m * Kp, K);
    }
}

/* ── Popcount distance with mask ───────────────────────────────────────── */

int32_t glyph_popcount_dist(
    const uint8_t* a, const uint8_t* b, const uint8_t* mask, int packed_bytes)
{
    int32_t total = 0;
    int i = 0;

#if GLYPH_HAS_NEON
    uint32x4_t acc = vdupq_n_u32(0);
    for (; i + 16 <= packed_bytes; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t vm = vld1q_u8(mask + i);
        uint8x16_t diff = vandq_u8(veorq_u8(va, vb), vm);
        /* VCNT → pairwise widen u8→u16 → u16→u32 → accumulate. */
        uint16x8_t cnt16 = vpaddlq_u8(vcntq_u8(diff));
        uint32x4_t cnt32 = vpaddlq_u16(cnt16);
        acc = vaddq_u32(acc, cnt32);
    }
    total = (int32_t)vaddvq_u32(acc);
#endif

    for (; i < packed_bytes; i++) {
        uint8_t x = (uint8_t)((a[i] ^ b[i]) & mask[i]);
        /* Kernighan popcount, 8-bit. */
        while (x) { total++; x = (uint8_t)(x & (uint8_t)(x - 1u)); }
    }
    return total;
}
