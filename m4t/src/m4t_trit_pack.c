/*
 * m4t_trit_pack.c — trit packing, unpacking, popcount distance
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Pure container-level operations. No MTFP, no float.
 */

#include "m4t_trit_pack.h"
#include "m4t_internal.h"
#include <assert.h>
#include <string.h>

/* Lock the 2-bit-per-trit packing: 4 trits per byte. Any refactor that
 * widens to sign-only (8 trits per byte) or narrows to a larger code
 * changes the ternary Hamming interpretation of m4t_popcount_dist — see
 * the guard comment on that function in m4t_trit_pack.h. */
_Static_assert(
    M4T_TRIT_PACKED_BYTES(4) == 1 && M4T_TRIT_PACKED_BYTES(8) == 2,
    "m4t_trit_pack: packing ratio is load-bearing (4 trits per byte, "
    "2 bits per trit). m4t_popcount_dist measures ternary Hamming only "
    "under this layout; see m4t_trit_pack.h guard comment."
);

/* Decode LUT: maps 2-bit trit codes to signed trit values.
 * Pattern repeats 4× so the same LUT drives vqtbl1q_s8 against any 16-byte
 * register of codes in [0, 3]. */
const int8_t M4T_TRIT_DECODE_LUT[16] __attribute__((aligned(16))) = {
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
     0,  1, -1,  0,
};

/* ── Pack / unpack ─────────────────────────────────────────────────────── */

static inline uint8_t trit_to_code(m4t_trit_t t) {
    /* {-1, 0, +1} → {0b10, 0b00, 0b01}.
     * In debug builds we fail loudly on out-of-range inputs — silently
     * mapping noise to "zero trit" hides bugs in trit generators. In
     * release builds (NDEBUG) the non-{-1,0,+1} fallback still collapses
     * to zero for defense in depth. */
    assert(t >= -1 && t <= 1);
    return (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
}

static inline m4t_trit_t code_to_trit(uint8_t code) {
    /* 0b00 → 0, 0b01 → +1, 0b10 → -1, 0b11 → 0 (reserved). */
    return (code == 0x01u) ? 1 : (code == 0x02u) ? -1 : 0;
}

void m4t_pack_trits_1d(uint8_t* dst, const m4t_trit_t* src, int n) {
    int nb = M4T_TRIT_PACKED_BYTES(n);
    memset(dst, 0, (size_t)nb);
    for (int i = 0; i < n; i++) {
        uint8_t code = trit_to_code(src[i]);
        dst[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

void m4t_unpack_trits_1d(m4t_trit_t* dst, const uint8_t* src, int n) {
    for (int i = 0; i < n; i++) {
        uint8_t code = (uint8_t)((src[i >> 2] >> ((i & 3) * 2)) & 0x3u);
        dst[i] = code_to_trit(code);
    }
}

void m4t_pack_trits_rowmajor(
    uint8_t* dst, const m4t_trit_t* src, int M, int K)
{
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    memset(dst, 0, (size_t)M * (size_t)Kp);
    for (int m = 0; m < M; m++) {
        m4t_pack_trits_1d(dst + (size_t)m * Kp, src + (size_t)m * K, K);
    }
}

void m4t_unpack_trits_rowmajor(
    m4t_trit_t* dst, const uint8_t* src, int M, int K)
{
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    for (int m = 0; m < M; m++) {
        m4t_unpack_trits_1d(dst + (size_t)m * K, src + (size_t)m * Kp, K);
    }
}

/* ── Popcount distance with mask ───────────────────────────────────────── */

int32_t m4t_popcount_dist(
    const uint8_t* a, const uint8_t* b, const uint8_t* mask, int packed_bytes)
{
    /* Ternary Hamming distance; see guard comment in m4t_trit_pack.h.
     * Correctness depends on the 2-bit trit codes (+1=0b01, 0=0b00,
     * -1=0b10). Per-position cost is 0/1/2 from XOR popcount on each
     * 2-bit field; max distance is 2·N trits, not N. */
    int32_t total = 0;
    int i = 0;

#if M4T_HAS_NEON
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
