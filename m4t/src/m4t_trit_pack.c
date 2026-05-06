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

/* Per project rule (feedback_function_over_speed_no_scalar): production
 * pack/unpack paths are NEON-only. _scalar helpers are reused by the
 * public _scalar_ref test oracles. */

static void pack_trits_1d_scalar(uint8_t* dst, const m4t_trit_t* src, int n) {
    int nb = M4T_TRIT_PACKED_BYTES(n);
    memset(dst, 0, (size_t)nb);
    for (int i = 0; i < n; i++) {
        uint8_t code = trit_to_code(src[i]);
        dst[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

static void unpack_trits_1d_scalar(m4t_trit_t* dst, const uint8_t* src, int n) {
    for (int i = 0; i < n; i++) {
        uint8_t code = (uint8_t)((src[i >> 2] >> ((i & 3) * 2)) & 0x3u);
        dst[i] = code_to_trit(code);
    }
}

#if M4T_HAS_NEON

/* TBL constants for 4-in-8 pack/unpack. */
static const uint8_t M4T_PACK_LUT[16] __attribute__((aligned(16))) = {
    /* (trit & 3) → 2-bit code. trit ∈ {-1, 0, +1}:
     *   +1 (0x01) & 3 = 1 → code 0b01 = 1
     *    0 (0x00) & 3 = 0 → code 0b00 = 0
     *   -1 (0xff) & 3 = 3 → code 0b10 = 2 */
    0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static const uint8_t M4T_PLACE[16] __attribute__((aligned(16))) = {
    /* Per-lane shift-equivalent: codes are 2 bits, place k = 4^(k%4). */
    1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64
};
static const uint8_t M4T_UNPACK_LUT[16] __attribute__((aligned(16))) = {
    /* Code (0..3) → trit:
     *   0b00 → 0,  0b01 → +1,  0b10 → -1,  0b11 → 0 (reserved) */
    0, 1, (uint8_t)-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
/* Replicate index pattern: each input byte i is replicated 4 times in lanes
 * 4i..4i+3 of the 16-lane output. Used by unpack to expand 4 bytes → 16 codes. */
static const uint8_t M4T_REPLICATE_4[16] __attribute__((aligned(16))) = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
};
/* Per-lane right-shift counts (signed, negative = right shift) to extract
 * 2-bit codes at positions 0/2/4/6 within each replicated byte. */
static const int8_t M4T_PACK_SHIFTS[16] __attribute__((aligned(16))) = {
    0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6
};

/* Pack 16 trits → 4 bytes. */
static inline void pack_block_16(uint8_t* dst4, const m4t_trit_t* src16,
                                  uint8x16_t lut, uint8x16_t place,
                                  uint8x16_t mask3)
{
    int8x16_t trits = vld1q_s8((const int8_t*)src16);
    uint8x16_t codes = vqtbl1q_u8(lut, vandq_u8(
        vreinterpretq_u8_s8(trits), mask3));
    uint8x16_t shifted = vmulq_u8(codes, place);
    /* 16-lane reduce-OR via two pairwise additions (codes occupy
     * disjoint bit positions, so add ≡ OR for the whole block). */
    uint8x16_t r1 = vpaddq_u8(shifted, shifted);
    uint8x16_t r2 = vpaddq_u8(r1, r1);
    /* Lanes 0..3 of r2 are the 4 output bytes. */
    uint32_t word;
    word = vgetq_lane_u32(vreinterpretq_u32_u8(r2), 0);
    memcpy(dst4, &word, 4);
}

/* Unpack 4 bytes → 16 trits. */
static inline void unpack_block_4(m4t_trit_t* dst16, const uint8_t* src4,
                                   uint8x16_t lut, uint8x16_t replicate,
                                   int8x16_t shifts, uint8x16_t mask3)
{
    /* Load 4 bytes as a 16-byte vector (only first 4 lanes valid). */
    uint32_t word;
    memcpy(&word, src4, 4);
    uint8x16_t v = vreinterpretq_u8_u32(vdupq_n_u32(word));
    /* Replicate: lanes 0..3 ← byte 0; 4..7 ← byte 1; 8..11 ← byte 2; 12..15 ← byte 3. */
    uint8x16_t replicated = vqtbl1q_u8(v, replicate);
    /* Right-shift each lane by 0/2/4/6 to put the desired 2-bit code in low bits. */
    uint8x16_t shifted = vreinterpretq_u8_s8(
        vshlq_s8(vreinterpretq_s8_u8(replicated), shifts));
    /* Extract low 2 bits → code in [0, 3]. */
    uint8x16_t codes = vandq_u8(shifted, mask3);
    /* Decode to trits via TBL. */
    int8x16_t trits = vqtbl1q_s8(vreinterpretq_s8_u8(lut), codes);
    vst1q_s8((int8_t*)dst16, trits);
}

static void pack_trits_1d_neon(uint8_t* dst, const m4t_trit_t* src, int n) {
    int nb = M4T_TRIT_PACKED_BYTES(n);
    if (nb == 0) return;
    /* Need to zero only the partial-tail byte; full blocks are
     * fully overwritten by the pack_block_16 store. */
    int n_full_blocks = n / 16;
    int n_full_trits  = n_full_blocks * 16;

    uint8x16_t lut   = vld1q_u8(M4T_PACK_LUT);
    uint8x16_t place = vld1q_u8(M4T_PLACE);
    uint8x16_t mask3 = vdupq_n_u8(0x03);

    for (int b = 0; b < n_full_blocks; b++) {
        pack_block_16(dst + b * 4, src + b * 16, lut, place, mask3);
    }
    /* Geometric scalar tail (n % 16 trits). */
    int trit_tail = n - n_full_trits;
    if (trit_tail > 0) {
        int dst_tail_start = n_full_blocks * 4;
        int tail_bytes = nb - dst_tail_start;
        memset(dst + dst_tail_start, 0, (size_t)tail_bytes);
        for (int i = 0; i < trit_tail; i++) {
            uint8_t code = trit_to_code(src[n_full_trits + i]);
            dst[dst_tail_start + (i >> 2)] |= (uint8_t)(code << ((i & 3) * 2));
        }
    }
}

static void unpack_trits_1d_neon(m4t_trit_t* dst, const uint8_t* src, int n) {
    int n_full_blocks = n / 16;
    int n_full_trits  = n_full_blocks * 16;

    uint8x16_t lut       = vld1q_u8(M4T_UNPACK_LUT);
    uint8x16_t replicate = vld1q_u8(M4T_REPLICATE_4);
    int8x16_t  shifts    = vld1q_s8((const int8_t*)M4T_PACK_SHIFTS);
    uint8x16_t mask3     = vdupq_n_u8(0x03);

    for (int b = 0; b < n_full_blocks; b++) {
        unpack_block_4(dst + b * 16, src + b * 4, lut, replicate, shifts, mask3);
    }
    /* Geometric scalar tail. */
    for (int i = n_full_trits; i < n; i++) {
        uint8_t code = (uint8_t)((src[i >> 2] >> ((i & 3) * 2)) & 0x3u);
        dst[i] = code_to_trit(code);
    }
}

#endif /* M4T_HAS_NEON */

void m4t_pack_trits_1d(uint8_t* dst, const m4t_trit_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    /* NEON-only production dispatch. Scalar reference via _scalar_ref. */
    pack_trits_1d_neon(dst, src, n);
}

void m4t_unpack_trits_1d(m4t_trit_t* dst, const uint8_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    unpack_trits_1d_neon(dst, src, n);
}

void m4t_pack_trits_1d_scalar_ref(uint8_t* dst, const m4t_trit_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    pack_trits_1d_scalar(dst, src, n);
}

void m4t_unpack_trits_1d_scalar_ref(m4t_trit_t* dst, const uint8_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    unpack_trits_1d_scalar(dst, src, n);
}

void m4t_pack_trits_rowmajor(
    uint8_t* dst, const m4t_trit_t* src, int M, int K)
{
    assert(M >= 0 && K >= 0);
    if (M == 0 || K == 0) return;
    assert(dst && src);
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    memset(dst, 0, (size_t)M * (size_t)Kp);
    for (int m = 0; m < M; m++) {
        m4t_pack_trits_1d(dst + (size_t)m * Kp, src + (size_t)m * K, K);
    }
}

void m4t_unpack_trits_rowmajor(
    m4t_trit_t* dst, const uint8_t* src, int M, int K)
{
    assert(M >= 0 && K >= 0);
    if (M == 0 || K == 0) return;
    assert(dst && src);
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    for (int m = 0; m < M; m++) {
        m4t_unpack_trits_1d(dst + (size_t)m * K, src + (size_t)m * Kp, K);
    }
}

/* ── §20 5-in-8 base-3 packing ────────────────────────────────────────── */

/* Per M4T_SUBSTRATE.md §20.
 *   trit_to_unsigned: -1 → 2, 0 → 0, +1 → 1.
 *   byte = u_0 + 3*u_1 + 9*u_2 + 27*u_3 + 81*u_4. */
static inline uint8_t trit5_to_unsigned(m4t_trit_t t) {
    assert(t >= -1 && t <= 1);
    /* {-1, 0, +1} → {2, 0, 1}. Fallback to 0 in NDEBUG for noise. */
    return (t == 1) ? 1u : (t == -1) ? 2u : 0u;
}

static inline m4t_trit_t unsigned_to_trit5(uint8_t u) {
    /* {0, 1, 2} → {0, +1, -1}. Other values map to 0 (defensive). */
    return (u == 1u) ? (m4t_trit_t)1
         : (u == 2u) ? (m4t_trit_t)-1
         :              (m4t_trit_t)0;
}

/* Per project rule (feedback_function_over_speed_no_scalar): production
 * 5-in-8 pack/unpack paths are NEON-only. _scalar helpers are reused
 * by the public _scalar_ref test oracles. */

static void pack_trits_5in8_1d_scalar(uint8_t* dst, const m4t_trit_t* src, int n) {
    int nb = M4T_TRIT_PACKED5_BYTES(n);
    memset(dst, 0, (size_t)nb);
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
    for (int i = 0; i < n; i++) {
        uint8_t u = trit5_to_unsigned(src[i]);
        int byte_idx  = i / 5;
        int digit_pos = i % 5;
        dst[byte_idx] = (uint8_t)(dst[byte_idx] + u * POW3[digit_pos]);
    }
}

static void unpack_trits_5in8_1d_scalar(m4t_trit_t* dst, const uint8_t* src, int n) {
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
    for (int i = 0; i < n; i++) {
        int byte_idx  = i / 5;
        int digit_pos = i % 5;
        uint8_t u = (uint8_t)((src[byte_idx] / POW3[digit_pos]) % 3u);
        dst[i] = unsigned_to_trit5(u);
    }
}

#if M4T_HAS_NEON

/* 5-in-8 split-LUT decode constants (mirror of the §20 matmul tables in
 * m4t_ternary_matmul.c — duplicated here per project rule of avoiding
 * cross-TU symbol coupling unless required). LUT contents map directly
 * to trit values: trit_value(u) = {0→0, 1→+1, 2→-1}.
 *
 * For 'low' input ∈ [0, 8] (= byte mod 9):
 *   d0 = (low) % 3       — first digit's unsigned code → trit
 *   d1 = (low) / 3       — second digit's unsigned code → trit
 * For 'high' input ∈ [0, 26] (= byte / 9):
 *   d2 = (high) % 3
 *   d3 = (high / 3) % 3
 *   d4 = (high / 9) % 3 */
static const int8_t M4T_5IN8_PACK_LOW_D0[16] __attribute__((aligned(16))) = {
     0,  1, -1,  0,  1, -1,  0,  1, -1,
};
static const int8_t M4T_5IN8_PACK_LOW_D1[16] __attribute__((aligned(16))) = {
     0,  0,  0,  1,  1,  1, -1, -1, -1,
};
static const int8_t M4T_5IN8_PACK_HIGH_D2[32] __attribute__((aligned(16))) = {
     0,  1, -1,  0,  1, -1,  0,  1, -1,
     0,  1, -1,  0,  1, -1,  0,  1, -1,
     0,  1, -1,  0,  1, -1,  0,  1, -1,
};
static const int8_t M4T_5IN8_PACK_HIGH_D3[32] __attribute__((aligned(16))) = {
     0,  0,  0,  1,  1,  1, -1, -1, -1,
     0,  0,  0,  1,  1,  1, -1, -1, -1,
     0,  0,  0,  1,  1,  1, -1, -1, -1,
};
static const int8_t M4T_5IN8_PACK_HIGH_D4[32] __attribute__((aligned(16))) = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,
     1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

/* Forward map for pack: trit → unsigned code via LUT[(trit & 3)]:
 *   +1 (0x01) & 3 = 1 → 1
 *    0 (0x00) & 3 = 0 → 0
 *   -1 (0xff) & 3 = 3 → 2 */
static const uint8_t M4T_5IN8_PACK_TRIT_TO_U[16] __attribute__((aligned(16))) = {
    0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* Stride-5 (de)interleave index tables. NEON has no LD5/ST5 instruction,
 * so we synthesize stride-5 access via vqtbl4q (4-vector / 64-lane lookup)
 * + vqtbl1q (16-lane lookup of the 5th vector) + vorrq combine. Out-of-
 * range table indices return 0 (vqtbl4q: idx≥64; vqtbl1q: idx≥16), which
 * lets the two TBLs OR cleanly without overlap.
 *
 * Pack (deinterleave) — 5 input src_vec[0..4] → 5 output stride-d vectors:
 *   For output stride d ∈ [0, 5), output lane i pulls input position 5i+d.
 *   With 5 input vectors of 16 lanes (= 80 input lanes), positions 0..63
 *   are in src_vec[0..3] (vqtbl4q domain) and 64..79 are in src_vec[4]
 *   (vqtbl1q domain).
 *
 * Unpack (interleave) — 5 stride-d input d_vec[0..4] → 5 output vectors:
 *   For output vector v ∈ [0, 5), output lane i (= overall lane 16v+i)
 *   pulls d=(16v+i)%5 from d_vec[d] at src_lane=(16v+i)/5. */
static const uint8_t M4T_5IN8_PACK_TBL4_IDX[5][16] __attribute__((aligned(16))) = {
    /* v_0 deinterleave: positions 0,5,10,...,75 — first 13 in tbl4 domain */
    { 0, 5,10,15,20,25,30,35,40,45,50,55,60,64,64,64 },
    /* v_1 */
    { 1, 6,11,16,21,26,31,36,41,46,51,56,61,64,64,64 },
    /* v_2 */
    { 2, 7,12,17,22,27,32,37,42,47,52,57,62,64,64,64 },
    /* v_3 */
    { 3, 8,13,18,23,28,33,38,43,48,53,58,63,64,64,64 },
    /* v_4 — position 64 is OOR for tbl4 (idx 64 == 0 lane); only first 12 valid */
    { 4, 9,14,19,24,29,34,39,44,49,54,59,64,64,64,64 },
};
static const uint8_t M4T_5IN8_PACK_TBL1_IDX[5][16] __attribute__((aligned(16))) = {
    /* v_0: only last 3 lanes use tbl1 (positions 65, 70, 75 → src[1, 6, 11]) */
    { 16,16,16,16,16,16,16,16,16,16,16,16,16, 1, 6,11 },
    { 16,16,16,16,16,16,16,16,16,16,16,16,16, 2, 7,12 },
    { 16,16,16,16,16,16,16,16,16,16,16,16,16, 3, 8,13 },
    { 16,16,16,16,16,16,16,16,16,16,16,16,16, 4, 9,14 },
    /* v_4: last 4 lanes use tbl1 (positions 64, 69, 74, 79 → src[0, 5, 10, 15]) */
    { 16,16,16,16,16,16,16,16,16,16,16,16, 0, 5,10,15 },
};
static const uint8_t M4T_5IN8_UNPACK_TBL4_IDX[5][16] __attribute__((aligned(16))) = {
    /* output_vec[0] — interleaves d0..d3 with OOR for d4 lanes */
    {  0,16,32,48,64, 1,17,33,49,64, 2,18,34,50,64, 3 },
    /* output_vec[1] */
    { 19,35,51,64, 4,20,36,52,64, 5,21,37,53,64, 6,22 },
    /* output_vec[2] */
    { 38,54,64, 7,23,39,55,64, 8,24,40,56,64, 9,25,41 },
    /* output_vec[3] */
    { 57,64,10,26,42,58,64,11,27,43,59,64,12,28,44,60 },
    /* output_vec[4] */
    { 64,13,29,45,61,64,14,30,46,62,64,15,31,47,63,64 },
};
static const uint8_t M4T_5IN8_UNPACK_TBL1_IDX[5][16] __attribute__((aligned(16))) = {
    /* d4 contributions; OOR (16) elsewhere */
    { 16,16,16,16, 0,16,16,16,16, 1,16,16,16,16, 2,16 },
    { 16,16,16, 3,16,16,16,16, 4,16,16,16,16, 5,16,16 },
    { 16,16, 6,16,16,16,16, 7,16,16,16,16, 8,16,16,16 },
    { 16, 9,16,16,16,16,10,16,16,16,16,11,16,16,16,16 },
    { 12,16,16,16,16,13,16,16,16,16,14,16,16,16,16,15 },
};

static void pack_trits_5in8_1d_neon(uint8_t* dst, const m4t_trit_t* src, int n) {
    int nb = M4T_TRIT_PACKED5_BYTES(n);
    if (nb == 0) return;

    int n_full_blocks = n / 80;     /* 80 trits → 16 packed bytes per iter */
    int n_full_trits  = n_full_blocks * 80;

    uint8x16_t lut_u = vld1q_u8(M4T_5IN8_PACK_TRIT_TO_U);
    uint8x16_t mask3 = vdupq_n_u8(0x03);

    /* Pre-load TBL index vectors (compile-time constants; hoisted by compiler). */
    uint8x16_t idx4_v0 = vld1q_u8(M4T_5IN8_PACK_TBL4_IDX[0]);
    uint8x16_t idx4_v1 = vld1q_u8(M4T_5IN8_PACK_TBL4_IDX[1]);
    uint8x16_t idx4_v2 = vld1q_u8(M4T_5IN8_PACK_TBL4_IDX[2]);
    uint8x16_t idx4_v3 = vld1q_u8(M4T_5IN8_PACK_TBL4_IDX[3]);
    uint8x16_t idx4_v4 = vld1q_u8(M4T_5IN8_PACK_TBL4_IDX[4]);
    uint8x16_t idx1_v0 = vld1q_u8(M4T_5IN8_PACK_TBL1_IDX[0]);
    uint8x16_t idx1_v1 = vld1q_u8(M4T_5IN8_PACK_TBL1_IDX[1]);
    uint8x16_t idx1_v2 = vld1q_u8(M4T_5IN8_PACK_TBL1_IDX[2]);
    uint8x16_t idx1_v3 = vld1q_u8(M4T_5IN8_PACK_TBL1_IDX[3]);
    uint8x16_t idx1_v4 = vld1q_u8(M4T_5IN8_PACK_TBL1_IDX[4]);

    for (int b = 0; b < n_full_blocks; b++) {
        const int8_t* p = (const int8_t*)(src + b * 80);
        /* Load 80 contiguous trits as 5 × 16-lane vectors. */
        int8x16x4_t tbl4 = { {
            vld1q_s8(p +  0), vld1q_s8(p + 16),
            vld1q_s8(p + 32), vld1q_s8(p + 48),
        } };
        int8x16_t tbl1 = vld1q_s8(p + 64);

        /* Stride-5 deinterleave: 5 vectors of 16 trits each, where v_d
         * holds trits at positions 5*lane + d. */
        int8x16_t v0 = vorrq_s8(vqtbl4q_s8(tbl4, idx4_v0), vqtbl1q_s8(tbl1, idx1_v0));
        int8x16_t v1 = vorrq_s8(vqtbl4q_s8(tbl4, idx4_v1), vqtbl1q_s8(tbl1, idx1_v1));
        int8x16_t v2 = vorrq_s8(vqtbl4q_s8(tbl4, idx4_v2), vqtbl1q_s8(tbl1, idx1_v2));
        int8x16_t v3 = vorrq_s8(vqtbl4q_s8(tbl4, idx4_v3), vqtbl1q_s8(tbl1, idx1_v3));
        int8x16_t v4 = vorrq_s8(vqtbl4q_s8(tbl4, idx4_v4), vqtbl1q_s8(tbl1, idx1_v4));

        /* Map each trit → unsigned code via LUT[(trit & 3)]. */
        uint8x16_t u0 = vqtbl1q_u8(lut_u, vandq_u8(vreinterpretq_u8_s8(v0), mask3));
        uint8x16_t u1 = vqtbl1q_u8(lut_u, vandq_u8(vreinterpretq_u8_s8(v1), mask3));
        uint8x16_t u2 = vqtbl1q_u8(lut_u, vandq_u8(vreinterpretq_u8_s8(v2), mask3));
        uint8x16_t u3 = vqtbl1q_u8(lut_u, vandq_u8(vreinterpretq_u8_s8(v3), mask3));
        uint8x16_t u4 = vqtbl1q_u8(lut_u, vandq_u8(vreinterpretq_u8_s8(v4), mask3));

        /* Multiply by POW3[d]: 1, 3, 9, 27, 81. Max product 2*81 = 162 < 256.
         * NEON has no vmulq_n_u8 (no scalar-by-vector u8 multiply intrinsic);
         * broadcast the scalar via vdupq_n_u8 and use vmulq_u8. */
        uint8x16_t s1 = vmulq_u8(u1, vdupq_n_u8(3));
        uint8x16_t s2 = vmulq_u8(u2, vdupq_n_u8(9));
        uint8x16_t s3 = vmulq_u8(u3, vdupq_n_u8(27));
        uint8x16_t s4 = vmulq_u8(u4, vdupq_n_u8(81));
        /* Sum: max 2 + 6 + 18 + 54 + 162 = 242 < 256, no overflow. */
        uint8x16_t byte = vaddq_u8(vaddq_u8(u0, s1),
                                   vaddq_u8(s2, vaddq_u8(s3, s4)));
        vst1q_u8(dst + b * 16, byte);
    }
    /* Geometric scalar tail (trits beyond the last full 80-trit block). */
    int trit_tail = n - n_full_trits;
    if (trit_tail > 0) {
        int dst_tail_start = n_full_blocks * 16;
        int tail_bytes = nb - dst_tail_start;
        memset(dst + dst_tail_start, 0, (size_t)tail_bytes);
        static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
        for (int i = 0; i < trit_tail; i++) {
            uint8_t u = trit5_to_unsigned(src[n_full_trits + i]);
            int byte_idx  = i / 5;
            int digit_pos = i % 5;
            dst[dst_tail_start + byte_idx] =
                (uint8_t)(dst[dst_tail_start + byte_idx] + u * POW3[digit_pos]);
        }
    }
}

static void unpack_trits_5in8_1d_neon(m4t_trit_t* dst, const uint8_t* src, int n) {
    int n_full_blocks = n / 80;
    int n_full_trits  = n_full_blocks * 80;

    const uint8x16_t nine_v = vdupq_n_u8(9);
    const int8x16_t  lut_d0 = vld1q_s8(M4T_5IN8_PACK_LOW_D0);
    const int8x16_t  lut_d1 = vld1q_s8(M4T_5IN8_PACK_LOW_D1);
    const int8x16x2_t lut_d2 = { {
        vld1q_s8(M4T_5IN8_PACK_HIGH_D2 +  0),
        vld1q_s8(M4T_5IN8_PACK_HIGH_D2 + 16),
    } };
    const int8x16x2_t lut_d3 = { {
        vld1q_s8(M4T_5IN8_PACK_HIGH_D3 +  0),
        vld1q_s8(M4T_5IN8_PACK_HIGH_D3 + 16),
    } };
    const int8x16x2_t lut_d4 = { {
        vld1q_s8(M4T_5IN8_PACK_HIGH_D4 +  0),
        vld1q_s8(M4T_5IN8_PACK_HIGH_D4 + 16),
    } };

    /* Pre-load interleave index vectors. */
    uint8x16_t idx4[5], idx1[5];
    for (int v = 0; v < 5; v++) {
        idx4[v] = vld1q_u8(M4T_5IN8_UNPACK_TBL4_IDX[v]);
        idx1[v] = vld1q_u8(M4T_5IN8_UNPACK_TBL1_IDX[v]);
    }

    for (int b = 0; b < n_full_blocks; b++) {
        uint8x16_t bv = vld1q_u8(src + b * 16);
        /* Split-LUT decode: high = byte/9 via magic mul (b*57)>>9; low = b - 9*high. */
        uint16x8_t lo16 = vshrq_n_u16(
            vmulq_n_u16(vmovl_u8(vget_low_u8(bv)), 57), 9);
        uint16x8_t hi16 = vshrq_n_u16(
            vmulq_n_u16(vmovl_u8(vget_high_u8(bv)), 57), 9);
        uint8x16_t high = vcombine_u8(vmovn_u16(lo16), vmovn_u16(hi16));
        uint8x16_t low  = vsubq_u8(bv, vmulq_u8(high, nine_v));

        /* 5 stride-d trit vectors (decoded from 16 packed bytes). */
        int8x16_t d0 = vqtbl1q_s8(lut_d0, low);
        int8x16_t d1 = vqtbl1q_s8(lut_d1, low);
        int8x16_t d2 = vqtbl2q_s8(lut_d2, high);
        int8x16_t d3 = vqtbl2q_s8(lut_d3, high);
        int8x16_t d4 = vqtbl2q_s8(lut_d4, high);

        /* Build 4-vector table for vqtbl4 (d0..d3) and 1-vector for vqtbl1 (d4). */
        int8x16x4_t tbl4 = { { d0, d1, d2, d3 } };

        /* Stride-5 interleave via 5 (vqtbl4 + vqtbl1 + OR) sequences →
         * 5 × 16-byte output vectors, written contiguously. */
        int8_t* out = (int8_t*)(dst + b * 80);
        for (int v = 0; v < 5; v++) {
            int8x16_t r4 = vqtbl4q_s8(tbl4, idx4[v]);
            int8x16_t r1 = vqtbl1q_s8(d4,   idx1[v]);
            vst1q_s8(out + v * 16, vorrq_s8(r4, r1));
        }
    }
    /* Geometric scalar tail. */
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
    for (int i = n_full_trits; i < n; i++) {
        int byte_idx  = i / 5;
        int digit_pos = i % 5;
        uint8_t u = (uint8_t)((src[byte_idx] / POW3[digit_pos]) % 3u);
        dst[i] = unsigned_to_trit5(u);
    }
}

#endif /* M4T_HAS_NEON */

void m4t_pack_trits_5in8_1d(uint8_t* dst, const m4t_trit_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    pack_trits_5in8_1d_neon(dst, src, n);
}

void m4t_unpack_trits_5in8_1d(m4t_trit_t* dst, const uint8_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    unpack_trits_5in8_1d_neon(dst, src, n);
}

void m4t_pack_trits_5in8_1d_scalar_ref(uint8_t* dst, const m4t_trit_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    pack_trits_5in8_1d_scalar(dst, src, n);
}

void m4t_unpack_trits_5in8_1d_scalar_ref(m4t_trit_t* dst, const uint8_t* src, int n) {
    assert(n >= 0);
    if (n == 0) return;
    assert(dst && src);
    unpack_trits_5in8_1d_scalar(dst, src, n);
}

/* ── Popcount distance with mask ───────────────────────────────────────── */

int32_t m4t_popcount_dist(
    const uint8_t* a, const uint8_t* b, const uint8_t* mask, int packed_bytes)
{
    assert(a && b && mask);
    /* Ternary Hamming distance; see guard comment in m4t_trit_pack.h.
     * Correctness depends on the 2-bit trit codes (+1=0b01, 0=0b00,
     * -1=0b10). Per-position cost is 0/1/2 from XOR popcount on each
     * 2-bit field; max distance is 2·N trits, not N.
     *
     * Three-tier implementation covering every N_PROJ regime cleanly:
     *
     *   1. NEON 16-byte loop (packed_bytes ≥ 16, i.e. N_PROJ ≥ 64).
     *      VEOR + VAND + VCNT + pairwise widen accumulate.
     *   2. Scalar 8-byte loop via __builtin_popcountll for any
     *      packed_bytes ≥ 8 that didn't divide into 16. Covers the
     *      N_PROJ=32 fast path (sig_bytes=8) and the 8-byte chunk
     *      of any tail after the NEON loop.
     *   3. Scalar 4-byte loop via __builtin_popcount for any
     *      packed_bytes ≥ 4 that didn't divide into 8. Covers the
     *      N_PROJ=16 fast path (sig_bytes=4) — critical because at
     *      N_PROJ=16 the NEON loop never triggers, so without this
     *      path every popcount_dist call fell through to a
     *      Kernighan-on-uint8 tail, ~24× slower than the hardware
     *      CNT instruction.
     *   4. Byte tail via __builtin_popcount for the last 0–3 bytes.
     *
     * The compiler lowers __builtin_popcount{,ll} to the ARMv8 CNT
     * instruction when targeting M-series silicon (repo is built with
     * -mcpu=native). Bit-exact equivalent to the prior scalar path;
     * accuracy is preserved because the operation being computed is
     * still popcount(XOR & mask). */
    int32_t total = 0;
    int i = 0;

#if M4T_HAS_NEON
    {
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
        total += (int32_t)vaddvq_u32(acc);
    }
#endif

    /* 8-byte chunks via __builtin_popcountll. Handles N_PROJ=32 and
     * the 8-byte portion of any tail after the NEON loop. Uses memcpy
     * for aligned-load-agnostic portability — the compiler folds it
     * into a single LDR instruction. */
    for (; i + 8 <= packed_bytes; i += 8) {
        uint64_t a64, b64, m64;
        memcpy(&a64, a    + i, 8);
        memcpy(&b64, b    + i, 8);
        memcpy(&m64, mask + i, 8);
        total += (int32_t)__builtin_popcountll((a64 ^ b64) & m64);
    }

    /* 4-byte chunks via __builtin_popcount. Handles the critical
     * N_PROJ=16 fast path (sig_bytes=4) and any 4-byte remainder
     * after the 8-byte loop. */
    for (; i + 4 <= packed_bytes; i += 4) {
        uint32_t a32, b32, m32;
        memcpy(&a32, a    + i, 4);
        memcpy(&b32, b    + i, 4);
        memcpy(&m32, mask + i, 4);
        total += (int32_t)__builtin_popcount((a32 ^ b32) & m32);
    }

    /* Byte tail via __builtin_popcount (0–3 bytes remaining). */
    for (; i < packed_bytes; i++) {
        total += (int32_t)__builtin_popcount(
            (unsigned int)((a[i] ^ b[i]) & mask[i]));
    }
    return total;
}
