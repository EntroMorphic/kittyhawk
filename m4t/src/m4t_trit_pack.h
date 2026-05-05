/*
 * m4t_trit_pack.h — trit packing, unpacking, and bitwise routing primitives
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Trits live in packed uint8_t buffers using 2-bit codes (see m4t_types.h).
 * This header provides the pack/unpack path and the popcount-based Hamming
 * distance used by the routing layer. Nothing here touches float or MTFP:
 * these are pure container-level operations.
 */

#ifndef M4T_TRIT_PACK_H
#define M4T_TRIT_PACK_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pack `n` trits from a flat m4t_trit_t buffer into 2-bit codes.
 * `dst` must have at least M4T_TRIT_PACKED_BYTES(n) bytes. */
void m4t_pack_trits_1d(uint8_t* dst, const m4t_trit_t* src, int n);

/* Unpack `n` trits back to a flat m4t_trit_t buffer. */
void m4t_unpack_trits_1d(m4t_trit_t* dst, const uint8_t* src, int n);

/* Pack an [M, K] row-major ternary weight matrix. Output stride is
 * Kp = M4T_TRIT_PACKED_BYTES(K) bytes per row. */
void m4t_pack_trits_rowmajor(
    uint8_t* dst,
    const m4t_trit_t* src,
    int M, int K
);

/* Unpack an [M, K] row-major packed ternary matrix. */
void m4t_unpack_trits_rowmajor(
    m4t_trit_t* dst,
    const uint8_t* src,
    int M, int K
);

/* Ternary Hamming distance between two packed trit buffers, masked by a
 * per-bit mask. The kernel is a byte-level XOR + VCNT popcount; its
 * *ternary* interpretation is load-bearing on the 2-bit-per-trit packing
 * convention established by `m4t_pack_trits_1d` / `trit_to_code`:
 *
 *     +1 → 0b01      0 → 0b00      −1 → 0b10
 *
 * With that convention, XOR of two trit codes produces:
 *
 *     same trit (any state)            → 0b00   popcount 0   cost 0 (agree)
 *     ±1 vs 0                          → 0b01   popcount 1   cost 1 (partial)
 *     +1 vs −1                         → 0b11   popcount 2   cost 2 (opposite)
 *
 * So this function returns a ternary Hamming distance with max = 2·N trits,
 * not a binary Hamming distance with max = N. Consumers rely on this in
 * `MAX_DIST = 2 * N_PROJ` (`tools/mnist_probe_nproj16.c` and the cascade
 * tools) and in `max_dist = 2 * N_PROJ` in `tools/mnist_routed_weighted.c`.
 *
 * WARNING — guard on the packing convention. If anyone ever changes the
 * packing to a 1-bit sign-only layout (collapsing +1 and −1 to one state),
 * this function will silently start measuring a binary Hamming distance
 * and every downstream distance/threshold/vote-rule calibration will drift
 * without a build error. The 2-bit code is encoded by `trit_to_code` and
 * decoded by `M4T_TRIT_DECODE_LUT`; both must move together. See §18 of
 * `m4t/docs/M4T_SUBSTRATE.md` — emission coverage is this primitive's
 * contract with the rest of the substrate.
 *
 * NEON path: VEOR + VAND + VCNT + VADDLP (widen) + VADDV. */
int32_t m4t_popcount_dist(
    const uint8_t* a,
    const uint8_t* b,
    const uint8_t* mask,
    int packed_bytes
);

/* Decode LUT shared by trit-consuming kernels. Maps a 2-bit code in
 * positions 0..3 (bits 0-1 of each nibble) to a signed trit value. Exposed
 * so that kernels can vqtbl1q_s8 against it without redefining the table. */
extern const int8_t M4T_TRIT_DECODE_LUT[16];

/* ── §20 5-in-8 base-3 packing (opt-in dense format) ──────────────────────
 *
 * Per m4t/docs/M4T_SUBSTRATE.md §20. Packs 5 trits per byte using positional
 * base-3 encoding:
 *
 *   trit_to_unsigned: -1 → 2, 0 → 0, +1 → 1
 *   byte = u_0 + 3*u_1 + 9*u_2 + 27*u_3 + 81*u_4
 *
 * Density: 1.6 bits/cell (vs the default 2 bits/cell of m4t_pack_trits_1d).
 * Approaches the information-theoretic floor log2(3) ≈ 1.585. NO base-2
 * representation can reach below 2 bits/cell (sign + mask are independent).
 *
 * Decode: u_i = (byte / 3^i) mod 3; trit_value(u) = {0→0, 1→+1, 2→-1}.
 *
 * Trade-offs vs the default 4-in-8 packing (m4t_pack_trits_1d):
 *   - 5-in-8 has 1.25× tighter storage.
 *   - 5-in-8 has more decode cost per byte (1× div-by-9 + 5 LUT lookups
 *     vs 1× TBL lookup for 4-in-8).
 *   - Use 5-in-8 when storage bandwidth dominates; default to 4-in-8 for
 *     compute-bound paths.
 *
 * Trailing trits beyond `5 * floor(n/5)` are zero-padded in the final byte
 * (preserves byte-level encoding when n % 5 != 0). */

/* Number of bytes required to store n trits in 5-in-8 format. */
#define M4T_TRIT_PACKED5_BYTES(n) (((n) + 4) / 5)

/* Pack `n` trits into 5-in-8 format. dst must have at least
 * M4T_TRIT_PACKED5_BYTES(n) bytes. */
void m4t_pack_trits_5in8_1d(uint8_t* dst, const m4t_trit_t* src, int n);

/* Unpack `n` trits from 5-in-8 format. Companion to m4t_pack_trits_5in8_1d. */
void m4t_unpack_trits_5in8_1d(m4t_trit_t* dst, const uint8_t* src, int n);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TRIT_PACK_H */
