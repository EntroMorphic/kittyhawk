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

/* Hamming-style distance between two packed trit buffers, masked by a
 * per-bit mask. Counts mismatching bits where the mask is set. Used as the
 * XOR+POPCNT routing distance; NEON uses VEOR + VAND + VCNT + VADDL. */
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

#ifdef __cplusplus
}
#endif

#endif /* M4T_TRIT_PACK_H */
