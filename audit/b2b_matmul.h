/*
 * audit/b2b_matmul.h — strong-claim comparison kernels
 *
 * Per journal/tristate_strong_synthesize.md. Three NEON kernels for the
 * strong-claim test:
 *
 *   Path A  base3_packed_matmul_neon       — base-3 ternary packed via SDOT
 *   Path B  b2b_honest_matmul_neon         — B2-B sign+mask, separate decode
 *   Path B' b2b_skip_matmul_neon           — B2-B with all-masked-block skip
 *
 * All three:
 *   - NEON only. No scalar fallback. No scalar reference.
 *   - Require K % 16 == 0 (no scalar tail).
 *   - Take ternary X (int8 in {-1, 0, +1}) as activations.
 *   - Output int32 Y.
 *   - Use 2-bit-per-cell packed W storage (same density: K/4 bytes per row).
 *
 * Storage format differs:
 *   Path A:  base-3 codes (substrate's existing m4t_pack_trits_1d encoding):
 *              0b00 → 0, 0b01 → +1, 0b10 → -1.
 *   Path B:  B2-B sign + mask:
 *              bit 0 = sign (0 → +1, 1 → -1); bit 1 = mask (0 → cell, 1 → 0).
 *              0b00 → +1, 0b01 → -1, 0b10 → 0, 0b11 → 0.
 *
 * Verification is NEON-vs-NEON cross-check: all three kernels MUST produce
 * bit-exact identical Y for the same logical inputs (same ternary values
 * stored in different formats).
 *
 * Op count per 16-cell inner block (target measurement):
 *   Path A:  ~7 ops    (load + DUP/SHIFT/MASK + TBL + load X + SDOT)
 *   Path B:  ~11 ops   (load + DUP/SHIFT/MASK + sign-extract + mask-extract +
 *                       sign-TBL + mask-multiplier + multiply + load X + SDOT)
 *   Path B': ~13 ops   (Path B + reduction + branch for skip check)
 *
 * Path B's optimal degenerates to Path A if it uses a unified LUT, but that
 * erases the conceptual distinction between B2-B's separate sign/mask
 * representation and base-3's native ternary value. The strong-claim test
 * compares the HONEST representations.
 */

#ifndef B2B_MATMUL_H
#define B2B_MATMUL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pack a ternary array {-1, 0, +1} into substrate's base-3 packed format.
 * Convenience wrapper that does NOT depend on libm4t — keeps audit/ self-
 * contained at link time. Bit-identical to m4t_pack_trits_1d. */
void base3_pack(uint8_t* dst, const int8_t* src, int n);

/* Pack a ternary array into B2-B (sign + mask) format.
 *   t == 0   → 0b10 (mask=1, sign=0)
 *   t == +1  → 0b00 (mask=0, sign=0)
 *   t == -1  → 0b01 (mask=0, sign=1)
 * 4 cells per byte, LSB-first. dst must have ≥ (n+3)/4 bytes. */
void b2b_pack(uint8_t* dst, const int8_t* src, int n);

/* Path A — base-3 packed W via SDOT.
 *
 * Y[M, N] = X[M, K] @ W^T[N, K].  Inner loop processes 16 cells per
 * iteration. Op count target: 7 NEON ops per 16-cell block.
 *
 * Preconditions: K % 16 == 0; X, W_packed, Y non-NULL when M*N*K > 0;
 * Y must not alias X or W_packed. */
void base3_packed_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_packed,
    int M, int K, int N);

/* Path B — B2-B sign+mask, honest separate decode.
 *
 * Same shape as Path A; storage is B2-B sign+mask. Decode separates the
 * sign and mask bits, computes sign value via TBL, applies mask via
 * multiply, then SDOT. Op count target: ~11 NEON ops per 16-cell block.
 *
 * Preconditions: same as Path A. */
void b2b_honest_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_b2b,
    int M, int K, int N);

/* Path B' — B2-B with all-masked-block skip check.
 *
 * Same as Path B but with an early-exit check at the start of each
 * 16-cell block: if all 16 mask bits are set (block is fully masked,
 * contributes zero to the accumulator), skip the rest of the block.
 *
 * Op count target: ~13 NEON ops per 16-cell block in the no-skip case
 * (extra reduction + branch). In the skip case, ops/block drops to
 * ~5 (load + extract mask + reduction + branch). Whether this wins
 * depends on the fraction of all-masked blocks in the workload.
 *
 * skip_count_out (nullable): if non-NULL, written with the number of
 * 16-cell blocks where the skip path fired. Used by red-team R-G3 to
 * empirically count skip firing rate vs theoretical prediction.
 *
 * Preconditions: same as Path A. */
void b2b_skip_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_b2b,
    int M, int K, int N,
    int* skip_count_out);

/* Path C — B2-B optimal: unified TBL decode (same op shape as Path A).
 *
 * Per red-team C1: an aggressively-optimized B2-B implementation uses
 * a 4-entry LUT mapping the 2-bit code directly to the signed value,
 * exactly like Path A. The kernel structure is byte-for-byte identical
 * to Path A; only the LUT contents differ:
 *   Path A LUT:  [0, +1, -1, 0]   (base-3 encoding)
 *   Path C LUT:  [+1, -1, 0, 0]   (B2-B encoding: bit0=sign, bit1=mask)
 *
 * Op count: identical to Path A (7 ops per 16-cell block).
 *
 * This kernel exists to surface that "base-3" vs "B2-B" is a labeling
 * choice at the 2-bits/cell density. A skilled implementer wouldn't
 * decode sign and mask separately (Path B); they would use a unified
 * LUT (Path C). Path C ≡ Path A in op shape; the structural-advantage
 * claim from the original synthesis is therefore contingent on
 * enforcing Path B's bit-by-bit decode, not Path C's optimal decode.
 *
 * Preconditions: same as Path A. Operates on B2-B-encoded packed W. */
void b2b_optimal_matmul_neon(
    int32_t* Y,
    const int8_t* X,
    const uint8_t* W_b2b,
    int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* B2B_MATMUL_H */
