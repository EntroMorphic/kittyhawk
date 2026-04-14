/*
 * m4t_ternary_matmul.h — MTFP19 activations × packed ternary weights → MTFP19
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The routing-native matmul for MTFP19 activations. This is Law #7 in
 * action: ternary projections (weights) applied to MTFP data (activations).
 * Not a dense matmul — the weights are {-1, 0, +1}, so 1/3 of every row
 * is zero by construction and the inner loop is conditional negate-and-add.
 *
 * Contract:
 *   - Activations X are m4t_mtfp_t mantissas (MTFP19).
 *   - Weights W are 2-bit packed trits in {-1, 0, +1}.
 *   - Output Y is m4t_mtfp_t mantissas (MTFP19).
 *   - Accumulation is int64; the final store is saturating clamp to
 *     ±M4T_MTFP_MAX_VAL. This is §8.5 Case S — fixed-output type, the
 *     result cannot widen without changing the caller's buffer.
 *   - Under the default-block-exponent convention: output mantissa
 *     lives at the same block_exp as the input activation mantissa
 *     (weights are pure ternary, carry no scale).
 *
 * Hardware shape:
 *   - MTFP19 × ternary is NOT SDOT-native (SDOT is int8 × int8 → int32).
 *     The SDOT-native path is MTFP4 × ternary → MTFP19, provided by
 *     m4t_mtfp4_sdot_matmul_bt. This kernel is for consumers that need
 *     full MTFP19 precision on activations.
 *   - Inner loop decodes 16 trits per iteration, conditionally negates
 *     MTFP19 activations via bit-select, and widen-accumulates into int64.
 *
 * Reference idiom: trit-decode (`vld1q_u8` → shift/mask → `vqtbl1q_s8`)
 * borrowed from trix-z's trix_ternary_matvec_i8. The MTFP19 accumulator
 * pattern is new.
 */

#ifndef M4T_TERNARY_MATMUL_H
#define M4T_TERNARY_MATMUL_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Y[M,N] = X[M,K] @ W^T[K,N]
 *
 * W is stored row-major [N, K] as packed trits: W_packed[j] is a row of
 * Kp = M4T_TRIT_PACKED_BYTES(K) bytes holding K trits LSB-first.
 *
 * Output is MTFP (int32 cells), accumulated internally in int64 and
 * clamped to ±M4T_MTFP_MAX_VAL on store.
 */
void m4t_mtfp_ternary_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_mtfp_t* X,
    const uint8_t* W_packed,
    int M, int K, int N
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TERNARY_MATMUL_H */
