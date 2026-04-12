/*
 * m4t_ternary_matmul.h — MTFP activations × packed ternary weights → MTFP
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * This is m4t's forward matmul kernel for routed FFN and routed projection
 * layers. Activations are m4t_mtfp_t (int32 ternary fixed-point); weights
 * are 2-bit packed trits; output is m4t_mtfp_t. Inner loop is pure integer
 * add/subtract (no multiply) — the "multiply" by a trit is a conditional
 * negate-and-add into an int64 lane accumulator.
 *
 * Reference idiom: trit-decode borrowed from trix-z's trix_ternary_matvec_i8
 * (`vld1q_u8` → shift/mask → `vqtbl1q_s8`). The accumulator is NEW — the
 * reference has no kernel that combines packed-trit weights with MTFP
 * activations.
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
