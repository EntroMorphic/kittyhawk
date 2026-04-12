/*
 * m4t_trit_reducers.h — masked-VCNT ternary reductions
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Two reducers that collapse a packed-trit vector into a scalar:
 *
 *   signed_sum:  count(+1 trits) - count(-1 trits)
 *   sparsity:    count(nonzero trits)
 *
 * Both use the masked-VCNT trick: AND a packed byte with 0x55 (even bits)
 * to isolate +1 codes, or 0xAA (odd bits) to isolate -1 codes, then VCNT
 * counts the set bits. The signed sum is the difference; sparsity is the
 * sum. ~15 NEON instructions per 64 trits.
 *
 * These are the building blocks for weight-derived signature computation
 * (column-sum → mean-subtract → sign-extract) in the routing layer.
 */

#ifndef M4T_TRIT_REDUCERS_H
#define M4T_TRIT_REDUCERS_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sum of trits: returns count(+1) - count(-1).
 * Equivalent to treating each trit as {-1, 0, +1} and summing.
 * Result range: [-n_trits, +n_trits]. */
int64_t m4t_trit_signed_sum(const uint8_t* packed, int n_trits);

/* Count of nonzero trits: returns count(+1) + count(-1).
 * Useful for sparsity statistics and routing-distance normalization.
 * Result range: [0, n_trits]. */
int64_t m4t_trit_sparsity(const uint8_t* packed, int n_trits);

/* Separate counts: returns count(+1) and count(-1) individually.
 * More informative than signed_sum when you need both polarities. */
void m4t_trit_counts(
    const uint8_t* packed, int n_trits,
    int64_t* out_pos, int64_t* out_neg
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TRIT_REDUCERS_H */
