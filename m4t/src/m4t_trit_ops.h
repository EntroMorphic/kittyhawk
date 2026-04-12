/*
 * m4t_trit_ops.h — TBL-based binary trit operations
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Six opcodes that operate element-wise on packed-trit buffers. Five use
 * a 16-byte NEON TBL lookup (one cycle per 64 trits); neg uses a bit-swap
 * (five instructions per 64 trits).
 *
 * Calling convention: all ops take two packed-trit inputs and a length in
 * trits. For unary ops (neg), the second input is ignored but present for
 * API uniformity. dst may alias a or b.
 */

#ifndef M4T_TRIT_OPS_H
#define M4T_TRIT_OPS_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* dst[i] = a[i] * b[i]   (F_3 multiply: {-1,0,+1} × {-1,0,+1} → {-1,0,+1}) */
void m4t_trit_mul(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

/* dst[i] = clamp(a[i] + b[i], -1, +1)   (saturating ternary add) */
void m4t_trit_sat_add(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

/* dst[i] = max(a[i], b[i])   (ordered: -1 < 0 < +1) */
void m4t_trit_max(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

/* dst[i] = min(a[i], b[i]) */
void m4t_trit_min(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

/* dst[i] = (a[i] == b[i]) ? +1 : 0 */
void m4t_trit_eq(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

/* dst[i] = -a[i]   (b ignored; sign flip via bit-swap, no TBL) */
void m4t_trit_neg(uint8_t* dst, const uint8_t* a, const uint8_t* b, int n_trits);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TRIT_OPS_H */
