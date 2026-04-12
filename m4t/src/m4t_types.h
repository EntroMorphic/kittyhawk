/*
 * m4t_types.h — M4 Ternary Extensions: core numeric and container types
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * No binary floating point (float, double, float16, bfloat16) anywhere in
 * libm4t. int8/int16/int32/int64 appear ONLY as MTFP cell containers at
 * clean trit boundaries — never as binary quantization types.
 *
 * Four cell widths, each a balanced-ternary fixed-point type:
 *
 *   MTFP4   int8    4 trits   16 NEON lanes   SDOT-native routing
 *   MTFP9   int16   9 trits    8 NEON lanes   narrow activations
 *   MTFP19  int32  19 trits    4 NEON lanes   general-purpose (default)
 *   MTFP39  int64  39 trits    2 NEON lanes   wide accumulation
 *
 * Every cell container is halved: MAX_VAL = (3^trits - 1) / 2. This
 * guarantees that the sum of two in-range cells never wraps the underlying
 * integer type, allowing non-saturating SIMD add followed by min/max clamp.
 */

#ifndef M4T_TYPES_H
#define M4T_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── MTFP4: 4 trits in int8 ──────────────────────────────────────────────
 *
 * The routing cell. 16 cells per NEON register. Unlocks vdotq_s32 (SDOT)
 * as a native ternary matmul: int8 MTFP4 activations x int8 trit weights
 * → int32 accumulator, 16 multiply-accumulates per cycle.
 *
 * radix=2, scale=3^2=9.  real = cell / 9.
 * range: ±(3^4-1)/2 / 9 = ±40/9 ≈ ±4.44.  resolution: 1/9 ≈ 0.111.
 */
typedef int8_t m4t_mtfp4_t;

#define M4T_MTFP4_RADIX       2
#define M4T_MTFP4_SCALE       9               /* 3^2 */
#define M4T_MTFP4_TRITS       4
#define M4T_MTFP4_MAX_VAL     ((m4t_mtfp4_t)40)  /* (3^4 - 1) / 2 */

/* ── MTFP9: 9 trits in int16 ────────────────────────────────────────────
 *
 * The narrow cell. 8 cells per NEON register. Useful for intermediate
 * scores, attention weights, and anywhere resolution > 0.111 but full
 * MTFP19 width is unnecessary.
 *
 * radix=5, scale=3^5=243.  real = cell / 243.
 * range: ±(3^9-1)/2 / 243 = ±9841/243 ≈ ±40.5.  resolution: 1/243 ≈ 0.0041.
 */
typedef int16_t m4t_mtfp9_t;

#define M4T_MTFP9_RADIX       5
#define M4T_MTFP9_SCALE       243             /* 3^5 */
#define M4T_MTFP9_TRITS       9
#define M4T_MTFP9_MAX_VAL     ((m4t_mtfp9_t)9841)  /* (3^9 - 1) / 2 */

/* ── MTFP19: 19 trits in int32 ──────────────────────────────────────────
 *
 * The default cell. 4 cells per NEON register. General-purpose for FFN
 * activations, layernorm, matmul, bias — the workhorse of M4T.
 *
 * radix=10, scale=3^10=59049.  real = cell / 59049.
 * range: ±(3^19-1)/2 / 59049 ≈ ±9842.  resolution: 1/59049 ≈ 1.69e-5.
 */
typedef int32_t m4t_mtfp_t;

#define M4T_MTFP_RADIX        10
#define M4T_MTFP_SCALE        59049           /* 3^10 */
#define M4T_MTFP_TRITS        19
#define M4T_MTFP_MAX_VAL      ((m4t_mtfp_t)581130733)  /* (3^19 - 1) / 2 */

/* ── MTFP39: 39 trits in int64 ──────────────────────────────────────────
 *
 * The wide cell. 2 cells per NEON register. For high-precision paths,
 * wide accumulation, and any application that needs > 19 trits of range.
 * MTFP21 (trix-z compat) is a documented clamp on MTFP39.
 *
 * radix=10, scale=3^10=59049.  real = cell / 59049.
 * range: ±(3^39-1)/2 / 59049 ≈ ±3.43e13.  resolution: 1/59049 ≈ 1.69e-5.
 */
typedef int64_t m4t_mtfp_w_t;

#define M4T_MTFPW_RADIX       10
#define M4T_MTFPW_SCALE       59049           /* 3^10 */
#define M4T_MTFPW_TRITS       39
#define M4T_MTFPW_MAX_VAL     ((m4t_mtfp_w_t)INT64_C(2026277576509488133)) /* (3^39 - 1) / 2 */

/* MTFP21 subset of MTFP39 (trix-z compatibility). Same storage, narrower clamp. */
#define M4T_MTFP21_MAX_VAL    ((m4t_mtfp_w_t)INT64_C(5230176601)) /* (3^21 - 1) / 2 */

/* ── Trits ───────────────────────────────────────────────────────────────
 *
 * A single base-3 digit: {-1, 0, +1}. Storage is int8_t. This is a
 * documentation-grade typedef, not a distinct C type.
 *
 * Packed trits live in uint8_t buffers using a 2-bit encoding:
 *   0b00 → 0, 0b01 → +1, 0b10 → -1, 0b11 → reserved (treated as 0)
 * Four trits per byte, LSB-first.
 */
typedef int8_t m4t_trit_t;

#define M4T_TRIT_PACKED_BYTES(n) (((n) + 3) / 4)

#ifdef __cplusplus
}
#endif

#endif /* M4T_TYPES_H */
