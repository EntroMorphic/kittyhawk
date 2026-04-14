/*
 * m4t_types.h — M4 Ternary Extensions: mantissa container types and constants
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * MTFP is base-3 floating-point. A value is `mantissa × 3^exponent`, where
 * the mantissa is an n-trit signed integer held in one of the cell types
 * below, and the exponent is sidecar metadata carried at the BLOCK level
 * (not per cell, not per tensor). See m4t/docs/M4T_SUBSTRATE.md for the
 * full contract.
 *
 * No binary floating point (float, double, float16, bfloat16) anywhere in
 * libm4t. Binary float appears only at build time, in m4t/tools/m4t_lut_gen.c.
 * int8/int16/int32/int64 appear only as mantissa containers at clean trit
 * boundaries — never as binary quantization types.
 *
 * Four cell widths. The cell holds the mantissa; cells per block follow
 * from the 16-byte block (one NEON vector):
 *
 *   type          container  trits  cells/block   role
 *   ------------  ---------  -----  -----------   -------------------------
 *   m4t_mtfp4_t   int8         4        16        SDOT-native routing
 *   m4t_mtfp9_t   int16        9         8        narrow activations
 *   m4t_mtfp_t    int32       19         4        general (MTFP19)
 *   m4t_mtfp_w_t  int64       39         2        wide accumulator (MTFP39)
 *
 * Every cell is balanced: MAX_VAL = (3^trits - 1) / 2. Two in-range cells
 * can be added in the container without wrapping, so SIMD arithmetic uses
 * non-saturating add followed by min/max clamp.
 *
 * Legacy note: M4T_MTFP_SCALE (= 3^10) and its siblings survived the
 * fixed-point era. Under the rebuilt spec they are NOT intrinsic to the
 * type — they are a *default block-exponent convention* used by legacy
 * consumers that have not yet been rewritten to carry explicit per-block
 * exponent arrays. A tensor using the default convention interprets every
 * mantissa as `mantissa × 3^(-RADIX)`. New consumers that manage their own
 * block exponents should ignore SCALE and track the exponent explicitly.
 */

#ifndef M4T_TYPES_H
#define M4T_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Block geometry ──────────────────────────────────────────────────────
 *
 * A block is one 128-bit NEON vector: 16 bytes. Cell count per block falls
 * out of the cell width. Derived from SDOT atomicity (16 int8 in / 4 int32
 * out) and TBL atomicity (16-byte index vector). Everything else in the
 * substrate composes cleanly around this unit. */

#define M4T_BLOCK_BYTES              16
#define M4T_MTFP4_CELLS_PER_BLOCK    16
#define M4T_MTFP9_CELLS_PER_BLOCK     8
#define M4T_MTFP_CELLS_PER_BLOCK      4
#define M4T_MTFPW_CELLS_PER_BLOCK     2

/* ── MTFP4: 4-trit mantissa in int8 ──────────────────────────────────────
 *
 * The routing mantissa. SDOT-native: 16 cells per 128-bit register, and
 * `sdot v.4s, a.16b, b.16b` computes 16 MTFP4 × MTFP4 products into 4
 * MTFP19 mantissas exactly (max output magnitude 16 × 40 × 40 = 25 600). */

typedef int8_t m4t_mtfp4_t;

#define M4T_MTFP4_RADIX       2
#define M4T_MTFP4_SCALE       9               /* 3^2 — default block-exponent convention */
#define M4T_MTFP4_TRITS       4
#define M4T_MTFP4_MAX_VAL     ((m4t_mtfp4_t)40)  /* (3^4 - 1) / 2 */

/* ── MTFP9: 9-trit mantissa in int16 ───────────────────────────────────── */

typedef int16_t m4t_mtfp9_t;

#define M4T_MTFP9_RADIX       5
#define M4T_MTFP9_SCALE       243             /* 3^5 — default block-exponent convention */
#define M4T_MTFP9_TRITS       9
#define M4T_MTFP9_MAX_VAL     ((m4t_mtfp9_t)9841)  /* (3^9 - 1) / 2 */

/* ── MTFP19: 19-trit mantissa in int32 ─────────────────────────────────── */

typedef int32_t m4t_mtfp_t;

#define M4T_MTFP_RADIX        10
#define M4T_MTFP_SCALE        59049           /* 3^10 — default block-exponent convention */
#define M4T_MTFP_TRITS        19
#define M4T_MTFP_MAX_VAL      ((m4t_mtfp_t)581130733)  /* (3^19 - 1) / 2 */

/* ── MTFP39: 39-trit mantissa in int64 ─────────────────────────────────── */

typedef int64_t m4t_mtfp_w_t;

#define M4T_MTFPW_RADIX       10
#define M4T_MTFPW_SCALE       59049           /* 3^10 — default block-exponent convention */
#define M4T_MTFPW_TRITS       39
#define M4T_MTFPW_MAX_VAL     ((m4t_mtfp_w_t)INT64_C(2026277576509488133)) /* (3^39 - 1) / 2 */

/* ── Trits ───────────────────────────────────────────────────────────────
 *
 * A single base-3 digit: {-1, 0, +1}. Storage is int8_t.
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
