/*
 * glyph_types.h — core numeric and container types for glyph
 *
 * GLYPH IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * No binary floating point (float, double, float16, bfloat16) and no small-int
 * numeric types (int8_t/int16_t/uint8_t/uint16_t as numbers) appear anywhere
 * in the glyph runtime library. Byte buffers holding packed trit containers
 * are permitted, because they are storage for base-3 values, not binary
 * numerics. int32_t is permitted only as a glyph_mtfp_t container cell.
 *
 * If you find yourself wanting a float here, stop and read docs/REMEDIATION_PLAN.md.
 */

#ifndef GLYPH_TYPES_H
#define GLYPH_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Multi-Trit Floating Point (MTFP) ────────────────────────────────────
 *
 * A glyph_mtfp_t is a balanced-ternary fixed-point cell. The underlying
 * storage is int32_t, but the cell is NOT a binary integer. Its real value
 * is (cell / GLYPH_MTFP_SCALE), where the scale is 3^10 = 59049.
 *
 *   resolution  = 1 / 59049      ≈ 1.69e-5
 *   clamp range = ±GLYPH_MTFP_MAX_VAL / GLYPH_MTFP_SCALE ≈ ±18180
 *
 * Treat glyph_mtfp_t as opaque at the type system level. Never write
 * `int32_t x = some_mtfp;` or `int y = mtfp + 7;` — always go through
 * glyph_mtfp_* arithmetic.
 */
typedef int32_t glyph_mtfp_t;

#define GLYPH_MTFP_RADIX     10
#define GLYPH_MTFP_SCALE     59049                     /* 3^10 */

/* Cell range is half of int32. The ÷2 is load-bearing, not legacy:
 *
 * Fast NEON add-then-clamp paths use non-saturating `vaddq_s32` followed
 * by `vminq/vmaxq` against ±MAX_VAL. For the pre-clamp intermediate to
 * never wrap int32 for any two in-range operands, we need:
 *
 *     2 · MAX_VAL  ≤  INT32_MAX
 *
 * which forces MAX_VAL ≤ INT32_MAX/2 = 1073741823. We sit exactly at
 * that bound. Changing this constant without auditing every non-
 * saturating NEON add site will silently wrap. See NEW2 in
 * docs/REDTEAM_FIXES.md. */
#define GLYPH_MTFP_MAX_VAL   ((glyph_mtfp_t)1073741823)

/* ── Trits ───────────────────────────────────────────────────────────────
 *
 * A glyph_trit_t is a single base-3 digit with legal values {-1, 0, +1}.
 * Underlying storage is int8_t; this is a documentation-grade typedef and
 * does not create a distinct C type. Callers MUST ensure that any buffer
 * typed `glyph_trit_t*` contains only values in {-1, 0, +1}. Arithmetic on
 * glyph_trit_t values outside that set is undefined in glyph.
 *
 * Packed trits live in uint8_t buffers using a 2-bit encoding:
 *
 *   0b00 → 0
 *   0b01 → +1
 *   0b10 → -1
 *   0b11 → reserved (treated as 0; do not emit)
 *
 * Four trits per byte, packed LSB-first within the byte.
 */
typedef int8_t glyph_trit_t;

/* Number of packed bytes required to hold `n` trits. */
#define GLYPH_TRIT_PACKED_BYTES(n) (((n) + 3) / 4)

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_TYPES_H */
