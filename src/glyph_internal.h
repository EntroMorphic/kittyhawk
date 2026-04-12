/*
 * glyph_internal.h — private platform macros shared across glyph .c files
 *
 * GLYPH IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Not exported. Do not install this header. Do not include it from a
 * public header — the compile-time identity of `GLYPH_HAS_NEON` etc. is
 * a private implementation detail of the .c translation units.
 */

#ifndef GLYPH_INTERNAL_H
#define GLYPH_INTERNAL_H

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#  define GLYPH_HAS_NEON 1
#else
#  define GLYPH_HAS_NEON 0
#  error "glyph requires ARM NEON (aarch64). Non-NEON targets are unsupported in v0. See docs/REMEDIATION_PLAN.md D5."
#endif

#ifdef __APPLE__
#  include <dispatch/dispatch.h>
#  define GLYPH_HAS_DISPATCH 1
#else
#  define GLYPH_HAS_DISPATCH 0
#endif

/* Small-problem threshold: below this many rows, run matmul / layernorm
 * serially instead of launching libdispatch. The break-even on an M4 is
 * around a dozen rows for our kernel shapes; 4 is safely below that. */
#define GLYPH_SERIAL_ROW_THRESHOLD 4

#endif /* GLYPH_INTERNAL_H */
