/*
 * m4t_internal.h — private platform macros and shared kernel utilities
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Not exported. Not installed. Not included from public headers.
 * M4T is single-threaded at the opcode level — no libdispatch, no
 * pthreads. Threading is a consumer concern, not a substrate concern.
 * Real silicon instructions don't spawn threads, and neither does M4T.
 */

#ifndef M4T_INTERNAL_H
#define M4T_INTERNAL_H

#include "m4t_mtfp.h"   /* M4T_MTFP_CELLS_PER_BLOCK, M4T_FLAG_* */

#include <stdint.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#  define M4T_HAS_NEON 1
#else
#  define M4T_HAS_NEON 0
#  error "M4T requires ARM NEON (aarch64). See docs/M4T_CONTRACT.md."
#endif

/* ── Shared kernel utility: per-block flag set ──────────────────────────── */

/* OR `event_bits` (M4T_FLAG_SATURATED | M4T_FLAG_ROUNDED) into the per-block
 * flag byte for cell `i`. See m4t_mtfp.h for the per-block layout contract.
 * Used by every kernel that produces caller-prescribed output (Case S/R
 * resolutions per §8.5). */
static inline void m4t_flag_or(uint8_t* flags, int i, uint8_t event_bits) {
    int block = i / M4T_MTFP_CELLS_PER_BLOCK;
    int slot  = i % M4T_MTFP_CELLS_PER_BLOCK;
    flags[block] |= (uint8_t)(event_bits << (slot * 2));
}

#endif /* M4T_INTERNAL_H */
