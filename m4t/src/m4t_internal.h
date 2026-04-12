/*
 * m4t_internal.h — private platform macros for M4T translation units
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

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#  define M4T_HAS_NEON 1
#else
#  define M4T_HAS_NEON 0
#  error "M4T requires ARM NEON (aarch64). See docs/M4T_CONTRACT.md."
#endif

#endif /* M4T_INTERNAL_H */
