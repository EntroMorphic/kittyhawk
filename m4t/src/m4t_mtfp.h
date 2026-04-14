/*
 * m4t_mtfp.h — MTFP19 mantissa-layer primitives
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * This header exposes the minimum surface that the kept routing primitives
 * need from the numeric core: saturating accumulator clamp and three
 * same-block vector ops. Everything that was dense or fixed-point-shaped
 * in the pre-reset m4t_mtfp.h lives in archive/m4t/src/m4t_mtfp.{c,h}.
 *
 * Contract:
 *   - Operands are mantissa cells (int32 mantissas with some implicit
 *     block exponent shared across the vector). All cells passed to a
 *     single call share the same (unspecified) block exponent; the
 *     substrate does not check this, the caller asserts it.
 *   - Per-cell saturation at ±M4T_MTFP_MAX_VAL. The §8.5 "widen, don't
 *     round" invariant applies: saturation happens only at cell overflow,
 *     never due to representational rounding.
 *   - Cross-block arithmetic with different block exponents is NOT
 *     provided. Consumers that need it must request a named opt-in
 *     variant (not yet implemented; see M4T_SUBSTRATE §14.2).
 *
 * All three vector functions operate on arrays of arbitrary length. The
 * NEON body processes full MTFP19 blocks (4 cells at a time); any tail is
 * handled scalar. This matches §14.3 (zero-pad convention) when the
 * caller provides aligned buffers.
 */

#ifndef M4T_MTFP_H
#define M4T_MTFP_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Saturating clamp of an int64 accumulator to an MTFP19 mantissa cell.
 *
 * Exact when |v| ≤ M4T_MTFP_MAX_VAL. Saturates at ±M4T_MTFP_MAX_VAL when
 * the accumulator would overflow the mantissa cell. Callers performing
 * widening accumulation (e.g. ternary matmul) use this on the final store
 * to realize the saturation edge of the §8.5 invariant.
 *
 * Not inlined via macro: static inline so the compiler can specialize. */

static inline m4t_mtfp_t m4t_mtfp_clamp64(int64_t v) {
    if (v >  (int64_t)M4T_MTFP_MAX_VAL) return  M4T_MTFP_MAX_VAL;
    if (v < -(int64_t)M4T_MTFP_MAX_VAL) return -M4T_MTFP_MAX_VAL;
    return (m4t_mtfp_t)v;
}

/* dst[0..n) = 0. */
void m4t_mtfp_vec_zero(m4t_mtfp_t* dst, int n);

/* dst[i] += a[i], clamped per cell to ±M4T_MTFP_MAX_VAL.
 * Same-block contract: dst and a are interpreted at the same block
 * exponent. */
void m4t_mtfp_vec_add_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n);

/* dst[i] -= a[i], clamped per cell to ±M4T_MTFP_MAX_VAL.
 * Same-block contract: dst and a are interpreted at the same block
 * exponent. */
void m4t_mtfp_vec_sub_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP_H */
