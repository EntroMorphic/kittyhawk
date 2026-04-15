/*
 * glyph_rng.h — xoshiro128+ RNG state and primitives.
 *
 * Small, fast, deterministic RNG used to generate random ternary
 * projection matrices. This is the "plus" variant of Blackman and
 * Vigna's xoshiro128 family: state update is the standard xoshiro128
 * step (xor + shift + rotate), output is `s[0] + s[3]` (NOT the
 * `starstar` output mix). This is the variant every cascade tool has
 * used historically; libglyph preserves it for reproducibility with
 * Phase 3 measurements.
 *
 * IMPORTANT: xoshiro requires at least one non-zero state element.
 * Seeding with (0,0,0,0) produces all zeros forever. Callers must
 * pick a non-degenerate seed quadruple.
 */

#ifndef GLYPH_RNG_H
#define GLYPH_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t s[4];
} glyph_rng_t;

/* Seed the RNG with four 32-bit values. Any combination works except
 * all-zero. */
void glyph_rng_seed(glyph_rng_t* r, uint32_t a, uint32_t b, uint32_t c, uint32_t d);

/* Draw the next 32-bit value from the xoshiro128** state. */
uint32_t glyph_rng_next(glyph_rng_t* r);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_RNG_H */
