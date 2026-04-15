/*
 * glyph_rng.h — xoshiro128** RNG state and primitives.
 *
 * The RNG is used for generating random ternary projection matrices
 * and deriving independent table seeds in multi-table LSH consumers.
 * Deterministic given the same seed quadruple.
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
