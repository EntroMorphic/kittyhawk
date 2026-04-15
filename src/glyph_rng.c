/*
 * glyph_rng.c — xoshiro128** reference implementation.
 */

#include "glyph_rng.h"

void glyph_rng_seed(glyph_rng_t* r, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    r->s[0] = a;
    r->s[1] = b;
    r->s[2] = c;
    r->s[3] = d;
}

uint32_t glyph_rng_next(glyph_rng_t* r) {
    uint32_t result = r->s[0] + r->s[3];
    uint32_t t = r->s[1] << 9;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = (r->s[3] << 11) | (r->s[3] >> 21);
    return result;
}
