/*
 * trix_rng.h — Shared xoshiro128+ PRNG for all trix native modules.
 *
 * Previously duplicated across trix_atoms.c, trix_multitrit.c, and
 * trix_ternary_route.c as at_rng_*, mt_rng_*, and tr_rng_* respectively.
 * Consolidated here to eliminate the triplication.
 */

#ifndef TRIX_RNG_H
#define TRIX_RNG_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t trix_rng_rotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

typedef struct { uint32_t s[4]; } trix_rng_t;

static inline uint32_t trix_rng_next(trix_rng_t* r) {
    uint32_t result = r->s[0] + r->s[3], t = r->s[1] << 9;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = trix_rng_rotl(r->s[3], 11);
    return result;
}

static inline float trix_rng_uniform(trix_rng_t* r) {
    return (float)(trix_rng_next(r) >> 8) / 16777216.0f;
}

static inline trix_rng_t trix_rng_seed(uint64_t seed) {
    trix_rng_t r;
    r.s[0] = (uint32_t)seed;
    r.s[1] = (uint32_t)(seed >> 32);
    r.s[2] = (uint32_t)(seed * 2654435761ULL);
    r.s[3] = (uint32_t)((seed * 2654435761ULL) >> 32);
    for (int i = 0; i < 16; i++) trix_rng_next(&r);
    return r;
}

static inline void trix_xavier_init(float* w, int fan_in, int fan_out, int n, trix_rng_t* rng) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < n; i++)
        w[i] = (2.0f * trix_rng_uniform(rng) - 1.0f) * limit;
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_RNG_H */
