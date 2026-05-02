/*
 * synth_proto.c — synthetic prototype-classification benchmark.
 *
 * Pure integer arithmetic, deterministic from seed, no float.
 */

#include "synth_proto.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

/* Local xorshift32 RNG. Test/benchmark code; not in any runtime kernel. */
static uint32_t xs32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Uniform balanced ternary: p(-1) = p(+1) = 1/3, p(0) = 1/3. */
static m4t_trit_t rand_balanced_trit(uint32_t* state) {
    uint32_t r = xs32(state) % 3u;
    if (r == 0) return -1;
    if (r == 1) return  0;
    return  1;
}

/* Sign-only ternary: p(+1) = p(-1) = 1/2, no zero. Used for class
 * prototypes: zero in a prototype's informative dim doesn't help the
 * classifier discriminate, so prototypes have ±1-only on informative
 * dims and 0 on noise dims. */
static m4t_trit_t rand_sign_trit(uint32_t* state) {
    uint32_t r = xs32(state) & 1u;
    return r ? (m4t_trit_t)1 : (m4t_trit_t)-1;
}

synth_proto_config_t synth_proto_default(void) {
    synth_proto_config_t cfg;
    cfg.n_classes       = 10;
    cfg.input_dim       = 64;
    cfg.informative_dim = 16;
    cfg.noise_pct       = 10;     /* 10% per-trit flip prob */
    cfg.seed            = 0xdeadbeefu;
    return cfg;
}

void synth_proto_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_proto_config_t* cfg)
{
    assert(out_prototypes && cfg);
    assert(cfg->n_classes > 0);
    assert(cfg->input_dim > 0);
    assert(cfg->informative_dim >= 0 && cfg->informative_dim <= cfg->input_dim);

    uint32_t state = cfg->seed;
    int C = cfg->n_classes;
    int D = cfg->input_dim;
    int K = cfg->informative_dim;

    for (int c = 0; c < C; c++) {
        m4t_trit_t* p = out_prototypes + (size_t)c * D;
        /* Informative dims: random ±1 (no zero — keeps class signal sharp). */
        for (int j = 0; j < K; j++) {
            p[j] = rand_sign_trit(&state);
        }
        /* Noise dims: zero (signal-free). Samples will fill these. */
        for (int j = K; j < D; j++) {
            p[j] = 0;
        }
    }
}

void synth_proto_generate_samples(
    m4t_trit_t* out_samples,
    int* out_labels,
    int n_samples,
    const m4t_trit_t* prototypes,
    const synth_proto_config_t* cfg,
    uint32_t sample_seed)
{
    assert(out_samples && out_labels && prototypes && cfg);
    assert(n_samples >= 0);

    int C = cfg->n_classes;
    int D = cfg->input_dim;
    int K = cfg->informative_dim;
    /* noise_pct is per-trit flip probability × 100. To compare to a
     * uniform [0, 100) sample: flip if rand < noise_pct. */
    uint32_t flip_threshold = (uint32_t)cfg->noise_pct;

    /* Deterministic state from cfg.seed XOR sample_seed. */
    uint32_t state = cfg->seed ^ sample_seed ^ 0x5a5a5a5au;

    for (int i = 0; i < n_samples; i++) {
        int c = (int)(xs32(&state) % (uint32_t)C);
        out_labels[i] = c;

        const m4t_trit_t* p = prototypes + (size_t)c * D;
        m4t_trit_t* s = out_samples + (size_t)i * D;

        /* Informative dims: copy prototype, then flip with prob noise_pct.
         * Flip rule: prototype = +1 → flip to -1 with prob noise_pct, else
         * 0 with prob noise_pct, else stays at +1. (Symmetric for -1.)
         * This is "noise on a 3-state lattice": per-trit, with prob
         * noise_pct the trit flips to one of the two non-prototype values
         * (uniformly). */
        for (int j = 0; j < K; j++) {
            uint32_t r = xs32(&state) % 100u;
            if (r < flip_threshold) {
                /* Flip to one of the two non-prototype values. */
                m4t_trit_t alt1, alt2;
                if (p[j] == 1)       { alt1 = 0;  alt2 = -1; }
                else if (p[j] == -1) { alt1 = 0;  alt2 =  1; }
                else                 { alt1 = -1; alt2 =  1; }
                s[j] = (xs32(&state) & 1u) ? alt1 : alt2;
            } else {
                s[j] = p[j];
            }
        }
        /* Noise dims: uniform-random balanced ternary (worst case for
         * classifiers: pure entropy in these dims, no class signal). */
        for (int j = K; j < D; j++) {
            s[j] = rand_balanced_trit(&state);
        }
    }
}
