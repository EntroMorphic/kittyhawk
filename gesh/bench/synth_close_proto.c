/* synth_close_proto.c — see header. */

#include "synth_close_proto.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

synth_close_proto_config_t synth_close_proto_default(void) {
    synth_close_proto_config_t cfg;
    cfg.n_classes       = 10;
    cfg.input_dim       = 64;
    cfg.informative_dim = 16;
    cfg.noise_pct       = 10;
    cfg.seed            = 0xdeadbeefu;
    return cfg;
}

static uint32_t xs32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

void synth_close_proto_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_close_proto_config_t* cfg)
{
    assert(out_prototypes && cfg);
    int K = cfg->informative_dim;
    int D = cfg->input_dim;
    int C = cfg->n_classes;
    assert(K <= D);
    assert(C - 1 <= K);  /* need K ≥ C-1 dims to flip */

    uint32_t state = cfg->seed ? cfg->seed : 1u;

    /* Base prototype: random ±1 in informative dims, 0 in noise dims. */
    m4t_trit_t* base = malloc((size_t)D * sizeof(m4t_trit_t));
    for (int j = 0; j < K; j++) base[j] = (xs32(&state) & 1u) ? 1 : -1;
    for (int j = K; j < D; j++) base[j] = 0;

    /* Class 0: identical to base. Class c (c ≥ 1): flip the FIRST c
     * informative dims. Hamming(class_a, class_b) = |a - b| × 2 (a flipped
     * trit costs 2 vs the unflipped). */
    for (int c = 0; c < C; c++) {
        m4t_trit_t* p = out_prototypes + (size_t)c * D;
        memcpy(p, base, (size_t)D * sizeof(m4t_trit_t));
        for (int f = 0; f < c; f++) {
            p[f] = (m4t_trit_t)(-p[f]);
        }
    }
    free(base);
}

void synth_close_proto_generate_samples(
    m4t_trit_t* out_samples,
    int* out_labels,
    int n_samples,
    const m4t_trit_t* prototypes,
    const synth_close_proto_config_t* cfg,
    uint32_t sample_seed)
{
    assert(out_samples && out_labels && prototypes && cfg);
    uint32_t state = (cfg->seed ^ sample_seed ^ 0x5a5a5a5au);
    if (state == 0) state = 1u;
    int K = cfg->informative_dim;
    int D = cfg->input_dim;

    for (int i = 0; i < n_samples; i++) {
        int c = (int)(xs32(&state) % (uint32_t)cfg->n_classes);
        out_labels[i] = c;
        const m4t_trit_t* p = prototypes + (size_t)c * D;
        m4t_trit_t* s = out_samples + (size_t)i * D;
        /* Informative dims: flip with prob noise_pct/100. */
        for (int j = 0; j < K; j++) {
            uint32_t r = xs32(&state);
            if ((r % 100u) < (uint32_t)cfg->noise_pct) {
                /* Pick a non-current value uniformly. */
                m4t_trit_t pv = p[j];
                if (pv == 1) s[j] = ((r >> 7) & 1u) ? 0 : -1;
                else if (pv == -1) s[j] = ((r >> 7) & 1u) ? 0 : 1;
                else s[j] = ((r >> 7) & 1u) ? 1 : -1;
            } else {
                s[j] = p[j];
            }
        }
        /* Noise dims: uniform random ternary. */
        for (int j = K; j < D; j++) {
            uint32_t r = xs32(&state) % 3u;
            s[j] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
        }
    }
}
