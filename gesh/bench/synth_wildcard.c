/*
 * synth_wildcard.c — see synth_wildcard.h.
 *
 * Substrate-claim P0-1 verification benchmark. Generates data with explicit
 * always-informative / sometimes-informative / never-informative dim splits
 * to expose the substrate's wildcard semantics.
 */

#include "synth_wildcard.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

synth_wildcard_config_t synth_wildcard_default(void) {
    synth_wildcard_config_t cfg;
    cfg.n_classes      = 10;
    cfg.input_dim      = 64;
    cfg.always_dim     = 16;   /* K */
    cfg.sometimes_dim  = 16;   /* M */
    cfg.noise_dim      = 32;   /* N = 64 - 16 - 16 */
    cfg.noise_pct      = 10;   /* 10% flip on informative dims */
    cfg.proto_seed     = 0xc0ffeebbu;
    return cfg;
}

/* xorshift32 — internal-only RNG; matches the gesh_train pattern. */
static uint32_t xs32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

void synth_wildcard_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_wildcard_config_t* cfg)
{
    assert(out_prototypes && cfg);
    assert(cfg->n_classes > 0);
    assert(cfg->always_dim >= 0 && cfg->sometimes_dim >= 0 && cfg->noise_dim >= 0);
    assert(cfg->input_dim == cfg->always_dim + cfg->sometimes_dim + cfg->noise_dim);

    uint32_t state = cfg->proto_seed ? cfg->proto_seed : 0x12345678u;

    int K = cfg->always_dim;
    int M = cfg->sometimes_dim;
    int N = cfg->noise_dim;
    int D = cfg->input_dim;

    for (int c = 0; c < cfg->n_classes; c++) {
        m4t_trit_t* p = out_prototypes + (size_t)c * D;
        /* K always-informative: random ±1 per class. */
        for (int j = 0; j < K; j++) {
            p[j] = (xs32(&state) & 1u) ? (m4t_trit_t)1 : (m4t_trit_t)-1;
        }
        /* M sometimes-informative: 50% chance ±1, 50% chance 0. */
        for (int j = K; j < K + M; j++) {
            uint32_t r = xs32(&state);
            if ((r & 1u) == 0) {
                p[j] = 0;  /* Class-c-irrelevant on this dim — wildcard target. */
            } else {
                p[j] = ((r >> 1) & 1u) ? (m4t_trit_t)1 : (m4t_trit_t)-1;
            }
        }
        /* N never-informative: 0 in prototype (samples will be random). */
        for (int j = K + M; j < D; j++) {
            p[j] = 0;
        }
        (void)N;  /* unused after assertion */
    }
}

void synth_wildcard_generate_samples(
    m4t_trit_t* out_samples,
    int* out_labels,
    int n_samples,
    const m4t_trit_t* prototypes,
    const synth_wildcard_config_t* cfg,
    uint32_t sample_seed)
{
    assert(out_samples && out_labels && prototypes && cfg);
    assert(n_samples >= 0);

    uint32_t state = (cfg->proto_seed ^ sample_seed ^ 0x5a5a5a5au);
    if (state == 0) state = 0x12345678u;

    int K = cfg->always_dim;
    int M = cfg->sometimes_dim;
    int D = cfg->input_dim;

    for (int i = 0; i < n_samples; i++) {
        int c = (int)(xs32(&state) % (uint32_t)cfg->n_classes);
        out_labels[i] = c;
        const m4t_trit_t* p = prototypes + (size_t)c * D;
        m4t_trit_t* s = out_samples + (size_t)i * D;

        /* Informative dims (always + sometimes-with-prototype-value): copy
         * prototype, possibly flip per noise_pct. Sometimes-with-zero-
         * prototype: leave the sample at zero (class genuinely doesn't
         * have a value here). */
        for (int j = 0; j < K + M; j++) {
            m4t_trit_t pv = p[j];
            if (pv == 0) {
                s[j] = 0;  /* class-irrelevant; sample stays zero */
            } else {
                /* Flip with probability noise_pct/100. */
                uint32_t r = xs32(&state);
                if ((r % 100u) < (uint32_t)cfg->noise_pct) {
                    /* Flip to one of the two non-current values. */
                    uint32_t flip = (r >> 7) % 2u;
                    if (pv == 1)  s[j] = flip ? (m4t_trit_t)0 : (m4t_trit_t)-1;
                    else          s[j] = flip ? (m4t_trit_t)0 : (m4t_trit_t)1;
                } else {
                    s[j] = pv;
                }
            }
        }
        /* Never-informative dims: uniform random ternary noise. */
        for (int j = K + M; j < D; j++) {
            uint32_t r = xs32(&state) % 3u;
            s[j] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
        }
    }
}
