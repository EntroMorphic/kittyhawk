/*
 * synth_compose_hier.c — implementation of hierarchical synthetic.
 */

#include "synth_compose_hier.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static uint32_t splitmix32(uint32_t* state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static int8_t random_pm1(uint32_t* state) {
    return (splitmix32(state) & 1u) ? +1 : -1;
}

synth_compose_hier_config_t synth_compose_hier_default(void) {
    synth_compose_hier_config_t cfg = {
        .n_super     = 4,
        .k_per_super = 5,    /* total 20 classes */
        .coarse_dim  = 16,
        .fine_dim    = 16,
        .noise_dim   = 32,   /* total input_dim = 64 */
        .noise_pct   = 10,
        .seed        = 0xCAFEFACEu,
    };
    return cfg;
}

int synth_compose_hier_input_dim(const synth_compose_hier_config_t* cfg) {
    return cfg->coarse_dim + cfg->fine_dim + cfg->noise_dim;
}

int synth_compose_hier_n_classes(const synth_compose_hier_config_t* cfg) {
    return cfg->n_super * cfg->k_per_super;
}

void synth_compose_hier_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_compose_hier_config_t* cfg)
{
    assert(out_prototypes && cfg);
    int D     = synth_compose_hier_input_dim(cfg);
    int C     = synth_compose_hier_n_classes(cfg);
    int Cd    = cfg->coarse_dim;
    int Fd    = cfg->fine_dim;
    int K     = cfg->k_per_super;

    uint32_t s = cfg->seed;

    /* Build per-super coarse prototypes once. */
    int8_t* super_proto = malloc((size_t)cfg->n_super * Cd);
    for (int g = 0; g < cfg->n_super; g++) {
        for (int j = 0; j < Cd; j++) super_proto[g * Cd + j] = random_pm1(&s);
    }

    /* For each sub-class: copy super coarse prototype, then random fine
     * prototype, then zero noise positions. */
    for (int c = 0; c < C; c++) {
        int g = c / K;
        m4t_trit_t* row = out_prototypes + (size_t)c * D;
        for (int j = 0; j < Cd; j++)        row[j]          = super_proto[g * Cd + j];
        for (int j = 0; j < Fd; j++)        row[Cd + j]     = random_pm1(&s);
        for (int j = 0; j < cfg->noise_dim; j++) row[Cd + Fd + j] = 0;
    }

    free(super_proto);
}

void synth_compose_hier_generate_samples(
    m4t_trit_t* out_samples,
    int* out_sub_labels,
    int* out_super_labels,
    int n_samples,
    const m4t_trit_t* prototypes,
    const synth_compose_hier_config_t* cfg,
    uint32_t sample_seed)
{
    assert(out_samples && out_sub_labels && prototypes && cfg);
    int D = synth_compose_hier_input_dim(cfg);
    int C = synth_compose_hier_n_classes(cfg);
    int K = cfg->k_per_super;
    int informative = cfg->coarse_dim + cfg->fine_dim;

    uint32_t s = sample_seed;
    for (int i = 0; i < n_samples; i++) {
        int c = (int)(splitmix32(&s) % (uint32_t)C);
        out_sub_labels[i] = c;
        if (out_super_labels) out_super_labels[i] = c / K;

        const m4t_trit_t* p = prototypes + (size_t)c * D;
        m4t_trit_t* row = out_samples + (size_t)i * D;

        /* Informative dims: prototype value, with noise_pct chance of
         * a flip to one of the OTHER two trit values. */
        for (int j = 0; j < informative; j++) {
            uint32_t r = splitmix32(&s);
            if ((int)(r % 100) < cfg->noise_pct) {
                int8_t p_val = (int8_t)p[j];
                int8_t other_a = (int8_t)((p_val == +1) ? -1 : +1);
                int8_t other_b = 0;
                row[j] = (r & 1u) ? other_a : other_b;
            } else {
                row[j] = p[j];
            }
        }

        /* Noise dims: uniform random ternary. */
        for (int j = informative; j < D; j++) {
            uint32_t r = splitmix32(&s);
            int v = (int)(r % 3u);
            row[j] = (v == 0) ? 0 : (v == 1 ? +1 : -1);
        }
    }
}
