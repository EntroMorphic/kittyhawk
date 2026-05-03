/*
 * synth_compose_hier.h — hierarchical synthetic for compositional
 * routing. Built specifically as the FAIR benchmark for P0-4
 * remediation: super-classes form coarse structure (a stage-1 router
 * can find them), sub-classes form fine structure within each super
 * (a stage-2 router refines them). Single-stage routing must do both
 * jobs at once and is expected to underperform.
 *
 * Construction:
 *   N_SUPER super-classes; K sub-classes per super; N_CLASSES = N_SUPER × K.
 *   coarse_dim trits encode super-class (random ±1 per super).
 *   fine_dim   trits encode sub-class (random ±1 per sub, orthogonal
 *              to coarse_dim across all sub-classes within a super).
 *   noise_dim  trits are sample-level noise.
 *
 *   Total input_dim = coarse_dim + fine_dim + noise_dim.
 *
 * Sample: super-prototype + sub-prototype + per-sample noise on
 *          informative dims at noise_pct rate; uniform random on
 *          noise_dim dims.
 *
 * label[i] is the SUB-class id (0..N_SUPER*K-1).
 * super_label[i] = label[i] / K.
 */

#ifndef GESH_SYNTH_COMPOSE_HIER_H
#define GESH_SYNTH_COMPOSE_HIER_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_super;        /* coarse classes */
    int k_per_super;    /* sub-classes per super */
    int coarse_dim;     /* informative dims for super-class */
    int fine_dim;       /* informative dims for sub-class */
    int noise_dim;      /* uninformative dims */
    int noise_pct;      /* sample noise on informative dims (0..100) */
    uint32_t seed;
} synth_compose_hier_config_t;

synth_compose_hier_config_t synth_compose_hier_default(void);

/* Returns total input_dim = coarse_dim + fine_dim + noise_dim. */
int synth_compose_hier_input_dim(const synth_compose_hier_config_t* cfg);

/* Returns total n_classes = n_super × k_per_super. */
int synth_compose_hier_n_classes(const synth_compose_hier_config_t* cfg);

/* Build per-class prototypes. out_prototypes is [n_classes × input_dim].
 * The first coarse_dim positions are the super-prototype (shared across
 * sub-classes of the super); the next fine_dim positions are the
 * sub-prototype; the last noise_dim positions are zero (samples will
 * fill them). */
void synth_compose_hier_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_compose_hier_config_t* cfg);

/* Generate samples + sub-class labels + super-class labels.
 * Each sample = prototype + noise on informative dims + uniform random
 * on noise dims. */
void synth_compose_hier_generate_samples(
    m4t_trit_t* out_samples,
    int* out_sub_labels,
    int* out_super_labels,
    int n_samples,
    const m4t_trit_t* prototypes,
    const synth_compose_hier_config_t* cfg,
    uint32_t sample_seed);

#ifdef __cplusplus
}
#endif

#endif
