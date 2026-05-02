/*
 * synth_proto.h — synthetic prototype-classification benchmark for Gesh.
 *
 * Closed-form deterministic generator. C classes, D input dimensions,
 * K of which are class-informative; the remaining D-K are noise.
 *
 * For each class c, a true prototype P_c ∈ {-1, 0, +1}^K (random-balanced)
 * extended with zeros to D dims. Training/eval samples flip each trit
 * independently with probability `noise_p`, then random-fill the noise
 * dimensions. Labels are the source class c.
 *
 * The "right" projection picks out the informative K dims and ignores
 * the rest; PCA on training data with class supervision finds them
 * (informative dims have higher class-conditional variance). Without
 * such a projection, Hamming distance over all D dims is degraded by
 * the noise dims.
 *
 * All integer arithmetic. Substrate-legal (no float anywhere).
 */

#ifndef GESH_SYNTH_PROTO_H
#define GESH_SYNTH_PROTO_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration for the synthetic benchmark. */
typedef struct {
    int n_classes;       /* C — number of classes (default 10) */
    int input_dim;       /* D — total input dims, packed-trit (default 64) */
    int informative_dim; /* K — informative dims (default 16); K <= D */
    int noise_pct;       /* per-trit flip probability × 100 (default 10 = 10%) */
    uint32_t seed;       /* deterministic seed */
} synth_proto_config_t;

/* Default config for Phase A.1 measurements. */
synth_proto_config_t synth_proto_default(void);

/* Generate the C class prototypes deterministically from `cfg.seed`.
 * out_prototypes layout: row-major [C × D] in unpacked m4t_trit_t.
 * Informative dims (first K) are class-distinct ±1 patterns; noise dims
 * (remaining D-K) are 0 in the prototype. */
void synth_proto_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_proto_config_t* cfg
);

/* Generate `n_samples` training/eval samples with their labels.
 * out_samples layout: row-major [n_samples × D] in unpacked m4t_trit_t.
 * out_labels layout: [n_samples] int.
 *
 * For each sample: pick a class uniformly; copy that class's prototype;
 * flip each informative-dim trit with probability `noise_p`; fill each
 * noise-dim trit with a uniform-random ternary (worst case).
 *
 * Sample-generation seed is derived from cfg.seed and a per-call salt
 * so that calling generate_samples with the same cfg twice produces
 * the same output. */
void synth_proto_generate_samples(
    m4t_trit_t* out_samples,
    int* out_labels,
    int n_samples,
    const m4t_trit_t* prototypes,    /* from generate_prototypes */
    const synth_proto_config_t* cfg,
    uint32_t sample_seed
);

#ifdef __cplusplus
}
#endif

#endif /* GESH_SYNTH_PROTO_H */
