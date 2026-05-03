/*
 * synth_close_proto.h — synthetic with classes whose prototypes are
 * CLOSE in trit space. The intended P0-3 verification benchmark:
 * exposes whether geometric training can pull near-degenerate class
 * tiles apart.
 *
 * Construction:
 *   - 1 base prototype P_base ∈ {-1, +1}^K (K informative dims).
 *   - Per class c: take P_base, FLIP c trits at fixed positions
 *     (cumulative offset = c). Produces C prototypes that differ
 *     from each other by Hamming distances 1..C-1.
 *   - Remaining D-K dims are noise per sample.
 *
 * Compared to synth_proto (random ±1 per class — already well-spread),
 * this is the regime where bank tiles START close together and
 * geometric training's margin maximization should help.
 */

#ifndef GESH_SYNTH_CLOSE_PROTO_H
#define GESH_SYNTH_CLOSE_PROTO_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_classes;
    int input_dim;
    int informative_dim;
    int noise_pct;
    uint32_t seed;
} synth_close_proto_config_t;

synth_close_proto_config_t synth_close_proto_default(void);

void synth_close_proto_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_close_proto_config_t* cfg);

void synth_close_proto_generate_samples(
    m4t_trit_t* out_samples,
    int* out_labels,
    int n_samples,
    const m4t_trit_t* prototypes,
    const synth_close_proto_config_t* cfg,
    uint32_t sample_seed);

#ifdef __cplusplus
}
#endif

#endif
