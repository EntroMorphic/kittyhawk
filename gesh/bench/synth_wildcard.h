/*
 * synth_wildcard.h — synthetic prototype benchmark with explicit
 * don't-care structure for P0-1 verification.
 *
 * Designed to expose the substrate's wildcard semantics. Three dim
 * categories per class:
 *   - K always-informative dims: every class has a ±1 prototype here.
 *   - M sometimes-informative dims: each class either has a ±1
 *     prototype OR zero (drawn at proto-gen time, fresh per class).
 *   - N never-informative dims: uniform-random ternary noise per sample.
 *
 * Total D = K + M + N. The bank should learn to:
 *   - Hold ±1 in always-informative positions per class.
 *   - Hold ±1 in this-class-relevant sometimes-informative positions
 *     and 0 (wildcard) in this-class-irrelevant ones.
 *   - Hold 0 (wildcard) in all never-informative positions.
 *
 * A wildcard-aware bank+kernel pair should outperform a tie-cancellation
 * bank+kernel on this benchmark by an amount proportional to (M + N) / D.
 */

#ifndef GESH_SYNTH_WILDCARD_H
#define GESH_SYNTH_WILDCARD_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_classes;          /* C — number of classes */
    int input_dim;          /* D = always_dim + sometimes_dim + noise_dim */
    int always_dim;         /* K — always-informative dims */
    int sometimes_dim;      /* M — per-class informative-or-not */
    int noise_dim;          /* N — never-informative noise */
    int noise_pct;           /* per-trit flip probability × 100 (default 10) */
    uint32_t proto_seed;    /* prototype generation seed */
} synth_wildcard_config_t;

synth_wildcard_config_t synth_wildcard_default(void);

/* Generate C prototypes deterministically from cfg.proto_seed.
 * out_prototypes layout: row-major [C × D] in unpacked m4t_trit_t.
 *   - First K dims: random ±1 per class.
 *   - Next M dims: per class, either ±1 (50% prob) or 0 (50% prob).
 *   - Last N dims: 0 in the prototype (but samples will be uniform random). */
void synth_wildcard_generate_prototypes(
    m4t_trit_t* out_prototypes,
    const synth_wildcard_config_t* cfg);

/* Generate n_samples training/eval samples with their labels.
 * For each sample:
 *   - Pick a class uniformly.
 *   - Copy prototype.
 *   - Per informative-position (always or sometimes-with-±1): flip with
 *     probability cfg->noise_pct/100.
 *   - Per never-informative position: fill with uniform random ternary. */
void synth_wildcard_generate_samples(
    m4t_trit_t* out_samples,
    int* out_labels,
    int n_samples,
    const m4t_trit_t* prototypes,
    const synth_wildcard_config_t* cfg,
    uint32_t sample_seed);

#ifdef __cplusplus
}
#endif

#endif /* GESH_SYNTH_WILDCARD_H */
