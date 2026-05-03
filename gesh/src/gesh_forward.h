/*
 * gesh_forward.h — Phase A.1 forward pass.
 *
 * Pipeline per query:
 *   1. (optional) project the input through a ternary projection R:
 *        s[i] = sign( Σ_j R[i,j] · x[j] )
 *      where R ∈ {-1, 0, +1}^{sig_dim × input_dim} and x ∈ {-1, 0, +1}^input_dim.
 *      Result: s ∈ {-1, 0, +1}^sig_dim, packed.
 *   2. compute Hamming distance from s to each of the bank's T tiles.
 *   3. select top-k tiles by smallest Hamming distance.
 *   4. classify by majority vote over the labels of the top-k tiles.
 *
 * If R is NULL the projection step is skipped; the input itself is
 * treated as the signature (input_dim must equal sig_dim).
 *
 * All steps are integer arithmetic. No floats anywhere in the pipeline.
 */

#ifndef GESH_FORWARD_H
#define GESH_FORWARD_H

#include "gesh_bank.h"
#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ternary projection from input_dim trits to sig_dim trits.
 * R is row-major [sig_dim × input_dim] in unpacked m4t_trit_t.
 * (Phase A.1 keeps R unpacked for ease of lattice-update probing in A.2;
 * a packed-trit variant is straightforward when profile demands it.) */
typedef struct {
    const m4t_trit_t* R;    /* nullable; NULL = identity (no projection) */
    int input_dim;
    int sig_dim;
} gesh_projection_t;

/* Forward pass for a batch of queries.
 *
 * queries:      [n_queries × input_dim] unpacked m4t_trit_t
 * bank:         the frozen bank (sig_dim must match projection output)
 * proj:         the projection (may be identity-projection per above)
 * top_k:        number of nearest tiles to consider in the vote
 * out_predictions: [n_queries] int — predicted class for each query
 *
 * Returns 0 on success.
 *
 * Memory: the function allocates O(bank->n_tiles + top_k) scratch
 * internally per call. Substantial-batch callers should consider a
 * persistent-scratch variant (not yet provided).
 */
int gesh_forward_classify(
    int* out_predictions,
    const m4t_trit_t* queries,
    int n_queries,
    const gesh_bank_t* bank,
    const gesh_projection_t* proj,
    int top_k
);

/* Wildcard-aware variant of gesh_forward_classify. Same shape, but the
 * per-tile distance metric is `m4t_route_wildcard_dist` (treating tile-
 * zero positions as wildcard / don't-care, §19 (II) interpretation)
 * instead of `m4t_popcount_dist` (§19 (I) symmetric Hamming).
 *
 * Per the §19.6 sanctioned-pairing constraint: callers MUST use a bank
 * constructed by `gesh_bank_build_class_wildcard` or another constructor
 * that places zeros DELIBERATELY. Using this with `gesh_bank_build_class_mean`
 * or `gesh_bank_build_kmeans_per_class` re-interprets emergent ties as
 * deliberate wildcards, over-promoting ambiguous matches. */
int gesh_forward_classify_wildcard(
    int* out_predictions,
    const m4t_trit_t* queries,
    int n_queries,
    const gesh_bank_t* bank,
    const gesh_projection_t* proj,
    int top_k
);

/* Hierarchical (two-stage compositional) forward pass. P0-4.
 *
 * ⚠ P0-4 NEGATIVE RESULT — DO NOT USE FOR CLASSIFICATION ⚠
 *
 * This design FAILS Gate 1 on synth_close_proto by -15.64pp vs
 * single-stage wildcard (CI [-19.19, -12.09]). The substrate-novelty
 * hook (stage-1 wildcards parameterize stage-2 dim selection) is
 * anti-discriminative: class-mean wildcards mark within-class noise,
 * not between-class boundaries. Even with the wildcard binding
 * removed (full mask), the hierarchical structure adds -8.66pp.
 *
 * Kept in source for archival reference (so the negative result is
 * documented at the API surface, not lost) and for the regression
 * probe (compose_probe.c). See journal/gesh_compositional_routing_design.md
 * for the full verdict and what it teaches.
 *
 * Per query:
 *   1. Project to sig.
 *   2. Stage-1: m4t_route_wildcard_dist over hbank->stage1 → t1.
 *   3. Stage-2: m4t_popcount_dist over hbank->stage2[t1] using
 *      hbank->stage2_masks[t1] as the active-dim selector. The mask is
 *      derived from stage-1's tile-c wildcard positions — i.e., the
 *      third state in stage-1 IS the dim-selection signal for stage-2.
 *   4. Predict = label of stage-2's nearest tile in bucket t1.
 *
 * Empty-bucket fallback: if the stage-2 sub-bank has all-zero tiles
 * (no samples landed there at build time), fall back to stage-1's
 * tile label (`hbank->stage1.labels[t1]`).
 *
 * Storage: stage1_tile_match arg is the t1 from stage-1, useful for
 * inspection / per-bucket statistics. May be NULL.
 */
int gesh_forward_classify_hierarchical(
    int* out_predictions,
    int* out_stage1_tile_match,        /* nullable: [n_queries] */
    const m4t_trit_t* queries,
    int n_queries,
    const gesh_bank_hier_t* hbank,
    const gesh_projection_t* proj
);

#ifdef __cplusplus
}
#endif

#endif /* GESH_FORWARD_H */
