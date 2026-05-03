/*
 * gesh_bank.h — frozen bank construction for Gesh Phase A.
 *
 * The bank is an array of T tiles, each a packed-trit signature of
 * `sig_dim` trits. Each tile carries a class label.
 *
 * Phase A.1 construction: class-conditional ternary mean.
 *
 *   For each class c:
 *     For each dim j:
 *       sum[j] = Σ_{i : label(i) == c} sample[i, j]
 *       tile[c, j] = sign(sum[j])  (with tie at zero → 0)
 *
 * One tile per class; T = C. Substrate-legal (integer arithmetic).
 *
 * Phase A.2 will add no new bank constructors — Phase A.2's training
 * operates on the projection R, not the bank. Bank stays class-mean.
 *
 * Phase B+ variants (each gated on a measured failure mode):
 *   - k-means with k tiles per class (T = C × k); needed if
 *     class-mean is too coarse for the chosen task.
 *   - PCA-derived prototypes; gated on whether projection learning
 *     can match the PCA prior anyway.
 *   - Learned bank (joint optimization with routing); gated on
 *     whether frozen-bank cost is too high.
 */

#ifndef GESH_BANK_H
#define GESH_BANK_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The frozen bank. tiles_packed is row-major [T × Dp] where
 * Dp = M4T_TRIT_PACKED_BYTES(sig_dim). labels[t] gives the class
 * label for tile t. Both are caller-allocated; bank construction
 * fills them.
 *
 * Label convention (CONSUMED BY gesh_forward_classify):
 *   labels[t] >= 0 for all t
 *   labels are dense from 0; max(labels) + 1 = number of classes
 *
 * The forward pass derives n_classes from max(labels) + 1 and asserts
 * non-negativity per tile. A future bank constructor with sparse or
 * sentinel-bearing labels MUST update both this header and the forward
 * pass — currently the assumption is enforced as an assert at
 * forward-time, not at bank-construction-time. */
typedef struct {
    uint8_t* tiles_packed;   /* [T × M4T_TRIT_PACKED_BYTES(sig_dim)] */
    int*     labels;         /* [T] — non-negative, dense from 0 */
    int      n_tiles;        /* T */
    int      sig_dim;        /* trits per tile signature */
} gesh_bank_t;

/* Build a bank with one tile per class, where each tile is the
 * class-conditional ternary mean of the training samples.
 *
 * samples: [n_samples × sig_dim] unpacked m4t_trit_t (ternary, ±1 or 0).
 * labels:  [n_samples] class indices in [0, n_classes).
 *
 * The bank is filled with n_classes tiles in class order (tile c has
 * label c). Tiles are packed to bank->tiles_packed; bank->labels[c] == c.
 *
 * The caller is responsible for sizing bank->tiles_packed and
 * bank->labels appropriately:
 *   bank->n_tiles  must equal n_classes
 *   bank->sig_dim  is the per-tile signature length (== sample dim)
 *   bank->tiles_packed must be at least n_classes × M4T_TRIT_PACKED_BYTES(sig_dim) bytes
 *   bank->labels        must be at least n_classes ints */
void gesh_bank_build_class_mean(
    gesh_bank_t* bank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes
);

/* Build a class-mean bank with deliberate wildcards (zero-state,
 * §19 Wildcard interpretation) at low signal-to-noise positions for
 * each class.
 *
 * Algorithm:
 *   For each class c:
 *     1. Compute per-dim signed sum across class-c samples (as in
 *        gesh_bank_build_class_mean).
 *     2. Compute per-dim sample count.
 *     3. Compute per-dim "signal magnitude" = |sum| × 1000 / count
 *        (permille of mean magnitude in [0, 1000]).
 *     4. If signal_pm < snr_threshold_permille → tile trit = 0
 *        (DELIBERATE wildcard, §19 (II)).
 *        Else → tile trit = sign(sum) (standard ±1).
 *     5. Pack tile.
 *
 * Output tiles use (II) Wildcard zero-state interpretation. Pair with
 * `m4t_route_wildcard_dist` for substrate-native decision-rule routing.
 * Pairing with `m4t_popcount_dist` (which uses (I) Tie-cancellation
 * symmetric Hamming) is a §19 violation — wildcards get re-interpreted
 * as half-cost ties, defeating their purpose.
 *
 * Determinism: same samples + labels + threshold → same output.
 *
 * Edge case: snr_threshold_permille = 0 produces output bit-identical to
 * gesh_bank_build_class_mean (no positions zeroed; threshold satisfied
 * by anything ≥ 0). Useful as a sanity check.
 *
 * Preconditions:
 *   bank->n_tiles == n_classes
 *   bank->sig_dim > 0
 *   bank->tiles_packed and bank->labels non-NULL and sized
 *   labels[i] in [0, n_classes)
 *   snr_threshold_permille >= 0 */
void gesh_bank_build_class_wildcard(
    gesh_bank_t* bank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int snr_threshold_permille
);

/* Confidence-aware bank: builds class-mean trit signatures AND per-class
 * confidence bitmaps. The confidence bit is set at positions where the
 * class-mean's signal strength exceeds tau_strong_permille, indicating
 * "this class is strongly characterized by this dim."
 *
 * Outputs are paired: bank->tiles_packed (trit signatures) + a separate
 * caller-allocated conf_bits buffer of size n_classes × ceil(sig_dim/8).
 *
 * Pair with `m4t_route_confidence_weighted_dist` for substrate-novel
 * magnitude-aware routing. */
void gesh_bank_build_class_mean_with_confidence(
    gesh_bank_t* bank,
    uint8_t* conf_bits,            /* [n_classes × ceil(sig_dim/8)] */
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int tau_strong_permille
);

/* Multi-prototype bank: k > 1 tiles per class via k-means clustering on
 * the sample space restricted to each class. Total tiles T = k_per_class
 * × n_classes; tile (c × k_per_class + j) carries label c.
 *
 * Algorithm:
 *   1. For each class c independently:
 *      a. Gather indices of class-c samples.
 *      b. Init k_per_class assignments via stratified mod-k partition
 *         over the per-class sample list (deterministic; no random
 *         weights, per the substrate-discipline rule).
 *      c. Iterate up to max_iters times:
 *         - Compute centroid for each cluster as the per-dim ternary
 *           mean (sign of int32 sum) of assigned samples.
 *         - Reassign each sample to the nearest centroid by Hamming
 *           distance over packed-trit signatures.
 *         - If no sample changed cluster, converged. Break.
 *      d. Pack converged centroids into bank tiles c × k_per_class
 *         through (c+1) × k_per_class − 1, all labeled c.
 *
 * Empty-cluster handling: if a cluster receives zero assignments after
 * a reassignment pass, its centroid is left at the previous iteration's
 * value (no update). Persistent empty clusters reduce the effective k.
 *
 * Preconditions:
 *   bank->n_tiles == k_per_class × n_classes
 *   k_per_class >= 1
 *   max_iters >= 1
 *   samples and labels of length n_samples
 *   labels[i] in [0, n_classes)
 *   bank->tiles_packed sized for k_per_class × n_classes × M4T_TRIT_PACKED_BYTES(sig_dim)
 *   bank->labels sized for k_per_class × n_classes ints
 *
 * `seed` is reserved for tie-breaking in future variants; currently
 * unused (the deterministic stratified init makes it irrelevant). */
void gesh_bank_build_kmeans_per_class(
    gesh_bank_t* bank,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int n_classes,
    int k_per_class,
    int max_iters,
    uint32_t seed
);

/* Hierarchical (two-stage compositional) bank for P0-4.
 *
 * Stage-1: a wildcard bank with one tile per class (built via
 *          gesh_bank_build_class_wildcard). Samples route to stage-1
 *          tiles by m4t_route_wildcard_dist.
 *
 * Stage-2: per stage-1 tile, a class-mean sub-bank built from the
 *          samples that land in that stage-1 bucket. Distance at
 *          stage-2 uses m4t_popcount_dist with a mask derived from
 *          the stage-1 tile's WILDCARD POSITIONS — i.e., stage-2
 *          differentiates only on the dims where stage-1 marked
 *          intra-bucket variation.
 *
 * Substrate-novel composition: the third state in stage-1's tile IS
 * the stage-2 dim-selection signal. No separate mask channel; the
 * wildcard pattern routes itself.
 *
 * Storage:
 *   stage1               — gesh_bank_t with n_stage1 tiles
 *   stage2[n_stage1]     — per-bucket sub-banks (class-mean)
 *   stage2_masks[n_stage1 × Dp] — precomputed wildcard-select masks,
 *                          one per stage-1 tile.
 *
 * The caller is responsible for memory ownership; gesh_bank_hier_alloc /
 * gesh_bank_hier_free are convenience helpers. */
typedef struct {
    gesh_bank_t  stage1;
    gesh_bank_t* stage2;          /* [n_stage1] sub-banks */
    uint8_t*     stage2_masks;    /* [n_stage1 × Dp] precomputed dim-selectors */
    int          n_stage1;
    int          sig_dim;
} gesh_bank_hier_t;

/* Allocate the hierarchical bank's owned buffers. After this call,
 * stage1.{tiles_packed,labels}, stage2[*].{tiles_packed,labels}, and
 * stage2_masks are heap-allocated and ready to be filled by the build
 * routine. n_stage1 == n_classes (one stage-1 tile per class).
 *
 * stage2[c].n_tiles == n_classes (one stage-2 tile per class within
 * each bucket; sparse buckets get all-zero stage-2 tiles which act
 * as inactive, see gesh_forward note). */
int gesh_bank_hier_alloc(
    gesh_bank_hier_t* hbank,
    int n_classes,
    int sig_dim
);

void gesh_bank_hier_free(gesh_bank_hier_t* hbank);

/* Build the hierarchical bank from projected training samples.
 *
 * Steps:
 *   1. Build stage-1 wildcard bank via gesh_bank_build_class_wildcard.
 *   2. For each stage-1 tile c:
 *      a. Precompute stage2_masks[c] = wildcard_select_mask(stage1.tiles[c]).
 *      b. Find samples whose nearest stage-1 tile (by wildcard_dist) is c.
 *      c. Build stage2[c] as a class-mean bank over those samples
 *         (full sig_dim; the mask gates compare-time, not build-time).
 *
 * If a stage-1 bucket has zero assigned samples, the corresponding
 * stage2[c] is left zero-filled and the forward pass should fall back
 * to stage-1's tile label.
 *
 * Substrate-discipline: build operates on packed-trit signatures; no
 * float, no random projection.
 *
 * Determinism: same inputs → same output (stable assignment by
 * lowest tile index on distance tie). */
void gesh_bank_build_hierarchical(
    gesh_bank_hier_t* hbank,
    const m4t_trit_t* samples,    /* [n_samples × sig_dim], unpacked */
    const int* labels,
    int n_samples,
    int n_classes,
    int snr_threshold_permille
);

#ifdef __cplusplus
}
#endif

#endif /* GESH_BANK_H */
