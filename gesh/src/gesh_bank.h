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

#ifdef __cplusplus
}
#endif

#endif /* GESH_BANK_H */
