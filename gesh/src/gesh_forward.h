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

#ifdef __cplusplus
}
#endif

#endif /* GESH_FORWARD_H */
