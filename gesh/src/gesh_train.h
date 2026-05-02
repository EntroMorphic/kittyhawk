/*
 * gesh_train.h — lattice-update training for the projection R.
 *
 * Phase A.2 mechanism. No STE, no shadow parameters, no Gumbel-softmax.
 * Coordinate descent over R's ternary trits with bit-exact loss deltas.
 *
 * Algorithm:
 *   For each epoch:
 *     1. Sample a training batch (with replacement).
 *     2. Compute baseline classification error on the batch.
 *     3. Repeat n_flip_evals_per_epoch times:
 *        a. Pick a random trit position (i, j) in R.
 *        b. For each of the two non-current ternary values v:
 *           - Tentatively flip R[i, j] = v.
 *           - Recompute classification error on the batch.
 *           - Track the flip that minimizes error.
 *        c. Apply the best flip (or revert if neither helps).
 *        d. Update baseline to the new error.
 *     4. Rebuild bank from current R (end-of-epoch refresh).
 *
 * The bank stales slightly during the per-epoch flip-evaluation loop
 * (it reflects the R at epoch start, not the current trit-flip-tested R).
 * The end-of-epoch refresh re-syncs. If staleness becomes a measured
 * problem (training instability, oscillation), refresh frequency can
 * tighten — that's a Phase A.2 tuning question, not a design question.
 *
 * No float anywhere. All arithmetic is integer.
 */

#ifndef GESH_TRAIN_H
#define GESH_TRAIN_H

#include "gesh_bank.h"
#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_flip_evals_per_epoch;  /* number of (i, j) positions to evaluate per epoch */
    int n_epochs;                 /* total epochs */
    int batch_size;               /* samples per error-evaluation batch */

    /* Intra-epoch refresh frequencies. Both default to 0, meaning
     * "refresh once per epoch" (the original behavior). Set to a
     * positive K to refresh every K flips within the epoch. Smaller K
     * gives a less-stale optimization landscape at higher per-flip
     * cost.
     *   bank_refresh_every — fixes the stale-bank issue (H1 in red-team).
     *   batch_refresh_every — fixes per-epoch batch overfitting (H2). */
    int bank_refresh_every;
    int batch_refresh_every;

    /* Early stopping. Halt training if no batch-error improvement for
     * `early_stop_patience` consecutive epochs. 0 = disabled. */
    int early_stop_patience;

    /* Initial trit distribution for `gesh_init_random_projection_balanced`
     * (does NOT affect `gesh_init_random_projection`, which always uses
     * ±1 for backwards compatibility). 0 = balanced ±1/0; 1 = sign-only. */
    int init_balanced;

    int log_per_epoch;            /* 1 = print per-epoch summary to stderr, 0 = silent */

    /* Seed for trit-position sampling and batch selection. NOTE: 0 is a
     * VALID seed and will be used as-is (no fallback to a default). */
    uint32_t seed;
} gesh_train_config_t;

/* Default config for Phase A.2 measurements. */
gesh_train_config_t gesh_train_default(void);

/* Initialize R with random ±1 ternary values (no zeros, for maximum
 * information per trit at init).
 *
 * R: [sig_dim × input_dim] unpacked ternary, caller-allocated.
 * seed: 0 is a valid seed and used as-is. */
void gesh_init_random_projection(
    m4t_trit_t* R,
    int sig_dim, int input_dim,
    uint32_t seed
);

/* Initialize R with balanced random ternary (-1, 0, +1 each with prob
 * ~1/3). Allows lattice update to reach zero values without traversing
 * a flip; useful when the underlying projection has signal-free dims
 * the optimizer should learn to gate out. */
void gesh_init_random_projection_balanced(
    m4t_trit_t* R,
    int sig_dim, int input_dim,
    uint32_t seed
);

/* Train R via lattice-update coordinate descent. R is in/out — caller
 * provides initial values (e.g., via gesh_init_random_projection); the
 * function refines them.
 *
 * out_bank is filled with the final (post-training) bank, rebuilt from
 * the current R applied to all training samples. Caller must allocate
 * out_bank->tiles_packed (size n_classes × M4T_TRIT_PACKED_BYTES(sig_dim))
 * and out_bank->labels (size n_classes); set out_bank->n_tiles = n_classes
 * and out_bank->sig_dim = sig_dim.
 *
 * Returns the final batch classification error (count of misclassifications
 * in the last training batch) on success, or -1 on internal failure.
 *
 * Aliasing preconditions: R must not alias train_samples or any field of
 * out_bank. */
int gesh_train_lattice_update(
    m4t_trit_t*       R,
    gesh_bank_t*      out_bank,
    const m4t_trit_t* train_samples,
    const int*        train_labels,
    int n_samples,
    int n_classes,
    int sig_dim,
    int input_dim,
    int top_k,
    const gesh_train_config_t* cfg
);

#ifdef __cplusplus
}
#endif

#endif /* GESH_TRAIN_H */
