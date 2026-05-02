/*
 * gesh_train.c — lattice-update coordinate descent for Gesh's projection.
 *
 * Pure integer arithmetic. No STE. The lattice IS the geometry; training
 * walks the lattice directly.
 */

#include "gesh_train.h"
#include "gesh_forward.h"
#include "gesh_bank.h"
#include "m4t_trit_pack.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test-local xorshift32. Training is not in any runtime kernel — this
 * RNG is for trit-position sampling and batch selection. */
static uint32_t xs32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

gesh_train_config_t gesh_train_default(void) {
    gesh_train_config_t cfg;
    cfg.n_flip_evals_per_epoch = 100;
    cfg.n_epochs               = 50;
    cfg.batch_size             = 64;
    cfg.log_per_epoch          = 0;
    cfg.seed                   = 0xfeedfaceu;
    return cfg;
}

void gesh_init_random_projection(
    m4t_trit_t* R, int sig_dim, int input_dim, uint32_t seed)
{
    assert(R && sig_dim > 0 && input_dim > 0);
    uint32_t state = seed ? seed : 0xdeadbeefu;
    int total = sig_dim * input_dim;
    for (int i = 0; i < total; i++) {
        /* ±1 only (no zero). Maximum information per trit at init —
         * lattice update can drive trits to zero where helpful. */
        R[i] = (xs32(&state) & 1u) ? (m4t_trit_t)1 : (m4t_trit_t)-1;
    }
}

/* Project all training samples through R; rebuild bank from the
 * projected signatures. scratch_projected is caller-allocated and
 * must be at least n_samples × sig_dim m4t_trit_t. */
static void rebuild_bank_from_projection(
    gesh_bank_t* bank,
    const m4t_trit_t* R,
    const m4t_trit_t* samples,
    const int* labels,
    int n_samples,
    int sig_dim, int input_dim, int n_classes,
    m4t_trit_t* scratch_projected)
{
    for (int i = 0; i < n_samples; i++) {
        const m4t_trit_t* x = samples + (size_t)i * input_dim;
        m4t_trit_t* s = scratch_projected + (size_t)i * sig_dim;
        for (int oi = 0; oi < sig_dim; oi++) {
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            int32_t acc = 0;
            for (int j = 0; j < input_dim; j++) {
                acc += (int32_t)r[j] * (int32_t)x[j];
            }
            s[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
        }
    }
    gesh_bank_build_class_mean(bank, scratch_projected, labels, n_samples, n_classes);
}

/* Count misclassifications on a batch with the given R + bank.
 * Returns the count, or -1 on internal error. */
static int count_errors_on_batch(
    const m4t_trit_t* R,
    const m4t_trit_t* batch_samples,
    const int* batch_labels,
    int batch_size,
    const gesh_bank_t* bank,
    int sig_dim, int input_dim, int top_k)
{
    gesh_projection_t proj = {
        .R = R, .input_dim = input_dim, .sig_dim = sig_dim
    };
    int* preds = malloc((size_t)batch_size * sizeof(int));
    int rc = gesh_forward_classify(preds, batch_samples, batch_size,
                                     bank, &proj, top_k);
    if (rc != 0) {
        free(preds);
        return -1;
    }
    int errors = 0;
    for (int i = 0; i < batch_size; i++) {
        if (preds[i] != batch_labels[i]) errors++;
    }
    free(preds);
    return errors;
}

int gesh_train_lattice_update(
    m4t_trit_t* R,
    gesh_bank_t* out_bank,
    const m4t_trit_t* train_samples,
    const int* train_labels,
    int n_samples,
    int n_classes,
    int sig_dim, int input_dim, int top_k,
    const gesh_train_config_t* cfg)
{
    assert(R && out_bank && train_samples && train_labels && cfg);
    assert(n_samples > 0 && n_classes > 0);
    assert(sig_dim > 0 && input_dim > 0);
    assert(top_k > 0 && top_k <= n_classes);
    assert(out_bank->sig_dim == sig_dim);
    assert(out_bank->n_tiles == n_classes);
    assert(out_bank->tiles_packed && out_bank->labels);

    /* Aliasing preconditions: R is the writable trainable parameter;
     * train_samples is read-only input. They must not share storage. */
    assert((const void*)R != (const void*)train_samples);
    assert((const void*)R != (const void*)out_bank->tiles_packed);

    int n_epochs = (cfg->n_epochs > 0) ? cfg->n_epochs : 50;
    int n_flips = (cfg->n_flip_evals_per_epoch > 0)
                  ? cfg->n_flip_evals_per_epoch : 100;
    int batch = (cfg->batch_size > 0) ? cfg->batch_size : 64;
    if (batch > n_samples) batch = n_samples;

    uint32_t state = cfg->seed ? cfg->seed : 0xfeedfaceu;

    /* Scratch buffers. */
    m4t_trit_t* scratch_proj = malloc(
        (size_t)n_samples * (size_t)sig_dim * sizeof(m4t_trit_t));
    m4t_trit_t* batch_samples = malloc(
        (size_t)batch * (size_t)input_dim * sizeof(m4t_trit_t));
    int* batch_labels = malloc((size_t)batch * sizeof(int));

    /* Initial bank from current R. */
    rebuild_bank_from_projection(out_bank, R, train_samples, train_labels,
                                   n_samples, sig_dim, input_dim, n_classes,
                                   scratch_proj);

    int last_train_errors = -1;
    int n_flips_total = 0;
    int n_flips_accepted = 0;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        /* Sample a fresh training batch (with replacement). */
        for (int b = 0; b < batch; b++) {
            int idx = (int)(xs32(&state) % (uint32_t)n_samples);
            memcpy(batch_samples + (size_t)b * input_dim,
                   train_samples + (size_t)idx * input_dim,
                   (size_t)input_dim * sizeof(m4t_trit_t));
            batch_labels[b] = train_labels[idx];
        }

        /* Baseline error against current R + bank. */
        int base_errors = count_errors_on_batch(
            R, batch_samples, batch_labels, batch, out_bank,
            sig_dim, input_dim, top_k);
        if (base_errors < 0) {
            free(scratch_proj); free(batch_samples); free(batch_labels);
            return -1;
        }

        for (int flip = 0; flip < n_flips; flip++) {
            int trit_idx = (int)(xs32(&state) % (uint32_t)(sig_dim * input_dim));
            int i = trit_idx / input_dim;
            int j = trit_idx % input_dim;
            m4t_trit_t orig = R[i * input_dim + j];

            int best_errors = base_errors;
            m4t_trit_t best_value = orig;

            /* Try the two non-current ternary values. */
            for (int v_idx = -1; v_idx <= 1; v_idx++) {
                m4t_trit_t v = (m4t_trit_t)v_idx;
                if (v == orig) continue;
                R[i * input_dim + j] = v;
                int errs = count_errors_on_batch(
                    R, batch_samples, batch_labels, batch, out_bank,
                    sig_dim, input_dim, top_k);
                if (errs < 0) {
                    R[i * input_dim + j] = orig;  /* restore on internal err */
                    free(scratch_proj); free(batch_samples); free(batch_labels);
                    return -1;
                }
                if (errs < best_errors) {
                    best_errors = errs;
                    best_value = v;
                }
            }
            R[i * input_dim + j] = best_value;
            n_flips_total++;
            if (best_value != orig) {
                n_flips_accepted++;
                base_errors = best_errors;
            }
        }

        /* End-of-epoch: rebuild bank from current R. */
        rebuild_bank_from_projection(out_bank, R, train_samples, train_labels,
                                       n_samples, sig_dim, input_dim, n_classes,
                                       scratch_proj);
        last_train_errors = base_errors;

        if (cfg->log_per_epoch) {
            fprintf(stderr,
                    "[gesh_train] epoch %d/%d: batch_errors=%d/%d  "
                    "accepted=%d/%d cumulative\n",
                    epoch + 1, n_epochs, last_train_errors, batch,
                    n_flips_accepted, n_flips_total);
        }
    }

    free(scratch_proj);
    free(batch_samples);
    free(batch_labels);
    return last_train_errors;
}
