/*
 * gesh_train.c — lattice-update coordinate descent for Gesh's projection.
 *
 * Pure integer arithmetic. No STE. The lattice IS the geometry; training
 * walks the lattice directly.
 *
 * Phase A.2 + red-team remediations:
 *   - Configurable intra-epoch bank refresh (H1).
 *   - Configurable intra-epoch batch resampling (H2).
 *   - Early stopping on epoch-loss plateau (M5).
 *   - Budget-vs-R-size sanity warning (M6).
 *   - seed = 0 is a valid seed, no fallback (L1).
 *   - Balanced random init variant (L2).
 *   - Persistent-scratch forward pass for the hot loop (M4).
 */

#include "gesh_train.h"
#include "gesh_forward.h"
#include "gesh_bank.h"
#include "m4t_trit_pack.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test/training-local xorshift32. */
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
    cfg.bank_refresh_every     = 0;   /* once per epoch */
    cfg.batch_refresh_every    = 0;   /* once per epoch */
    cfg.early_stop_patience    = 0;   /* disabled */
    cfg.init_balanced          = 0;
    cfg.log_per_epoch          = 0;
    cfg.seed                   = 0xfeedfaceu;
    return cfg;
}

void gesh_init_random_projection(
    m4t_trit_t* R, int sig_dim, int input_dim, uint32_t seed)
{
    assert(R && sig_dim > 0 && input_dim > 0);
    /* L1 fix: seed = 0 is a valid seed. Use as-is. */
    uint32_t state = seed;
    /* xorshift32 with state 0 produces all zeros. Force a non-zero
     * state by mixing in a constant — preserves user's "deliberate 0
     * seed" intent while avoiding the all-zeros degenerate sequence. */
    if (state == 0) state = 0x12345678u;
    int total = sig_dim * input_dim;
    for (int i = 0; i < total; i++) {
        R[i] = (xs32(&state) & 1u) ? (m4t_trit_t)1 : (m4t_trit_t)-1;
    }
}

void gesh_init_random_projection_balanced(
    m4t_trit_t* R, int sig_dim, int input_dim, uint32_t seed)
{
    assert(R && sig_dim > 0 && input_dim > 0);
    uint32_t state = seed;
    if (state == 0) state = 0x12345678u;
    int total = sig_dim * input_dim;
    for (int i = 0; i < total; i++) {
        uint32_t r = xs32(&state) % 3u;
        R[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }
}

/* Project all training samples through R; rebuild bank from the
 * projected signatures. */
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

/* Resample a fresh training batch. */
static void resample_batch(
    m4t_trit_t* batch_samples, int* batch_labels,
    int batch_size,
    const m4t_trit_t* train, const int* train_lbl,
    int n_samples, int input_dim,
    uint32_t* state)
{
    for (int b = 0; b < batch_size; b++) {
        int idx = (int)(xs32(state) % (uint32_t)n_samples);
        memcpy(batch_samples + (size_t)b * input_dim,
               train + (size_t)idx * input_dim,
               (size_t)input_dim * sizeof(m4t_trit_t));
        batch_labels[b] = train_lbl[idx];
    }
}

/* M4 fix: persistent-scratch forward pass for the trainer's hot loop.
 * Allocates everything once outside the loop; reuses across calls. */
typedef struct {
    m4t_trit_t* unpacked_sig;     /* [sig_dim] */
    uint8_t*    packed_sig;        /* [Dp_sig] */
    uint8_t*    mask_packed;       /* [Dp_sig] */
    int32_t*    dists;             /* [n_tiles] */
    int*        topk_idx;          /* [top_k] */
    int*        vote;              /* [n_classes] */
    int*        preds;             /* [batch_size] */
    /* Top-k internal scratch (was malloc'd inside topk_smallest_indices). */
    int32_t*    topk_buf_d;        /* [top_k] */
} gesh_train_scratch_t;

static void scratch_alloc(
    gesh_train_scratch_t* s,
    int sig_dim, int n_tiles, int top_k, int n_classes, int batch_size)
{
    int Dp_sig = M4T_TRIT_PACKED_BYTES(sig_dim);
    s->unpacked_sig = malloc((size_t)sig_dim * sizeof(m4t_trit_t));
    s->packed_sig   = malloc((size_t)Dp_sig);
    s->mask_packed  = malloc((size_t)Dp_sig);
    s->dists        = malloc((size_t)n_tiles * sizeof(int32_t));
    s->topk_idx     = malloc((size_t)top_k * sizeof(int));
    s->vote         = malloc((size_t)n_classes * sizeof(int));
    s->preds        = malloc((size_t)batch_size * sizeof(int));
    s->topk_buf_d   = malloc((size_t)top_k * sizeof(int32_t));

    /* Mask construction (one-time). */
    memset(s->mask_packed, 0xFF, (size_t)Dp_sig);
    int tail_trits = sig_dim & 3;
    if (tail_trits > 0) {
        uint8_t tail_mask = (uint8_t)((1u << (tail_trits * 2)) - 1u);
        s->mask_packed[Dp_sig - 1] = tail_mask;
    }
}

static void scratch_free(gesh_train_scratch_t* s) {
    free(s->unpacked_sig); free(s->packed_sig); free(s->mask_packed);
    free(s->dists); free(s->topk_idx); free(s->vote);
    free(s->preds); free(s->topk_buf_d);
}

/* Hot-loop forward + error-count using pre-allocated scratch. Mirrors
 * gesh_forward_classify but skips the per-call allocs.
 *
 * Returns the number of misclassifications on the batch. */
static int count_errors_scratch(
    const m4t_trit_t* R,
    const m4t_trit_t* batch_samples, const int* batch_labels,
    int batch_size,
    const gesh_bank_t* bank,
    int sig_dim, int input_dim, int top_k, int n_classes,
    gesh_train_scratch_t* sc)
{
    int Dp_sig = M4T_TRIT_PACKED_BYTES(sig_dim);
    int errors = 0;

    for (int q = 0; q < batch_size; q++) {
        const m4t_trit_t* x = batch_samples + (size_t)q * input_dim;

        /* Project x through R into a ternary signature. */
        for (int oi = 0; oi < sig_dim; oi++) {
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            int32_t acc = 0;
            for (int j = 0; j < input_dim; j++) {
                acc += (int32_t)r[j] * (int32_t)x[j];
            }
            sc->unpacked_sig[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
        }
        m4t_pack_trits_1d(sc->packed_sig, sc->unpacked_sig, sig_dim);

        /* Hamming distance to each tile. */
        int T = bank->n_tiles;
        for (int t = 0; t < T; t++) {
            const uint8_t* tile = bank->tiles_packed + (size_t)t * Dp_sig;
            sc->dists[t] = m4t_popcount_dist(sc->packed_sig, tile,
                                               sc->mask_packed, Dp_sig);
        }

        /* Top-k smallest (insertion sort over T into sc->topk_buf_d). */
        for (int k = 0; k < top_k; k++) {
            sc->topk_idx[k] = -1;
            sc->topk_buf_d[k] = INT32_MAX;
        }
        for (int t = 0; t < T; t++) {
            int32_t d = sc->dists[t];
            if (d >= sc->topk_buf_d[top_k - 1]) continue;
            int ins = top_k - 1;
            while (ins > 0 && d < sc->topk_buf_d[ins - 1]) ins--;
            for (int sh = top_k - 1; sh > ins; sh--) {
                sc->topk_buf_d[sh] = sc->topk_buf_d[sh - 1];
                sc->topk_idx[sh]   = sc->topk_idx[sh - 1];
            }
            sc->topk_buf_d[ins] = d;
            sc->topk_idx[ins]   = t;
        }

        /* Class vote. */
        memset(sc->vote, 0, (size_t)n_classes * sizeof(int));
        for (int k = 0; k < top_k; k++) {
            int idx = sc->topk_idx[k];
            if (idx < 0) continue;
            sc->vote[bank->labels[idx]]++;
        }
        int best_class = 0;
        int best_vote = sc->vote[0];
        for (int c = 1; c < n_classes; c++) {
            if (sc->vote[c] > best_vote) {
                best_vote = sc->vote[c];
                best_class = c;
            }
        }
        if (best_class != batch_labels[q]) errors++;
    }

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
    assert((const void*)R != (const void*)train_samples);
    assert((const void*)R != (const void*)out_bank->tiles_packed);

    int n_epochs = (cfg->n_epochs > 0) ? cfg->n_epochs : 50;
    int n_flips  = (cfg->n_flip_evals_per_epoch > 0)
                   ? cfg->n_flip_evals_per_epoch : 100;
    int batch    = (cfg->batch_size > 0) ? cfg->batch_size : 64;
    if (batch > n_samples) batch = n_samples;

    /* M6: budget-vs-R-size sanity warning. */
    long total_trits = (long)sig_dim * (long)input_dim;
    long total_flips = (long)n_epochs * (long)n_flips;
    if (cfg->log_per_epoch && total_flips < total_trits) {
        fprintf(stderr,
                "[gesh_train] WARNING: total flip budget %ld < R size %ld; "
                "each trit visited <1 time on average.\n",
                total_flips, total_trits);
    }

    /* L1: seed = 0 is valid; use as-is (with all-zero protection). */
    uint32_t state = cfg->seed;
    if (state == 0) state = 0x12345678u;

    /* Refresh frequencies. 0 → once-per-epoch (= n_flips). */
    int bank_refresh = (cfg->bank_refresh_every > 0)
                       ? cfg->bank_refresh_every : n_flips;
    int batch_refresh = (cfg->batch_refresh_every > 0)
                        ? cfg->batch_refresh_every : n_flips;

    /* Scratch buffers. */
    m4t_trit_t* scratch_proj = malloc(
        (size_t)n_samples * (size_t)sig_dim * sizeof(m4t_trit_t));
    m4t_trit_t* batch_samples = malloc(
        (size_t)batch * (size_t)input_dim * sizeof(m4t_trit_t));
    int* batch_labels = malloc((size_t)batch * sizeof(int));

    gesh_train_scratch_t sc;
    scratch_alloc(&sc, sig_dim, n_classes, top_k, n_classes, batch);

    /* Initial bank from current R. */
    rebuild_bank_from_projection(out_bank, R, train_samples, train_labels,
                                   n_samples, sig_dim, input_dim, n_classes,
                                   scratch_proj);

    int last_train_errors = -1;
    int n_flips_total = 0;
    int n_flips_accepted = 0;

    /* Early stopping bookkeeping. */
    int best_epoch_errors = INT32_MAX;
    int epochs_no_improve = 0;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        /* Initial batch for this epoch. */
        resample_batch(batch_samples, batch_labels, batch,
                       train_samples, train_labels, n_samples, input_dim,
                       &state);

        int base_errors = count_errors_scratch(
            R, batch_samples, batch_labels, batch, out_bank,
            sig_dim, input_dim, top_k, n_classes, &sc);

        for (int flip = 0; flip < n_flips; flip++) {
            int trit_idx = (int)(xs32(&state) % (uint32_t)(sig_dim * input_dim));
            int i = trit_idx / input_dim;
            int j = trit_idx % input_dim;
            m4t_trit_t orig = R[i * input_dim + j];

            int best_errors = base_errors;
            m4t_trit_t best_value = orig;

            for (int v_idx = -1; v_idx <= 1; v_idx++) {
                m4t_trit_t v = (m4t_trit_t)v_idx;
                if (v == orig) continue;
                R[i * input_dim + j] = v;
                int errs = count_errors_scratch(
                    R, batch_samples, batch_labels, batch, out_bank,
                    sig_dim, input_dim, top_k, n_classes, &sc);
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

            /* H1: intra-epoch bank refresh. */
            if (((flip + 1) % bank_refresh) == 0 && (flip + 1) < n_flips) {
                rebuild_bank_from_projection(
                    out_bank, R, train_samples, train_labels,
                    n_samples, sig_dim, input_dim, n_classes, scratch_proj);
                /* Re-measure baseline against fresh bank. */
                base_errors = count_errors_scratch(
                    R, batch_samples, batch_labels, batch, out_bank,
                    sig_dim, input_dim, top_k, n_classes, &sc);
            }

            /* H2: intra-epoch batch refresh. */
            if (((flip + 1) % batch_refresh) == 0 && (flip + 1) < n_flips) {
                resample_batch(batch_samples, batch_labels, batch,
                               train_samples, train_labels, n_samples, input_dim,
                               &state);
                base_errors = count_errors_scratch(
                    R, batch_samples, batch_labels, batch, out_bank,
                    sig_dim, input_dim, top_k, n_classes, &sc);
            }
        }

        /* End-of-epoch bank refresh (always, for consistent post-epoch state). */
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

        /* M5: early stopping. */
        if (cfg->early_stop_patience > 0) {
            if (last_train_errors < best_epoch_errors) {
                best_epoch_errors = last_train_errors;
                epochs_no_improve = 0;
            } else {
                epochs_no_improve++;
                if (epochs_no_improve >= cfg->early_stop_patience) {
                    if (cfg->log_per_epoch) {
                        fprintf(stderr,
                                "[gesh_train] early stop at epoch %d "
                                "(no improvement for %d epochs)\n",
                                epoch + 1, cfg->early_stop_patience);
                    }
                    break;
                }
            }
        }
    }

    free(scratch_proj);
    free(batch_samples);
    free(batch_labels);
    scratch_free(&sc);
    return last_train_errors;
}
