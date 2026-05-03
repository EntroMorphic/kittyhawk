/*
 * mnist_full_run.c — single decisive MNIST run, no sweep, no multi-seed.
 *
 * Config (per explicit request):
 *   n_train      = 60,000 (full MNIST training set)
 *   n_test       = 10,000 (full MNIST test set)
 *   sig_dim      = 64
 *   batch_size   = 128
 *   top_k        = 3
 *   n_epochs     = 64
 *   early_stop   = OFF (patience = 0)
 *   flip budget  = 5 × sig_dim × D = 250,880
 *   refresh      = n_flips/4 (same cadence as other probes)
 *
 * Seeds: init = 0xc0ffeebb, train = 0xa5a5a5a5 (matching position 0 of
 * existing probe seed lists for traceability).
 *
 * Note: this probe still has scalar math in consumer code (per the
 * audit: per-row widen, bank class-sum, top-k insertion, vote argmax,
 * R init, etc.). Substrate-purification (full A+B) runs as a separate
 * cycle AFTER this measurement. The numerical results will be bit-
 * identical pre- and post-purification because the scalar math is
 * deterministic integer arithmetic; only the call site of the math
 * changes.
 */

#include "image_canon.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_project.h"
#include "gesh_train.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SIG_DIM     64
#define BATCH_SIZE  128
#define TOP_K       1
#define N_EPOCHS    64

static const char* DEFAULT_MNIST_DIR =
    "/Users/aaronjosserand-austin/Projects/glyph/01MAY26_archived/data/mnist";

static int eval_test_accuracy_pm(
    const m4t_trit_t* R,
    const gesh_bank_t* bank,
    const m4t_trit_t* test, const int* test_lbl, int n_test,
    int sig_dim, int input_dim, int top_k)
{
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    int rc = gesh_forward_classify(preds, test, n_test, bank, &proj, top_k);
    if (rc != 0) { free(preds); return -1; }
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        if (preds[i] == test_lbl[i]) correct++;
    }
    free(preds);
    return (correct * 1000) / n_test;
}

static int run_random(
    const m4t_trit_t* train, const int* train_lbl, int n_train,
    const m4t_trit_t* test,  const int* test_lbl,  int n_test,
    int sig_dim, int input_dim, int n_classes, int top_k,
    uint32_t init_seed)
{
    m4t_trit_t* R = malloc((size_t)sig_dim * input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, input_dim, init_seed);

    gesh_bank_t bank;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    bank.tiles_packed = malloc((size_t)n_classes * (size_t)Dp);
    bank.labels = malloc((size_t)n_classes * sizeof(int));
    bank.n_tiles = n_classes;
    bank.sig_dim = sig_dim;

    m4t_trit_t* projected = malloc((size_t)n_train * (size_t)sig_dim
                                     * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(projected, train, n_train,
                                  R, sig_dim, input_dim);
    gesh_bank_build_class_mean(&bank, projected, train_lbl, n_train, n_classes);
    free(projected);

    int pm = eval_test_accuracy_pm(R, &bank, test, test_lbl, n_test,
                                     sig_dim, input_dim, top_k);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static int run_trained(
    const m4t_trit_t* train, const int* train_lbl, int n_train,
    const m4t_trit_t* test,  const int* test_lbl,  int n_test,
    int sig_dim, int input_dim, int n_classes, int top_k,
    int flip_budget, uint32_t init_seed, uint32_t train_seed)
{
    m4t_trit_t* R = malloc((size_t)sig_dim * input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, input_dim, init_seed);

    gesh_bank_t bank;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    bank.tiles_packed = malloc((size_t)n_classes * (size_t)Dp);
    bank.labels = malloc((size_t)n_classes * sizeof(int));
    bank.n_tiles = n_classes;
    bank.sig_dim = sig_dim;

    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = N_EPOCHS;
    cfg.n_flip_evals_per_epoch = flip_budget / cfg.n_epochs;
    if (cfg.n_flip_evals_per_epoch < 10) cfg.n_flip_evals_per_epoch = 10;
    cfg.batch_size = BATCH_SIZE;
    cfg.bank_refresh_every = cfg.n_flip_evals_per_epoch / 4;
    if (cfg.bank_refresh_every < 1) cfg.bank_refresh_every = 1;
    cfg.batch_refresh_every = cfg.n_flip_evals_per_epoch / 4;
    if (cfg.batch_refresh_every < 1) cfg.batch_refresh_every = 1;
    cfg.early_stop_patience = 0;  /* OFF, per the request */
    cfg.log_per_epoch = 1;        /* per-epoch progress to stderr */
    cfg.seed = train_seed;

    fprintf(stderr, "Trained config: epochs=%d, flips/epoch=%d, batch=%d, "
                     "bank_refresh=%d, batch_refresh=%d, early_stop=OFF, "
                     "top_k=%d\n",
            cfg.n_epochs, cfg.n_flip_evals_per_epoch, cfg.batch_size,
            cfg.bank_refresh_every, cfg.batch_refresh_every, top_k);
    fflush(stderr);

    int rc = gesh_train_lattice_update(R, &bank, train, train_lbl,
                                          n_train, n_classes,
                                          sig_dim, input_dim, top_k, &cfg);
    if (rc < 0) {
        free(R); free(bank.tiles_packed); free(bank.labels);
        return -1;
    }
    int pm = eval_test_accuracy_pm(R, &bank, test, test_lbl, n_test,
                                     sig_dim, input_dim, top_k);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : DEFAULT_MNIST_DIR;
    image_canon_dataset_t ds;
    printf("# MNIST single-run probe — full 60K train, full 10K test\n");
    printf("# sig_dim=%d, top_k=%d, batch=%d, n_epochs=%d, early_stop=OFF\n",
           SIG_DIM, TOP_K, BATCH_SIZE, N_EPOCHS);
    printf("# Loading MNIST from %s\n", dir);
    fflush(stdout);
    if (image_canon_load_mnist(&ds, dir) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    printf("# Loaded train=%d test=%d input_dim=%d (%dx%d)\n",
           ds.n_train, ds.n_test, ds.input_dim, ds.img_w, ds.img_h);
    image_canon_normalize(&ds);
    int64_t tau = image_canon_quantize_tau(ds.x_train, 1000, ds.input_dim, 0.60);
    printf("# tau = %lld (density 0.60, sample 1000)\n", (long long)tau);

    m4t_trit_t* train_trits = malloc((size_t)ds.n_train * ds.input_dim
                                       * sizeof(m4t_trit_t));
    m4t_trit_t* test_trits  = malloc((size_t)ds.n_test  * ds.input_dim
                                       * sizeof(m4t_trit_t));
    image_canon_quantize_unpacked_batch(ds.x_train, ds.n_train, ds.input_dim,
                                          tau, train_trits);
    image_canon_quantize_unpacked_batch(ds.x_test,  ds.n_test,  ds.input_dim,
                                          tau, test_trits);
    printf("# Quantized %d train + %d test images.\n",
           ds.n_train, ds.n_test);
    fflush(stdout);

    int flip_budget = 5 * SIG_DIM * ds.input_dim;
    printf("# flip budget = 5 * sig_dim * D = %d\n", flip_budget);
    printf("\n");

    /* Random R baseline (single trial, deterministic). */
    clock_t t0 = clock();
    int rand_pm = run_random(train_trits, ds.y_train, ds.n_train,
                                test_trits,  ds.y_test,  ds.n_test,
                                SIG_DIM, ds.input_dim, 10, TOP_K,
                                0xc0ffeebbu);
    double rand_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Trained R (single decisive run). */
    clock_t t1 = clock();
    int train_pm = run_trained(train_trits, ds.y_train, ds.n_train,
                                  test_trits,  ds.y_test,  ds.n_test,
                                  SIG_DIM, ds.input_dim, 10, TOP_K,
                                  flip_budget,
                                  0xc0ffeebbu, 0xa5a5a5a5u);
    double train_s = (double)(clock() - t1) / CLOCKS_PER_SEC;

    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("\n");
    printf("## Results (single seed: init=0xc0ffeebb, train=0xa5a5a5a5)\n\n");
    printf("| variant       | accuracy | runtime |\n");
    printf("|---------------|----------|---------|\n");
    printf("| random R      | %5.1f%%   | %5.1fs  |\n", rand_pm / 10.0, rand_s);
    printf("| trained R     | %5.1f%%   | %5.1fs  |\n", train_pm / 10.0, train_s);
    printf("| gain          | %+5.1fpp |          |\n",
           (train_pm - rand_pm) / 10.0);
    printf("\nTotal runtime: %.1fs\n", total_s);

    free(train_trits); free(test_trits);
    image_canon_free(&ds);
    return 0;
}
