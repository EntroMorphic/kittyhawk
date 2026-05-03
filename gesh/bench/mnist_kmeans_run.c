/*
 * mnist_kmeans_run.c — k_per_class sweep on MNIST with multi-prototype bank.
 *
 * Sweeps k ∈ {1, 2, 4, 6, 8, 12, 16, 24, 32} prototypes per class with
 * a fixed random R (no training). Maps the bank-capacity-vs-accuracy
 * curve. Reports test accuracy at top_k=1 (the dominant choice from
 * the prior data; top_k>1 dilutes votes regardless of bank shape).
 *
 * Config:
 *   n_train      = 60,000 (full MNIST)
 *   n_test       = 10,000 (full MNIST)
 *   sig_dim      = 64
 *   top_k        = 1
 *   max_iters    = 50 (k-means iterations)
 *   R            = single random seed (no training)
 *
 * k=1 sanity check: should equal single-prototype class-mean exactly,
 * since the k-means routine with k_per_class=1 collapses to the same
 * algorithm.
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

#define SIG_DIM         64
#define KMEANS_ITERS    50
#define N_CLASSES       10
#define TOP_K           1

static const char* DEFAULT_MNIST_DIR =
    "/Users/aaronjosserand-austin/Projects/glyph/01MAY26_archived/data/mnist";

static int eval_test_accuracy_pm(
    const m4t_trit_t* R, const gesh_bank_t* bank,
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

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : DEFAULT_MNIST_DIR;
    image_canon_dataset_t ds;

    printf("# MNIST k-means k_per_class sweep — random R, no training\n");
    printf("# sig_dim=%d, top_k=%d, kmeans_iters=%d\n",
           SIG_DIM, TOP_K, KMEANS_ITERS);
    printf("# Loading MNIST from %s\n", dir);
    fflush(stdout);
    if (image_canon_load_mnist(&ds, dir) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    printf("# Loaded train=%d test=%d\n", ds.n_train, ds.n_test);
    image_canon_normalize(&ds);
    int64_t tau = image_canon_quantize_tau(ds.x_train, 1000, ds.input_dim, 0.60);

    m4t_trit_t* train_trits = malloc((size_t)ds.n_train * ds.input_dim
                                       * sizeof(m4t_trit_t));
    m4t_trit_t* test_trits  = malloc((size_t)ds.n_test  * ds.input_dim
                                       * sizeof(m4t_trit_t));
    image_canon_quantize_unpacked_batch(ds.x_train, ds.n_train, ds.input_dim,
                                          tau, train_trits);
    image_canon_quantize_unpacked_batch(ds.x_test,  ds.n_test,  ds.input_dim,
                                          tau, test_trits);

    /* Random R — same seed as prior MNIST runs. */
    m4t_trit_t* R = malloc((size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, ds.input_dim, 0xc0ffeebbu);

    /* Project all training samples once. */
    m4t_trit_t* train_proj = malloc((size_t)ds.n_train * (size_t)SIG_DIM
                                       * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train_trits, ds.n_train,
                                  R, SIG_DIM, ds.input_dim);

    /* Single-prototype baseline. */
    int Dp_sig = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    gesh_bank_t bank_single;
    bank_single.tiles_packed = malloc((size_t)N_CLASSES * (size_t)Dp_sig);
    bank_single.labels = malloc((size_t)N_CLASSES * sizeof(int));
    bank_single.n_tiles = N_CLASSES;
    bank_single.sig_dim = SIG_DIM;
    gesh_bank_build_class_mean(&bank_single, train_proj, ds.y_train,
                                  ds.n_train, N_CLASSES);
    int single_pm = eval_test_accuracy_pm(R, &bank_single, test_trits,
                                             ds.y_test, ds.n_test, SIG_DIM,
                                             ds.input_dim, TOP_K);

    printf("\n## Sweep k_per_class (top_k=%d, full 10K test, random R)\n\n",
           TOP_K);
    printf("| k_per_class | T   | accuracy | runtime | Δ vs single |\n");
    printf("|-------------|-----|----------|---------|-------------|\n");
    printf("| (single)    | %3d | %5.1f%%   |    -    |     -       |\n",
           N_CLASSES, single_pm / 10.0);

    int k_values[] = { 1, 2, 4, 6, 8, 12, 16, 24, 32 };
    int n_k = (int)(sizeof(k_values) / sizeof(k_values[0]));

    for (int i = 0; i < n_k; i++) {
        int k = k_values[i];
        int T = k * N_CLASSES;
        gesh_bank_t bank;
        bank.tiles_packed = malloc((size_t)T * (size_t)Dp_sig);
        bank.labels = malloc((size_t)T * sizeof(int));
        bank.n_tiles = T;
        bank.sig_dim = SIG_DIM;

        clock_t t0 = clock();
        gesh_bank_build_kmeans_per_class(&bank, train_proj, ds.y_train,
                                            ds.n_train, N_CLASSES, k,
                                            KMEANS_ITERS, 0xa5a5a5a5u);
        int pm = eval_test_accuracy_pm(R, &bank, test_trits, ds.y_test,
                                          ds.n_test, SIG_DIM, ds.input_dim, TOP_K);
        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;

        double delta = (pm - single_pm) / 10.0;
        printf("| %11d | %3d | %5.1f%%   | %5.2fs  | %+5.1fpp     |\n",
               k, T, pm / 10.0, dt, delta);
        fflush(stdout);

        free(bank.tiles_packed); free(bank.labels);
    }

    free(train_proj); free(R);
    free(bank_single.tiles_packed); free(bank_single.labels);
    free(train_trits); free(test_trits);
    image_canon_free(&ds);
    return 0;
}
