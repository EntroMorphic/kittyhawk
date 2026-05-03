/*
 * mnist_kmeans_trained.c — MNIST single-run probe with TRAINED R and
 * multi-prototype k-means bank refreshing in the training loop.
 *
 * Setup:
 *   n_train     = 60,000 (full)
 *   n_test      = 10,000 (full)
 *   sig_dim     = 64
 *   k_per_class = 8 (T = 80 tiles)
 *   top_k       = 1
 *   n_epochs    = 64
 *   flip budget = 5 × sig_dim × D = 250,880
 *   refresh     = n_flips/4 (kmeans rebuild + batch resample)
 *   early_stop  = OFF
 *
 * Random R baseline at the same bank shape and the trained number print
 * side by side to isolate the training contribution at multi-prototype.
 *
 * Predicted: trained R + k-means bank in the loop should beat the random
 * R + k-means bank baseline (64.1% in the prior k=8 sweep) by 5-10pp.
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
#define BATCH_SIZE      128
#define TOP_K           1
#define N_EPOCHS        64
#define K_PER_CLASS     8
#define KMEANS_ITERS    50
#define N_CLASSES       10

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

static int run_random_kmeans(
    const m4t_trit_t* train, const int* train_lbl, int n_train,
    const m4t_trit_t* test,  const int* test_lbl,  int n_test,
    int sig_dim, int input_dim, int n_classes, int k_per_class,
    int top_k, uint32_t init_seed)
{
    m4t_trit_t* R = malloc((size_t)sig_dim * input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, input_dim, init_seed);

    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int T = k_per_class * n_classes;
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)T * (size_t)Dp);
    bank.labels = malloc((size_t)T * sizeof(int));
    bank.n_tiles = T;
    bank.sig_dim = sig_dim;

    /* Project training samples + build k-means bank. */
    m4t_trit_t* train_proj = malloc((size_t)n_train * (size_t)sig_dim
                                       * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, n_train,
                                  R, sig_dim, input_dim);
    gesh_bank_build_kmeans_per_class(&bank, train_proj, train_lbl,
                                        n_train, n_classes, k_per_class,
                                        KMEANS_ITERS, 0xa5a5a5a5u);
    free(train_proj);

    int pm = eval_test_accuracy_pm(R, &bank, test, test_lbl, n_test,
                                     sig_dim, input_dim, top_k);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static int run_trained_kmeans(
    const m4t_trit_t* train, const int* train_lbl, int n_train,
    const m4t_trit_t* test,  const int* test_lbl,  int n_test,
    int sig_dim, int input_dim, int n_classes, int k_per_class,
    int top_k, int flip_budget, uint32_t init_seed, uint32_t train_seed)
{
    m4t_trit_t* R = malloc((size_t)sig_dim * input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, input_dim, init_seed);

    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int T = k_per_class * n_classes;
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)T * (size_t)Dp);
    bank.labels = malloc((size_t)T * sizeof(int));
    bank.n_tiles = T;
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
    cfg.early_stop_patience = 0;
    cfg.log_per_epoch = 1;
    cfg.seed = train_seed;
    cfg.k_per_class = k_per_class;
    cfg.kmeans_iters = KMEANS_ITERS;

    fprintf(stderr, "Trained k-means config: epochs=%d, flips/epoch=%d, "
                     "batch=%d, k=%d, T=%d, top_k=%d\n",
            cfg.n_epochs, cfg.n_flip_evals_per_epoch, cfg.batch_size,
            cfg.k_per_class, T, top_k);
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
    printf("# MNIST trained-R + k-means bank probe (single run)\n");
    printf("# sig_dim=%d, top_k=%d, batch=%d, n_epochs=%d, "
           "k_per_class=%d, T=%d, early_stop=OFF\n",
           SIG_DIM, TOP_K, BATCH_SIZE, N_EPOCHS, K_PER_CLASS,
           K_PER_CLASS * N_CLASSES);
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

    int flip_budget = 5 * SIG_DIM * ds.input_dim;
    printf("# flip budget = 5 × sig_dim × D = %d\n", flip_budget);
    printf("\n");

    /* Random R + k-means bank baseline. */
    clock_t t0 = clock();
    int rand_pm = run_random_kmeans(train_trits, ds.y_train, ds.n_train,
                                       test_trits,  ds.y_test,  ds.n_test,
                                       SIG_DIM, ds.input_dim, N_CLASSES,
                                       K_PER_CLASS, TOP_K, 0xc0ffeebbu);
    double rand_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Trained R + k-means bank. */
    clock_t t1 = clock();
    int train_pm = run_trained_kmeans(train_trits, ds.y_train, ds.n_train,
                                          test_trits,  ds.y_test,  ds.n_test,
                                          SIG_DIM, ds.input_dim, N_CLASSES,
                                          K_PER_CLASS, TOP_K, flip_budget,
                                          0xc0ffeebbu, 0xa5a5a5a5u);
    double train_s = (double)(clock() - t1) / CLOCKS_PER_SEC;
    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("\n");
    printf("## Results (single seed: init=0xc0ffeebb, train=0xa5a5a5a5)\n\n");
    printf("| variant            | T  | k  | top_k | accuracy | runtime |\n");
    printf("|--------------------|----|----|-------|----------|---------|\n");
    printf("| random R + k-means | %2d | %2d | %3d   | %5.1f%%   | %5.1fs  |\n",
           K_PER_CLASS * N_CLASSES, K_PER_CLASS, TOP_K, rand_pm / 10.0, rand_s);
    printf("| trained R + k-means| %2d | %2d | %3d   | %5.1f%%   | %5.1fs  |\n",
           K_PER_CLASS * N_CLASSES, K_PER_CLASS, TOP_K, train_pm / 10.0, train_s);
    printf("| gain               |    |    |       | %+5.1fpp |          |\n",
           (train_pm - rand_pm) / 10.0);
    printf("\nTotal runtime: %.1fs\n", total_s);

    free(train_trits); free(test_trits);
    image_canon_free(&ds);
    return 0;
}
