/* P0-3 Gate 4 (H1 fix): MNIST regression check for geometric training.
 * Uses subsampled n_train to keep per-flip rebuild cost manageable.
 * Single seed; PASS = no regression beyond -2pp vs error-trained baseline. */

#include "image_canon.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_project.h"
#include "gesh_train.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_TRAIN_SUB 2000     /* subsampled to keep per-flip rebuild fast */
#define SIG_DIM 64
#define TOP_K 1
#define N_EPOCHS 30
#define N_FLIPS_PER_EPOCH 200

static int eval_pm(const m4t_trit_t* R, const gesh_bank_t* bank,
                      const m4t_trit_t* test, const int* test_lbl, int n_test,
                      int sig_dim, int input_dim) {
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    gesh_forward_classify(preds, test, n_test, bank, &proj, TOP_K);
    int correct = 0;
    for (int i = 0; i < n_test; i++) if (preds[i] == test_lbl[i]) correct++;
    free(preds);
    return (correct * 1000) / n_test;
}

int main(void) {
    const char* dir = "/Users/aaronjosserand-austin/Projects/glyph/01MAY26_archived/data/mnist";
    image_canon_dataset_t ds;
    if (image_canon_load_mnist(&ds, dir) != 0) return 1;
    image_canon_normalize(&ds);
    int64_t tau = image_canon_quantize_tau(ds.x_train, 1000, ds.input_dim, 0.60);

    m4t_trit_t* train = malloc((size_t)ds.n_train * ds.input_dim * sizeof(m4t_trit_t));
    m4t_trit_t* test  = malloc((size_t)ds.n_test  * ds.input_dim * sizeof(m4t_trit_t));
    image_canon_quantize_unpacked_batch(ds.x_train, ds.n_train, ds.input_dim, tau, train);
    image_canon_quantize_unpacked_batch(ds.x_test,  ds.n_test,  ds.input_dim, tau, test);

    /* Subsample training. Deterministic stride. */
    int stride = ds.n_train / N_TRAIN_SUB;
    m4t_trit_t* train_sub = malloc((size_t)N_TRAIN_SUB * ds.input_dim * sizeof(m4t_trit_t));
    int* train_sub_lbl = malloc((size_t)N_TRAIN_SUB * sizeof(int));
    for (int i = 0; i < N_TRAIN_SUB; i++) {
        int idx = i * stride;
        memcpy(train_sub + (size_t)i * ds.input_dim,
               train + (size_t)idx * ds.input_dim,
               (size_t)ds.input_dim * sizeof(m4t_trit_t));
        train_sub_lbl[i] = ds.y_train[idx];
    }

    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    int C = 10;

    m4t_trit_t* R_random = malloc((size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));
    m4t_trit_t* R_error  = malloc((size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));
    m4t_trit_t* R_geom   = malloc((size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R_random, SIG_DIM, ds.input_dim, 0xc0ffeebbu);
    memcpy(R_error, R_random, (size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));
    memcpy(R_geom,  R_random, (size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));

    gesh_bank_t b_rand, b_err, b_geom;
    b_rand.tiles_packed = malloc((size_t)C * (size_t)Dp);
    b_rand.labels = malloc((size_t)C * sizeof(int));
    b_rand.n_tiles = C; b_rand.sig_dim = SIG_DIM;
    b_err = b_rand; b_err.tiles_packed = malloc((size_t)C * (size_t)Dp);
    b_err.labels = malloc((size_t)C * sizeof(int));
    b_geom = b_rand; b_geom.tiles_packed = malloc((size_t)C * (size_t)Dp);
    b_geom.labels = malloc((size_t)C * sizeof(int));

    /* Random R baseline. */
    clock_t t0 = clock();
    m4t_trit_t* train_proj = malloc((size_t)N_TRAIN_SUB * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train_sub, N_TRAIN_SUB,
                                  R_random, SIG_DIM, ds.input_dim);
    gesh_bank_build_class_mean(&b_rand, train_proj, train_sub_lbl, N_TRAIN_SUB, C);
    int random_pm = eval_pm(R_random, &b_rand, test, ds.y_test, ds.n_test,
                              SIG_DIM, ds.input_dim);
    double random_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Error-trained. */
    t0 = clock();
    gesh_train_config_t cfg_err = gesh_train_default();
    cfg_err.n_epochs = N_EPOCHS;
    cfg_err.n_flip_evals_per_epoch = N_FLIPS_PER_EPOCH;
    cfg_err.batch_size = 128;
    cfg_err.bank_refresh_every = N_FLIPS_PER_EPOCH / 4;
    cfg_err.batch_refresh_every = N_FLIPS_PER_EPOCH / 4;
    cfg_err.early_stop_patience = 5;
    cfg_err.seed = 0xa5a5a5a5u;
    gesh_train_lattice_update(R_error, &b_err, train_sub, train_sub_lbl,
                                 N_TRAIN_SUB, C, SIG_DIM, ds.input_dim,
                                 TOP_K, &cfg_err);
    int error_pm = eval_pm(R_error, &b_err, test, ds.y_test, ds.n_test,
                              SIG_DIM, ds.input_dim);
    double error_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Geometric-trained. */
    t0 = clock();
    gesh_train_config_t cfg_geom = gesh_train_default();
    cfg_geom.n_epochs = N_EPOCHS;
    cfg_geom.n_flip_evals_per_epoch = N_FLIPS_PER_EPOCH;
    cfg_geom.early_stop_patience = 5;
    cfg_geom.seed = 0xb7b7b7b7u;
    gesh_train_lattice_update_geometric(R_geom, &b_geom, train_sub, train_sub_lbl,
                                           N_TRAIN_SUB, C, SIG_DIM, ds.input_dim,
                                           &cfg_geom);
    int geom_pm = eval_pm(R_geom, &b_geom, test, ds.y_test, ds.n_test,
                            SIG_DIM, ds.input_dim);
    double geom_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("MNIST P0-3 Gate 4 (sig_dim=%d, top_k=1, n_train_sub=%d, full 10K test)\n",
           SIG_DIM, N_TRAIN_SUB);
    printf("  random R:       %.1f%% (%.2fs)\n", random_pm/10.0, random_s);
    printf("  error-trained:  %.1f%% (%.2fs)\n", error_pm/10.0, error_s);
    printf("  geometric:      %.1f%% (%.2fs)\n", geom_pm/10.0, geom_s);
    printf("  geom vs error: %+.1fpp\n", (geom_pm - error_pm)/10.0);
    printf("  Gate 4 (no regression beyond -2pp): %s\n",
           ((geom_pm - error_pm) >= -20) ? "PASS" : "FAIL");

    free(train); free(test); free(train_sub); free(train_sub_lbl);
    free(R_random); free(R_error); free(R_geom);
    free(b_rand.tiles_packed); free(b_rand.labels);
    free(b_err.tiles_packed);  free(b_err.labels);
    free(b_geom.tiles_packed); free(b_geom.labels);
    free(train_proj);
    image_canon_free(&ds);
    return 0;
}
