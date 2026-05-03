/* P0-1 Gate 4: MNIST regression check.
 * Wildcard bank vs class_mean bank, both with standard Hamming.
 * One seed; PASS = within ±2pp. */

#include "image_canon.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_project.h"
#include "gesh_train.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>

#define SIG_DIM 64
#define TOP_K   1
#define WILDCARD_SNR_THRESHOLD_PERMILLE 200

static int eval_pm(
    const m4t_trit_t* R, const gesh_bank_t* bank,
    const m4t_trit_t* test, const int* test_lbl, int n_test,
    int sig_dim, int input_dim)
{
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    int rc = gesh_forward_classify(preds, test, n_test, bank, &proj, TOP_K);
    if (rc != 0) { free(preds); return -1; }
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

    m4t_trit_t* R = malloc((size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, ds.input_dim, 0xc0ffeebbu);

    m4t_trit_t* train_proj = malloc((size_t)ds.n_train * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, ds.n_train, R, SIG_DIM, ds.input_dim);

    int Dp_sig = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    gesh_bank_t bm, bw;
    bm.tiles_packed = malloc((size_t)10 * (size_t)Dp_sig);
    bm.labels = malloc((size_t)10 * sizeof(int));
    bm.n_tiles = 10; bm.sig_dim = SIG_DIM;
    gesh_bank_build_class_mean(&bm, train_proj, ds.y_train, ds.n_train, 10);

    bw.tiles_packed = malloc((size_t)10 * (size_t)Dp_sig);
    bw.labels = malloc((size_t)10 * sizeof(int));
    bw.n_tiles = 10; bw.sig_dim = SIG_DIM;
    gesh_bank_build_class_wildcard(&bw, train_proj, ds.y_train, ds.n_train, 10,
                                       WILDCARD_SNR_THRESHOLD_PERMILLE);

    int pm_mean = eval_pm(R, &bm, test, ds.y_test, ds.n_test, SIG_DIM, ds.input_dim);
    int pm_wild = eval_pm(R, &bw, test, ds.y_test, ds.n_test, SIG_DIM, ds.input_dim);

    printf("MNIST Gate 4 (sig_dim=%d, top_k=1, full 60K train, full 10K test):\n", SIG_DIM);
    printf("  class_mean     + Hamming: %.1f%%\n", pm_mean / 10.0);
    printf("  class_wildcard + Hamming: %.1f%%\n", pm_wild / 10.0);
    printf("  delta: %+.1fpp\n", (pm_wild - pm_mean) / 10.0);
    printf("  Gate 4 (within ±2pp): %s\n",
           (abs(pm_wild - pm_mean) <= 20) ? "PASS" : "FAIL");

    free(train); free(test); free(R); free(train_proj);
    free(bm.tiles_packed); free(bm.labels);
    free(bw.tiles_packed); free(bw.labels);
    image_canon_free(&ds);
    return 0;
}
