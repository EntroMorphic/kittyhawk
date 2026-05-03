/* P0-4 Gate 4 (RED-TEAM REMEDIATION C3): MNIST regression check.
 * Originally skipped citing "synth FAIL is decisive." Red-team called
 * that — running it here at subsampled scale to confirm. */

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

#define N_TRAIN_SUB 4000
#define SIG_DIM 64
#define WILDCARD_SNR_PERMILLE 200

static int eval_pm(int* preds_buf, int (*forward)(int*, const m4t_trit_t*, int,
                       const gesh_bank_t*, const gesh_projection_t*, int),
                       const gesh_bank_t* bank, const gesh_projection_t* proj,
                       const m4t_trit_t* test, const int* lbl, int n_test) {
    forward(preds_buf, test, n_test, bank, proj, 1);
    int c = 0; for (int i = 0; i < n_test; i++) if (preds_buf[i] == lbl[i]) c++;
    return (c * 1000) / n_test;
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

    m4t_trit_t* R = malloc((size_t)SIG_DIM * ds.input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, ds.input_dim, 0xc0ffeebbu);
    gesh_projection_t proj = { .R = R, .input_dim = ds.input_dim, .sig_dim = SIG_DIM };

    /* Pre-project subsampled training. */
    m4t_trit_t* train_proj = malloc((size_t)N_TRAIN_SUB * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train_sub, N_TRAIN_SUB, R, SIG_DIM, ds.input_dim);

    int* preds = malloc((size_t)ds.n_test * sizeof(int));

    /* V1 wildcard single-stage. */
    gesh_bank_t b_w = { .tiles_packed = malloc((size_t)C * Dp),
                          .labels = malloc((size_t)C * sizeof(int)),
                          .n_tiles = C, .sig_dim = SIG_DIM };
    gesh_bank_build_class_wildcard(&b_w, train_proj, train_sub_lbl, N_TRAIN_SUB,
                                      C, WILDCARD_SNR_PERMILLE);
    int wc_pm = eval_pm(preds, gesh_forward_classify_wildcard,
                          &b_w, &proj, test, ds.y_test, ds.n_test);

    /* V3 (P0-4 as-shipped). */
    gesh_bank_hier_t hb;
    gesh_bank_hier_alloc(&hb, C, SIG_DIM);
    gesh_bank_build_hierarchical(&hb, train_proj, train_sub_lbl, N_TRAIN_SUB,
                                    C, WILDCARD_SNR_PERMILLE);
    gesh_forward_classify_hierarchical(preds, NULL, test, ds.n_test, &hb, &proj);
    int correct_h = 0;
    for (int i = 0; i < ds.n_test; i++) if (preds[i] == ds.y_test[i]) correct_h++;
    int hier_pm = (correct_h * 1000) / ds.n_test;

    printf("MNIST P0-4 Gate 4 (subsampled n_train=%d, sig_dim=%d, full 10K test)\n",
           N_TRAIN_SUB, SIG_DIM);
    printf("  V1 wildcard single-stage    : %.1f%%\n", wc_pm/10.0);
    printf("  V3 P0-4 hier (as-shipped)   : %.1f%%\n", hier_pm/10.0);
    printf("  V3 vs V1                     : %+.1fpp\n", (hier_pm - wc_pm)/10.0);
    printf("  Gate 4 (no regression beyond -2pp): %s\n",
           ((hier_pm - wc_pm) >= -20) ? "PASS" : "FAIL");

    free(preds);
    free(b_w.tiles_packed); free(b_w.labels);
    gesh_bank_hier_free(&hb);
    free(train_proj); free(R);
    free(train); free(test); free(train_sub); free(train_sub_lbl);
    image_canon_free(&ds);
    return 0;
}
