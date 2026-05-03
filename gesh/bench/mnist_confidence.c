/* P0-2 Gate 4: MNIST regression check for confidence routing. */

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

#define SIG_DIM 64
#define TOP_K 1
#define TAU_STRONG_PERMILLE 600

static int classify_baseline(const m4t_trit_t* test, int n_test,
                                int input_dim, const m4t_trit_t* R,
                                const gesh_bank_t* bank,
                                const int* test_lbl) {
    int* preds = malloc((size_t)n_test * sizeof(int));
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = SIG_DIM };
    gesh_forward_classify(preds, test, n_test, bank, &proj, TOP_K);
    int correct = 0;
    for (int i = 0; i < n_test; i++) if (preds[i] == test_lbl[i]) correct++;
    free(preds);
    return (correct * 1000) / n_test;
}

static int classify_confidence(const m4t_trit_t* test, int n_test,
                                  int input_dim, const m4t_trit_t* R,
                                  const uint8_t* tiles_packed,
                                  const uint8_t* tile_conf,
                                  int T, const int* labels,
                                  const int* test_lbl,
                                  int64_t tau_weak, int64_t tau_strong) {
    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    int conf_bytes = (SIG_DIM + 7) / 8;
    uint8_t mask[256];
    memset(mask, 0xFF, (size_t)Dp);
    int tail = SIG_DIM & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);

    int correct = 0;
    int64_t* row = malloc((size_t)SIG_DIM * sizeof(int64_t));
    uint8_t* qt = malloc((size_t)Dp);
    uint8_t* qc = malloc((size_t)conf_bytes);

    for (int q = 0; q < n_test; q++) {
        const m4t_trit_t* x = test + (size_t)q * input_dim;
        for (int oi = 0; oi < SIG_DIM; oi++) {
            int64_t acc = 0;
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            for (int j = 0; j < input_dim; j++) acc += (int64_t)r[j] * (int64_t)x[j];
            row[oi] = acc;
        }
        m4t_route_threshold_extract_dual(qt, qc, row, tau_weak, tau_strong, SIG_DIM);

        int best_class = 0;
        int32_t best_dist = INT32_MAX;
        for (int t = 0; t < T; t++) {
            int32_t d = m4t_route_confidence_weighted_dist(
                qt, qc,
                tiles_packed + (size_t)t * Dp,
                tile_conf    + (size_t)t * conf_bytes,
                mask, SIG_DIM);
            if (d < best_dist) { best_dist = d; best_class = labels[t]; }
        }
        if (best_class == test_lbl[q]) correct++;
    }
    free(row); free(qt); free(qc);
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

    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    int conf_bytes_per_class = (SIG_DIM + 7) / 8;
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)10 * (size_t)Dp);
    bank.labels = malloc((size_t)10 * sizeof(int));
    bank.n_tiles = 10; bank.sig_dim = SIG_DIM;
    gesh_bank_build_class_mean(&bank, train_proj, ds.y_train, ds.n_train, 10);

    int baseline_pm = classify_baseline(test, ds.n_test, ds.input_dim, R,
                                           &bank, ds.y_test);

    /* Build confidence bank. */
    uint8_t* tile_conf = malloc((size_t)10 * (size_t)conf_bytes_per_class);
    gesh_bank_build_class_mean_with_confidence(
        &bank, tile_conf, train_proj, ds.y_train, ds.n_train, 10,
        TAU_STRONG_PERMILLE);

    /* tau_strong calibrated for MNIST scale: D=784, ternary samples
     * with ~60% zero density mean ~314 nonzero per sample; sums of
     * 314 ±1 terms have stddev ≈ √314 ≈ 18; tau_strong = 30 catches
     * roughly the top half of nonzero accumulators. */
    int conf_pm = classify_confidence(test, ds.n_test, ds.input_dim, R,
                                          bank.tiles_packed, tile_conf, 10,
                                          bank.labels, ds.y_test,
                                          /*tau_weak*/0, /*tau_strong*/30);

    printf("MNIST P0-2 Gate 4 (sig_dim=%d, top_k=1, full 60K/10K):\n", SIG_DIM);
    printf("  baseline (class_mean + Hamming):                  %.1f%%\n", baseline_pm / 10.0);
    printf("  confidence (mean_with_conf + weighted dist):      %.1f%%\n", conf_pm / 10.0);
    printf("  delta: %+.1fpp\n", (conf_pm - baseline_pm) / 10.0);
    printf("  Gate 4 (within ±2pp): %s\n",
           (abs(conf_pm - baseline_pm) <= 20) ? "PASS" : "FAIL");

    free(train); free(test); free(R); free(train_proj); free(tile_conf);
    free(bank.tiles_packed); free(bank.labels);
    image_canon_free(&ds);
    return 0;
}
