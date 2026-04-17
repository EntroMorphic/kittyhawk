/*
 * bruteforce_nproj.c — brute-force routed k-NN at arbitrary N_PROJ.
 *
 * No bucket index, no multi-probe. Computes popcount_dist between
 * the query and ALL training prototypes at the specified N_PROJ.
 * Dense application shape, routing-native kernels. Measures the
 * ceiling of what a given N_PROJ can achieve without filter
 * constraints.
 *
 * Sweeps N_PROJ ∈ {16, 32, 64, 128, 256, 512, 1024} with M tables
 * each, reporting 1-NN and k=5 accuracy at every point.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_rng.h"
#include "glyph_sig.h"
#include "m4t_trit_pack.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10
#define N_NPROJ  7

static const int nproj_values[N_NPROJ] = {16, 32, 64, 128, 256, 512, 1024};

static void derive_seed(uint32_t m, const uint32_t base[4], uint32_t out[4]) {
    if (m == 0) { out[0]=base[0]; out[1]=base[1]; out[2]=base[2]; out[3]=base[3]; return; }
    out[0] = 2654435761u * m + 1013904223u;
    out[1] = 1597334677u * m + 2246822519u;
    out[2] = 3266489917u * m +  668265263u;
    out[3] =  374761393u * m + 3266489917u;
}

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) {
        fprintf(stderr, "failed to load dataset from %s\n", cfg.data_dir);
        return 1;
    }
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    const int M = cfg.m_max;
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    const int KNN_K = 5;

    printf("bruteforce_nproj: brute-force routed k-NN ceiling\n");
    printf("  data_dir=%s  deskew=%s  density=%.2f  M=%d\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on", cfg.density, M);
    printf("  n_train=%d  n_test=%d  input_dim=%d\n\n",
           ds.n_train, ds.n_test, ds.input_dim);

    printf("  N_PROJ  sig_bytes   1-NN      k=5-NN    build(s)  sweep(s)\n");

    for (int ni = 0; ni < N_NPROJ; ni++) {
        int n_proj = nproj_values[ni];
        int sig_bytes = M4T_TRIT_PACKED_BYTES(n_proj);

        /* Build M sig encoders + encode train and test. */
        clock_t t_build = clock();
        glyph_sig_builder_t* builders = calloc((size_t)M, sizeof(glyph_sig_builder_t));
        uint8_t** train_sigs = calloc((size_t)M, sizeof(uint8_t*));
        uint8_t** test_sigs  = calloc((size_t)M, sizeof(uint8_t*));

        for (int m = 0; m < M; m++) {
            uint32_t seeds[4]; derive_seed((uint32_t)m, cfg.base_seed, seeds);
            glyph_sig_builder_init(&builders[m], n_proj, ds.input_dim, cfg.density,
                                    seeds[0], seeds[1], seeds[2], seeds[3],
                                    ds.x_train, n_calib);
            train_sigs[m] = calloc((size_t)ds.n_train * sig_bytes, 1);
            test_sigs[m]  = calloc((size_t)ds.n_test  * sig_bytes, 1);
            glyph_sig_encode_batch(&builders[m], ds.x_train, ds.n_train, train_sigs[m]);
            glyph_sig_encode_batch(&builders[m], ds.x_test,  ds.n_test,  test_sigs[m]);
        }
        double build_sec = (double)(clock() - t_build) / CLOCKS_PER_SEC;

        /* Brute-force sweep: for each test query, compute summed
         * popcount_dist to every training prototype across M tables. */
        uint8_t* mask = malloc(sig_bytes);
        memset(mask, 0xFF, sig_bytes);

        int correct_1nn = 0, correct_knn = 0;
        clock_t t_sweep = clock();

        for (int qi = 0; qi < ds.n_test; qi++) {
            int y = ds.y_test[qi];

            /* Top-K tracking. */
            typedef struct { int32_t score; int label; } topk_t;
            topk_t topk[64];
            int n_topk = 0;
            int32_t best_score = INT32_MAX;
            int     best_label = -1;

            for (int ti = 0; ti < ds.n_train; ti++) {
                int32_t score = 0;
                for (int m = 0; m < M; m++) {
                    score += m4t_popcount_dist(
                        test_sigs[m] + (size_t)qi * sig_bytes,
                        train_sigs[m] + (size_t)ti * sig_bytes,
                        mask, sig_bytes);
                }

                /* 1-NN. */
                if (score < best_score) {
                    best_score = score;
                    best_label = ds.y_train[ti];
                }

                /* Top-K insertion. */
                int lbl = ds.y_train[ti];
                if (n_topk < KNN_K) {
                    int pos = n_topk;
                    while (pos > 0 && topk[pos-1].score > score) {
                        topk[pos] = topk[pos-1]; pos--;
                    }
                    topk[pos].score = score;
                    topk[pos].label = lbl;
                    n_topk++;
                } else if (score < topk[KNN_K-1].score) {
                    int pos = KNN_K - 1;
                    while (pos > 0 && topk[pos-1].score > score) {
                        topk[pos] = topk[pos-1]; pos--;
                    }
                    topk[pos].score = score;
                    topk[pos].label = lbl;
                }
            }

            if (best_label == y) correct_1nn++;

            /* k-NN rank-weighted vote. */
            int cvotes[N_CLASSES] = {0};
            for (int i = 0; i < n_topk; i++)
                cvotes[topk[i].label] += (KNN_K - i);
            int kpred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cvotes[c] > cvotes[kpred]) kpred = c;
            if (kpred == y) correct_knn++;
        }
        double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

        printf("  %4d     %3d     %6.2f%%    %6.2f%%    %6.1f    %6.1f\n",
               n_proj, sig_bytes,
               100.0 * correct_1nn / ds.n_test,
               100.0 * correct_knn / ds.n_test,
               build_sec, sweep_sec);
        fflush(stdout);

        /* Cleanup this N_PROJ. */
        free(mask);
        for (int m = 0; m < M; m++) {
            glyph_sig_builder_free(&builders[m]);
            free(train_sigs[m]); free(test_sigs[m]);
        }
        free(builders); free(train_sigs); free(test_sigs);
    }

    glyph_dataset_free(&ds);
    return 0;
}
