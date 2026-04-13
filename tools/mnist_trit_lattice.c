/*
 * mnist_trit_lattice.c — MNIST via Trit Lattice LSH. Zero float.
 *
 * Two-stage architecture:
 *   Stage 1: Random ternary projection → L1 nearest centroid → top-K candidates
 *   Stage 2: Pairwise ternary refinement among candidates → final prediction
 *
 * Usage: ./mnist_trit_lattice <mnist_dir>
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INPUT_DIM 784
#define N_CLASSES 10
#define N_PAIRS   45  /* C(10, 2) */

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = (int)read_u32_be(f);
    int rows = (int)read_u32_be(f), cols = (int)read_u32_be(f);
    int dim = rows * cols; size_t total = (size_t)(*n) * dim;
    uint8_t* raw = malloc(total); fread(raw, 1, total, f); fclose(f);
    m4t_mtfp_t* data = malloc(total * sizeof(m4t_mtfp_t));
    for (size_t i = 0; i < total; i++)
        data[i] = (m4t_mtfp_t)(((int32_t)raw[i] * M4T_MTFP_SCALE + 127) / 255);
    free(raw); return data;
}
static int* load_labels(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = (int)read_u32_be(f);
    uint8_t* raw = malloc(*n); fread(raw, 1, *n, f); fclose(f);
    int* labels = malloc(*n * sizeof(int));
    for (int i = 0; i < *n; i++) labels[i] = (int)raw[i];
    free(raw); return labels;
}

static uint32_t rng_s[4] = { 42, 123, 456, 789 };
static uint32_t rng_next(void) {
    uint32_t result = rng_s[0] + rng_s[3];
    uint32_t t = rng_s[1] << 9;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = (rng_s[3] << 11) | (rng_s[3] >> 21);
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <mnist_dir>\n", argv[0]); return 1; }

    char path[512]; int n_train, n_test;
    snprintf(path, 512, "%s/train-images-idx3-ubyte", argv[1]);
    m4t_mtfp_t* x_train = load_images_mtfp(path, &n_train);
    snprintf(path, 512, "%s/train-labels-idx1-ubyte", argv[1]);
    int* y_train = load_labels(path, &n_train);
    snprintf(path, 512, "%s/t10k-images-idx3-ubyte", argv[1]);
    m4t_mtfp_t* x_test = load_images_mtfp(path, &n_test);
    snprintf(path, 512, "%s/t10k-labels-idx1-ubyte", argv[1]);
    int* y_test = load_labels(path, &n_test);

    printf("Trit Lattice LSH — Two-Stage MNIST (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    int class_counts[N_CLASSES];
    memset(class_counts, 0, sizeof(class_counts));
    for (int i = 0; i < n_train; i++) class_counts[y_train[i]]++;

    /* ── Random ternary projection ─────────────────────────────────────── */

    int n_proj_vals[] = { 256, 512, 1024 };
    int n_proj_count = 3;

    for (int pi = 0; pi < n_proj_count; pi++) {
        int N_PROJ = n_proj_vals[pi];

        /* Reset PRNG for reproducibility across projection counts */
        rng_s[0] = 42; rng_s[1] = 123; rng_s[2] = 456; rng_s[3] = 789;

        printf("=== N_PROJ = %d ===\n", N_PROJ);

        m4t_trit_t* proj_w = malloc((size_t)N_PROJ * INPUT_DIM);
        for (int i = 0; i < N_PROJ * INPUT_DIM; i++) {
            uint32_t r = rng_next() % 3;
            proj_w[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
        }
        int proj_Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
        uint8_t* proj_packed = malloc((size_t)N_PROJ * proj_Dp);
        m4t_pack_trits_rowmajor(proj_packed, proj_w, N_PROJ, INPUT_DIM);

        /* Project training images */
        m4t_mtfp_t* train_proj = malloc((size_t)n_train * N_PROJ * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_train; i++)
            m4t_mtfp_ternary_matmul_bt(train_proj + (size_t)i * N_PROJ,
                x_train + (size_t)i * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);

        /* Class centroids in projection space */
        int64_t pcs[N_CLASSES][1024]; /* max N_PROJ */
        memset(pcs, 0, sizeof(int64_t) * N_CLASSES * N_PROJ);
        for (int i = 0; i < n_train; i++) {
            int c = y_train[i];
            for (int p = 0; p < N_PROJ; p++)
                pcs[c][p] += (int64_t)train_proj[(size_t)i * N_PROJ + p];
        }
        int32_t centroids[N_CLASSES][1024];
        for (int c = 0; c < N_CLASSES; c++)
            for (int p = 0; p < N_PROJ; p++)
                centroids[c][p] = (int32_t)(pcs[c][p] / class_counts[c]);

        /* Pairwise ternary signatures in projection space */
        m4t_trit_t pair_sigs[N_PAIRS][1024];
        int pair_i[N_PAIRS], pair_j[N_PAIRS];
        int np = 0;
        for (int i = 0; i < N_CLASSES; i++)
            for (int j = i + 1; j < N_CLASSES; j++) {
                pair_i[np] = i; pair_j[np] = j;
                for (int p = 0; p < N_PROJ; p++) {
                    int32_t d = centroids[i][p] - centroids[j][p];
                    pair_sigs[np][p] = (d > 0) ? 1 : (d < 0) ? -1 : 0;
                }
                np++;
            }

        /* Pack pairwise signatures */
        int pair_Dp = M4T_TRIT_PACKED_BYTES(N_PROJ);
        uint8_t* pair_packed = malloc((size_t)N_PAIRS * pair_Dp);
        m4t_pack_trits_rowmajor(pair_packed, (const m4t_trit_t*)pair_sigs,
                                 N_PAIRS, N_PROJ);

        /* ── Inference ─────────────────────────────────────────────────── */

        int correct_l1 = 0;       /* L1 only */
        int correct_2stage_3 = 0; /* L1 top-3 → pairwise refine */
        int correct_2stage_5 = 0; /* L1 top-5 → pairwise refine */
        m4t_mtfp_t test_proj[1024];
        m4t_mtfp_t pair_scores[N_PAIRS];

        for (int s = 0; s < n_test; s++) {
            m4t_mtfp_ternary_matmul_bt(test_proj,
                x_test + (size_t)s * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);

            /* Stage 1: L1 distance to all centroids */
            int64_t dists[N_CLASSES];
            for (int c = 0; c < N_CLASSES; c++) {
                int64_t d = 0;
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t x = (int64_t)test_proj[p] - (int64_t)centroids[c][p];
                    d += (x >= 0) ? x : -x;
                }
                dists[c] = d;
            }

            /* Rank by distance (insertion sort, N=10 is tiny) */
            int ranked[N_CLASSES];
            for (int c = 0; c < N_CLASSES; c++) ranked[c] = c;
            for (int i = 1; i < N_CLASSES; i++) {
                int key = ranked[i]; int64_t kd = dists[key];
                int j = i - 1;
                while (j >= 0 && dists[ranked[j]] > kd) { ranked[j+1] = ranked[j]; j--; }
                ranked[j+1] = key;
            }

            /* L1-only prediction */
            if (ranked[0] == y_test[s]) correct_l1++;

            /* Stage 2: pairwise refinement among top-K candidates.
             * Compute all 45 pairwise dot products in one ternary matmul. */
            m4t_mtfp_ternary_matmul_bt(pair_scores, test_proj, pair_packed,
                                        1, N_PROJ, N_PAIRS);

            /* Top-3 refinement */
            {
                int K = 3;
                int64_t votes[N_CLASSES];
                memset(votes, 0, sizeof(votes));
                for (int pi2 = 0; pi2 < N_PAIRS; pi2++) {
                    int ci = pair_i[pi2], cj = pair_j[pi2];
                    /* Only consider pairs where BOTH classes are in top-K */
                    int ci_in = 0, cj_in = 0;
                    for (int k = 0; k < K; k++) {
                        if (ranked[k] == ci) ci_in = 1;
                        if (ranked[k] == cj) cj_in = 1;
                    }
                    if (!ci_in || !cj_in) continue;
                    /* Positive score → class i wins, negative → class j wins */
                    if (pair_scores[pi2] > 0) votes[ci] += (int64_t)pair_scores[pi2];
                    else votes[cj] -= (int64_t)pair_scores[pi2];
                }
                int pred = ranked[0];
                int64_t best = votes[ranked[0]];
                for (int k = 1; k < K; k++)
                    if (votes[ranked[k]] > best) { best = votes[ranked[k]]; pred = ranked[k]; }
                if (pred == y_test[s]) correct_2stage_3++;
            }

            /* Top-5 refinement */
            {
                int K = 5;
                int64_t votes[N_CLASSES];
                memset(votes, 0, sizeof(votes));
                for (int pi2 = 0; pi2 < N_PAIRS; pi2++) {
                    int ci = pair_i[pi2], cj = pair_j[pi2];
                    int ci_in = 0, cj_in = 0;
                    for (int k = 0; k < K; k++) {
                        if (ranked[k] == ci) ci_in = 1;
                        if (ranked[k] == cj) cj_in = 1;
                    }
                    if (!ci_in || !cj_in) continue;
                    if (pair_scores[pi2] > 0) votes[ci] += (int64_t)pair_scores[pi2];
                    else votes[cj] -= (int64_t)pair_scores[pi2];
                }
                int pred = ranked[0];
                int64_t best = votes[ranked[0]];
                for (int k = 1; k < K; k++)
                    if (votes[ranked[k]] > best) { best = votes[ranked[k]]; pred = ranked[k]; }
                if (pred == y_test[s]) correct_2stage_5++;
            }
        }

        printf("  L1 centroid only:              %d/%d = %d.%02d%%\n",
               correct_l1, n_test, correct_l1*100/n_test, (correct_l1*10000/n_test)%100);
        printf("  L1 top-3 → pairwise refine:    %d/%d = %d.%02d%%\n",
               correct_2stage_3, n_test, correct_2stage_3*100/n_test, (correct_2stage_3*10000/n_test)%100);
        printf("  L1 top-5 → pairwise refine:    %d/%d = %d.%02d%%\n\n",
               correct_2stage_5, n_test, correct_2stage_5*100/n_test, (correct_2stage_5*10000/n_test)%100);

        free(proj_w); free(proj_packed); free(train_proj); free(pair_packed);
    }

    printf("Zero float. Zero gradients. Pure lattice geometry.\n");
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
