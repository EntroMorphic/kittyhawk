/*
 * mnist_trit_lattice.c — MNIST via Trit Lattice LSH. Zero float.
 *
 * Three approaches tested:
 *   1. Centroid signatures (sign of centroid - global) → ternary matmul
 *   2. Random ternary projections → L1 nearest centroid
 *   3. Multi-trit routes: random projection → MTFP4 route weights →
 *      weighted accumulation of class-specific tile contributions
 *
 * Usage: ./mnist_trit_lattice <mnist_dir>
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_mtfp4.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INPUT_DIM 784
#define N_CLASSES 10

/* ── Data loading — zero float ─────────────────────────────────────────── */

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

/* ── PRNG ──────────────────────────────────────────────────────────────── */

static uint32_t rng_s[4] = { 42, 123, 456, 789 };

static uint32_t rng_next(void) {
    uint32_t result = rng_s[0] + rng_s[3];
    uint32_t t = rng_s[1] << 9;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = (rng_s[3] << 11) | (rng_s[3] >> 21);
    return result;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mnist_dir>\n", argv[0]);
        return 1;
    }

    char path[512]; int n_train, n_test;
    snprintf(path, 512, "%s/train-images-idx3-ubyte", argv[1]);
    m4t_mtfp_t* x_train = load_images_mtfp(path, &n_train);
    snprintf(path, 512, "%s/train-labels-idx1-ubyte", argv[1]);
    int* y_train = load_labels(path, &n_train);
    snprintf(path, 512, "%s/t10k-images-idx3-ubyte", argv[1]);
    m4t_mtfp_t* x_test = load_images_mtfp(path, &n_test);
    snprintf(path, 512, "%s/t10k-labels-idx1-ubyte", argv[1]);
    int* y_test = load_labels(path, &n_test);

    printf("Trit Lattice LSH — MNIST (zero float)\n");
    printf("Loaded %d train, %d test\n\n", n_train, n_test);

    /* ── Compute class counts ──────────────────────────────────────────── */

    int class_counts[N_CLASSES];
    memset(class_counts, 0, sizeof(class_counts));
    for (int i = 0; i < n_train; i++) class_counts[y_train[i]]++;

    /* ── Experiment: Multi-trit routed projection ──────────────────────── */
    /*
     * Architecture:
     *   1. Random ternary projection: [N_PROJ, 784] → project images to N_PROJ dims
     *   2. Per-class, compute multi-trit "route" weights in projection space:
     *      route_c[p] = quantize(class_centroid_proj[p] - global_centroid_proj[p])
     *      to MTFP4 (4-trit, range ±40)
     *   3. Classify: score_c = sum_p(route_c[p] * projected_image[p])
     *      This is a WEIGHTED sum — each projection dimension contributes
     *      proportionally to how discriminative it is for class c.
     *
     * The route weights are multi-trit: they carry magnitude, not just sign.
     * The multiplication route × projection is int8 × int32 → int32.
     * No __int128. No SCALE division. The route is a small integer.
     */

    #define N_PROJ 256

    printf("=== Multi-trit routed projection (N_PROJ=%d) ===\n\n", N_PROJ);

    /* Step 1: Generate random ternary projection matrix */
    printf("Generating %d random ternary projections...\n", N_PROJ);
    m4t_trit_t* proj_weights = malloc((size_t)N_PROJ * INPUT_DIM);
    for (int i = 0; i < N_PROJ * INPUT_DIM; i++) {
        uint32_t r = rng_next() % 3;
        proj_weights[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }
    int proj_Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* proj_packed = malloc((size_t)N_PROJ * proj_Dp);
    m4t_pack_trits_rowmajor(proj_packed, proj_weights, N_PROJ, INPUT_DIM);

    /* Step 2: Project all training images */
    printf("Projecting %d training images...\n", n_train);
    m4t_mtfp_t* train_proj = malloc((size_t)n_train * N_PROJ * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_train; i++) {
        m4t_mtfp_ternary_matmul_bt(train_proj + (size_t)i * N_PROJ,
            x_train + (size_t)i * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);
    }

    /* Step 3: Class centroids in projection space */
    printf("Computing class centroids in projection space...\n");
    int64_t proj_class_sums[N_CLASSES][N_PROJ];
    memset(proj_class_sums, 0, sizeof(proj_class_sums));
    for (int i = 0; i < n_train; i++) {
        int c = y_train[i];
        for (int p = 0; p < N_PROJ; p++)
            proj_class_sums[c][p] += (int64_t)train_proj[(size_t)i * N_PROJ + p];
    }
    int32_t proj_centroids[N_CLASSES][N_PROJ];
    for (int c = 0; c < N_CLASSES; c++)
        for (int p = 0; p < N_PROJ; p++)
            proj_centroids[c][p] = (int32_t)(proj_class_sums[c][p] / class_counts[c]);

    /* Global centroid in projection space */
    int32_t global_proj[N_PROJ];
    for (int p = 0; p < N_PROJ; p++) {
        int64_t sum = 0;
        for (int c = 0; c < N_CLASSES; c++) sum += (int64_t)proj_centroids[c][p];
        global_proj[p] = (int32_t)(sum / N_CLASSES);
    }

    /* Step 4: Multi-trit route weights = MTFP4(class_centroid - global) */
    printf("Computing multi-trit route weights (MTFP4)...\n");

    /* Scale: map the centroid diffs to MTFP4 range (±40).
     * Find max absolute diff to set the scale. */
    int64_t max_abs_diff = 0;
    for (int c = 0; c < N_CLASSES; c++)
        for (int p = 0; p < N_PROJ; p++) {
            int64_t d = (int64_t)proj_centroids[c][p] - (int64_t)global_proj[p];
            if (d < 0) d = -d;
            if (d > max_abs_diff) max_abs_diff = d;
        }
    /* Scale so max diff maps to ±M4T_MTFP4_MAX_VAL (40) */
    int64_t route_scale = (max_abs_diff > 0) ? max_abs_diff / M4T_MTFP4_MAX_VAL : 1;
    if (route_scale < 1) route_scale = 1;

    m4t_mtfp4_t route_weights[N_CLASSES][N_PROJ];
    for (int c = 0; c < N_CLASSES; c++) {
        for (int p = 0; p < N_PROJ; p++) {
            int64_t diff = (int64_t)proj_centroids[c][p] - (int64_t)global_proj[p];
            int32_t q = (int32_t)(diff / route_scale);
            if (q > M4T_MTFP4_MAX_VAL) q = M4T_MTFP4_MAX_VAL;
            if (q < -M4T_MTFP4_MAX_VAL) q = -M4T_MTFP4_MAX_VAL;
            route_weights[c][p] = (m4t_mtfp4_t)q;
        }
    }

    printf("  route_scale = %lld (max_abs_diff = %lld)\n",
           (long long)route_scale, (long long)max_abs_diff);

    /* Step 5: Inference — multi-trit weighted scoring */
    printf("Running inference (multi-trit routes)...\n");

    /* Method A: Single-trit (sign only) — baseline for comparison */
    int correct_1trit = 0;
    /* Method B: Multi-trit (MTFP4 weighted) */
    int correct_mtrit = 0;
    /* Method C: L1 centroid (from previous experiment) */
    int correct_l1 = 0;

    m4t_mtfp_t test_proj[N_PROJ];

    for (int s = 0; s < n_test; s++) {
        m4t_mtfp_t* img = x_test + (size_t)s * INPUT_DIM;
        m4t_mtfp_ternary_matmul_bt(test_proj, img, proj_packed, 1, INPUT_DIM, N_PROJ);

        /* Method A: single-trit scoring (sign of centroid diff × projected image) */
        {
            int64_t scores[N_CLASSES];
            memset(scores, 0, sizeof(scores));
            for (int c = 0; c < N_CLASSES; c++) {
                for (int p = 0; p < N_PROJ; p++) {
                    int32_t diff = proj_centroids[c][p] - global_proj[p];
                    int8_t sign = (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
                    scores[c] += (int64_t)sign * (int64_t)test_proj[p];
                }
            }
            int pred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (scores[c] > scores[pred]) pred = c;
            if (pred == y_test[s]) correct_1trit++;
        }

        /* Method B: multi-trit scoring (MTFP4 route × projected image) */
        {
            int64_t scores[N_CLASSES];
            memset(scores, 0, sizeof(scores));
            for (int c = 0; c < N_CLASSES; c++) {
                for (int p = 0; p < N_PROJ; p++) {
                    scores[c] += (int64_t)route_weights[c][p] * (int64_t)test_proj[p];
                }
            }
            int pred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (scores[c] > scores[pred]) pred = c;
            if (pred == y_test[s]) correct_mtrit++;
        }

        /* Method C: L1 nearest centroid in projection space */
        {
            int pred = 0;
            int64_t best_dist = INT64_MAX;
            for (int c = 0; c < N_CLASSES; c++) {
                int64_t dist = 0;
                for (int p = 0; p < N_PROJ; p++) {
                    int64_t d = (int64_t)test_proj[p] - (int64_t)proj_centroids[c][p];
                    dist += (d >= 0) ? d : -d;
                }
                if (dist < best_dist) { best_dist = dist; pred = c; }
            }
            if (pred == y_test[s]) correct_l1++;
        }
    }

    printf("\nResults (N_PROJ=%d):\n", N_PROJ);
    printf("  Single-trit routes (sign only):   %d/%d = %d.%02d%%\n",
           correct_1trit, n_test,
           correct_1trit * 100 / n_test, (correct_1trit * 10000 / n_test) % 100);
    printf("  Multi-trit routes (MTFP4):        %d/%d = %d.%02d%%\n",
           correct_mtrit, n_test,
           correct_mtrit * 100 / n_test, (correct_mtrit * 10000 / n_test) % 100);
    printf("  L1 nearest centroid:              %d/%d = %d.%02d%%\n",
           correct_l1, n_test,
           correct_l1 * 100 / n_test, (correct_l1 * 10000 / n_test) % 100);
    printf("\nZero float. Zero gradients. Pure lattice geometry.\n");

    free(x_train); free(y_train); free(x_test); free(y_test);
    free(proj_weights); free(proj_packed); free(train_proj);
    return 0;
}
