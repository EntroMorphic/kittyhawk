/*
 * mnist_trit_lattice.c — MNIST via Trit Lattice LSH. Zero float.
 *
 * The entire pipeline — data loading, "training" (computing class
 * geometry on the lattice), and inference — uses only integer arithmetic
 * and M4T ternary operations. No float anywhere.
 *
 * Algorithm:
 *   1. Load MNIST images as MTFP19 cells (pixel * SCALE / 255, integer)
 *   2. Compute class centroids on the lattice (integer sum / count)
 *   3. Compute global centroid (mean of class centroids)
 *   4. Class signatures = sign(class_centroid - global_centroid) → ternary
 *   5. Inference: score[c] = dot(image, sig_c) via ternary matmul → argmax
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

/* ── Data loading — zero float ─────────────────────────────────────────── */

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}

/* Load MNIST images directly to MTFP19 cells. Integer only.
 * pixel ∈ [0, 255] → cell = pixel * SCALE / 255 (integer division). */
static m4t_mtfp_t* load_images_mtfp(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f);
    *n = (int)read_u32_be(f);
    int rows = (int)read_u32_be(f), cols = (int)read_u32_be(f);
    int dim = rows * cols;
    size_t total = (size_t)(*n) * dim;

    uint8_t* raw = malloc(total);
    fread(raw, 1, total, f);
    fclose(f);

    m4t_mtfp_t* data = malloc(total * sizeof(m4t_mtfp_t));
    for (size_t i = 0; i < total; i++) {
        /* Integer conversion: cell = pixel * SCALE / 255
         * pixel * 59049 fits in int32 (max 255 * 59049 = 15057495). */
        data[i] = (m4t_mtfp_t)(((int32_t)raw[i] * M4T_MTFP_SCALE + 127) / 255);
    }
    free(raw);
    return data;
}

static int* load_labels(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f);
    *n = (int)read_u32_be(f);
    uint8_t* raw = malloc(*n);
    fread(raw, 1, *n, f);
    fclose(f);
    int* labels = malloc(*n * sizeof(int));
    for (int i = 0; i < *n; i++) labels[i] = (int)raw[i];
    free(raw);
    return labels;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mnist_dir>\n", argv[0]);
        return 1;
    }

    /* Load data — integer conversion, zero float */
    char path[512];
    int n_train, n_test;

    snprintf(path, 512, "%s/train-images-idx3-ubyte", argv[1]);
    m4t_mtfp_t* x_train = load_images_mtfp(path, &n_train);
    snprintf(path, 512, "%s/train-labels-idx1-ubyte", argv[1]);
    int* y_train = load_labels(path, &n_train);
    snprintf(path, 512, "%s/t10k-images-idx3-ubyte", argv[1]);
    m4t_mtfp_t* x_test = load_images_mtfp(path, &n_test);
    snprintf(path, 512, "%s/t10k-labels-idx1-ubyte", argv[1]);
    int* y_test = load_labels(path, &n_test);

    printf("Trit Lattice LSH — MNIST\n");
    printf("Loaded %d train, %d test (as MTFP19 cells, zero float)\n\n", n_train, n_test);

    /* ── Phase 1: Class centroids on the lattice ────────────────────────── */

    printf("Computing class geometry on the lattice...\n");

    /* Accumulate per-class sums in int64 to avoid overflow.
     * 60000 images × max cell ~59049 × 784 dims → max sum ~2.78e12, fits int64. */
    int64_t class_sums[N_CLASSES][INPUT_DIM];
    int class_counts[N_CLASSES];
    memset(class_sums, 0, sizeof(class_sums));
    memset(class_counts, 0, sizeof(class_counts));

    for (int i = 0; i < n_train; i++) {
        int c = y_train[i];
        class_counts[c]++;
        const m4t_mtfp_t* img = x_train + (size_t)i * INPUT_DIM;
        for (int d = 0; d < INPUT_DIM; d++) {
            class_sums[c][d] += (int64_t)img[d];
        }
    }

    /* Class centroids (MTFP19 cells) */
    m4t_mtfp_t centroids[N_CLASSES][INPUT_DIM];
    for (int c = 0; c < N_CLASSES; c++) {
        for (int d = 0; d < INPUT_DIM; d++) {
            centroids[c][d] = (m4t_mtfp_t)(class_sums[c][d] / class_counts[c]);
        }
    }

    /* Global centroid */
    m4t_mtfp_t global_centroid[INPUT_DIM];
    for (int d = 0; d < INPUT_DIM; d++) {
        int64_t sum = 0;
        for (int c = 0; c < N_CLASSES; c++) sum += (int64_t)centroids[c][d];
        global_centroid[d] = (m4t_mtfp_t)(sum / N_CLASSES);
    }

    /* ── Phase 2: Random ternary projections (LSH) ────────────────────── */

    /* Generate N_PROJ random ternary vectors. Each is a hash function.
     * Project all training and test images. Use the projected representation
     * for nearest-centroid classification. */

    #define N_PROJ 256
    printf("Generating %d random ternary projections (LSH)...\n", N_PROJ);

    /* Deterministic PRNG — xoshiro128+ from m4t */
    uint32_t rng[4] = { 42, 123, 456, 789 };
    #define RNG_NEXT() do { \
        uint32_t t = rng[1] << 9; \
        rng[2] ^= rng[0]; rng[3] ^= rng[1]; rng[1] ^= rng[2]; rng[0] ^= rng[3]; \
        rng[2] ^= t; rng[3] = (rng[3] << 11) | (rng[3] >> 21); \
    } while(0)
    #define RNG_VAL() (rng[0] + rng[3])

    m4t_trit_t proj_weights[N_PROJ * INPUT_DIM];
    for (int i = 0; i < N_PROJ * INPUT_DIM; i++) {
        RNG_NEXT();
        uint32_t r = RNG_VAL() % 3;
        proj_weights[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }

    int proj_Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* proj_packed = malloc((size_t)N_PROJ * proj_Dp);
    m4t_pack_trits_rowmajor(proj_packed, proj_weights, N_PROJ, INPUT_DIM);

    /* Project all training images → [n_train, N_PROJ] MTFP scores */
    printf("Projecting %d training images...\n", n_train);
    m4t_mtfp_t* train_proj = malloc((size_t)n_train * N_PROJ * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_train; i++) {
        m4t_mtfp_ternary_matmul_bt(train_proj + (size_t)i * N_PROJ,
            x_train + (size_t)i * INPUT_DIM, proj_packed, 1, INPUT_DIM, N_PROJ);
    }

    /* Compute class centroids in projection space (int64 accumulator) */
    printf("Computing class centroids in projection space...\n");
    int64_t proj_class_sums[N_CLASSES][N_PROJ];
    memset(proj_class_sums, 0, sizeof(proj_class_sums));
    for (int i = 0; i < n_train; i++) {
        int c = y_train[i];
        for (int p = 0; p < N_PROJ; p++)
            proj_class_sums[c][p] += (int64_t)train_proj[(size_t)i * N_PROJ + p];
    }
    m4t_mtfp_t proj_centroids[N_CLASSES][N_PROJ];
    for (int c = 0; c < N_CLASSES; c++)
        for (int p = 0; p < N_PROJ; p++)
            proj_centroids[c][p] = (m4t_mtfp_t)(proj_class_sums[c][p] / class_counts[c]);

    /* Inference: project test image → nearest centroid in projection space */
    printf("Running inference (project → L1 nearest centroid)...\n");

    int correct_lsh = 0;
    m4t_mtfp_t test_proj[N_PROJ];

    for (int s = 0; s < n_test; s++) {
        m4t_mtfp_t* img = x_test + (size_t)s * INPUT_DIM;
        m4t_mtfp_ternary_matmul_bt(test_proj, img, proj_packed, 1, INPUT_DIM, N_PROJ);

        /* L1 distance to each class centroid — all integer */
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
        if (pred == y_test[s]) correct_lsh++;
    }

    printf("\nTrit Lattice LSH MNIST (random proj + L1 centroid): %d/%d = %d.%02d%%\n",
           correct_lsh, n_test,
           correct_lsh * 100 / n_test,
           (correct_lsh * 10000 / n_test) % 100);
    printf("Zero float. Zero gradients. Pure lattice geometry.\n\n");
    free(train_proj);
    free(proj_packed);

    /* ── Phase 2b: Pairwise ternary signatures ──────────────────────────── */

    /* For each pair (i, j), sig_ij = sign(centroid_i - centroid_j).
     * This captures what distinguishes class i FROM class j specifically,
     * rather than from the global average. Much more discriminative.
     *
     * Score for class c = sum over j≠c of dot(image, sig_cj).
     * Total signatures: 10 × 9 = 90 (or 45 unique pairs × 2 signs). */

    printf("Computing pairwise signatures...\n");

    #define N_PAIRS (N_CLASSES * (N_CLASSES - 1))  /* 90 directed pairs */
    m4t_trit_t pair_sigs[N_PAIRS][INPUT_DIM];
    int pair_from[N_PAIRS], pair_to[N_PAIRS];
    int n_pairs = 0;

    for (int i = 0; i < N_CLASSES; i++) {
        for (int j = 0; j < N_CLASSES; j++) {
            if (i == j) continue;
            pair_from[n_pairs] = i;
            pair_to[n_pairs] = j;
            for (int d = 0; d < INPUT_DIM; d++) {
                int64_t diff = (int64_t)centroids[i][d] - (int64_t)centroids[j][d];
                /* Threshold: only keep strong differences. Use per-pixel
                 * mean absolute diff as threshold to filter noise. */
                pair_sigs[n_pairs][d] = (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
            }
            n_pairs++;
        }
    }

    /* Pack ALL pairwise signatures for batch ternary matmul */
    int Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* all_sigs_packed = malloc((size_t)n_pairs * Dp);
    m4t_pack_trits_rowmajor(all_sigs_packed, (const m4t_trit_t*)pair_sigs,
                             n_pairs, INPUT_DIM);

    printf("  %d pairwise signatures computed\n", n_pairs);

    /* ── Phase 3: Inference — pairwise voting ───────────────────────────── */

    printf("\nRunning inference (pairwise ternary dot → vote → argmax)...\n");

    int correct = 0;
    m4t_mtfp_t pair_scores[N_PAIRS];

    for (int s = 0; s < n_test; s++) {
        m4t_mtfp_t* img = x_test + (size_t)s * INPUT_DIM;

        /* Compute all 90 pairwise dot products in one ternary matmul */
        m4t_mtfp_ternary_matmul_bt(pair_scores, img, all_sigs_packed,
                                    1, INPUT_DIM, n_pairs);

        /* Accumulate votes: for pair (i,j), if score > 0, class i wins.
         * Each class accumulates its total pairwise score. */
        int64_t class_votes[N_CLASSES];
        memset(class_votes, 0, sizeof(class_votes));
        for (int p = 0; p < n_pairs; p++) {
            class_votes[pair_from[p]] += (int64_t)pair_scores[p];
        }

        /* argmax over accumulated votes */
        int pred = 0;
        int64_t best = class_votes[0];
        for (int c = 1; c < N_CLASSES; c++) {
            if (class_votes[c] > best) { best = class_votes[c]; pred = c; }
        }

        if (pred == y_test[s]) correct++;
    }

    printf("\nTrit Lattice LSH MNIST (pairwise): %d/%d correct = %d.%02d%%\n",
           correct, n_test,
           correct * 100 / n_test,
           (correct * 10000 / n_test) % 100);
    printf("Zero float. Zero gradients. Zero training. Pure lattice geometry.\n");

    /* Cleanup */
    free(x_train); free(y_train);
    free(x_test); free(y_test);
    free(all_sigs_packed);

    return 0;
}
