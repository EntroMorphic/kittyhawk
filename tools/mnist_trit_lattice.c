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

    /* ── Phase 2: Ternary signatures = sign(centroid - global) ──────────── */

    int64_t diffs[INPUT_DIM];
    m4t_trit_t signatures[N_CLASSES][INPUT_DIM];

    for (int c = 0; c < N_CLASSES; c++) {
        for (int d = 0; d < INPUT_DIM; d++) {
            diffs[d] = (int64_t)centroids[c][d] - (int64_t)global_centroid[d];
        }
        /* sign_extract: positive → +1, negative → -1, zero → 0 */
        for (int d = 0; d < INPUT_DIM; d++) {
            signatures[c][d] = (diffs[d] > 0) ? 1 : (diffs[d] < 0) ? -1 : 0;
        }
    }

    /* Pack signatures for ternary matmul */
    int Dp = M4T_TRIT_PACKED_BYTES(INPUT_DIM);
    uint8_t* sigs_packed = malloc((size_t)N_CLASSES * Dp);
    m4t_pack_trits_rowmajor(sigs_packed, (const m4t_trit_t*)signatures,
                             N_CLASSES, INPUT_DIM);

    /* Print signature stats */
    for (int c = 0; c < N_CLASSES; c++) {
        int pos = 0, neg = 0, zer = 0;
        for (int d = 0; d < INPUT_DIM; d++) {
            if (signatures[c][d] == 1) pos++;
            else if (signatures[c][d] == -1) neg++;
            else zer++;
        }
        printf("  class %d: +1=%d, -1=%d, 0=%d\n", c, pos, neg, zer);
    }

    /* ── Phase 3: Inference — ternary matmul + argmax ───────────────────── */

    printf("\nRunning inference (ternary matmul → argmax)...\n");

    int correct = 0;
    m4t_mtfp_t scores[N_CLASSES];

    for (int s = 0; s < n_test; s++) {
        m4t_mtfp_t* img = x_test + (size_t)s * INPUT_DIM;

        /* scores[c] = dot(img, sig_c) via ternary matmul */
        m4t_mtfp_ternary_matmul_bt(scores, img, sigs_packed, 1, INPUT_DIM, N_CLASSES);

        /* argmax */
        int pred = 0;
        m4t_mtfp_t best = scores[0];
        for (int c = 1; c < N_CLASSES; c++) {
            if (scores[c] > best) { best = scores[c]; pred = c; }
        }

        if (pred == y_test[s]) correct++;
    }

    printf("\nTrit Lattice LSH MNIST: %d/%d correct = %d.%02d%%\n",
           correct, n_test,
           correct * 100 / n_test,
           (correct * 10000 / n_test) % 100);
    printf("Zero float. Zero gradients. Zero training. Pure lattice geometry.\n");

    /* Cleanup */
    free(x_train); free(y_train);
    free(x_test); free(y_test);
    free(sigs_packed);

    return 0;
}
