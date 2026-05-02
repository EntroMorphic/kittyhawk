/*
 * image_canon.c — see image_canon.h.
 *
 * Lifted from 01MAY26_archived/src/{glyph_dataset.c, glyph_sig.c}; trimmed
 * to MNIST + normalize + ternary-quantize-to-unpacked. Same arithmetic
 * choices as the archived path (validated across that cycle's sweeps).
 */

#include "image_canon.h"
#include "m4t_types.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── IDX loader ────────────────────────────────────────────────────────── */

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

#define IMAGE_CANON_IDX_MAGIC_IMAGES 0x00000803u
#define IMAGE_CANON_IDX_MAGIC_LABELS 0x00000801u

static m4t_mtfp_t* load_images_mtfp(const char* path, int* n,
                                     int* rows, int* cols) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "image_canon: cannot open %s\n", path);
        return NULL;
    }
    uint32_t magic = read_u32_be(f);
    if (magic != IMAGE_CANON_IDX_MAGIC_IMAGES) {
        fprintf(stderr,
            "image_canon: %s magic 0x%08x, expected 0x%08x\n",
            path, magic, IMAGE_CANON_IDX_MAGIC_IMAGES);
        fclose(f);
        return NULL;
    }
    *n = (int)read_u32_be(f);
    *rows = (int)read_u32_be(f);
    *cols = (int)read_u32_be(f);
    if (*n <= 0 || *rows <= 0 || *cols <= 0) {
        fprintf(stderr,
            "image_canon: %s invalid shape n=%d rows=%d cols=%d\n",
            path, *n, *rows, *cols);
        fclose(f);
        return NULL;
    }
    int dim = (*rows) * (*cols);
    size_t total = (size_t)(*n) * (size_t)dim;
    uint8_t* raw = malloc(total);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, total, f) != total) {
        fprintf(stderr, "image_canon: %s short read (expected %zu)\n",
                path, total);
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    m4t_mtfp_t* data = malloc(total * sizeof(m4t_mtfp_t));
    if (!data) { free(raw); return NULL; }
    /* Sanctioned conversion site (one-shot, non-runtime; see
     * M4T_SUBSTRATE.md §12). Maps [0,255] → [0, M4T_MTFP_SCALE] with
     * round-to-nearest. No float intermediate. */
    for (size_t i = 0; i < total; i++) {
        data[i] = (m4t_mtfp_t)(((int32_t)raw[i] * M4T_MTFP_SCALE + 127) / 255);
    }
    free(raw);
    return data;
}

static int* load_labels(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "image_canon: cannot open %s\n", path);
        return NULL;
    }
    uint32_t magic = read_u32_be(f);
    if (magic != IMAGE_CANON_IDX_MAGIC_LABELS) {
        fprintf(stderr,
            "image_canon: %s magic 0x%08x, expected 0x%08x\n",
            path, magic, IMAGE_CANON_IDX_MAGIC_LABELS);
        fclose(f);
        return NULL;
    }
    *n = (int)read_u32_be(f);
    if (*n <= 0) {
        fprintf(stderr, "image_canon: %s invalid count n=%d\n", path, *n);
        fclose(f);
        return NULL;
    }
    uint8_t* raw = malloc((size_t)(*n));
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)(*n), f) != (size_t)(*n)) {
        fprintf(stderr, "image_canon: %s short read\n", path);
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    int* l = malloc((size_t)(*n) * sizeof(int));
    if (!l) { free(raw); return NULL; }
    for (int i = 0; i < *n; i++) l[i] = (int)raw[i];
    free(raw);
    return l;
}

int image_canon_load_mnist(image_canon_dataset_t* ds, const char* dir) {
    memset(ds, 0, sizeof(*ds));
    char path[1024];
    if (strlen(dir) > sizeof(path) - 64) {
        fprintf(stderr, "image_canon: data path too long\n");
        return 1;
    }

    int tr_rows = 0, tr_cols = 0;
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", dir);
    ds->x_train = load_images_mtfp(path, &ds->n_train, &tr_rows, &tr_cols);
    if (!ds->x_train) return 1;

    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", dir);
    {
        int n_labels = 0;
        ds->y_train = load_labels(path, &n_labels);
        if (!ds->y_train || n_labels != ds->n_train) return 1;
    }

    int te_rows = 0, te_cols = 0;
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", dir);
    ds->x_test = load_images_mtfp(path, &ds->n_test, &te_rows, &te_cols);
    if (!ds->x_test) return 1;

    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", dir);
    {
        int n_labels = 0;
        ds->y_test = load_labels(path, &n_labels);
        if (!ds->y_test || n_labels != ds->n_test) return 1;
    }

    if (tr_rows != te_rows || tr_cols != te_cols) {
        fprintf(stderr, "image_canon: train/test shape mismatch\n");
        return 1;
    }
    ds->img_h = tr_rows;
    ds->img_w = tr_cols;
    ds->input_dim = tr_rows * tr_cols;
    return 0;
}

/* ── Normalize (per-image zero-mean unit-variance) ────────────────────── */

static int64_t isqrt64(int64_t n) {
    if (n <= 0) return 0;
    int64_t x = n;
    int64_t y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}

static void normalize_one(m4t_mtfp_t* img, int dim) {
    int64_t sum = 0;
    for (int d = 0; d < dim; d++) sum += (int64_t)img[d];
    int64_t mean = sum / dim;
    for (int d = 0; d < dim; d++) img[d] -= (m4t_mtfp_t)mean;

    int64_t sq = 0;
    for (int d = 0; d < dim; d++) {
        int64_t v = (int64_t)img[d];
        sq += v * v;
    }
    int64_t var = sq / dim;
    int64_t sd = isqrt64(var);
    if (sd == 0) return;

    /* Map ±1σ → ±SCALE; clip at ±3σ to keep int32. */
    for (int d = 0; d < dim; d++) {
        int64_t s = (int64_t)img[d] * M4T_MTFP_SCALE / sd;
        if (s >  3 * (int64_t)M4T_MTFP_SCALE) s =  3 * (int64_t)M4T_MTFP_SCALE;
        if (s < -3 * (int64_t)M4T_MTFP_SCALE) s = -3 * (int64_t)M4T_MTFP_SCALE;
        img[d] = (m4t_mtfp_t)s;
    }
}

void image_canon_normalize(image_canon_dataset_t* ds) {
    for (int i = 0; i < ds->n_train; i++)
        normalize_one(ds->x_train + (size_t)i * ds->input_dim, ds->input_dim);
    for (int i = 0; i < ds->n_test; i++)
        normalize_one(ds->x_test + (size_t)i * ds->input_dim, ds->input_dim);
}

/* ── Direct ternary quantization ──────────────────────────────────────── */

static int cmp_i64(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

int64_t image_canon_quantize_tau(const m4t_mtfp_t* x_sample,
                                  int n_sample, int n_dims,
                                  double density) {
    if (n_sample <= 0 || n_dims <= 0) return 0;
    if (density <= 0.0) return 0;
    size_t total = (size_t)n_sample * (size_t)n_dims;
    int64_t* abs_vals = malloc(total * sizeof(int64_t));
    if (!abs_vals) return 0;
    for (size_t i = 0; i < total; i++) {
        int64_t v = (int64_t)x_sample[i];
        abs_vals[i] = (v >= 0) ? v : -v;
    }
    qsort(abs_vals, total, sizeof(int64_t), cmp_i64);
    size_t idx = (density >= 1.0) ? total - 1
                                  : (size_t)(density * (double)total);
    if (idx >= total) idx = total - 1;
    int64_t tau = abs_vals[idx];
    free(abs_vals);
    return tau;
}

void image_canon_quantize_unpacked_batch(const m4t_mtfp_t* x_batch,
                                          int n, int n_dims,
                                          int64_t tau,
                                          m4t_trit_t* out_trits) {
    assert((const void*)out_trits != (const void*)x_batch);
    for (int i = 0; i < n; i++) {
        const m4t_mtfp_t* x = x_batch + (size_t)i * n_dims;
        m4t_trit_t* o = out_trits + (size_t)i * n_dims;
        for (int d = 0; d < n_dims; d++) {
            int64_t v = (int64_t)x[d];
            o[d] = (v > tau) ? (m4t_trit_t)1
                  : (v < -tau) ? (m4t_trit_t)-1
                  : (m4t_trit_t)0;
        }
    }
}

void image_canon_free(image_canon_dataset_t* ds) {
    if (!ds) return;
    free(ds->x_train);
    free(ds->y_train);
    free(ds->x_test);
    free(ds->y_test);
    memset(ds, 0, sizeof(*ds));
}
