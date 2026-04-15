/*
 * glyph_dataset.c — MNIST IDX loader + integer-moment deskew.
 *
 * Extracted from repeated use across tools/mnist_*.c. Same arithmetic,
 * same deskew formula, just centralized into libglyph.
 */

#include "glyph_dataset.h"
#include "m4t_mtfp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) |
           ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8) |
           (uint32_t)b[3];
}

static m4t_mtfp_t* load_images_mtfp(const char* path, int* n, int* rows, int* cols) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "glyph_dataset: cannot open %s\n", path);
        return NULL;
    }
    read_u32_be(f);                     /* magic number, unused */
    *n = (int)read_u32_be(f);
    *rows = (int)read_u32_be(f);
    *cols = (int)read_u32_be(f);
    int dim = (*rows) * (*cols);
    size_t total = (size_t)(*n) * dim;
    uint8_t* raw = malloc(total);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, total, f) != total) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    m4t_mtfp_t* data = malloc(total * sizeof(m4t_mtfp_t));
    if (!data) { free(raw); return NULL; }
    for (size_t i = 0; i < total; i++) {
        data[i] = (m4t_mtfp_t)(((int32_t)raw[i] * M4T_MTFP_SCALE + 127) / 255);
    }
    free(raw);
    return data;
}

static int* load_labels(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "glyph_dataset: cannot open %s\n", path);
        return NULL;
    }
    read_u32_be(f);                     /* magic number, unused */
    *n = (int)read_u32_be(f);
    uint8_t* raw = malloc(*n);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, *n, f) != (size_t)*n) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    int* l = malloc((size_t)(*n) * sizeof(int));
    if (!l) { free(raw); return NULL; }
    for (int i = 0; i < *n; i++) l[i] = (int)raw[i];
    free(raw);
    return l;
}

int glyph_dataset_load_mnist(glyph_dataset_t* ds, const char* dir) {
    memset(ds, 0, sizeof(*ds));

    char path[1024];

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
        fprintf(stderr, "glyph_dataset: train/test image shape mismatch\n");
        return 1;
    }
    ds->img_h = tr_rows;
    ds->img_w = tr_cols;
    ds->input_dim = tr_rows * tr_cols;
    return 0;
}

static void deskew_image(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                         int img_w, int img_h, int input_dim)
{
    int64_t sum_p = 0, sum_xp = 0, sum_yp = 0;
    for (int y = 0; y < img_h; y++)
        for (int x = 0; x < img_w; x++) {
            int64_t p = (int64_t)src[y * img_w + x];
            sum_p += p;
            sum_xp += (int64_t)x * p;
            sum_yp += (int64_t)y * p;
        }
    if (sum_p == 0) {
        memcpy(dst, src, (size_t)input_dim * sizeof(m4t_mtfp_t));
        return;
    }
    int64_t Mxy = 0, Myy = 0;
    for (int y = 0; y < img_h; y++) {
        int64_t dy = (int64_t)y * sum_p - sum_yp;
        for (int x = 0; x < img_w; x++) {
            int64_t p = (int64_t)src[y * img_w + x];
            int64_t dx = (int64_t)x * sum_p - sum_xp;
            Mxy += dx * dy / sum_p * p / sum_p;
            Myy += dy * dy / sum_p * p / sum_p;
        }
    }
    memset(dst, 0, (size_t)input_dim * sizeof(m4t_mtfp_t));
    for (int y = 0; y < img_h; y++) {
        int32_t shift = 0;
        if (Myy != 0) {
            int64_t dy = (int64_t)y * sum_p - sum_yp;
            shift = (int32_t)(-(dy * Mxy) / (Myy * sum_p));
        }
        for (int x = 0; x < img_w; x++) {
            int nx = x + shift;
            if (nx >= 0 && nx < img_w) dst[y * img_w + nx] = src[y * img_w + x];
        }
    }
}

static void deskew_all(m4t_mtfp_t* images, int n, int img_w, int img_h, int input_dim) {
    m4t_mtfp_t* buf = malloc((size_t)input_dim * sizeof(m4t_mtfp_t));
    if (!buf) return;
    for (int i = 0; i < n; i++) {
        deskew_image(buf, images + (size_t)i * input_dim, img_w, img_h, input_dim);
        memcpy(images + (size_t)i * input_dim, buf, (size_t)input_dim * sizeof(m4t_mtfp_t));
    }
    free(buf);
}

void glyph_dataset_deskew(glyph_dataset_t* ds) {
    deskew_all(ds->x_train, ds->n_train, ds->img_w, ds->img_h, ds->input_dim);
    deskew_all(ds->x_test,  ds->n_test,  ds->img_w, ds->img_h, ds->input_dim);
}

void glyph_dataset_free(glyph_dataset_t* ds) {
    if (!ds) return;
    free(ds->x_train);
    free(ds->y_train);
    free(ds->x_test);
    free(ds->y_test);
    memset(ds, 0, sizeof(*ds));
}
