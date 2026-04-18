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

/* IDX1/IDX3 magic numbers per http://yann.lecun.com/exdb/mnist/.
 * 0x00000803 = 3D unsigned byte (images), 0x00000801 = 1D unsigned byte
 * (labels). Validating these guards against the user pointing --data
 * at the wrong directory or swapping image/label file roles, which
 * would otherwise silently load garbage and produce a confusing low
 * accuracy. */
#define GLYPH_IDX_MAGIC_IMAGES 0x00000803u
#define GLYPH_IDX_MAGIC_LABELS 0x00000801u

static m4t_mtfp_t* load_images_mtfp(const char* path, int* n, int* rows, int* cols) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "glyph_dataset: cannot open %s\n", path);
        return NULL;
    }
    uint32_t magic = read_u32_be(f);
    if (magic != GLYPH_IDX_MAGIC_IMAGES) {
        fprintf(stderr,
            "glyph_dataset: %s has magic 0x%08x, expected 0x%08x "
            "(IDX 3D unsigned byte images)\n",
            path, magic, GLYPH_IDX_MAGIC_IMAGES);
        fclose(f);
        return NULL;
    }
    *n = (int)read_u32_be(f);
    *rows = (int)read_u32_be(f);
    *cols = (int)read_u32_be(f);
    if (*n <= 0 || *rows <= 0 || *cols <= 0) {
        fprintf(stderr,
            "glyph_dataset: %s has invalid shape n=%d rows=%d cols=%d\n",
            path, *n, *rows, *cols);
        fclose(f);
        return NULL;
    }
    int dim = (*rows) * (*cols);
    size_t total = (size_t)(*n) * dim;
    uint8_t* raw = malloc(total);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, total, f) != total) {
        fprintf(stderr, "glyph_dataset: %s short read (expected %zu bytes)\n",
                path, total);
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
    uint32_t magic = read_u32_be(f);
    if (magic != GLYPH_IDX_MAGIC_LABELS) {
        fprintf(stderr,
            "glyph_dataset: %s has magic 0x%08x, expected 0x%08x "
            "(IDX 1D unsigned byte labels)\n",
            path, magic, GLYPH_IDX_MAGIC_LABELS);
        fclose(f);
        return NULL;
    }
    *n = (int)read_u32_be(f);
    if (*n <= 0) {
        fprintf(stderr, "glyph_dataset: %s has invalid count n=%d\n", path, *n);
        fclose(f);
        return NULL;
    }
    uint8_t* raw = malloc((size_t)(*n));
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)(*n), f) != (size_t)(*n)) {
        fprintf(stderr, "glyph_dataset: %s short read (expected %d bytes)\n",
                path, *n);
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

/* ----------------------------------------------------------------
 * CIFAR-10 binary loader
 * ----------------------------------------------------------------
 * Reads raw float32/int32 dumps exported from the Python pipeline.
 * Float→MTFP conversion happens once at load time.
 * ---------------------------------------------------------------- */

static long file_size(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

static m4t_mtfp_t* load_float32_images(const char* path, int dim, int* n_out) {
    long sz = file_size(path);
    if (sz < 0) {
        fprintf(stderr, "glyph_dataset: cannot open %s\n", path);
        return NULL;
    }
    size_t expected_per = (size_t)dim * sizeof(float);
    if (sz == 0 || (size_t)sz % expected_per != 0) {
        fprintf(stderr, "glyph_dataset: %s size %ld not divisible by %zu\n",
                path, sz, expected_per);
        return NULL;
    }
    int n = (int)((size_t)sz / expected_per);
    *n_out = n;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    size_t total = (size_t)n * dim;
    float* raw = malloc(total * sizeof(float));
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, sizeof(float), total, f) != total) {
        fprintf(stderr, "glyph_dataset: %s short read\n", path);
        free(raw); fclose(f); return NULL;
    }
    fclose(f);

    m4t_mtfp_t* data = malloc(total * sizeof(m4t_mtfp_t));
    if (!data) { free(raw); return NULL; }
    for (size_t i = 0; i < total; i++) {
        float v = raw[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        data[i] = (m4t_mtfp_t)(v * (float)M4T_MTFP_SCALE);
    }
    free(raw);
    return data;
}

static int* load_int32_labels(const char* path, int* n_out) {
    long sz = file_size(path);
    if (sz < 0) {
        fprintf(stderr, "glyph_dataset: cannot open %s\n", path);
        return NULL;
    }
    if (sz == 0 || (size_t)sz % sizeof(int32_t) != 0) {
        fprintf(stderr, "glyph_dataset: %s size %ld not divisible by 4\n", path, sz);
        return NULL;
    }
    int n = (int)((size_t)sz / sizeof(int32_t));
    *n_out = n;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t* raw = malloc((size_t)n * sizeof(int32_t));
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, sizeof(int32_t), (size_t)n, f) != (size_t)n) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);

    int* labels = malloc((size_t)n * sizeof(int));
    if (!labels) { free(raw); return NULL; }
    for (int i = 0; i < n; i++) labels[i] = (int)raw[i];
    free(raw);
    return labels;
}

int glyph_dataset_load_cifar10(glyph_dataset_t* ds, const char* dir) {
    memset(ds, 0, sizeof(*ds));
    const int dim = 3072;  /* 32 × 32 × 3 */
    char path[1024];

    snprintf(path, sizeof(path), "%s/train_images.bin", dir);
    ds->x_train = load_float32_images(path, dim, &ds->n_train);
    if (!ds->x_train) return 1;

    snprintf(path, sizeof(path), "%s/train_labels.bin", dir);
    {
        int n_labels = 0;
        ds->y_train = load_int32_labels(path, &n_labels);
        if (!ds->y_train || n_labels != ds->n_train) return 1;
    }

    snprintf(path, sizeof(path), "%s/test_images.bin", dir);
    ds->x_test = load_float32_images(path, dim, &ds->n_test);
    if (!ds->x_test) return 1;

    snprintf(path, sizeof(path), "%s/test_labels.bin", dir);
    {
        int n_labels = 0;
        ds->y_test = load_int32_labels(path, &n_labels);
        if (!ds->y_test || n_labels != ds->n_test) return 1;
    }

    ds->img_h = 32;
    ds->img_w = 32;
    ds->input_dim = dim;
    return 0;
}

int glyph_dataset_load_auto(glyph_dataset_t* ds, const char* dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", dir);
    FILE* f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return glyph_dataset_load_mnist(ds, dir);
    }
    snprintf(path, sizeof(path), "%s/train_images.bin", dir);
    f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return glyph_dataset_load_cifar10(ds, dir);
    }
    fprintf(stderr, "glyph_dataset: no recognized dataset format in %s\n", dir);
    return 1;
}

static int64_t isqrt64(int64_t n) {
    if (n <= 0) return 0;
    int64_t x = n;
    int64_t y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}

static void normalize_images(m4t_mtfp_t* images, int n, int dim) {
    for (int i = 0; i < n; i++) {
        m4t_mtfp_t* img = images + (size_t)i * dim;

        /* Step 1: compute mean. */
        int64_t sum = 0;
        for (int d = 0; d < dim; d++) sum += (int64_t)img[d];
        int64_t mean = sum / dim;

        /* Step 2: subtract mean. */
        for (int d = 0; d < dim; d++) img[d] -= (m4t_mtfp_t)mean;

        /* Step 3: compute variance. */
        int64_t var_sum = 0;
        for (int d = 0; d < dim; d++) {
            int64_t v = (int64_t)img[d];
            var_sum += v * v / dim;
        }

        /* Step 4: integer stddev. */
        int64_t stddev = isqrt64(var_sum);
        if (stddev == 0) continue;

        /* Step 5: rescale to target range.
         * Map ±1σ to ±MTFP_SCALE. Clip at ±3σ to stay in int32. */
        for (int d = 0; d < dim; d++) {
            int64_t scaled = (int64_t)img[d] * M4T_MTFP_SCALE / stddev;
            if (scaled > 3 * (int64_t)M4T_MTFP_SCALE)
                scaled = 3 * (int64_t)M4T_MTFP_SCALE;
            if (scaled < -3 * (int64_t)M4T_MTFP_SCALE)
                scaled = -3 * (int64_t)M4T_MTFP_SCALE;
            img[d] = (m4t_mtfp_t)scaled;
        }
    }
}

void glyph_dataset_normalize(glyph_dataset_t* ds) {
    normalize_images(ds->x_train, ds->n_train, ds->input_dim);
    normalize_images(ds->x_test,  ds->n_test,  ds->input_dim);
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
