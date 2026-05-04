/*
 * test_image_canon.c — smoke test for image_canon.{c,h}.
 *
 * Writes a tiny synthetic IDX file pair to /tmp, loads it, normalizes,
 * computes tau, and quantizes. Asserts basic invariants: sample counts,
 * value-range bounds after normalize, structural-zero rate matches the
 * tau density.
 *
 * Does NOT exercise large-scale numerical correctness against a Python
 * reference; that would belong in a separate validation step.
 */

#include "image_canon.h"
#include "m4t_types.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define IDX_DIR "/tmp/gesh_test_image_canon"
#define N_TRAIN 8
#define N_TEST  4
#define IMG_W   4
#define IMG_H   4

static void write_be_u32(FILE* f, uint32_t v) {
    uint8_t b[4] = {
        (uint8_t)(v >> 24), (uint8_t)(v >> 16),
        (uint8_t)(v >>  8), (uint8_t)v };
    fwrite(b, 1, 4, f);
}

static void write_idx_images(const char* path, int n, int rows, int cols) {
    FILE* f = fopen(path, "wb");
    assert(f);
    write_be_u32(f, 0x00000803u);  /* magic */
    write_be_u32(f, (uint32_t)n);
    write_be_u32(f, (uint32_t)rows);
    write_be_u32(f, (uint32_t)cols);
    /* Pixel pattern: image i, pixel j = (i * 7 + j * 11) & 0xff. Diverse,
     * deterministic, not all-equal (so normalize has work to do). */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < rows * cols; j++) {
            uint8_t v = (uint8_t)((i * 7 + j * 11) & 0xff);
            fwrite(&v, 1, 1, f);
        }
    }
    fclose(f);
}

static void write_idx_labels(const char* path, int n) {
    FILE* f = fopen(path, "wb");
    assert(f);
    write_be_u32(f, 0x00000801u);
    write_be_u32(f, (uint32_t)n);
    for (int i = 0; i < n; i++) {
        uint8_t v = (uint8_t)(i % 10);
        fwrite(&v, 1, 1, f);
    }
    fclose(f);
}

static void make_dataset_files(void) {
    char path[1024];
    /* mkdir -p IDX_DIR */
    mkdir(IDX_DIR, 0777);  /* OK if it already exists */

    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", IDX_DIR);
    write_idx_images(path, N_TRAIN, IMG_H, IMG_W);
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", IDX_DIR);
    write_idx_labels(path, N_TRAIN);
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", IDX_DIR);
    write_idx_images(path, N_TEST, IMG_H, IMG_W);
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", IDX_DIR);
    write_idx_labels(path, N_TEST);
}

static void test_load_basic(void) {
    image_canon_dataset_t ds;
    int rc = image_canon_load_mnist(&ds, IDX_DIR);
    if (rc != 0) { fprintf(stderr, "load failed\n"); exit(1); }
    assert(ds.n_train == N_TRAIN);
    assert(ds.n_test  == N_TEST);
    assert(ds.input_dim == IMG_W * IMG_H);
    assert(ds.img_w == IMG_W);
    assert(ds.img_h == IMG_H);
    /* Pixel range pre-normalize: [0, MTFP_SCALE]. */
    for (int i = 0; i < N_TRAIN * (IMG_W * IMG_H); i++) {
        assert(ds.x_train[i] >= 0);
        assert(ds.x_train[i] <= M4T_MTFP_SCALE);
    }
    image_canon_free(&ds);
    printf("  PASS test_load_basic\n");
}

static void test_normalize_invariants(void) {
    image_canon_dataset_t ds;
    /* IMPORTANT: load_mnist() must NOT be inside assert() — under -DNDEBUG
     * (Release builds), assert(EXPR) becomes ((void)0) and EXPR is never
     * evaluated. The call would be silently elided and ds left uninitialized. */
    int rc = image_canon_load_mnist(&ds, IDX_DIR);
    if (rc != 0) { fprintf(stderr, "load failed\n"); exit(1); }
    image_canon_normalize(&ds);

    /* Per-image post-normalize: clipped at ±3 × MTFP_SCALE. */
    int dim = ds.input_dim;
    for (int i = 0; i < ds.n_train; i++) {
        int64_t sum = 0;
        for (int d = 0; d < dim; d++) {
            int64_t v = ds.x_train[(size_t)i * dim + d];
            if (v < -3 * (int64_t)M4T_MTFP_SCALE ||
                v >  3 * (int64_t)M4T_MTFP_SCALE) {
                fprintf(stderr, "post-normalize out of range\n"); exit(1);
            }
            sum += v;
        }
        /* Two-tier principled bound (V4-G4 tightening of V3-G2):
         *
         *   normalize_one's TWO integer-division steps each introduce drift:
         *   (a) centering: img[d] -= sum/dim. Truncation drift per element
         *       bounded by 1; sum drift after centering: ≤ dim.
         *   (b) rescaling: img[d] = img[d] * SCALE / sd. Integer divide
         *       truncation per element bounded by 1; this amplifies the
         *       centering drift by SCALE/sd per element.
         *
         *   LOOSE bound (catches order-of-magnitude bugs):
         *     |sum| < dim * SCALE / 10. Says "post-normalize mean within
         *     10% of unit scale." For dim=16, bound ≈ 94K. Catches
         *     broken-mean-centering bugs (drift ~ SCALE).
         *
         *   TIGHT bound (catches 2-3x regressions):
         *     Worst case derivation:
         *       - After centering: residual sum ≤ dim.
         *       - After rescaling: per-element rescaling truncation drift
         *         bounded by 1, summed = dim. Plus the residual sum gets
         *         multiplied by SCALE/sd.
         *       - For pixel data with range ~SCALE, sd ≈ SCALE/4 to
         *         SCALE/3, so SCALE/sd ≤ 4.
         *       - |sum after rescale| ≤ dim + dim * 4 = 5 * dim.
         *     Apply 2x safety factor: tight_bound = 10 * dim.
         *     For dim=16, tight_bound = 160. Observed drift on this
         *     synthetic data is ≤ ~80 (V2 pinpoint). Catches any
         *     regression that 2x's the drift.
         *
         * Historical: this assert was ±dim and silently disabled in
         * Release (V2 pinpoint). V3 enabled tests with -UNDEBUG and
         * derived the loose bound. V4 added the tight bound to catch
         * regressions the loose bound would miss. */
        int64_t loose_bound = (int64_t)dim * (int64_t)M4T_MTFP_SCALE / 10;
        if (sum < -loose_bound || sum > loose_bound) {
            fprintf(stderr, "LOOSE mean drift %lld (bound +/- %lld)\n",
                    (long long)sum, (long long)loose_bound); exit(1);
        }
        int64_t tight_bound = 10 * (int64_t)dim;
        if (sum < -tight_bound || sum > tight_bound) {
            fprintf(stderr, "TIGHT mean drift %lld (bound +/- %lld) — "
                            "regression: drift exceeds analytical worst "
                            "case (5*dim) by 2x\n",
                    (long long)sum, (long long)tight_bound); exit(1);
        }
    }
    image_canon_free(&ds);
    printf("  PASS test_normalize_invariants\n");
}

static void test_quantize_density(void) {
    image_canon_dataset_t ds;
    int rc = image_canon_load_mnist(&ds, IDX_DIR);
    if (rc != 0) { fprintf(stderr, "load failed\n"); exit(1); }
    image_canon_normalize(&ds);
    int dim = ds.input_dim;

    int64_t tau60 = image_canon_quantize_tau(ds.x_train, ds.n_train, dim, 0.60);
    assert(tau60 > 0);

    m4t_trit_t* out = malloc((size_t)ds.n_train * dim * sizeof(m4t_trit_t));
    image_canon_quantize_unpacked_batch(ds.x_train, ds.n_train, dim,
                                          tau60, out);

    /* At density=0.60, ~60% of trits should be zero (within ±10pp for
     * small samples). */
    int n_zero = 0;
    int total = ds.n_train * dim;
    for (int i = 0; i < total; i++) {
        assert(out[i] == -1 || out[i] == 0 || out[i] == 1);
        if (out[i] == 0) n_zero++;
    }
    double zero_pct = 100.0 * (double)n_zero / (double)total;
    /* Tolerance is wide because total = 8 × 16 = 128 trits — small sample. */
    assert(zero_pct >= 45.0 && zero_pct <= 75.0);

    free(out);
    image_canon_free(&ds);
    printf("  PASS test_quantize_density (zero rate %.1f%%)\n", zero_pct);
}

static void test_aliasing_assert_disabled_in_release(void) {
    /* Just verifying the function runs with distinct buffers — the
     * assert path is exercised in debug builds, where calling with
     * out_trits == x_batch would abort. We don't trip the assert here;
     * we just confirm the normal path works. */
    image_canon_dataset_t ds;
    int rc = image_canon_load_mnist(&ds, IDX_DIR);
    if (rc != 0) { fprintf(stderr, "load failed\n"); exit(1); }
    image_canon_normalize(&ds);
    int dim = ds.input_dim;
    int64_t tau = image_canon_quantize_tau(ds.x_train, ds.n_train, dim, 0.50);
    m4t_trit_t* out = malloc((size_t)ds.n_train * dim * sizeof(m4t_trit_t));
    image_canon_quantize_unpacked_batch(ds.x_train, ds.n_train, dim,
                                          tau, out);
    free(out);
    image_canon_free(&ds);
    printf("  PASS test_aliasing_assert_disabled_in_release\n");
}

int main(void) {
    printf("test_image_canon: writing test IDX files to %s/\n", IDX_DIR);
    make_dataset_files();
    test_load_basic();
    test_normalize_invariants();
    test_quantize_density();
    test_aliasing_assert_disabled_in_release();
    printf("ALL PASS test_image_canon\n");
    return 0;
}
