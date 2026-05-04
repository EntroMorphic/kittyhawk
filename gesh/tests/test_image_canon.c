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

/* Integer square root (Newton iteration). Local to this test file because
 * image_canon.c's isqrt64 is static and not exposed.
 *   isqrt64(0) = 0, isqrt64(1) = 1, isqrt64(2) = 1.
 *   For n >= 0, returns floor(sqrt(n)). */
static int64_t test_isqrt64(int64_t n) {
    if (n < 2) return n < 0 ? 0 : n;
    int64_t x = n;
    int64_t y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}

/* V4-residual-2 closure: derive per-image tight bound from the image's
 * pre-normalize standard deviation. Decoupled from any specific synthetic
 * pixel pattern — change the test data, the bound auto-recalibrates.
 *
 * Drift derivation, working through normalize_one (see image_canon.c):
 *
 *   Step (a) centering: img[d] -= sum/dim. Let R = sum after centering;
 *   |R| < dim (R = original_sum mod dim under integer divide).
 *
 *   Step (b) rescaling: img[d] = floor(img[d] * SCALE / sd). For each
 *   element, |floor(c*SCALE/sd) - c*SCALE/sd_real| ≤ 1. Summed across
 *   dim elements, the per-element truncation contributes ≤ dim total.
 *   The centering residual R, scaled, contributes |R| * SCALE/sd_real.
 *
 *   So |sum after normalize| ≤ |R| * SCALE/sd_real + dim
 *                            ≤ dim * SCALE/sd_real + dim
 *                            = dim * (1 + SCALE/sd_real).
 *
 *   Two layers of integer-truncation pessimize the COMPUTED bound vs
 *   the strict math:
 *     (i)  We compute var = sq/dim and sd = isqrt(var). Both truncate,
 *          so computed sd is a LOWER bound on true sd_real.
 *     (ii) Computed (SCALE/sd) is therefore already an UPPER bound on
 *          true (SCALE/sd_real). Adding +1 to the integer divide
 *          handles (SCALE/sd) itself truncating: floor(SCALE/sd) ≤
 *          true SCALE/sd_real ≤ floor(SCALE/sd) + 1.
 *
 *   Final formula: bound_math = dim * (1 + (floor(SCALE/sd_computed) + 1)).
 *   Apply 2x safety factor: bound = 2 * dim * (1 + scale_over_sd_ub),
 *   where scale_over_sd_ub = floor(SCALE/sd_computed) + 1.
 *
 *   For dim=16, sd_pre ≈ SCALE/5: scale_over_sd_ub = 6, bound = 224.
 *   Observed drift on this synthetic ≤ 76; headroom ≈ 2.95x.
 *
 * Edge case: sd == 0 (all pixels identical). normalize_one early-returns
 * before rescaling; centered values are all 0; post-normalize sum = 0.
 * Use 2*dim as a generous floor-bound check. */
static int64_t derive_tight_bound(const m4t_mtfp_t* img, int dim) {
    if (dim <= 0) return 0;
    int64_t sum = 0;
    for (int d = 0; d < dim; d++) sum += (int64_t)img[d];
    int64_t mean = sum / dim;
    int64_t sq = 0;
    for (int d = 0; d < dim; d++) {
        int64_t centered = (int64_t)img[d] - mean;
        sq += centered * centered;
    }
    int64_t var = sq / dim;
    int64_t sd = test_isqrt64(var);
    if (sd == 0) return 2 * (int64_t)dim;
    int64_t scale_over_sd_ub = (int64_t)M4T_MTFP_SCALE / sd + 1;
    return 2 * (int64_t)dim * (1 + scale_over_sd_ub);
}

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
    int dim = ds.input_dim;

    /* V4-residual-2 closure: derive per-image tight bound from each image's
     * pre-normalize sd BEFORE calling normalize (which destroys the input).
     * Decouples the bound from any specific synthetic pixel pattern —
     * change the test data, the bound auto-recalibrates per image. */
    int64_t* tight_bounds = malloc((size_t)ds.n_train * sizeof(int64_t));
    if (!tight_bounds) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (int i = 0; i < ds.n_train; i++) {
        tight_bounds[i] = derive_tight_bound(
            ds.x_train + (size_t)i * dim, dim);
    }

    image_canon_normalize(&ds);

    /* Per-image post-normalize: clipped at ±3 × MTFP_SCALE; mean drift
     * within both the loose (data-independent) and tight (per-image,
     * data-derived) bounds. */
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
        /* LOOSE bound (data-independent backstop): |sum| < dim*SCALE/10.
         * Says "post-normalize mean within 10% of unit scale." For
         * dim=16, bound ≈ 94K. Tight will fail before loose for any
         * realistic data, so this primarily serves as a safety net if
         * derive_tight_bound itself has a future bug (e.g., underestimates
         * sd, producing a too-loose tight bound). Catches catastrophic
         * mean-centering breakage (drift ~ SCALE). */
        int64_t loose_bound = (int64_t)dim * (int64_t)M4T_MTFP_SCALE / 10;
        if (sum < -loose_bound || sum > loose_bound) {
            fprintf(stderr, "LOOSE mean drift %lld (bound +/- %lld)\n",
                    (long long)sum, (long long)loose_bound); exit(1);
        }
        /* TIGHT bound (per-image, data-derived): see derive_tight_bound
         * for the math. Catches 2x drift regressions for ANY test data
         * pattern, not just this synthetic. */
        int64_t tight_bound = tight_bounds[i];
        if (sum < -tight_bound || sum > tight_bound) {
            fprintf(stderr, "TIGHT mean drift img=%d sum=%lld "
                            "(derived bound +/- %lld) — regression: drift "
                            "exceeds analytical worst-case by 2x\n",
                    i, (long long)sum, (long long)tight_bound); exit(1);
        }
    }
    free(tight_bounds);
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
