/*
 * test_gesh_project.c — bit-equivalence between the kernel-routed
 * projection wrapper (gesh_project.c) and a reference open-coded
 * loop. Establishes that the substrate-discipline cleanup did not
 * change semantics.
 *
 * The reference loop is intentionally identical to the original
 * open-coded path that lived scattered across gesh_forward.c,
 * gesh_train.c, and the bench code. If the kernel path drifts from
 * the open-coded path (or the kernel itself changes contract), this
 * test catches it.
 */

#include "gesh_project.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reference open-coded ternary projection. Identical to what was
 * scattered across the codebase prior to the substrate-discipline
 * cleanup. The single source of truth for what gesh_project must
 * reproduce. */
static void reference_project_batch(
    m4t_trit_t* out_batch,
    const m4t_trit_t* x_batch,
    int n,
    const m4t_trit_t* R,
    int sig_dim, int input_dim)
{
    for (int i = 0; i < n; i++) {
        const m4t_trit_t* x = x_batch + (size_t)i * input_dim;
        m4t_trit_t* s = out_batch + (size_t)i * sig_dim;
        for (int oi = 0; oi < sig_dim; oi++) {
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            int32_t acc = 0;
            for (int j = 0; j < input_dim; j++) {
                acc += (int32_t)r[j] * (int32_t)x[j];
            }
            s[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
        }
    }
}

/* xorshift32 for deterministic random ternary. */
static uint32_t xs32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static void fill_random_ternary(m4t_trit_t* dst, int n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    for (int i = 0; i < n; i++) {
        uint32_t r = xs32(&s) % 3u;
        dst[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }
}

static void test_equiv_batch(int n, int sig_dim, int input_dim, uint32_t seed) {
    m4t_trit_t* x = malloc((size_t)n * (size_t)input_dim * sizeof(m4t_trit_t));
    m4t_trit_t* R = malloc((size_t)sig_dim * (size_t)input_dim * sizeof(m4t_trit_t));
    m4t_trit_t* out_ref = malloc((size_t)n * (size_t)sig_dim * sizeof(m4t_trit_t));
    m4t_trit_t* out_kern = malloc((size_t)n * (size_t)sig_dim * sizeof(m4t_trit_t));

    fill_random_ternary(x, n * input_dim, seed);
    fill_random_ternary(R, sig_dim * input_dim, seed ^ 0x9e3779b9u);

    reference_project_batch(out_ref, x, n, R, sig_dim, input_dim);
    gesh_project_batch_unpacked(out_kern, x, n, R, sig_dim, input_dim);

    int n_diff = 0;
    int total = n * sig_dim;
    for (int i = 0; i < total; i++) {
        if (out_ref[i] != out_kern[i]) n_diff++;
    }
    if (n_diff != 0) {
        fprintf(stderr,
                "FAIL batch n=%d sig_dim=%d input_dim=%d seed=0x%08x: "
                "%d/%d output trits differ between kernel and reference\n",
                n, sig_dim, input_dim, seed, n_diff, total);
        exit(1);
    }
    free(x); free(R); free(out_ref); free(out_kern);
    printf("  PASS batch n=%-4d sig_dim=%-4d input_dim=%-4d seed=0x%08x\n",
           n, sig_dim, input_dim, seed);
}

static void test_threshold_int32(int n, uint32_t seed) {
    /* Build a values[] with mixed signs, exact zeros, and large
     * magnitudes — exercise all three ternary output states. */
    int32_t* values = malloc((size_t)n * sizeof(int32_t));
    m4t_trit_t* out_ref = malloc((size_t)n * sizeof(m4t_trit_t));
    m4t_trit_t* out_kern = malloc((size_t)n * sizeof(m4t_trit_t));

    uint32_t s = seed ? seed : 1u;
    for (int i = 0; i < n; i++) {
        uint32_t r = xs32(&s);
        int32_t v;
        switch (r % 5u) {
        case 0: v = 0; break;
        case 1: v = (int32_t)(r % 1000u) - 500; break;
        case 2: v = -(int32_t)(r & 0xFFFF); break;
        case 3: v =  (int32_t)(r & 0xFFFF); break;
        default: v = (int32_t)(r ^ 0xDEADBEEF) >> 8; break;
        }
        values[i] = v;
        out_ref[i] = (v > 0) ? 1 : (v < 0) ? -1 : 0;
    }
    gesh_threshold_int32_to_trit(out_kern, values, n);

    int n_diff = 0;
    int n_pos = 0, n_neg = 0, n_zero = 0;
    for (int i = 0; i < n; i++) {
        if (out_ref[i] != out_kern[i]) n_diff++;
        if (out_kern[i] == 1)  n_pos++;
        if (out_kern[i] == -1) n_neg++;
        if (out_kern[i] == 0)  n_zero++;
    }
    if (n_diff != 0) {
        fprintf(stderr,
                "FAIL threshold n=%d seed=0x%08x: %d/%d differ\n",
                n, seed, n_diff, n);
        exit(1);
    }
    /* Emission coverage for the threshold call: all three states must
     * appear at least once in this seeded test. */
    if (!(n_pos > 0 && n_neg > 0 && n_zero > 0)) {
        fprintf(stderr, "FAIL threshold emission coverage seed=0x%08x: "
                         "pos=%d neg=%d zero=%d (need all > 0)\n",
                seed, n_pos, n_neg, n_zero);
        exit(1);
    }
    free(values); free(out_ref); free(out_kern);
    printf("  PASS threshold n=%-5d seed=0x%08x (pos=%d neg=%d zero=%d)\n",
           n, seed, n_pos, n_neg, n_zero);
}

int main(void) {
    printf("test_gesh_project: substrate-routed projection equivalence\n");

    /* Sweep diverse shapes; multiple seeds per shape. */
    int shapes[][3] = {
        /* n,  sig_dim, input_dim */
        { 1,    8,        16 },
        { 1,    32,       64 },
        { 4,    64,       64 },
        { 16,   128,      784 },     /* MNIST shape */
        { 64,   256,      784 },     /* MNIST mid-batch */
        { 128,  64,       128 },     /* expansion regime synthetic */
        { 100,  16,       64 },      /* synthetic compression peak */
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    uint32_t seeds[] = { 0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu };
    int n_seeds = sizeof(seeds) / sizeof(seeds[0]);

    for (int s = 0; s < n_shapes; s++) {
        for (int k = 0; k < n_seeds; k++) {
            test_equiv_batch(shapes[s][0], shapes[s][1], shapes[s][2],
                              seeds[k]);
        }
    }

    /* Threshold equivalence + emission coverage. */
    int thresh_sizes[] = { 64, 128, 256, 1024 };
    int n_thresh = sizeof(thresh_sizes) / sizeof(thresh_sizes[0]);
    for (int s = 0; s < n_thresh; s++) {
        for (int k = 0; k < n_seeds; k++) {
            test_threshold_int32(thresh_sizes[s], seeds[k]);
        }
    }

    printf("ALL PASS test_gesh_project\n");
    return 0;
}
