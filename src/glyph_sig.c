/*
 * glyph_sig.c — ternary signature builder implementation.
 *
 * Centralizes the "random ternary projection + density-calibrated τ +
 * threshold_extract" pipeline previously duplicated across tools.
 */

#include "glyph_sig.h"
#include "glyph_rng.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"
#include "m4t_route.h"

#include <stdlib.h>
#include <string.h>

static int cmp_i64(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static int64_t tau_for_density(int64_t* v, size_t n, double d) {
    if (n == 0 || d <= 0.0) return 0;
    if (d >= 1.0) return v[n - 1] + 1;
    qsort(v, n, sizeof(int64_t), cmp_i64);
    size_t idx = (size_t)(d * (double)n);
    if (idx >= n) idx = n - 1;
    return v[idx];
}

int glyph_sig_builder_init(
    glyph_sig_builder_t* sb,
    int n_proj,
    int input_dim,
    double density,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const m4t_mtfp_t* calibration_set,
    int n_calib)
{
    memset(sb, 0, sizeof(*sb));
    sb->n_proj = n_proj;
    sb->input_dim = input_dim;
    sb->sig_bytes = M4T_TRIT_PACKED_BYTES(n_proj);
    sb->density = density;
    sb->seed[0] = s0; sb->seed[1] = s1; sb->seed[2] = s2; sb->seed[3] = s3;

    /* Allocations tracked for goto-cleanup on any OOM path. */
    m4t_trit_t* proj_w    = NULL;
    m4t_mtfp_t* calib_proj = NULL;
    int64_t*    buf        = NULL;
    int rc = 1;

    /* Build the random ternary projection matrix. */
    glyph_rng_t rng;
    glyph_rng_seed(&rng, s0, s1, s2, s3);

    proj_w = malloc((size_t)n_proj * input_dim);
    if (!proj_w) goto cleanup;
    for (int i = 0; i < n_proj * input_dim; i++) {
        uint32_t r = glyph_rng_next(&rng) % 3;
        proj_w[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }

    int proj_Dp = M4T_TRIT_PACKED_BYTES(input_dim);
    sb->proj_packed = malloc((size_t)n_proj * proj_Dp);
    if (!sb->proj_packed) goto cleanup;
    m4t_pack_trits_rowmajor(sb->proj_packed, proj_w, n_proj, input_dim);

    /* Calibrate tau from the |projection| distribution of the calibration
     * subset. Uses the same percentile-at-density rule every cascade tool
     * used before this refactor. */
    calib_proj = malloc((size_t)n_calib * n_proj * sizeof(m4t_mtfp_t));
    if (!calib_proj) goto cleanup;
    for (int i = 0; i < n_calib; i++) {
        m4t_mtfp_ternary_matmul_bt(
            calib_proj + (size_t)i * n_proj,
            calibration_set + (size_t)i * input_dim,
            sb->proj_packed, 1, input_dim, n_proj);
    }
    size_t total = (size_t)n_calib * n_proj;
    buf = malloc(total * sizeof(int64_t));
    if (!buf) goto cleanup;
    for (int i = 0; i < n_calib; i++) {
        for (int p = 0; p < n_proj; p++) {
            int64_t v = calib_proj[(size_t)i * n_proj + p];
            buf[(size_t)i * n_proj + p] = (v >= 0) ? v : -v;
        }
    }
    sb->tau_q = tau_for_density(buf, total, density);
    rc = 0;

cleanup:
    free(proj_w);
    free(calib_proj);
    free(buf);
    if (rc != 0) {
        /* Release any partial state owned by sb so the caller sees a
         * clean zero-initialized builder after a failed init. */
        free(sb->proj_packed);
        memset(sb, 0, sizeof(*sb));
    }
    return rc;
}

void glyph_sig_encode(const glyph_sig_builder_t* sb,
                      const m4t_mtfp_t* x,
                      uint8_t* out_sig)
{
    m4t_mtfp_t* proj_row = malloc((size_t)sb->n_proj * sizeof(m4t_mtfp_t));
    int64_t*    tmp      = malloc((size_t)sb->n_proj * sizeof(int64_t));
    if (!proj_row || !tmp) { free(proj_row); free(tmp); return; }
    m4t_mtfp_ternary_matmul_bt(
        proj_row, x, sb->proj_packed, 1, sb->input_dim, sb->n_proj);
    for (int p = 0; p < sb->n_proj; p++) tmp[p] = (int64_t)proj_row[p];
    memset(out_sig, 0, (size_t)sb->sig_bytes);
    m4t_route_threshold_extract(out_sig, tmp, sb->tau_q, sb->n_proj);
    free(proj_row);
    free(tmp);
}

void glyph_sig_encode_batch(const glyph_sig_builder_t* sb,
                            const m4t_mtfp_t* x_batch,
                            int n,
                            uint8_t* out_sigs)
{
    m4t_mtfp_t* proj = malloc((size_t)n * sb->n_proj * sizeof(m4t_mtfp_t));
    int64_t*    tmp  = malloc((size_t)sb->n_proj * sizeof(int64_t));
    if (!proj || !tmp) { free(proj); free(tmp); return; }
    for (int i = 0; i < n; i++) {
        m4t_mtfp_ternary_matmul_bt(
            proj + (size_t)i * sb->n_proj,
            x_batch + (size_t)i * sb->input_dim,
            sb->proj_packed, 1, sb->input_dim, sb->n_proj);
    }
    memset(out_sigs, 0, (size_t)n * sb->sig_bytes);
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < sb->n_proj; p++)
            tmp[p] = (int64_t)proj[(size_t)i * sb->n_proj + p];
        m4t_route_threshold_extract(
            out_sigs + (size_t)i * sb->sig_bytes, tmp, sb->tau_q, sb->n_proj);
    }
    free(proj);
    free(tmp);
}

void glyph_sig_builder_free(glyph_sig_builder_t* sb) {
    if (!sb) return;
    free(sb->proj_packed);
    memset(sb, 0, sizeof(*sb));
}
