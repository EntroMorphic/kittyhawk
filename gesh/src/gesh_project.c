/*
 * gesh_project.c — implementation of gesh_project.h.
 *
 * One-shot wrappers that route ternary projection and sign-threshold
 * through libm4t kernels, eliminating the open-coded multiply-accumulate
 * and threshold-quantize sites that the Phase B kernel-use audit found.
 *
 * Allocation policy: all scratch is malloc'd per call. Callers in hot
 * loops (gesh_train) should NOT call these directly per flip-eval; they
 * own their own pre-packed R + persistent scratch via the same kernel
 * pipeline (see gesh_train.c). This file is the canonical wrapper for
 * one-off projection (gesh_forward), bank construction (gesh_bank), and
 * bench code that doesn't have hot-loop pressure.
 */

#include "gesh_project.h"
#include "m4t_types.h"
#include "m4t_trit_pack.h"
#include "m4t_route.h"
#include "m4t_ternary_matmul.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void gesh_project_batch_unpacked(
    m4t_trit_t* out_batch,
    const m4t_trit_t* x_batch,
    int n,
    const m4t_trit_t* R,
    int sig_dim,
    int input_dim)
{
    assert(out_batch && x_batch && R);
    assert(n > 0 && sig_dim > 0 && input_dim > 0);
    assert((const void*)out_batch != (const void*)x_batch);
    assert((const void*)out_batch != (const void*)R);

    int Rp_per_row = M4T_TRIT_PACKED_BYTES(input_dim);
    int Op_per_row = M4T_TRIT_PACKED_BYTES(sig_dim);

    /* Pack R rows once (amortized over n queries). */
    uint8_t* R_packed = malloc((size_t)sig_dim * (size_t)Rp_per_row);
    for (int oi = 0; oi < sig_dim; oi++) {
        m4t_pack_trits_1d(R_packed + (size_t)oi * Rp_per_row,
                          R + (size_t)oi * input_dim,
                          input_dim);
    }

    /* Widen x batch ternary → MTFP19. The matmul kernel expects MTFP19
     * activations; ternary trits widen by sign-extending int8 to int32. */
    m4t_mtfp_t* X_mtfp = malloc((size_t)n * (size_t)input_dim
                                  * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) {
        const m4t_trit_t* xi = x_batch + (size_t)i * input_dim;
        m4t_mtfp_t* Xi = X_mtfp + (size_t)i * input_dim;
        for (int j = 0; j < input_dim; j++) Xi[j] = (m4t_mtfp_t)xi[j];
    }

    /* Y[n, sig_dim] = X[n, input_dim] @ R^T[input_dim, sig_dim]. */
    m4t_mtfp_t* Y_mtfp = malloc((size_t)n * (size_t)sig_dim
                                  * sizeof(m4t_mtfp_t));
    m4t_mtfp_ternary_matmul_bt(Y_mtfp, X_mtfp, R_packed, NULL,
                                 n, input_dim, sig_dim);

    /* Per-row threshold-extract through the kernel; then unpack to
     * caller's m4t_trit_t format. Per-row int64 widen amortizes only
     * O(sig_dim) per row. */
    int64_t* row64 = malloc((size_t)sig_dim * sizeof(int64_t));
    uint8_t* row_packed = malloc((size_t)Op_per_row);
    for (int i = 0; i < n; i++) {
        const m4t_mtfp_t* Yi = Y_mtfp + (size_t)i * sig_dim;
        for (int oi = 0; oi < sig_dim; oi++) row64[oi] = (int64_t)Yi[oi];
        m4t_route_threshold_extract(row_packed, row64, /*tau=*/0, sig_dim);
        m4t_unpack_trits_1d(out_batch + (size_t)i * sig_dim,
                              row_packed, sig_dim);
    }

    free(row_packed);
    free(row64);
    free(Y_mtfp);
    free(X_mtfp);
    free(R_packed);
}

void gesh_project_one_packed(
    uint8_t* out_packed,
    const m4t_trit_t* x,
    const m4t_trit_t* R,
    int sig_dim,
    int input_dim)
{
    assert(out_packed && x && R);
    assert(sig_dim > 0 && input_dim > 0);
    assert((const void*)out_packed != (const void*)x);
    assert((const void*)out_packed != (const void*)R);

    int Rp_per_row = M4T_TRIT_PACKED_BYTES(input_dim);

    /* Pack R rows. */
    uint8_t* R_packed = malloc((size_t)sig_dim * (size_t)Rp_per_row);
    for (int oi = 0; oi < sig_dim; oi++) {
        m4t_pack_trits_1d(R_packed + (size_t)oi * Rp_per_row,
                          R + (size_t)oi * input_dim,
                          input_dim);
    }

    /* Widen single x. */
    m4t_mtfp_t* X_mtfp = malloc((size_t)input_dim * sizeof(m4t_mtfp_t));
    for (int j = 0; j < input_dim; j++) X_mtfp[j] = (m4t_mtfp_t)x[j];

    /* Matmul: 1 × input_dim × sig_dim. */
    m4t_mtfp_t* Y_mtfp = malloc((size_t)sig_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_ternary_matmul_bt(Y_mtfp, X_mtfp, R_packed, NULL,
                                 1, input_dim, sig_dim);

    /* Widen → threshold-extract → out_packed. */
    int64_t* Y64 = malloc((size_t)sig_dim * sizeof(int64_t));
    for (int oi = 0; oi < sig_dim; oi++) Y64[oi] = (int64_t)Y_mtfp[oi];
    m4t_route_threshold_extract(out_packed, Y64, /*tau=*/0, sig_dim);

    free(Y64);
    free(Y_mtfp);
    free(X_mtfp);
    free(R_packed);
}

void gesh_project_scratch_init(
    gesh_project_scratch_t* sc,
    int sig_dim, int input_dim, int n_max)
{
    assert(sc && sig_dim > 0 && input_dim > 0 && n_max > 0);
    int Rp_per_row = M4T_TRIT_PACKED_BYTES(input_dim);
    int Op_per_row = M4T_TRIT_PACKED_BYTES(sig_dim);
    sc->sig_dim   = sig_dim;
    sc->input_dim = input_dim;
    sc->n_max     = n_max;
    sc->R_packed  = malloc((size_t)sig_dim * (size_t)Rp_per_row);
    sc->X_mtfp    = malloc((size_t)n_max * (size_t)input_dim
                              * sizeof(m4t_mtfp_t));
    sc->Y_mtfp    = malloc((size_t)n_max * (size_t)sig_dim
                              * sizeof(m4t_mtfp_t));
    sc->row64     = malloc((size_t)sig_dim * sizeof(int64_t));
    sc->row_packed = malloc((size_t)Op_per_row);
}

void gesh_project_scratch_free(gesh_project_scratch_t* sc) {
    if (!sc) return;
    free(sc->R_packed); free(sc->X_mtfp); free(sc->Y_mtfp);
    free(sc->row64); free(sc->row_packed);
    sc->R_packed = NULL;
    sc->X_mtfp = NULL;
    sc->Y_mtfp = NULL;
    sc->row64 = NULL;
    sc->row_packed = NULL;
}

void gesh_project_batch_unpacked_scratch(
    m4t_trit_t* out_batch,
    const m4t_trit_t* x_batch,
    int n,
    const m4t_trit_t* R,
    int sig_dim, int input_dim,
    gesh_project_scratch_t* sc)
{
    assert(sc && out_batch && x_batch && R);
    assert(n > 0 && sig_dim > 0 && input_dim > 0);
    assert(n <= sc->n_max && sig_dim <= sc->sig_dim
            && input_dim <= sc->input_dim);
    assert((const void*)out_batch != (const void*)x_batch);
    assert((const void*)out_batch != (const void*)R);

    int Rp_per_row = M4T_TRIT_PACKED_BYTES(input_dim);

    /* Pack R into scratch.R_packed (fresh each call; R may have changed). */
    for (int oi = 0; oi < sig_dim; oi++) {
        m4t_pack_trits_1d(sc->R_packed + (size_t)oi * Rp_per_row,
                            R + (size_t)oi * input_dim, input_dim);
    }

    /* Widen x batch ternary → MTFP19 in scratch.X_mtfp. */
    for (int i = 0; i < n; i++) {
        const m4t_trit_t* xi = x_batch + (size_t)i * input_dim;
        m4t_mtfp_t* Xi = sc->X_mtfp + (size_t)i * input_dim;
        for (int j = 0; j < input_dim; j++) Xi[j] = (m4t_mtfp_t)xi[j];
    }

    /* Matmul into scratch.Y_mtfp. */
    m4t_mtfp_ternary_matmul_bt(sc->Y_mtfp, sc->X_mtfp, sc->R_packed, NULL,
                                  n, input_dim, sig_dim);

    /* Per-row threshold-extract via kernel + unpack. */
    for (int i = 0; i < n; i++) {
        const m4t_mtfp_t* Yi = sc->Y_mtfp + (size_t)i * sig_dim;
        for (int oi = 0; oi < sig_dim; oi++) sc->row64[oi] = (int64_t)Yi[oi];
        m4t_route_threshold_extract(sc->row_packed, sc->row64, /*tau=*/0,
                                       sig_dim);
        m4t_unpack_trits_1d(out_batch + (size_t)i * sig_dim,
                              sc->row_packed, sig_dim);
    }
}

void gesh_threshold_int32_to_trit(
    m4t_trit_t* out,
    const int32_t* values,
    int n)
{
    assert(out && values);
    assert(n > 0);
    assert((const void*)out != (const void*)values);

    int64_t* v64 = malloc((size_t)n * sizeof(int64_t));
    for (int i = 0; i < n; i++) v64[i] = (int64_t)values[i];

    int Dp = M4T_TRIT_PACKED_BYTES(n);
    uint8_t* packed = malloc((size_t)Dp);
    m4t_route_threshold_extract(packed, v64, /*tau=*/0, n);
    m4t_unpack_trits_1d(out, packed, n);

    free(packed);
    free(v64);
}
