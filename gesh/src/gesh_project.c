/*
 * gesh_project.c — implementation of gesh_project.h.
 *
 * Substrate-routed ternary projection. The matmul kernel of choice is
 * `m4t_mtfp4_sdot_matmul_bt` (NOT `m4t_mtfp_ternary_matmul_bt`):
 *
 *   - Both our activations (x) and weights (R) are unpacked ternary
 *     `m4t_trit_t` (= int8_t), values in {-1, 0, +1}.
 *   - `m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N)` accepts X as
 *     `m4t_mtfp4_t* (= int8_t*)` and W as `m4t_trit_t* (= int8_t*)`.
 *     Internally it uses `vdotq_s32` (1-cycle 16-lane SDOT) to compute
 *     signed int8 × int8 → int32 accumulate. No decode, no packing,
 *     no widening. This is the hardware-native MAC for our input class.
 *   - `m4t_mtfp_ternary_matmul_bt` decodes packed-trit weights via
 *     ~30 NEON ops per 16 trits — appropriate when weights are
 *     storage-optimized in 2-bit-packed format and activations are
 *     full-precision MTFP19. Wrong tool for ternary × ternary.
 *
 * Substrate-discipline: ternary trits are a strict subset of the MTFP4
 * mantissa range (|m| ≤ 40), so passing ternary `m4t_trit_t*` to the
 * MTFP4 SDOT kernel via cast is within the kernel's declared input
 * class. K is bounded by M4T_SDOT_K_MAX_EXACT = 14.5M — well above
 * any sig_dim or input_dim we use.
 *
 * Threshold step still uses `m4t_route_threshold_extract`. The SDOT
 * output is `m4t_mtfp_t* (int32)`; we widen to int64 per row before
 * threshold-extract (kernel takes int64).
 */

#include "gesh_project.h"
#include "m4t_types.h"
#include "m4t_trit_pack.h"
#include "m4t_route.h"
#include "m4t_mtfp4.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Thin alias: ternary × ternary → MTFP19 via the SDOT path. */
static inline void ternary_matmul_sdot(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_trit_t* W,
    int M, int K, int N)
{
    m4t_mtfp4_sdot_matmul_bt(Y, (const m4t_mtfp4_t*)X, W, M, K, N);
}

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

    int Op_per_row = M4T_TRIT_PACKED_BYTES(sig_dim);

    /* Y[n, sig_dim] = X[n, input_dim] @ R^T via SDOT. No packing of R,
     * no widening of X — both are already int8 ternary. */
    m4t_mtfp_t* Y_mtfp = malloc((size_t)n * (size_t)sig_dim
                                  * sizeof(m4t_mtfp_t));
    ternary_matmul_sdot(Y_mtfp, x_batch, R, n, input_dim, sig_dim);

    /* Per-row threshold-extract through the kernel; then unpack to
     * caller's m4t_trit_t format. Per-row int64 widen since the
     * threshold-extract kernel takes int64. */
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

    /* Single-query SDOT: M=1. */
    m4t_mtfp_t* Y_mtfp = malloc((size_t)sig_dim * sizeof(m4t_mtfp_t));
    ternary_matmul_sdot(Y_mtfp, x, R, 1, input_dim, sig_dim);

    /* Widen → threshold-extract → out_packed. */
    int64_t* Y64 = malloc((size_t)sig_dim * sizeof(int64_t));
    for (int oi = 0; oi < sig_dim; oi++) Y64[oi] = (int64_t)Y_mtfp[oi];
    m4t_route_threshold_extract(out_packed, Y64, /*tau=*/0, sig_dim);

    free(Y64);
    free(Y_mtfp);
}

void gesh_project_scratch_init(
    gesh_project_scratch_t* sc,
    int sig_dim, int input_dim, int n_max)
{
    assert(sc && sig_dim > 0 && input_dim > 0 && n_max > 0);
    int Op_per_row = M4T_TRIT_PACKED_BYTES(sig_dim);
    sc->sig_dim   = sig_dim;
    sc->input_dim = input_dim;
    sc->n_max     = n_max;
    /* SDOT path: no R packing, no X widening. R_packed and X_mtfp are
     * kept NULL for binary-compat with the struct shape but unused. */
    sc->R_packed  = NULL;
    sc->X_mtfp    = NULL;
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

    /* Matmul into scratch.Y_mtfp via SDOT — no R-pack or X-widen. */
    ternary_matmul_sdot(sc->Y_mtfp, x_batch, R, n, input_dim, sig_dim);

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
