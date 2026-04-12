/*
 * m4t_route.c — ternary routing primitives
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 */

#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_mtfp.h"
#include "m4t_internal.h"

#include <string.h>
#include <assert.h>

/* ── Sign extraction ───────────────────────────────────────────────────── */

void m4t_route_sign_extract(
    uint8_t* dst_packed, const int64_t* values, int n)
{
    int n_bytes = M4T_TRIT_PACKED_BYTES(n);
    memset(dst_packed, 0, (size_t)n_bytes);

    for (int i = 0; i < n; i++) {
        m4t_trit_t t = (values[i] > 0) ? 1 : (values[i] < 0) ? -1 : 0;
        uint8_t code = (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
        dst_packed[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

/* ── Distance batch ────────────────────────────────────────────────────── */

void m4t_route_distance_batch(
    int32_t* distances,
    const uint8_t* query_packed,
    const uint8_t* tile_sigs_packed,
    const uint8_t* mask,
    int T, int sig_dim)
{
    assert(distances && query_packed && tile_sigs_packed && mask);
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);

    for (int t = 0; t < T; t++) {
        distances[t] = m4t_popcount_dist(
            query_packed,
            tile_sigs_packed + (size_t)t * Dp,
            mask, Dp);
    }
}

/* ── Top-k selection ───────────────────────────────────────────────────── */

void m4t_route_topk_abs(
    m4t_route_decision_t* decisions,
    const int32_t* scores,
    int T, int k)
{
    assert(decisions && scores);
    assert(k >= 0 && T >= 0);

    /* Track which tiles have been selected. T is small (≤ 16 typically). */
    int8_t used[64];
    assert(T <= 64);
    memset(used, 0, (size_t)T);

    for (int sel = 0; sel < k; sel++) {
        int best_idx = -1;
        int32_t best_abs = -1;

        for (int t = 0; t < T; t++) {
            if (used[t]) continue;
            int32_t a = scores[t];
            int32_t abs_a = (a >= 0) ? a : -a;
            if (abs_a > best_abs) {
                best_abs = abs_a;
                best_idx = t;
            }
        }

        if (best_idx < 0 || best_abs == 0) {
            /* No more nonzero tiles. Fill remaining with sentinel. */
            for (int r = sel; r < k; r++) {
                decisions[r].tile_idx = -1;
                decisions[r].sign = 0;
            }
            break;
        }

        used[best_idx] = 1;
        decisions[sel].tile_idx = best_idx;
        decisions[sel].sign = (scores[best_idx] > 0) ? 1 : -1;
    }
}

/* ── Signed accumulation ───────────────────────────────────────────────── */

void m4t_route_apply_signed(
    m4t_mtfp_t* result,
    const m4t_mtfp_t* tile_outs,
    const m4t_route_decision_t* decisions,
    int k, int dim)
{
    assert(result && tile_outs && decisions);

    for (int sel = 0; sel < k; sel++) {
        int idx = decisions[sel].tile_idx;
        m4t_trit_t sign = decisions[sel].sign;
        if (idx < 0) continue;

        const m4t_mtfp_t* tile = tile_outs + (size_t)idx * dim;

        if (sign == 1) {
            m4t_mtfp_vec_add_inplace(result, tile, dim);
        } else if (sign == -1) {
            /* Subtract: result[d] -= tile[d]. No vec_sub_inplace exists,
             * so we negate and add. Use scalar to avoid a temp buffer. */
            for (int d = 0; d < dim; d++) {
                result[d] = m4t_mtfp_sub(result[d], tile[d]);
            }
        }
    }
}

/* ── Signature update ──────────────────────────────────────────────────── */

void m4t_route_signature_update(
    uint8_t* signatures,
    const uint8_t* weights,
    int64_t* scratch,
    int T, int H, int D)
{
    assert(signatures && weights && scratch);
    assert(T > 0 && H > 0 && D > 0);

    int Dp = M4T_TRIT_PACKED_BYTES(D);

    /* scratch layout: [0 .. T*D-1] = column sums, [T*D .. T*D+D-1] = means */
    int64_t* col_sums = scratch;           /* [T, D] */
    int64_t* means    = scratch + T * D;   /* [D]    */

    /* Phase 1: column sums. For each tile, sum each column of the [H, D]
     * weight matrix. Unpack one row at a time to avoid large temp buffers. */
    memset(col_sums, 0, (size_t)T * D * sizeof(int64_t));

    /* Allocate a small row buffer on the stack. D is the model dim, typically
     * 32–256. This is bounded and small. */
    m4t_trit_t row_buf[4096];
    assert(D <= 4096);

    for (int t = 0; t < T; t++) {
        int64_t* cs = col_sums + (size_t)t * D;
        for (int h = 0; h < H; h++) {
            const uint8_t* row_packed = weights + ((size_t)t * H + h) * Dp;
            m4t_unpack_trits_1d(row_buf, row_packed, D);
            for (int d = 0; d < D; d++) {
                cs[d] += (int64_t)row_buf[d];
            }
        }
    }

    /* Phase 2: mean across tiles per dimension. */
    memset(means, 0, (size_t)D * sizeof(int64_t));
    for (int t = 0; t < T; t++) {
        const int64_t* cs = col_sums + (size_t)t * D;
        for (int d = 0; d < D; d++) {
            means[d] += cs[d];
        }
    }
    for (int d = 0; d < D; d++) {
        means[d] /= T;
    }

    /* Phase 3: sign(col_sum - mean) → packed-trit signature per tile.
     * Reuse the col_sums buffer for the difference values. */
    for (int t = 0; t < T; t++) {
        int64_t* cs = col_sums + (size_t)t * D;
        for (int d = 0; d < D; d++) {
            cs[d] -= means[d];
        }
        m4t_route_sign_extract(
            signatures + (size_t)t * Dp, cs, D);
    }
}
