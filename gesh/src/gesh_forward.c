/*
 * gesh_forward.c — Phase A.1 forward pass.
 *
 * Integer-only. Composes existing m4t primitives (popcount_dist for
 * Hamming, optional projection inline) with a small top-k-smallest
 * helper and a histogram-vote classifier.
 */

#include "gesh_forward.h"
#include "gesh_project.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Distance kernel signature shared between standard Hamming and wildcard
 * forward variants. Both kernels have the same arg shape. */
typedef int32_t (*dist_kernel_t)(
    const uint8_t* query_packed,
    const uint8_t* tile_packed,
    const uint8_t* mask,
    int packed_bytes_or_sig_dim);

/* Trampolines unify the calling convention. m4t_popcount_dist takes
 * packed_bytes; m4t_route_wildcard_dist takes sig_dim. Both produce the
 * same int32 distance under their declared semantics. */
static int32_t dist_hamming_trampoline(
    const uint8_t* a, const uint8_t* b, const uint8_t* mask,
    int packed_bytes_or_sig_dim)
{
    return m4t_popcount_dist(a, b, mask, packed_bytes_or_sig_dim);
}
static int32_t dist_wildcard_trampoline(
    const uint8_t* a, const uint8_t* b, const uint8_t* mask,
    int packed_bytes_or_sig_dim)
{
    return m4t_route_wildcard_dist(a, b, mask, packed_bytes_or_sig_dim);
}

/* Find the indices of the top_k smallest values in `dists[0..T)`.
 * Writes top_k indices to `out_idx`; output order is ascending by value.
 * Stable on ties: lower index wins.
 *
 * Implementation: insertion-sort-style maintenance of a top-k buffer.
 * O(T · top_k) — fine for small top_k and moderate T (Phase A: T = C =
 * 10, top_k typically 1 or 3). */
static void topk_smallest_indices(
    int* out_idx,
    const int32_t* dists,
    int T, int top_k)
{
    /* Initialize with sentinel max distance. */
    for (int k = 0; k < top_k; k++) {
        out_idx[k] = -1;
    }
    int32_t* buf_d = malloc((size_t)top_k * sizeof(int32_t));
    for (int k = 0; k < top_k; k++) buf_d[k] = INT32_MAX;

    for (int t = 0; t < T; t++) {
        int32_t d = dists[t];
        /* Find insertion point in the sorted buf. Buf[0..k-1] is sorted
         * ascending; if d < buf[k-1] (worst kept), it displaces. */
        if (d >= buf_d[top_k - 1]) continue;
        /* Find insertion index. */
        int ins = top_k - 1;
        while (ins > 0 && d < buf_d[ins - 1]) ins--;
        /* Shift right from ins to top_k - 1. */
        for (int s = top_k - 1; s > ins; s--) {
            buf_d[s]   = buf_d[s - 1];
            out_idx[s] = out_idx[s - 1];
        }
        buf_d[ins] = d;
        out_idx[ins] = t;
    }

    free(buf_d);
}

/* Shared core for both forward variants. The only behavioral difference
 * between gesh_forward_classify and gesh_forward_classify_wildcard is the
 * per-tile distance kernel; everything else (projection, top-k, vote)
 * is identical. The `dist_kernel` parameter selects between them.
 *
 * The `dist_kernel_arg_is_sig_dim` flag handles the API-shape difference:
 *   m4t_popcount_dist takes packed_bytes
 *   m4t_route_wildcard_dist takes sig_dim
 * Both compute the same byte-level work; only the argument convention
 * differs. */
static int forward_classify_impl(
    int* out_predictions,
    const m4t_trit_t* queries,
    int n_queries,
    const gesh_bank_t* bank,
    const gesh_projection_t* proj,
    int top_k,
    dist_kernel_t dist_kernel,
    int dist_kernel_arg_is_sig_dim)
{
    assert(out_predictions && bank && proj);
    assert(n_queries >= 0);
    assert(top_k > 0 && top_k <= bank->n_tiles);
    assert(bank->sig_dim == proj->sig_dim);
    /* Substrate-discipline assert: sig_dim must be positive (else the
     * mask construction and popcount_dist semantics are undefined). */
    assert(bank->sig_dim > 0);
    if (proj->R == NULL) {
        /* Identity projection: input dim must equal signature dim. */
        assert(proj->input_dim == proj->sig_dim);
    } else {
        assert(proj->input_dim > 0 && proj->sig_dim > 0);
    }
    if (n_queries == 0) return 0;
    assert(queries);

    /* Aliasing preconditions. The substrate's writable-output kernels
     * all assert dst-distinct-from-input; gesh inherits the convention.
     * out_predictions writes the per-query class label; queries,
     * bank->tiles_packed, and proj->R are read-only inputs. None may
     * share storage with out_predictions. */
    assert((const void*)out_predictions != (const void*)queries);
    assert((const void*)out_predictions != (const void*)bank->tiles_packed);
    assert(proj->R == NULL ||
           (const void*)out_predictions != (const void*)proj->R);

    int input_dim = proj->input_dim;
    int sig_dim = bank->sig_dim;
    int T = bank->n_tiles;
    int Dp_sig = M4T_TRIT_PACKED_BYTES(sig_dim);

    /* Per-call scratch. unpacked_sig is no longer needed: the
     * kernel-routed projection writes directly to packed_sig. */
    uint8_t*    packed_sig   = malloc((size_t)Dp_sig);
    uint8_t*    mask_packed  = malloc((size_t)Dp_sig);
    int32_t*    dists        = malloc((size_t)T * sizeof(int32_t));
    int*        topk_idx     = malloc((size_t)top_k * sizeof(int));

    /* Mask = all ones for sig_dim trits, zero in any tail bits. The
     * popcount_dist mask is byte-level (one bit per packed-trit byte
     * position is too coarse here; use full-mask for in-tensor cells
     * and rely on the packing to leave tail bits zero). */
    memset(mask_packed, 0xFF, (size_t)Dp_sig);
    /* Zero tail bits if sig_dim is not a multiple of 4. */
    int tail_trits = sig_dim & 3;
    if (tail_trits > 0) {
        uint8_t tail_mask = (uint8_t)((1u << (tail_trits * 2)) - 1u);
        mask_packed[Dp_sig - 1] = tail_mask;
    }

    /* Class-id range derivation: assumes bank->labels are non-negative
     * and dense from 0 (the convention gesh_bank_build_class_mean
     * establishes). The assert below catches accidental violations
     * (sentinel -1, sparse labels) that would silently misbehave —
     * a future bank constructor with sparse labels needs an explicit
     * n_classes parameter, not implicit derivation from max(labels). */
    int max_label = 0;
    for (int t = 0; t < T; t++) {
        assert(bank->labels[t] >= 0);
        if (bank->labels[t] > max_label) max_label = bank->labels[t];
    }
    int n_classes = max_label + 1;
    int* vote = malloc((size_t)n_classes * sizeof(int));

    for (int q = 0; q < n_queries; q++) {
        const m4t_trit_t* x = queries + (size_t)q * input_dim;

        /* 1. Project query to signature (or identity if no projection).
         *    Substrate-discipline: projection is fully kernel-routed via
         *    gesh_project_one_packed (m4t_mtfp_ternary_matmul_bt +
         *    m4t_route_threshold_extract). No open-coded MAC or sign
         *    threshold in this path. */
        if (proj->R != NULL) {
            gesh_project_one_packed(packed_sig, x, proj->R,
                                      sig_dim, input_dim);
        } else {
            m4t_pack_trits_1d(packed_sig, x, sig_dim);
        }

        /* 2. Compute distance to each tile via the selected kernel. */
        int dist_arg = dist_kernel_arg_is_sig_dim ? sig_dim : Dp_sig;
        for (int t = 0; t < T; t++) {
            const uint8_t* tile = bank->tiles_packed + (size_t)t * Dp_sig;
            dists[t] = dist_kernel(packed_sig, tile, mask_packed, dist_arg);
        }

        /* 3. Find top_k smallest distances. */
        topk_smallest_indices(topk_idx, dists, T, top_k);

        /* 4. Class-vote across the top_k tiles' labels. */
        memset(vote, 0, (size_t)n_classes * sizeof(int));
        for (int k = 0; k < top_k; k++) {
            int idx = topk_idx[k];
            if (idx < 0) continue;
            vote[bank->labels[idx]]++;
        }
        /* Argmax of vote (lower class index wins on ties). */
        int best_class = 0;
        int best_vote = vote[0];
        for (int c = 1; c < n_classes; c++) {
            if (vote[c] > best_vote) {
                best_vote = vote[c];
                best_class = c;
            }
        }
        out_predictions[q] = best_class;
    }

    free(packed_sig);
    free(mask_packed);
    free(dists);
    free(topk_idx);
    free(vote);
    return 0;
}

int gesh_forward_classify(
    int* out_predictions,
    const m4t_trit_t* queries,
    int n_queries,
    const gesh_bank_t* bank,
    const gesh_projection_t* proj,
    int top_k)
{
    /* Standard ternary Hamming routing — §19 (I) Tie-cancellation
     * symmetric semantics for zero-state. */
    return forward_classify_impl(out_predictions, queries, n_queries,
                                  bank, proj, top_k,
                                  dist_hamming_trampoline,
                                  /*dist_kernel_arg_is_sig_dim=*/0);
}

int gesh_forward_classify_wildcard(
    int* out_predictions,
    const m4t_trit_t* queries,
    int n_queries,
    const gesh_bank_t* bank,
    const gesh_projection_t* proj,
    int top_k)
{
    /* Wildcard routing — §19 (II) Wildcard semantics for tile-zero
     * positions. Caller-responsible for using a deliberate-wildcard bank
     * (e.g. gesh_bank_build_class_wildcard) per the §19.6 sanctioned-
     * pairing constraint. */
    return forward_classify_impl(out_predictions, queries, n_queries,
                                  bank, proj, top_k,
                                  dist_wildcard_trampoline,
                                  /*dist_kernel_arg_is_sig_dim=*/1);
}
