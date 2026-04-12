/*
 * m4t_route.h — ternary routing primitives
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Five primitives that decompose the k-of-T ternary routing algorithm
 * from trix-z into reusable, independently testable opcodes:
 *
 *   sign_extract:      int64 values → packed-trit signs
 *   distance_batch:    query sig × T tile sigs → T distances
 *   topk_abs:          T scores → k (tile, sign) decisions
 *   apply_signed:      k decisions × tile outputs → accumulated result
 *   signature_update:  T×[H,D] weights → T×[D] signatures (compound)
 *
 * A routing pass composes them:
 *   1. signature_update (once at load time)
 *   2. distance_batch (per token)
 *   3. topk_abs (per token)
 *   4. per-tile matmul via m4t_mtfp_ternary_matmul_bt (per selected tile)
 *   5. apply_signed (per token)
 */

#ifndef M4T_ROUTE_H
#define M4T_ROUTE_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Types ─────────────────────────────────────────────────────────────── */

typedef struct {
    int32_t tile_idx;
    m4t_trit_t sign;  /* +1 or -1 */
} m4t_route_decision_t;

/* ── Sign extraction ───────────────────────────────────────────────────── */

/* For each value[i], output sign(value[i]) as a packed trit:
 *   value > 0 → +1 (code 0b01)
 *   value < 0 → -1 (code 0b10)
 *   value == 0 → 0 (code 0b00)
 * dst must have at least M4T_TRIT_PACKED_BYTES(n) bytes. */
void m4t_route_sign_extract(
    uint8_t* dst_packed,
    const int64_t* values,
    int n
);

/* ── Distance batch ────────────────────────────────────────────────────── */

/* Compute routing distance from one query signature to T tile signatures.
 * All signatures are packed trits of sig_dim trits each.
 *
 * distances[t] = popcount_dist(query, tile_sigs + t * Dp, mask, Dp)
 * where Dp = M4T_TRIT_PACKED_BYTES(sig_dim). */
void m4t_route_distance_batch(
    int32_t* distances,
    const uint8_t* query_packed,
    const uint8_t* tile_sigs_packed,
    const uint8_t* mask,
    int T,
    int sig_dim
);

/* ── Top-k selection ───────────────────────────────────────────────────── */

/* Select the k tiles with the largest |score|. For each selected tile,
 * record (tile_idx, sign(score)). Tiles with score == 0 are skipped.
 *
 * If fewer than k tiles have nonzero scores, the remaining decisions
 * have tile_idx = -1 and sign = 0.
 *
 * T is typically small (4–16). Uses simple selection, not a heap. */
void m4t_route_topk_abs(
    m4t_route_decision_t* decisions,
    const int32_t* scores,
    int T,
    int k
);

/* ── Signed accumulation ───────────────────────────────────────────────── */

/* Accumulate k signed tile contributions into result:
 *   result[d] += sign * tile_outs[tile_idx * dim + d]   for d in [0, dim)
 *
 * result must be pre-zeroed or contain a prior accumulation.
 * Decisions with tile_idx < 0 are skipped (sentinel from topk_abs). */
void m4t_route_apply_signed(
    m4t_mtfp_t* result,
    const m4t_mtfp_t* tile_outs,
    const m4t_route_decision_t* decisions,
    int k,
    int dim
);

/* ── Signature update (compound) ───────────────────────────────────────── */

/* Compute weight-derived ternary signatures from packed-trit weights.
 *
 * Algorithm (from trix-z):
 *   For each tile t, dim d:  raw[t,d] = sum over h of W_t[h, d]
 *   For each dim d:          mean[d]  = sum_t(raw[t,d]) / T
 *   For each tile t, dim d:  sig[t,d] = sign(raw[t,d] - mean[d])
 *
 * weights layout: T tiles concatenated, each tile is H rows of Dp packed
 *   bytes. Total: T * H * Dp bytes. Row [t * H + h] holds the h-th row
 *   of tile t's weight matrix.
 *
 * signatures layout: T rows of Dp packed bytes.
 *
 * scratch: caller-provided buffer of at least (T + 1) * D int64_t values.
 *   Used for column sums ([T * D]) and means ([D]).
 *
 * Constraint: D ≤ 4096 (M4T_ROUTE_MAX_DIM). Uses a fixed stack buffer
 * for row unpacking. This is a setup-time function, not a hot-path opcode. */
#define M4T_ROUTE_MAX_DIM 4096
void m4t_route_signature_update(
    uint8_t* signatures,
    const uint8_t* weights,
    int64_t* scratch,
    int T,
    int H,
    int D
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_ROUTE_H */
