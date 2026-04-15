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

/* ── Threshold extraction ─────────────────────────────────────────────────
 *
 * For each value[i], produce a packed trit based on a three-state
 * magnitude classification:
 *
 *   value >  tau    → +1 (code 0b01)  strong positive
 *   value < -tau    → -1 (code 0b10)  strong negative
 *   |value| <= tau  →  0 (code 0b00)  within neutral band
 *
 * Preconditions:
 *   tau >= 0
 *   n >= 0
 *   dst_packed has at least M4T_TRIT_PACKED_BYTES(n) bytes
 *
 * Sanctioned input-class contract (see M4T_SUBSTRATE §18):
 *   - tau > 0 and input values with magnitudes spanning across tau:
 *     emission coverage holds — all three output states are produced
 *     non-trivially. This is the primary sanctioned deployment.
 *   - tau = 0 and input values that can be exactly zero with
 *     non-trivial probability (integer arithmetic with potential
 *     exact equality; e.g. col_sum − mean in signature_update):
 *     emission coverage holds.
 *   - tau = 0 and continuous-valued inputs (MTFP projection outputs
 *     from ternary_matmul, etc.): emission coverage FAILS — the zero
 *     state is measure-zero. The primitive produces a sign-only
 *     classification in practice. Consumers in this class should pass
 *     tau > 0 sized to the expected noise floor. */
void m4t_route_threshold_extract(
    uint8_t* dst_packed,
    const int64_t* values,
    int64_t tau,
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
 * T is capped at M4T_ROUTE_MAX_T (uniqueness tracked via a uint64_t
 * bitmask). Uses simple selection, not a heap.
 *
 * §18 contract (output-side emission coverage):
 *   Enumerated output state: decision.sign ∈ {+1, -1, 0} where +1/-1
 *     encode the sign of a selected tile's score and 0 is the sentinel
 *     for "no tile selected" (tile_idx == -1).
 *   Sanctioned input class: score arrays with mixed-sign nonzero
 *     entries plus k configured so that at least one of {k > #nonzero,
 *     k ≤ #nonzero} is realized across representative calls. Under
 *     this class all three sign states occur.
 *   Coverage test: test_topk_abs_basic / test_topk_abs_with_zeros /
 *     test_topk_abs_all_tiles in test_m4t_route.c (union covers all
 *     three states). See M4T_SUBSTRATE.md §18. */
#define M4T_ROUTE_MAX_T 64
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
 * Decisions with tile_idx < 0 are skipped (sentinel from topk_abs).
 *
 * §18 contract (input-side emission coverage):
 *   Three-state API locus: the decision.sign field consumed from each
 *     decision ∈ {+1, -1, 0-sentinel} drives three distinct branches:
 *     +1 → add tile_outs, -1 → subtract, 0/sentinel → skip. The
 *     primitive's three-way character is exercised iff decisions
 *     realize all three sign states across the call.
 *   Sanctioned input class: decisions produced by m4t_route_topk_abs
 *     under its sanctioned input class; the three sign states arise
 *     naturally from mixed-sign scores with some sentinel overflow.
 *   Coverage test: test_apply_signed (+1 add, -1 sub branches),
 *     test_apply_signed_sentinel (0-sentinel skip branch) in
 *     test_m4t_route.c. See M4T_SUBSTRATE.md §18. */
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
 * Rounding note. `mean[d]` uses C integer division (truncation toward
 * zero), not mathematical rounding. On inputs where the mathematical
 * mean lies strictly between two integers, the truncated mean differs
 * by up to 1, which can flip the sign for tiles whose `raw[t,d]` lands
 * exactly on the mathematical mean but not the truncated one. The
 * behavior is deterministic; document it at call sites if it matters.
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
 * D is not capped by the substrate. The row unpack buffer is heap-allocated.
 * This is a setup-time function, not a hot-path opcode. */
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
