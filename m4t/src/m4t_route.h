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

/* ── Dual-threshold extract (§19 5-state encoding) ─────────────────────── */

/* Threshold-extract that emits BOTH a packed-trit signature (using
 * tau_weak) AND a parallel packed-bit confidence bitmap (positions
 * where |value| > tau_strong). The composed (trit, confidence) pair
 * is a 5-state encoding {-strong, -weak, 0, +weak, +strong}.
 *
 * Per-position outputs:
 *   value > tau_strong   → trit=+1, conf=1
 *   tau_weak < value ≤ tau_strong → trit=+1, conf=0
 *   |value| ≤ tau_weak   → trit=0,  conf=0
 *   value < -tau_strong  → trit=-1, conf=1
 *  -tau_strong ≤ value < -tau_weak → trit=-1, conf=0
 *
 * Substrate-novel: magnitude information that single-tau threshold
 * extract discards. Base-2 with quantize-then-route loses magnitude
 * after quantization; recovering it requires a parallel mantissa
 * channel (≥1.5× storage).
 *
 * Buffer sizes:
 *   dst_trit_packed: ≥ M4T_TRIT_PACKED_BYTES(n) bytes
 *   dst_conf_bits:   ≥ ceil(n / 8) bytes (1 bit per position, LSB-first)
 *
 * Preconditions: 0 ≤ tau_weak ≤ tau_strong; n ≥ 0; non-aliasing outputs. */
void m4t_route_threshold_extract_dual(
    uint8_t* dst_trit_packed,
    uint8_t* dst_conf_bits,
    const int64_t* values,
    int64_t tau_weak,
    int64_t tau_strong,
    int n
);

/* ── Confidence-weighted distance ──────────────────────────────────────── */

/* Hamming-style distance over (trit, confidence) signature pairs. The
 * trit signatures are packed-trit (2 bits/pos); the confidence bitmaps
 * are packed-bit (1 bit/pos). Cost table per position:
 *
 *   (q.trit ↔ t.trit same nonzero sign):    0    (agreement, any confidence)
 *   (q.trit = 0, t.trit = 0):                0    (mutual abstain, current Hamming)
 *   (q.trit ±1, t.trit = 0):                 1    (asymmetric abstain, current Hamming)
 *   (q.trit = 0, t.trit ±1):                 1    (asymmetric abstain, current Hamming)
 *   (q.trit ↔ t.trit opposite sign):
 *     - both confidence bits 0:              2    (current Hamming, both uncertain)
 *     - exactly one confidence bit 1:        3    (one side asserted strongly)
 *     - both confidence bits 1:              4    (mutual high-confidence disagreement)
 *
 * §19 (III) Abstain semantics for trit zero (cost depends on whether
 * the OTHER side asserts). §19 has no formal interpretation for the
 * confidence bit alone; it's a magnitude class indicator that modifies
 * mismatch cost when both sides have a sign opinion.
 *
 * Returns int32 cost; max possible cost per position is 4 (vs 2 for
 * standard Hamming). Distance is scaled accordingly.
 *
 * Preconditions: same as m4t_popcount_dist; sig_dim ≥ 0; mask sized for
 * trit signature. The conf bitmaps are NOT masked separately — the trit
 * mask governs which positions count. */
int32_t m4t_route_confidence_weighted_dist(
    const uint8_t* query_trit_packed,
    const uint8_t* query_conf_bits,
    const uint8_t* tile_trit_packed,
    const uint8_t* tile_conf_bits,
    const uint8_t* mask,
    int sig_dim
);

/* ── Wildcard distance ─────────────────────────────────────────────────── */

/* Wildcard-Hamming distance between a query signature and a tile
 * signature, where TILE-zero is interpreted as wildcard (don't-care /
 * match-anything). Distinct from `m4t_popcount_dist` which costs
 * (q=±1, t=0) at 1; this kernel costs it at 0.
 *
 * Per-position cost table:
 *   (q=+1, t=+1)  → 0   (full match)
 *   (q=-1, t=-1)  → 0   (full match)
 *   (q=0,  t=0)   → 0   (mutual abstention; same as popcount_dist)
 *   (q=±1, t=0)   → 0   (wildcard match — KEY DISTINCTION FROM popcount_dist)
 *   (q=0,  t=±1)  → 1   (query abstains, tile asserts; partial mismatch)
 *   (q=+1, t=-1)  → 2   (full mismatch)
 *   (q=-1, t=+1)  → 2   (full mismatch)
 *
 * §19 zero-state interpretation: the kernel produces (II) Wildcard
 * semantics for tile-side zeros; (III) Abstain semantics for query-side
 * zeros (cost 1 against tile-±1, cost 0 against tile-0). See
 * `m4t/docs/M4T_SUBSTRATE.md` §19 for the substrate-level discussion.
 *
 * §19 sanctioned input class: tile signatures from constructors that
 * place zeros DELIBERATELY (`gesh_bank_build_class_wildcard`,
 * sparse-coded constructors, feature-pruning constructors). NOT for
 * use against `gesh_bank_build_class_mean` or `gesh_bank_build_kmeans_per_class`
 * outputs where zeros emerge from sample-cancellation ties — there
 * the wildcard interpretation over-promotes ambiguous matches.
 *
 * Implementation: computes standard ternary Hamming via
 * `m4t_popcount_dist`, then subtracts a wildcard correction equal to
 * the count of positions where tile-trit==0 AND query-trit!=0 (those
 * positions contribute 1 to standard Hamming but 0 to wildcard).
 *
 * Preconditions:
 *   query_packed, tile_packed, mask non-NULL.
 *   sig_dim ≥ 0.
 *   query_packed and tile_packed contain valid trit codes
 *     (0b00, 0b01, 0b10); 0b11 is undefined behavior. */
int32_t m4t_route_wildcard_dist(
    const uint8_t* query_packed,
    const uint8_t* tile_packed,
    const uint8_t* mask,
    int sig_dim
);

/* ── Pairwise Hamming sum (lattice-native geometric loss) ──────────────── */

/* Sum of ternary Hamming distances over all (i, j) tile pairs with i<j.
 * Used as a lattice-native geometric loss for substrate-novel training:
 * maximizing this sum maximizes class-tile separation in trit space.
 *
 * Substrate-distinct: ternary Hamming is the trit lattice's natural
 * distance metric. Operates entirely on packed-trit signatures via
 * existing m4t_popcount_dist; no labels, no float, no continuous
 * relaxation. The lattice IS the geometry.
 *
 * Returns int32 sum. Max possible value = T*(T-1)/2 * 2*sig_dim.
 *
 * Preconditions: T >= 0; sig_dim >= 0; tile signatures + mask non-NULL
 * when T*(T-1) > 0. */
int32_t m4t_route_pairwise_hamming_sum(
    const uint8_t* tile_sigs_packed,    /* [T × M4T_TRIT_PACKED_BYTES(sig_dim)] */
    const uint8_t* mask,
    int T,
    int sig_dim
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
 * scratch: caller-owned temporary buffer of at least (T + 1) * D int64_t
 *   values. Contents are overwritten; no useful data on return.
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

/* ── Emission-coverage helper (§18 testability) ────────────────────────── */

/* Inspect a decisions[] array and report which sign states actually appeared.
 *
 * Outputs three booleans (any non-NULL pointer is written; NULL skips):
 *   *out_has_pos  ← true iff some decision has sign == +1
 *   *out_has_neg  ← true iff some decision has sign == -1
 *   *out_has_zero ← true iff some decision has sign ==  0 (sentinel)
 *
 * Consumer integration tests use this to demonstrate that the §18 input-
 * class contract is honored *at the call site* — i.e., that the routing
 * pass actually exercises all three sign states across representative
 * inputs. A test that calls apply_signed and asserts
 * (has_pos && has_neg && has_zero) has positively demonstrated emission
 * coverage; the contract is no longer an unstated assumption.
 *
 * Pure inspection — does not allocate, does not modify decisions.
 *
 * Preconditions:
 *   decisions is non-NULL when k > 0
 *   k >= 0 */
void m4t_route_decisions_emit_coverage(
    const m4t_route_decision_t* decisions,
    int k,
    int* out_has_pos,
    int* out_has_neg,
    int* out_has_zero
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_ROUTE_H */
