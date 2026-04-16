/*
 * glyph_resolver.h — candidate-set resolver variants for routed k-NN.
 *
 * A resolver reads a candidate union (produced by multi-table bucket
 * lookup + multi-probe) and returns a predicted label. Three variants
 * measured in Axis 6:
 *
 *   VOTE — argmax class by weighted vote sum over the union. Counts
 *          "how many tables voted for each candidate" weighted by
 *          vote count, aggregated per class. No distance arithmetic.
 *
 *   SUM  — argmin candidate by Σ_m popcount_dist(q_sig_m, cand_sig_m)
 *          across M active tables. Picks the candidate with the
 *          smallest composite Hamming distance. Structurally equivalent
 *          to 1-NN over a composite M×N_PROJ-trit signature, restricted
 *          to the union.
 *
 *   PTM  — per-table 1-NN majority. For each table m, find the
 *          candidate in the union with the smallest popcount_dist
 *          under that table's projection; that yields M candidate
 *          labels; majority-vote them.
 *
 * Empirical ranking (Axis 6, N_PROJ=16 deskewed MNIST):
 *   SUM > PTM > VOTE  at every M ≥ 2.
 *   At M=32, SUM=97.24%, PTM=94.25%, VOTE=88.50%.
 */

#ifndef GLYPH_RESOLVER_H
#define GLYPH_RESOLVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time cap on the number of classes a resolver handles. Stack
 * class-tally buffers are sized to this. Must be >= the largest class
 * cardinality ever passed via glyph_union_t.n_classes. 256 covers
 * MNIST (10), Fashion-MNIST (10), EMNIST (47), CIFAR-100 (100) and
 * ImageNet-scale 200-class benchmarks with room to spare. Larger class
 * cardinalities require either a resolver API change (caller-supplied
 * scratch buffer) or bumping this constant. Runtime asserts in the
 * resolver implementations guard against overruns. */
#define GLYPH_MAX_CLASSES 256

/* A union of candidate prototype indices, together with per-prototype
 * vote counts and the shared training-label table the resolvers read
 * from. All fields are borrowed; the caller owns the memory.
 *
 * Lifecycle contract (important):
 *
 *   hit_list — array of prototype indices currently in the union.
 *              Caller appends a proto_idx exactly once when it first
 *              enters the union (detected via votes[idx] == 0) and
 *              leaves it sorted by nothing in particular. Max length
 *              is the caller's per-query union cap.
 *
 *   n_hit    — number of valid entries in hit_list.
 *
 *   votes    — dense array of length n_train, indexed by prototype
 *              index (NOT by hit-list position). votes[idx] is the
 *              count of tables that placed prototype idx in the
 *              query's multi-probe neighborhood. Must be allocated
 *              with size >= n_train. Caller must zero entries for
 *              proto_idxs that were in a prior query's union; the
 *              efficient pattern is "lazy zero" — iterate hit_list
 *              at query end and set votes[hit_list[j]] = 0.
 *
 *   y_train  — prototype labels, length n_train. Each label must be
 *              in [0, n_classes). The resolvers read y_train[idx]
 *              for every idx in hit_list.
 *
 *   n_classes — number of distinct classes. Must be <= GLYPH_MAX_CLASSES.
 *              Runtime-asserted in each resolver implementation.
 *
 * The resolvers do NOT mutate any of these fields; they are declared
 * const where the C type system allows. Callers that invalidate the
 * union between queries (new n_hit, new hit_list contents, new votes
 * state) may reuse the same struct across many queries. */
typedef struct {
    const int32_t*  hit_list;
    int             n_hit;
    const uint16_t* votes;
    const int*      y_train;
    int             n_classes;
} glyph_union_t;

/* VOTE resolver — O(n_hit) time, no distance arithmetic. */
int glyph_resolver_vote(const glyph_union_t* u);

/* SUM resolver — O(n_hit × m_active) popcount_dist calls.
 *
 * This is the reference implementation. Accepts any sig_bytes value
 * supported by m4t_popcount_dist. Use this when sig_bytes != 4 or
 * when architectural portability matters.
 */
int glyph_resolver_sum(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask);

/* SUM resolver — NEON-batched variant specialized for sig_bytes=4
 * (N_PROJ=16). Honors the mask parameter in full generality — the
 * mask is broadcast to all four 4-byte lanes and ANDed with the
 * XOR result before VCNT, matching m4t_popcount_dist's semantics
 * bit-for-bit regardless of mask value.
 *
 * Processes 4 candidates per 16-byte NEON vector: gathers 4 training
 * signatures into a uint8x16_t register via scalar loads, XORs with
 * a broadcast of the query's 4-byte signature, ANDs with a broadcast
 * mask, applies VCNT for per-byte popcount, and pairwise-sums inside
 * each 4-byte lane to produce 4 per-candidate distances per NEON op.
 * Accumulates per-candidate scores across all m_active tables.
 *
 * Bit-exact equivalent to glyph_resolver_sum when sig_bytes=4 for
 * any mask. The equivalence test lives in test_glyph_libglyph.c.
 *
 * Returns -1 if sig_bytes != 4 (falls through; caller should have
 * used glyph_resolver_sum).
 */
int glyph_resolver_sum_neon4(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask);

/* RADIUS-AWARE SUM resolver — filter-to-resolver rerouting variant.
 *
 * Score each candidate as:
 *
 *     score(c) = sum_dist(c) + lambda × min_radius[c]
 *
 * where min_radius[c] is the smallest multi-probe ternary Hamming
 * radius at which ANY table placed c in the query's neighborhood.
 * Candidates found at radius 0 (exact-match bucket hit in at least
 * one table) get no penalty; candidates only reachable via r=1 get
 * lambda added to their score; r=2 adds 2*lambda.
 *
 * Rationale: a candidate found via exact-bucket-match in some table
 * is structurally closer to the query than one only reachable via
 * multi-probe expansion. The current SUM resolver collapses this
 * distinction — it sees only the summed signature distance, not
 * how the candidate entered the union. This variant reads filter-
 * stage geometry that scalar SUM discards.
 *
 * The caller passes min_radius as a dense array of length n_train,
 * indexed by prototype index (same lifecycle pattern as votes):
 * caller tracks per-query min_radius during the probe pass, then
 * lazy-zeros the touched entries via hit_list at query end.
 *
 * lambda is caller-chosen. A starting value of 8 corresponds to
 * "one radius step is worth one byte of Hamming distance" at
 * N_PROJ=16 (sig_bytes=4, max per-table distance 32). Lower
 * lambda shrinks toward scalar SUM; higher lambda shrinks toward
 * "strict radius-0 preference."
 *
 * Not bit-exact equivalent to glyph_resolver_sum — this is a
 * different algorithm producing a different ranking.
 */
int glyph_resolver_sum_radiusaware(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask,
    const uint8_t*       min_radius,
    int                  lambda);

/* VOTE-WEIGHTED SUM resolver — filter-to-resolver rerouting variant.
 *
 * Same O(n_hit × m_active) popcount_dist calls as glyph_resolver_sum,
 * but scores each candidate as:
 *
 *     score(c) = sum_dist(c) / (1.0 + votes[c])
 *
 * where votes[c] is the number of tables whose multi-probe neighborhood
 * included c (the same u->votes field the VOTE resolver reads). The
 * division amortizes the summed distance by the filter-stage vote
 * count — candidates that more tables agreed were near the query get
 * their sum_dist effectively discounted.
 *
 * This is the simplest way to fold filter-stage information into the
 * resolver's ranking. The current SUM resolver reads only the geometric
 * sum of signature distances; the vote count is a separate dimension
 * of filter-stage agreement that SUM ignores. On datasets where
 * signature-space distance is a clean proxy for class similarity
 * (MNIST), vote-weighting should be marginal or neutral. On datasets
 * where signature-space distance is noisy (Fashion-MNIST, CIFAR-10),
 * vote-weighting may recover some of the 15-point resolver gap by
 * preferring candidates that multiple independent hashes agreed on.
 *
 * Computed in a single pass over the hit_list with integer arithmetic:
 * score_int(c) = sum_dist(c) × 1024 / (1 + votes[c]) to avoid float.
 * The 1024 scale keeps integer precision sufficient for the argmin
 * tie-break across typical sum_dist and votes magnitudes.
 *
 * Not bit-exact equivalent to glyph_resolver_sum — this is a different
 * algorithm producing a different ranking. Equivalence on the scalar
 * argmin is only expected when all candidates have identical votes.
 */
int glyph_resolver_sum_voteweighted(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask);

/* PTM resolver — O(n_hit × m_active) popcount_dist calls, different
 * access pattern than SUM. */
int glyph_resolver_per_table_majority(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_RESOLVER_H */
