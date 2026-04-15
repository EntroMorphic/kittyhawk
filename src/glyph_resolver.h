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

/* A union of candidate prototype indices, with per-prototype vote
 * counts indexed by prototype index (not by hit-list position).
 * Caller owns both arrays. */
typedef struct {
    const int32_t*  hit_list;   /* [n_hit] prototype indices in the union  */
    int             n_hit;
    const uint16_t* votes;      /* [max_proto_idx+1] vote counts by idx    */
    const int*      y_train;    /* [n_train] prototype labels              */
    int             n_classes;
} glyph_union_t;

/* VOTE resolver — O(n_hit) time, no distance arithmetic. */
int glyph_resolver_vote(const glyph_union_t* u);

/* SUM resolver — O(n_hit × m_active) popcount_dist calls. */
int glyph_resolver_sum(
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
