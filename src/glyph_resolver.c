/*
 * glyph_resolver.c — candidate-set resolver implementations.
 */

#include "glyph_resolver.h"
#include "m4t_trit_pack.h"

#include <limits.h>
#include <stddef.h>

int glyph_resolver_vote(const glyph_union_t* u) {
    int class_votes[32] = {0};   /* u->n_classes is expected to be ≤ 32 */
    for (int j = 0; j < u->n_hit; j++) {
        int idx = u->hit_list[j];
        class_votes[u->y_train[idx]] += u->votes[idx];
    }
    int pred = 0;
    for (int c = 1; c < u->n_classes; c++)
        if (class_votes[c] > class_votes[pred]) pred = c;
    return pred;
}

int glyph_resolver_sum(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask)
{
    int32_t best_score = INT32_MAX;
    int     best_label = -1;
    for (int j = 0; j < u->n_hit; j++) {
        int idx = u->hit_list[j];
        int32_t score = 0;
        for (int m = 0; m < m_active; m++) {
            score += m4t_popcount_dist(
                query_sigs[m],
                table_train_sigs[m] + (size_t)idx * sig_bytes,
                mask, sig_bytes);
        }
        if (score < best_score) {
            best_score = score;
            best_label = u->y_train[idx];
        }
    }
    return best_label;
}

int glyph_resolver_per_table_majority(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask)
{
    int label_votes[32] = {0};
    for (int m = 0; m < m_active; m++) {
        int32_t best_d = INT32_MAX;
        int     best_label = -1;
        for (int j = 0; j < u->n_hit; j++) {
            int idx = u->hit_list[j];
            int32_t d = m4t_popcount_dist(
                query_sigs[m],
                table_train_sigs[m] + (size_t)idx * sig_bytes,
                mask, sig_bytes);
            if (d < best_d) {
                best_d = d;
                best_label = u->y_train[idx];
            }
        }
        if (best_label >= 0) label_votes[best_label]++;
    }
    int pred = 0;
    for (int c = 1; c < u->n_classes; c++)
        if (label_votes[c] > label_votes[pred]) pred = c;
    return pred;
}
