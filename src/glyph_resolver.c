/*
 * glyph_resolver.c — candidate-set resolver implementations.
 */

#include "glyph_resolver.h"
#include "m4t_trit_pack.h"

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define GLYPH_RESOLVER_HAS_NEON 1
#else
#define GLYPH_RESOLVER_HAS_NEON 0
#endif

int glyph_resolver_vote(const glyph_union_t* u) {
    assert(u->n_classes > 0 && u->n_classes <= GLYPH_MAX_CLASSES);
    int class_votes[GLYPH_MAX_CLASSES];
    memset(class_votes, 0, (size_t)u->n_classes * sizeof(int));
    for (int j = 0; j < u->n_hit; j++) {
        int idx = u->hit_list[j];
        int label = u->y_train[idx];
        /* Label must be in [0, n_classes). Silently clamping would
         * hide dataset corruption; assert instead. */
        assert(label >= 0 && label < u->n_classes);
        class_votes[label] += u->votes[idx];
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

/* NEON-batched SUM resolver for sig_bytes=4.
 *
 * Layout of the hot loop:
 *
 *   for each group of 4 consecutive candidates in hit_list:
 *       for each table m:
 *           gather the 4 candidates' 4-byte sigs from train_sigs[m]
 *               via 4 scalar loads into a contiguous 16-byte buffer
 *           broadcast the query's 4-byte sig at table m to a uint32x4_t
 *           XOR the gathered vector with the broadcast vector
 *           VCNT → uint8x16_t of per-byte popcounts (0..8 each)
 *           pairwise-sum twice (vpaddlq_u8 → vpaddlq_u16) to collapse
 *               each 4-byte lane into a single u32 popcount
 *           add the 4 per-candidate partial scores to 4 local accumulators
 *       argmin the 4 scores against the running best
 *
 *   scalar tail for (n_hit % 4) candidates uses the per-call
 *       __builtin_popcount path via a tight inlined helper
 *
 * Gather cost: each 4-candidate group needs 4 × scalar 4-byte loads
 * into a contiguous local buffer (16 bytes). At M=64 that's 64 gathers
 * per group × 4 scalar loads each = 256 scalar loads per 4 candidates.
 * Compared to the baseline 64 × 4 = 256 independent popcount_dist
 * calls, this is the same number of loads but the downstream work
 * collapses to 1 NEON op per 4 candidates.
 *
 * The mask is honored at the per-byte level — it's broadcast into the
 * NEON register the same way the query signature is, and ANDed with
 * the XOR result before VCNT. This is bit-exact equivalent to
 * m4t_popcount_dist for every mask value, not just all-ones.
 */
int glyph_resolver_sum_neon4(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask)
{
    if (sig_bytes != 4) return -1;

#if !GLYPH_RESOLVER_HAS_NEON
    /* On a non-NEON target, fall back to the scalar SUM resolver.
     * The equivalence test covers both paths. */
    return glyph_resolver_sum(
        u, m_active, sig_bytes, table_train_sigs, query_sigs, mask);
#else
    assert(mask != NULL);

    /* Broadcast the mask to all four 4-byte lanes once per query
     * (mask is invariant across candidates and tables). */
    uint32_t m32;
    memcpy(&m32, mask, 4);
    uint32x4_t mbcast = vdupq_n_u32(m32);
    uint8x16_t mbytes = vreinterpretq_u8_u32(mbcast);

    int32_t best_score = INT32_MAX;
    int     best_label = -1;

    /* Contiguous 16-byte scratch buffer for gathered candidate sigs. */
    uint8_t cand_buf[16] __attribute__((aligned(16)));

    int j = 0;
    const int n_hit = u->n_hit;

    /* Main loop: 4 candidates per iteration. */
    for (; j + 4 <= n_hit; j += 4) {
        const int32_t idx0 = u->hit_list[j + 0];
        const int32_t idx1 = u->hit_list[j + 1];
        const int32_t idx2 = u->hit_list[j + 2];
        const int32_t idx3 = u->hit_list[j + 3];

        /* Accumulators for the 4 candidates' scores. */
        uint32x4_t scores = vdupq_n_u32(0);

        for (int m = 0; m < m_active; m++) {
            const uint8_t* tsigs = table_train_sigs[m];

            /* Gather 4 candidates' 4-byte sigs from table m into
             * cand_buf (16 bytes). Four scalar 4-byte loads. */
            memcpy(cand_buf +  0, tsigs + (size_t)idx0 * 4, 4);
            memcpy(cand_buf +  4, tsigs + (size_t)idx1 * 4, 4);
            memcpy(cand_buf +  8, tsigs + (size_t)idx2 * 4, 4);
            memcpy(cand_buf + 12, tsigs + (size_t)idx3 * 4, 4);

            /* Load the 4-candidate gather into a NEON register. */
            uint8x16_t cands = vld1q_u8(cand_buf);

            /* Broadcast the query's 4-byte sig to all 4 lanes. */
            uint32_t q32;
            memcpy(&q32, query_sigs[m], 4);
            uint32x4_t qbcast = vdupq_n_u32(q32);
            uint8x16_t qbytes = vreinterpretq_u8_u32(qbcast);

            /* XOR to get per-byte bit differences, then AND with the
             * broadcast mask so masked-off bits never contribute to
             * the popcount — honors the general-case API contract. */
            uint8x16_t diff = veorq_u8(cands, qbytes);
            diff = vandq_u8(diff, mbytes);

            /* VCNT: per-byte popcount. Result: uint8x16_t where each
             * byte holds its own popcount in [0, 8]. */
            uint8x16_t cnt8 = vcntq_u8(diff);

            /* Horizontal sum within each 4-byte lane. Two pairwise
             * widening adds collapse 16 u8 values into 4 u32 values,
             * each being the sum of 4 adjacent bytes. */
            uint16x8_t cnt16 = vpaddlq_u8(cnt8);
            uint32x4_t cnt32 = vpaddlq_u16(cnt16);

            /* Accumulate into per-candidate scores. */
            scores = vaddq_u32(scores, cnt32);
        }

        /* Extract the 4 candidate scores and argmin against best. */
        uint32_t s[4];
        vst1q_u32(s, scores);

        const int32_t idxs[4] = {idx0, idx1, idx2, idx3};
        for (int k = 0; k < 4; k++) {
            if ((int32_t)s[k] < best_score) {
                best_score = (int32_t)s[k];
                best_label = u->y_train[idxs[k]];
            }
        }
    }

    /* Scalar tail for the last n_hit % 4 candidates. Reuses the
     * optimized m4t_popcount_dist path from Fix 1 (builtin popcount
     * for 4-byte sigs). */
    for (; j < n_hit; j++) {
        int idx = u->hit_list[j];
        int32_t score = 0;
        for (int m = 0; m < m_active; m++) {
            score += m4t_popcount_dist(
                query_sigs[m],
                table_train_sigs[m] + (size_t)idx * 4,
                mask, 4);
        }
        if (score < best_score) {
            best_score = score;
            best_label = u->y_train[idx];
        }
    }

    return best_label;
#endif
}

int glyph_resolver_sum_voteweighted(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask)
{
    /* Integer scaling factor to keep the division (sum_dist /
     * (1 + votes)) in int64 without losing precision at the argmin
     * tie-break. 1024 is enough headroom: sum_dist ≤ M × 2 × N_PROJ
     * (bounded by max Hamming distance per table × tables), so for
     * M=64, N_PROJ=16 the max is 2048. 2048 × 1024 = 2^21, well
     * within int64. */
    enum { VW_SCALE = 1024 };

    int64_t best_score = INT64_MAX;
    int     best_label = -1;
    for (int j = 0; j < u->n_hit; j++) {
        int idx = u->hit_list[j];
        int32_t sum_dist = 0;
        for (int m = 0; m < m_active; m++) {
            sum_dist += m4t_popcount_dist(
                query_sigs[m],
                table_train_sigs[m] + (size_t)idx * sig_bytes,
                mask, sig_bytes);
        }
        /* Amortize by (1 + votes). +1 guards against vote=0 and
         * ensures the ranking is well-defined for any union. */
        int64_t denom = 1 + (int64_t)u->votes[idx];
        int64_t score = ((int64_t)sum_dist * VW_SCALE) / denom;
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
    assert(u->n_classes > 0 && u->n_classes <= GLYPH_MAX_CLASSES);
    int label_votes[GLYPH_MAX_CLASSES];
    memset(label_votes, 0, (size_t)u->n_classes * sizeof(int));
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
