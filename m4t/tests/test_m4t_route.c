/*
 * test_m4t_route.c — tests for ternary routing primitives.
 *
 * Golden values are hand-derived. No float.
 */

#include "m4t_types.h"
#include "m4t_trit_pack.h"
#include "m4t_mtfp.h"
#include "m4t_route.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_EQ_I32(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %d, expected %d (line %d)\n", \
                (msg), (int)(actual), (int)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

#define ASSERT_EQ_I64(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %lld, expected %lld (line %d)\n", \
                (msg), (long long)(actual), (long long)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

/* ── threshold_extract ───────────────────────────────────────────────── */

/* tau=0 degenerate: exact sign-extraction semantics.
 * v > 0 → +1, v < 0 → -1, v == 0 → 0. */
static int test_threshold_extract_tau0(void) {
    int64_t values[7] = { 5, -3, 0, 100, -1, 0, 42 };
    uint8_t packed[M4T_TRIT_PACKED_BYTES(7)];
    m4t_route_threshold_extract(packed, values, 0, 7);

    m4t_trit_t result[7];
    m4t_unpack_trits_1d(result, packed, 7);

    const m4t_trit_t expected[7] = { 1, -1, 0, 1, -1, 0, 1 };
    for (int i = 0; i < 7; i++) {
        ASSERT_EQ_I32(result[i], expected[i], "threshold_extract tau=0");
    }
    return 0;
}

/* tau>0: values strictly above +tau → +1, strictly below -tau → -1,
 * |v| <= tau → 0. Tests boundary cases at ±tau (inclusive band). */
static int test_threshold_extract_tau5(void) {
    int64_t values[9] = { 6, 5, 4, 0, -4, -5, -6, 100, -100 };
    uint8_t packed[M4T_TRIT_PACKED_BYTES(9)];
    m4t_route_threshold_extract(packed, values, 5, 9);

    m4t_trit_t result[9];
    m4t_unpack_trits_1d(result, packed, 9);

    /*           6   5   4   0  -4  -5  -6  100 -100 */
    /* expect: +1   0   0   0   0   0  -1   +1  -1   */
    const m4t_trit_t expected[9] = { 1, 0, 0, 0, 0, 0, -1, 1, -1 };
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ_I32(result[i], expected[i], "threshold_extract tau=5");
    }
    return 0;
}

/* All inputs within the band → all zeros out. Emission coverage fails for
 * the +1 and -1 states here, as expected (this is exactly the input class
 * the primitive's contract documents as borderline). */
static int test_threshold_extract_all_within_band(void) {
    int64_t values[5] = { 3, -2, 0, 5, -5 };
    uint8_t packed[M4T_TRIT_PACKED_BYTES(5)];
    m4t_route_threshold_extract(packed, values, 5, 5);

    m4t_trit_t result[5];
    m4t_unpack_trits_1d(result, packed, 5);

    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(result[i], 0, "threshold_extract all within band");
    }
    return 0;
}

/* ── distance_batch ────────────────────────────────────────────────────── */

static int test_distance_batch(void) {
    /* 4 trits, 3 tiles.
     * query  = [+1, -1, +1, 0]
     * tile 0 = [+1, -1, +1, 0]  → distance = 0 (identical)
     * tile 1 = [-1, +1, -1, 0]  → all 3 nonzero trits differ
     * tile 2 = [+1, -1, 0,  0]  → trit 2 differs (+1 vs 0)
     *
     * Popcount distance counts mismatching BITS in the packed encoding.
     * query packed:  01 10 01 00 = 0x19
     * tile0 packed:  01 10 01 00 = 0x19  → XOR = 0x00, popcount = 0
     * tile1 packed:  10 01 10 00 = 0x26  → XOR = 0x3F, popcount = 6
     * tile2 packed:  01 10 00 00 = 0x09  → XOR = 0x10, popcount = 1
     */
    enum { D = 4, T = 3 };

    m4t_trit_t q[4]  = { 1, -1,  1, 0 };
    m4t_trit_t t0[4] = { 1, -1,  1, 0 };
    m4t_trit_t t1[4] = {-1,  1, -1, 0 };
    m4t_trit_t t2[4] = { 1, -1,  0, 0 };

    uint8_t qp[M4T_TRIT_PACKED_BYTES(D)];
    uint8_t tp[T * M4T_TRIT_PACKED_BYTES(D)];
    uint8_t mask[M4T_TRIT_PACKED_BYTES(D)];
    memset(mask, 0xFF, sizeof(mask));

    m4t_pack_trits_1d(qp, q, D);
    m4t_pack_trits_1d(tp + 0 * M4T_TRIT_PACKED_BYTES(D), t0, D);
    m4t_pack_trits_1d(tp + 1 * M4T_TRIT_PACKED_BYTES(D), t1, D);
    m4t_pack_trits_1d(tp + 2 * M4T_TRIT_PACKED_BYTES(D), t2, D);

    int32_t dist[T];
    m4t_route_distance_batch(dist, qp, tp, mask, T, D);

    ASSERT_EQ_I32(dist[0], 0, "dist tile0 (identical)");
    ASSERT_EQ_I32(dist[1], 6, "dist tile1 (all differ)");
    ASSERT_EQ_I32(dist[2], 1, "dist tile2 (one differs)");
    return 0;
}

/* ── topk_abs ──────────────────────────────────────────────────────────── */

static int test_topk_abs_basic(void) {
    /* T=4 tiles, scores = [3, -7, 1, -5].
     * |scores| = [3, 7, 1, 5].
     * Top-2: tile 1 (|7|, sign=-1), tile 3 (|5|, sign=-1).
     */
    int32_t scores[4] = { 3, -7, 1, -5 };
    m4t_route_decision_t decisions[2];
    m4t_route_topk_abs(decisions, scores, 4, 2);

    ASSERT_EQ_I32(decisions[0].tile_idx, 1, "topk[0] idx");
    ASSERT_EQ_I32(decisions[0].sign, -1, "topk[0] sign");
    ASSERT_EQ_I32(decisions[1].tile_idx, 3, "topk[1] idx");
    ASSERT_EQ_I32(decisions[1].sign, -1, "topk[1] sign");
    return 0;
}

static int test_topk_abs_with_zeros(void) {
    /* T=4, scores = [0, 5, 0, 0], k=3.
     * Only tile 1 has nonzero score. Remaining decisions are sentinels. */
    int32_t scores[4] = { 0, 5, 0, 0 };
    m4t_route_decision_t decisions[3];
    m4t_route_topk_abs(decisions, scores, 4, 3);

    ASSERT_EQ_I32(decisions[0].tile_idx, 1, "topk-zeros[0] idx");
    ASSERT_EQ_I32(decisions[0].sign, 1, "topk-zeros[0] sign");
    ASSERT_EQ_I32(decisions[1].tile_idx, -1, "topk-zeros[1] sentinel");
    ASSERT_EQ_I32(decisions[2].tile_idx, -1, "topk-zeros[2] sentinel");
    return 0;
}

static int test_topk_abs_all_tiles(void) {
    /* k == T: select all. */
    int32_t scores[3] = { -2, 3, -1 };
    m4t_route_decision_t decisions[3];
    m4t_route_topk_abs(decisions, scores, 3, 3);

    /* Order: |3|=3, |-2|=2, |-1|=1 → tiles 1, 0, 2 */
    ASSERT_EQ_I32(decisions[0].tile_idx, 1, "topk-all[0] idx");
    ASSERT_EQ_I32(decisions[0].sign, 1, "topk-all[0] sign");
    ASSERT_EQ_I32(decisions[1].tile_idx, 0, "topk-all[1] idx");
    ASSERT_EQ_I32(decisions[1].sign, -1, "topk-all[1] sign");
    ASSERT_EQ_I32(decisions[2].tile_idx, 2, "topk-all[2] idx");
    ASSERT_EQ_I32(decisions[2].sign, -1, "topk-all[2] sign");
    return 0;
}

/* ── apply_signed ──────────────────────────────────────────────────────── */

static int test_apply_signed(void) {
    /* 2 tiles, dim=4. tile_outs:
     *   tile 0: [10, 20, 30, 40]
     *   tile 1: [1,  2,  3,  4]
     *
     * Decisions: tile 0 sign=+1, tile 1 sign=-1.
     * result = +[10,20,30,40] - [1,2,3,4] = [9, 18, 27, 36]
     */
    const int D = 4;
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    m4t_mtfp_t tile_outs[8] = {
        10*S, 20*S, 30*S, 40*S,
         1*S,  2*S,  3*S,  4*S
    };

    m4t_route_decision_t decisions[2] = {
        { .tile_idx = 0, .sign =  1 },
        { .tile_idx = 1, .sign = -1 }
    };

    m4t_mtfp_t result[4];
    m4t_mtfp_vec_zero(result, D);
    m4t_route_apply_signed(result, tile_outs, decisions, 2, D);

    ASSERT_EQ_I32(result[0],  9*S, "apply[0]");
    ASSERT_EQ_I32(result[1], 18*S, "apply[1]");
    ASSERT_EQ_I32(result[2], 27*S, "apply[2]");
    ASSERT_EQ_I32(result[3], 36*S, "apply[3]");
    return 0;
}

static int test_apply_signed_sentinel(void) {
    /* Decision with tile_idx=-1 is skipped. */
    const int D = 2;
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    m4t_mtfp_t tile_outs[2] = { 5*S, 7*S };

    m4t_route_decision_t decisions[2] = {
        { .tile_idx =  0, .sign = 1 },
        { .tile_idx = -1, .sign = 0 }
    };

    m4t_mtfp_t result[2] = { 0, 0 };
    m4t_route_apply_signed(result, tile_outs, decisions, 2, D);

    ASSERT_EQ_I32(result[0], 5*S, "apply-sent[0]");
    ASSERT_EQ_I32(result[1], 7*S, "apply-sent[1]");
    return 0;
}

/* ── signature_update ──────────────────────────────────────────────────── */

static int test_signature_update(void) {
    /* T=2 tiles, H=3 hidden rows, D=4 dims.
     *
     * Tile 0 weights (3 rows × 4 cols):
     *   [+1, +1, -1, 0]
     *   [+1, -1, -1, 0]
     *   [+1,  0, +1, 0]
     * Column sums: [3, 0, -1, 0]
     *
     * Tile 1 weights:
     *   [-1, +1, +1, +1]
     *   [-1, +1, -1, +1]
     *   [ 0, +1, +1, -1]
     * Column sums: [-2, 3, 1, 1]
     *
     * Means per dim: [(3+(-2))/2, (0+3)/2, (-1+1)/2, (0+1)/2]
     *              = [0, 1, 0, 0]  (integer division)
     *
     * Differences:
     *   Tile 0: [3-0, 0-1, -1-0, 0-0] = [3, -1, -1, 0]
     *   Tile 1: [-2-0, 3-1, 1-0, 1-0] = [-2, 2, 1, 1]
     *
     * Signs:
     *   Tile 0: [+1, -1, -1, 0]
     *   Tile 1: [-1, +1, +1, +1]
     */
    enum { T = 2, H = 3, D = 4 };
    int Dp = M4T_TRIT_PACKED_BYTES(D);

    m4t_trit_t w0[12] = {
         1,  1, -1,  0,
         1, -1, -1,  0,
         1,  0,  1,  0
    };
    m4t_trit_t w1[12] = {
        -1,  1,  1,  1,
        -1,  1, -1,  1,
         0,  1,  1, -1
    };

    uint8_t weights[T * H * Dp];
    for (int h = 0; h < H; h++)
        m4t_pack_trits_1d(weights + (0 * H + h) * Dp, w0 + h * D, D);
    for (int h = 0; h < H; h++)
        m4t_pack_trits_1d(weights + (1 * H + h) * Dp, w1 + h * D, D);

    uint8_t signatures[T * Dp];
    int64_t scratch[(T + 1) * D];

    m4t_route_signature_update(signatures, weights, scratch, T, H, D);

    m4t_trit_t sig0[D], sig1[D];
    m4t_unpack_trits_1d(sig0, signatures + 0 * Dp, D);
    m4t_unpack_trits_1d(sig1, signatures + 1 * Dp, D);

    const m4t_trit_t exp0[4] = { 1, -1, -1, 0 };
    const m4t_trit_t exp1[4] = {-1,  1,  1, 1 };

    for (int d = 0; d < D; d++) {
        ASSERT_EQ_I32(sig0[d], exp0[d], "sig0");
        ASSERT_EQ_I32(sig1[d], exp1[d], "sig1");
    }
    return 0;
}

/* ── End-to-end mini routing pass ──────────────────────────────────────── */

static int test_route_e2e(void) {
    /* T=2, k=1, D=4. Query matches tile 0 exactly.
     *
     * Sigs: tile0=[+1,-1,+1,+1], tile1=[-1,+1,-1,-1], query=tile0.
     * Packed: query=0x59, tile0=0x59, tile1=0xA6.
     * Distances: XOR popcount → dist0=0, dist1=8.
     *
     * Convert distance to affinity: score = -distance.
     * scores = [0, -8]. topk_abs picks tile 1 (|score|=8, sign=-1).
     * The -1 sign means "anti-expert" — this tile is maximally different.
     *
     * Tile 1 output: [100, 200, 300, 400] * S.
     * apply_signed with sign=-1: result = -tile1_out. */
    enum { T = 2, D = 4 };
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    int Dp = M4T_TRIT_PACKED_BYTES(D);

    m4t_trit_t q_trits[4]  = { 1, -1,  1,  1 };
    m4t_trit_t t0_trits[4] = { 1, -1,  1,  1 };
    m4t_trit_t t1_trits[4] = {-1,  1, -1, -1 };

    uint8_t qp[1], tp[2];
    uint8_t mask[1] = { 0xFF };
    m4t_pack_trits_1d(qp, q_trits, D);
    m4t_pack_trits_1d(tp + 0, t0_trits, D);
    m4t_pack_trits_1d(tp + 1 * Dp, t1_trits, D);

    /* Step 1: distances */
    int32_t dist[T];
    m4t_route_distance_batch(dist, qp, tp, mask, T, D);
    ASSERT_EQ_I32(dist[0], 0, "e2e dist0");
    ASSERT_EQ_I32(dist[1], 8, "e2e dist1");

    /* Step 2: convert distance → affinity (negate), then topk */
    int32_t scores[T];
    for (int t = 0; t < T; t++) scores[t] = -dist[t];

    m4t_route_decision_t decisions[1];
    m4t_route_topk_abs(decisions, scores, T, 1);
    ASSERT_EQ_I32(decisions[0].tile_idx, 1, "e2e topk idx");
    ASSERT_EQ_I32(decisions[0].sign, -1, "e2e topk sign (anti-expert)");

    /* Step 3: tile outputs */
    m4t_mtfp_t tile_outs[8] = {
        0, 0, 0, 0,
        100*S, 200*S, 300*S, 400*S
    };

    /* Step 4: apply with sign=-1 → subtract */
    m4t_mtfp_t result[4] = { 0, 0, 0, 0 };
    m4t_route_apply_signed(result, tile_outs, decisions, 1, D);

    ASSERT_EQ_I32(result[0], -100*S, "e2e result[0] (anti)");
    ASSERT_EQ_I32(result[1], -200*S, "e2e result[1] (anti)");
    ASSERT_EQ_I32(result[2], -300*S, "e2e result[2] (anti)");
    ASSERT_EQ_I32(result[3], -400*S, "e2e result[3] (anti)");
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    if (test_threshold_extract_tau0())             return 1;
    if (test_threshold_extract_tau5())             return 1;
    if (test_threshold_extract_all_within_band())  return 1;
    if (test_distance_batch())        return 1;
    if (test_topk_abs_basic())        return 1;
    if (test_topk_abs_with_zeros())   return 1;
    if (test_topk_abs_all_tiles())    return 1;
    if (test_apply_signed())          return 1;
    if (test_apply_signed_sentinel()) return 1;
    if (test_signature_update())      return 1;
    if (test_route_e2e())             return 1;
    printf("m4t_route: all tests passed\n");
    return 0;
}
