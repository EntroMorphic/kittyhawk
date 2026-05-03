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

/* ── threshold_extract ─────────────────────────────────────────────────
 *
 * §18 coverage test group — output-side emission coverage for
 * m4t_route_threshold_extract. The three test functions below collectively
 * verify that all three output codes ({+1, 0, -1}) are emitted under the
 * sanctioned input classes (tau=0 + integer-zero-realizing inputs, tau>0
 * + values spanning ±tau, plus the band-only-inputs defensive case). */

/* §18 coverage test: tau=0 degenerate path. Input [5, -3, 0, 100, -1, 0, 42]
 * realizes all three output codes via exact zeros in the input. */
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

/* §18 coverage test: tau>0 path — the primary sanctioned deployment.
 * Values strictly above +tau → +1, strictly below -tau → -1, |v| <= tau → 0.
 * Tests boundary cases at ±tau (inclusive band). All three output codes
 * realized under the sanctioned input class. */
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

/* §18 coverage test: defensive — demonstrates that when the sanctioned
 * input class is violated (all inputs within the band), the primitive
 * correctly produces only the zero output code. Not an assertion of
 * coverage, but a verification that the primitive's behavior is
 * deterministic and spec-compliant at the coverage boundary. */
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

/* Edge case: n = 0. Must not write to dst, must not crash. */
static int test_threshold_extract_n_zero(void) {
    uint8_t packed[1] = { 0xFF };
    int64_t values[1] = { 42 };
    m4t_route_threshold_extract(packed, values, 0, 0);
    ASSERT_EQ_I32(packed[0], 0xFF, "threshold_extract n=0 must not touch dst");
    return 0;
}

/* Pack-byte boundary coverage: n values at {3, 5, 7, 8} exercise the bit
 * placement across byte edges. Each trit pair lives at a specific bit
 * position within a byte; this test verifies placement is correct at and
 * across byte boundaries. */
static int test_threshold_extract_pack_boundaries(void) {
    /* Pattern: alternating +1, -1 with a zero in the middle. */
    const int64_t src[8] = { 10, -10, 10, 0, -10, 10, -10, 10 };
    const m4t_trit_t expected[8] = { 1, -1, 1, 0, -1, 1, -1, 1 };

    int sizes[] = { 3, 5, 7, 8 };
    for (size_t s_idx = 0; s_idx < sizeof(sizes)/sizeof(sizes[0]); s_idx++) {
        int n = sizes[s_idx];
        uint8_t packed[M4T_TRIT_PACKED_BYTES(8)];
        m4t_route_threshold_extract(packed, src, 0, n);

        m4t_trit_t result[8];
        m4t_unpack_trits_1d(result, packed, n);

        for (int i = 0; i < n; i++) {
            if (result[i] != expected[i]) {
                fprintf(stderr,
                    "FAIL: pack_boundaries n=%d i=%d got %+d expected %+d (line %d)\n",
                    n, i, (int)result[i], (int)expected[i], __LINE__);
                return 1;
            }
        }
    }
    return 0;
}

/* Extreme value coverage: INT64_MAX and INT64_MIN+1 on both sides of
 * several tau values. Avoids INT64_MIN (per L-RT1A: -INT64_MIN is UB
 * even though tau ≥ 0 is the documented precondition, values can legally
 * be any int64; we exclude INT64_MIN from the test because it isn't a
 * realistic MTFP mantissa anyway and the comparison is well-defined
 * regardless: INT64_MIN < -tau for any tau ≥ 0, so result is -1). */
static int test_threshold_extract_extremes(void) {
    /* tau = 0: sign of INT64_MAX is +1, sign of INT64_MIN+1 is -1. */
    {
        int64_t vals[2] = { INT64_MAX, INT64_MIN + 1 };
        uint8_t packed[M4T_TRIT_PACKED_BYTES(2)];
        m4t_route_threshold_extract(packed, vals, 0, 2);
        m4t_trit_t r[2]; m4t_unpack_trits_1d(r, packed, 2);
        ASSERT_EQ_I32(r[0],  1, "extremes tau=0 INT64_MAX");
        ASSERT_EQ_I32(r[1], -1, "extremes tau=0 INT64_MIN+1");
    }
    /* tau = 1,000,000: same extremes, same expected output. */
    {
        int64_t vals[2] = { INT64_MAX, INT64_MIN + 1 };
        uint8_t packed[M4T_TRIT_PACKED_BYTES(2)];
        m4t_route_threshold_extract(packed, vals, 1000000, 2);
        m4t_trit_t r[2]; m4t_unpack_trits_1d(r, packed, 2);
        ASSERT_EQ_I32(r[0],  1, "extremes tau=1e6 INT64_MAX");
        ASSERT_EQ_I32(r[1], -1, "extremes tau=1e6 INT64_MIN+1");
    }
    /* Also verify INT64_MIN (allowed input value; -tau ≥ 0 so comparison
     * v < -tau is well-defined for INT64_MIN regardless). */
    {
        int64_t vals[1] = { INT64_MIN };
        uint8_t packed[M4T_TRIT_PACKED_BYTES(1)];
        m4t_route_threshold_extract(packed, vals, 0, 1);
        m4t_trit_t r[1]; m4t_unpack_trits_1d(r, packed, 1);
        ASSERT_EQ_I32(r[0], -1, "extremes tau=0 INT64_MIN");
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

/* ── topk_abs ──────────────────────────────────────────────────────────
 *
 * §18 coverage test group — output-side emission coverage for the
 * decision.sign field of m4t_route_topk_abs. The three test functions
 * below collectively verify that all three output sign states
 * ({+1, -1, 0-sentinel}) are emitted under the sanctioned input class. */

/* §18 coverage test: exercises the -1 sign state (two negative-score tiles
 * selected). Combined with test_topk_abs_all_tiles, covers {+1, -1}. */
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

/* §18 coverage test: exercises the 0-sentinel sign state (fewer nonzero
 * tiles than k → remaining decisions are sentinels with sign=0). */
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

/* §18 coverage test: exercises both +1 and -1 sign states in a single
 * call (mixed-sign scores, k == T). */
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

/* ── apply_signed ──────────────────────────────────────────────────────
 *
 * §18 coverage test group — input-side emission coverage for
 * m4t_route_apply_signed. Three-way branch driven by decision.sign:
 * +1 → add, -1 → sub, 0/sentinel → skip. The two tests together
 * exercise all three branches. */

/* §18 coverage test: exercises +1 (add) and -1 (sub) branches. */
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

/* §18 coverage test: exercises the 0-sentinel (skip) branch. */
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

/* ── signature_update ──────────────────────────────────────────────────
 *
 * §18 coverage test: compound primitive that internally uses
 * threshold_extract(tau=0) on col_sum − mean. Sanctioned input class is
 * integer weight matrices where col_sum-vs-mean can realize exact equality.
 * Expected signatures below include all three trit states, verifying
 * coverage end-to-end. */
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

/* ── Emission-coverage helper ──────────────────────────────────────────── */

/* Verifies that decisions_emit_coverage correctly reports which sign states
 * were realized. This is the §18 testability primitive — consumers integrate
 * it to demonstrate the input-class contract is honored at the call site. */
static int test_decisions_emit_coverage(void) {
    /* All three states present. */
    {
        m4t_route_decision_t d[3] = {
            { .tile_idx = 0,  .sign =  1 },
            { .tile_idx = 1,  .sign = -1 },
            { .tile_idx = -1, .sign =  0 },
        };
        int has_pos = -1, has_neg = -1, has_zero = -1;
        m4t_route_decisions_emit_coverage(d, 3, &has_pos, &has_neg, &has_zero);
        if (!(has_pos == 1 && has_neg == 1 && has_zero == 1)) {
            printf("FAIL: emit_coverage all-three\n"); return 1;
        }
    }

    /* Only positive. */
    {
        m4t_route_decision_t d[2] = {
            { .tile_idx = 0, .sign = 1 },
            { .tile_idx = 1, .sign = 1 },
        };
        int has_pos = -1, has_neg = -1, has_zero = -1;
        m4t_route_decisions_emit_coverage(d, 2, &has_pos, &has_neg, &has_zero);
        if (!(has_pos == 1 && has_neg == 0 && has_zero == 0)) {
            printf("FAIL: emit_coverage pos-only\n"); return 1;
        }
    }

    /* k = 0 → all states absent. */
    {
        int has_pos = -1, has_neg = -1, has_zero = -1;
        m4t_route_decisions_emit_coverage(NULL, 0, &has_pos, &has_neg, &has_zero);
        if (!(has_pos == 0 && has_neg == 0 && has_zero == 0)) {
            printf("FAIL: emit_coverage empty\n"); return 1;
        }
    }

    /* NULL out-pointers are skipped without crashing. */
    {
        m4t_route_decision_t d[1] = { { .tile_idx = 0, .sign = 1 } };
        m4t_route_decisions_emit_coverage(d, 1, NULL, NULL, NULL);
    }

    return 0;
}

/* ── wildcard_dist (§19) ───────────────────────────────────────────────
 *
 * §19 zero-state semantic test group. The wildcard kernel uses (II)
 * Wildcard interpretation for tile-side zeros and (III) Abstain for
 * query-side zeros — distinct from m4t_popcount_dist's (I) Tie-
 * cancellation symmetric semantic. §19.6 review gate requires:
 *   (a) declared zero-state interpretation in docstring (done in m4t_route.h)
 *   (b) behavior-difference test demonstrating different output for
 *       same input vs the existing kernel (test_wildcard_vs_hamming below)
 *   (c) §19 audit table entry (done in m4t/docs/M4T_SUBSTRATE.md §19.4)
 *   (d) sanctioned-pairing constraint — documented in m4t_route.h
 */

/* Build a single-byte packed signature from an array of 4 trits in {-1,0,+1}.
 * trits[0] occupies bits 0-1, trits[1] bits 2-3, etc. */
static uint8_t pack4(const m4t_trit_t* trits) {
    uint8_t b = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t code = (trits[i] == 1)  ? 0x01u :
                       (trits[i] == -1) ? 0x02u :
                                          0x00u;
        b |= (uint8_t)(code << (i * 2));
    }
    return b;
}

/* Cost-table verification: every (q, t) ∈ {-1, 0, +1}² combination
 * produces the documented wildcard cost. */
static int test_wildcard_dist_cost_table(void) {
    /* Build two single-byte signatures (4 trits each) covering every
     * needed (q, t) pair. Use a separate test per pair for clarity. */
    int cases[][3] = {
        /*  q,  t, expected_cost */
        {  1,  1, 0 },   /*  full match  */
        { -1, -1, 0 },   /*  full match  */
        {  0,  0, 0 },   /*  mutual abstention */
        {  1,  0, 0 },   /*  WILDCARD: tile-zero, query-+1 → free match */
        { -1,  0, 0 },   /*  WILDCARD: tile-zero, query--1 → free match */
        {  0,  1, 1 },   /*  query abstains, tile asserts */
        {  0, -1, 1 },   /*  query abstains, tile asserts */
        {  1, -1, 2 },   /*  full mismatch */
        { -1,  1, 2 },   /*  full mismatch */
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    for (int c = 0; c < n_cases; c++) {
        m4t_trit_t q_trits[4] = { (m4t_trit_t)cases[c][0], 0, 0, 0 };
        m4t_trit_t t_trits[4] = { (m4t_trit_t)cases[c][1], 0, 0, 0 };
        uint8_t q_packed = pack4(q_trits);
        uint8_t t_packed = pack4(t_trits);
        /* Mask: only position 0 is active (sig_dim = 1, packed_bytes = 1
         * with low 2 bits set). Other 3 fields in the byte are masked off. */
        uint8_t mask = 0x03u;  /* low 2 bits active = position 0 only */
        int32_t got = m4t_route_wildcard_dist(&q_packed, &t_packed, &mask, 1);
        ASSERT_EQ_I32(got, cases[c][2], "wildcard cost-table cell");
    }
    return 0;
}

/* §19.6(b) behavior-difference test: same inputs MUST produce different
 * results from m4t_popcount_dist where the wildcard interpretation
 * differs. Specifically: q=±1, t=0 costs 1 in popcount_dist but 0 in
 * wildcard_dist. */
static int test_wildcard_vs_hamming_behavior_diff(void) {
    /* Signature: 8 trits, all positions set up so that tile has 4 zeros
     * and query has 4 ±1s aligned with them. Standard Hamming = 4
     * (one cost-1 per tile-zero-vs-query-±1). Wildcard = 0 (all are
     * wildcards). */
    m4t_trit_t q_trits[8] = {  1, -1,  1, -1,  1, -1,  1, -1 };
    m4t_trit_t t_trits[8] = {  0,  0,  0,  0,  0,  0,  0,  0 };
    uint8_t q_packed[2] = { pack4(q_trits), pack4(q_trits + 4) };
    uint8_t t_packed[2] = { pack4(t_trits), pack4(t_trits + 4) };
    uint8_t mask[2] = { 0xFFu, 0xFFu };

    int32_t hamming  = m4t_popcount_dist(q_packed, t_packed, mask, 2);
    int32_t wildcard = m4t_route_wildcard_dist(q_packed, t_packed, mask, 8);

    ASSERT_EQ_I32(hamming,  8, "popcount_dist on q=±1 t=0  — should be 8 (4 cells × cost 1, but actually...)");
    /* WAIT: ternary Hamming costs (q=±1, t=0) at popcount(XOR & mask)
     * where XOR(0b01, 0b00) = 0b01 (popcount=1) and XOR(0b10, 0b00) =
     * 0b10 (popcount=1). So 8 trits × cost 1 each = 8. ✓ */
    ASSERT_EQ_I32(wildcard, 0, "wildcard_dist on q=±1 t=0 — all wildcard matches, cost 0");
    /* Behavior difference: hamming=8, wildcard=0. The kernels are
     * operationally distinct on this input. §19.6(b) gate satisfied. */
    return 0;
}

/* Equivalence test: when tile has NO zeros (all ±1), wildcard and
 * Hamming must produce identical results — the wildcard correction
 * is zero by construction. */
static int test_wildcard_equals_hamming_no_tile_zeros(void) {
    m4t_trit_t q_trits[16] = {  1, -1,  1,  0,  1, -1,  0,  1,
                                 -1, -1,  1,  1,  0,  1, -1,  0 };
    m4t_trit_t t_trits[16] = {  1,  1, -1,  1, -1, -1,  1, -1,
                                  1, -1,  1,  1, -1,  1,  1, -1 };
    uint8_t q_packed[4], t_packed[4];
    for (int i = 0; i < 4; i++) {
        q_packed[i] = pack4(q_trits + i*4);
        t_packed[i] = pack4(t_trits + i*4);
    }
    uint8_t mask[4] = { 0xFFu, 0xFFu, 0xFFu, 0xFFu };
    int32_t hamming  = m4t_popcount_dist(q_packed, t_packed, mask, 4);
    int32_t wildcard = m4t_route_wildcard_dist(q_packed, t_packed, mask, 16);
    ASSERT_EQ_I32(wildcard, hamming, "wildcard==hamming when tile has no zeros");
    return 0;
}

/* Mask-respect test: positions outside the mask must contribute zero
 * to both kernels, including the wildcard correction. */
static int test_wildcard_respects_mask(void) {
    /* sig_dim=4. Position 0: q=+1, t=0 (would be wildcard correction).
     * Positions 1-3: q=±1, t=±1 mismatches (would contribute Hamming). */
    m4t_trit_t q_trits[4] = {  1,  1,  1,  1 };
    m4t_trit_t t_trits[4] = {  0, -1, -1, -1 };
    uint8_t q_packed = pack4(q_trits);
    uint8_t t_packed = pack4(t_trits);

    /* Mask only position 0 active. */
    uint8_t mask_pos0 = 0x03u;
    int32_t got = m4t_route_wildcard_dist(&q_packed, &t_packed, &mask_pos0, 1);
    /* With only position 0 active and that being q=+1 t=0 (wildcard match),
     * cost should be 0. */
    ASSERT_EQ_I32(got, 0, "wildcard cost 0 with only wildcard position active");

    /* Mask only positions 1-3 active. */
    uint8_t mask_pos123 = 0xFCu;  /* 0b11111100 — fields 1, 2, 3 active */
    got = m4t_route_wildcard_dist(&q_packed, &t_packed, &mask_pos123, 4);
    /* Three full-mismatches (q=+1, t=-1) at cost 2 each = 6. */
    ASSERT_EQ_I32(got, 6, "wildcard cost from 3 full-mismatches with wildcard masked off");
    return 0;
}

/* Multi-byte test exercising the 8-byte and 4-byte loops. */
static int test_wildcard_multi_byte(void) {
    /* sig_dim = 64 → packed_bytes = 16. Build a signature where:
     *   - First 32 trits: tile=0 (wildcard), query alternates +1/-1.
     *     Hamming would be 32 (each cost 1); wildcard is 0.
     *   - Last 32 trits: q=t=+1 (full match). Both 0.
     * Wildcard total = 0; Hamming total = 32. */
    m4t_trit_t q_trits[64], t_trits[64];
    for (int i = 0; i < 32; i++) {
        q_trits[i] = (m4t_trit_t)((i & 1) ? -1 : 1);
        t_trits[i] = 0;
    }
    for (int i = 32; i < 64; i++) {
        q_trits[i] = 1;
        t_trits[i] = 1;
    }
    uint8_t q_packed[16], t_packed[16];
    for (int i = 0; i < 16; i++) {
        q_packed[i] = pack4(q_trits + i*4);
        t_packed[i] = pack4(t_trits + i*4);
    }
    uint8_t mask[16];
    memset(mask, 0xFFu, 16);

    int32_t hamming  = m4t_popcount_dist(q_packed, t_packed, mask, 16);
    int32_t wildcard = m4t_route_wildcard_dist(q_packed, t_packed, mask, 64);
    ASSERT_EQ_I32(hamming,  32, "multi-byte Hamming sanity");
    ASSERT_EQ_I32(wildcard,  0, "multi-byte wildcard total — all 32 trits wildcard-matched");
    return 0;
}

/* ── threshold_extract_dual + confidence_weighted_dist (P0-2) ──────────
 *
 * 5-state encoding via (trit, confidence) pair. Tests the cost-table
 * extension and the dual-extract output structure. */

static int test_threshold_extract_dual_basic(void) {
    /* Inputs spanning -strong, -weak, 0, +weak, +strong. */
    int64_t values[5] = { -100, -20, 0, 20, 100 };
    int64_t tau_weak = 10, tau_strong = 50;
    uint8_t trit_packed[2] = {0};
    uint8_t conf_bits = 0;

    m4t_route_threshold_extract_dual(trit_packed, &conf_bits,
                                       values, tau_weak, tau_strong, 5);

    /* Expected trits: -1, -1, 0, +1, +1 → codes 0b10, 0b10, 0b00, 0b01, 0b01. */
    /* Byte 0 (positions 0..3): 0b01 0b00 0b10 0b10 = 0b01_00_10_10 = 0x4A. */
    ASSERT_EQ_I32(trit_packed[0], 0x4Au, "dual-extract trit byte 0");
    /* Byte 1 (position 4): 0b01 in low bits = 0x01. */
    ASSERT_EQ_I32(trit_packed[1], 0x01u, "dual-extract trit byte 1");

    /* Expected conf bits: 1, 0, 0, 0, 1 → bits set at position 0, 4. */
    /* Bit 0 (position 0): 1; bit 1 (pos 1): 0; bit 4 (pos 4): 1.
     * Byte = 0b00010001 = 0x11. */
    ASSERT_EQ_I32(conf_bits, 0x11u, "dual-extract confidence byte");
    return 0;
}

static int test_confidence_weighted_dist_cost_table(void) {
    /* Build position-0-only tests for each (q_trit, t_trit, q_conf, t_conf)
     * combination of interest. Single-byte signatures; mask = 0x03 (low
     * 2 bits = position 0 active). */
    struct case_t {
        m4t_trit_t q_trit, t_trit;
        int q_conf, t_conf;
        int32_t expected_cost;
    } cases[] = {
        /* Same-sign agreement: cost 0 regardless of confidence */
        {  1,  1, 0, 0, 0 },
        {  1,  1, 1, 1, 0 },
        /* Mutual abstain */
        {  0,  0, 0, 0, 0 },
        /* Asymmetric abstain (current Hamming, cost 1) */
        {  1,  0, 1, 0, 1 },
        {  0,  1, 0, 1, 1 },
        /* Opposite-sign, no confidence: cost 2 (current Hamming) */
        {  1, -1, 0, 0, 2 },
        /* Opposite-sign, one confident: cost 3 */
        {  1, -1, 1, 0, 3 },
        {  1, -1, 0, 1, 3 },
        /* Opposite-sign, both confident: cost 4 */
        {  1, -1, 1, 1, 4 },
        { -1,  1, 1, 1, 4 },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int c = 0; c < n_cases; c++) {
        m4t_trit_t q_t[4] = { cases[c].q_trit, 0, 0, 0 };
        m4t_trit_t t_t[4] = { cases[c].t_trit, 0, 0, 0 };
        uint8_t q_packed = pack4(q_t);
        uint8_t t_packed = pack4(t_t);
        uint8_t q_conf_byte = (uint8_t)(cases[c].q_conf & 1u);
        uint8_t t_conf_byte = (uint8_t)(cases[c].t_conf & 1u);
        uint8_t mask = 0x03u;
        int32_t got = m4t_route_confidence_weighted_dist(
            &q_packed, &q_conf_byte,
            &t_packed, &t_conf_byte,
            &mask, 1);
        ASSERT_EQ_I32(got, cases[c].expected_cost, "confidence_dist case");
    }
    return 0;
}

/* §19.6(b) behavior-difference: same trit inputs, different confidence,
 * MUST produce different distances. The cost difference reveals that
 * the kernel uses confidence as a substrate-distinct signal. */
static int test_confidence_weighted_vs_hamming_behavior_diff(void) {
    /* Single position: q=+1, t=-1 (full mismatch).
     * Hamming costs 2; confidence weighting:
     *   no conf: 2; one conf: 3; both conf: 4.
     * So the kernel must produce different values for these inputs. */
    m4t_trit_t q[4] = {  1, 0, 0, 0 };
    m4t_trit_t t[4] = { -1, 0, 0, 0 };
    uint8_t q_packed = pack4(q);
    uint8_t t_packed = pack4(t);
    uint8_t mask = 0x03u;

    uint8_t qc_off = 0, tc_off = 0;
    uint8_t qc_on  = 1, tc_on  = 1;

    int32_t hamming  = m4t_popcount_dist(&q_packed, &t_packed, &mask, 1);
    int32_t conf_off = m4t_route_confidence_weighted_dist(
        &q_packed, &qc_off, &t_packed, &tc_off, &mask, 1);
    int32_t conf_one = m4t_route_confidence_weighted_dist(
        &q_packed, &qc_on,  &t_packed, &tc_off, &mask, 1);
    int32_t conf_both = m4t_route_confidence_weighted_dist(
        &q_packed, &qc_on,  &t_packed, &tc_on,  &mask, 1);

    ASSERT_EQ_I32(hamming,    2, "Hamming on full mismatch");
    ASSERT_EQ_I32(conf_off,   2, "conf-off matches Hamming on full mismatch");
    ASSERT_EQ_I32(conf_one,   3, "one-confident extends mismatch by 1");
    ASSERT_EQ_I32(conf_both,  4, "both-confident extends mismatch by 2");
    /* Behavior-difference: conf_one != hamming, conf_both != hamming. */
    if (conf_one == hamming || conf_both == hamming) {
        fprintf(stderr, "FAIL: conf kernel didn't differentiate from Hamming\n");
        return 1;
    }
    return 0;
}

/* H2: multi-byte loop boundaries + mask handling. */
static int test_confidence_weighted_multi_byte_and_mask(void) {
    /* 16 trits = 4 packed bytes (covers >8-byte? no, 4 bytes; this is the
     * 4-byte path). Use 32 trits = 8 packed bytes for the 8-byte path. */
    m4t_trit_t q[32], t[32];
    for (int i = 0; i < 32; i++) {
        q[i] = (m4t_trit_t)((i & 1) ? 1 : -1);
        t[i] = (m4t_trit_t)((i & 1) ? -1 : 1);  /* every position is full mismatch */
    }
    uint8_t q_packed[8], t_packed[8];
    for (int i = 0; i < 8; i++) {
        q_packed[i] = pack4(q + i*4);
        t_packed[i] = pack4(t + i*4);
    }
    uint8_t mask[8];
    memset(mask, 0xFFu, 8);

    /* All confidence ON: every position contributes 4 cost = 32 × 4 = 128. */
    uint8_t conf_all[4] = { 0xFFu, 0xFFu, 0xFFu, 0xFFu };
    int32_t got_all = m4t_route_confidence_weighted_dist(
        q_packed, conf_all, t_packed, conf_all, mask, 32);
    ASSERT_EQ_I32(got_all, 128, "32 full-mismatches × cost 4 (both-conf)");

    /* All confidence OFF: each position contributes 2 cost = 32 × 2 = 64. */
    uint8_t conf_none[4] = { 0, 0, 0, 0 };
    int32_t got_none = m4t_route_confidence_weighted_dist(
        q_packed, conf_none, t_packed, conf_none, mask, 32);
    ASSERT_EQ_I32(got_none, 64, "32 full-mismatches × cost 2 (no-conf)");

    /* Mask out the second 16 trits (last 4 bytes inactive). */
    uint8_t mask_half[8] = { 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0, 0, 0, 0 };
    int32_t got_masked = m4t_route_confidence_weighted_dist(
        q_packed, conf_all, t_packed, conf_all, mask_half, 32);
    ASSERT_EQ_I32(got_masked, 64, "16 active full-mismatches × cost 4 (mask honored)");
    return 0;
}

static int test_confidence_weighted_equals_hamming_no_confidence(void) {
    /* When all confidence bits are 0, weighted dist == Hamming. */
    m4t_trit_t q[16] = {  1, -1,  0,  1, -1,  0,  1, -1,
                            0,  1, -1,  0,  1, -1,  0,  1 };
    m4t_trit_t t[16] = {  1,  1, -1,  0,  1, -1,  0,  1,
                            -1, -1,  1,  0, -1,  0,  1,  1 };
    uint8_t q_packed[4], t_packed[4];
    for (int i = 0; i < 4; i++) {
        q_packed[i] = pack4(q + i*4);
        t_packed[i] = pack4(t + i*4);
    }
    uint8_t mask[4] = { 0xFFu, 0xFFu, 0xFFu, 0xFFu };
    uint8_t q_conf[2] = { 0, 0 };
    uint8_t t_conf[2] = { 0, 0 };

    int32_t hamming  = m4t_popcount_dist(q_packed, t_packed, mask, 4);
    int32_t weighted = m4t_route_confidence_weighted_dist(
        q_packed, q_conf, t_packed, t_conf, mask, 16);
    ASSERT_EQ_I32(weighted, hamming, "weighted == hamming when no confidence");
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    if (test_threshold_extract_tau0())             return 1;
    if (test_threshold_extract_tau5())             return 1;
    if (test_threshold_extract_all_within_band())  return 1;
    if (test_threshold_extract_n_zero())           return 1;
    if (test_threshold_extract_pack_boundaries())  return 1;
    if (test_threshold_extract_extremes())         return 1;
    if (test_distance_batch())        return 1;
    if (test_topk_abs_basic())        return 1;
    if (test_topk_abs_with_zeros())   return 1;
    if (test_topk_abs_all_tiles())    return 1;
    if (test_apply_signed())          return 1;
    if (test_apply_signed_sentinel()) return 1;
    if (test_signature_update())      return 1;
    if (test_route_e2e())             return 1;
    if (test_decisions_emit_coverage()) return 1;
    if (test_wildcard_dist_cost_table())               return 1;
    if (test_wildcard_vs_hamming_behavior_diff())      return 1;
    if (test_wildcard_equals_hamming_no_tile_zeros())  return 1;
    if (test_wildcard_respects_mask())                 return 1;
    if (test_wildcard_multi_byte())                    return 1;
    if (test_threshold_extract_dual_basic())                   return 1;
    if (test_confidence_weighted_dist_cost_table())                return 1;
    if (test_confidence_weighted_vs_hamming_behavior_diff())       return 1;
    if (test_confidence_weighted_multi_byte_and_mask())            return 1;
    if (test_confidence_weighted_equals_hamming_no_confidence())   return 1;
    printf("m4t_route: all tests passed\n");
    return 0;
}
