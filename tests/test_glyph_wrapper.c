/*
 * test_glyph_wrapper.c — verify glyph wrapper headers alias M4T correctly.
 *
 * This keeps the public glyph surface honest: if an alias drifts, the test
 * fails either at compile time (missing declaration / wrong type) or at
 * runtime (different result from the underlying m4t_* entry point).
 */

#include "glyph_types.h"
#include "glyph_trit_pack.h"
#include "glyph_ternary_matmul.h"
#include "glyph_route.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_EQ_I32(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %d, expected %d (line %d)\n", \
                (msg), (int)(actual), (int)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

static int test_types(void) {
    ASSERT_EQ_I32((int)sizeof(glyph_mtfp4_t), 1, "glyph_mtfp4_t size");
    ASSERT_EQ_I32((int)sizeof(glyph_mtfp9_t), 2, "glyph_mtfp9_t size");
    ASSERT_EQ_I32((int)sizeof(glyph_mtfp_t), 4, "glyph_mtfp_t size");
    ASSERT_EQ_I32((int)sizeof(glyph_trit_t), 1, "glyph_trit_t size");
    ASSERT_EQ_I32((int)sizeof(glyph_route_decision_t),
                  (int)sizeof(m4t_route_decision_t),
                  "glyph_route_decision_t size");
    ASSERT_EQ_I32(GLYPH_MTFP_SCALE, M4T_MTFP_SCALE, "scale alias");
    ASSERT_EQ_I32(GLYPH_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, "max alias");
    ASSERT_EQ_I32(GLYPH_TRIT_PACKED_BYTES(7), M4T_TRIT_PACKED_BYTES(7), "packed-bytes alias");
    return 0;
}

static int test_trit_pack_aliases(void) {
    glyph_trit_t src[7] = { 1, -1, 0, 1, -1, 0, 1 };
    uint8_t glyph_packed[GLYPH_TRIT_PACKED_BYTES(7)];
    uint8_t m4t_packed[M4T_TRIT_PACKED_BYTES(7)];
    glyph_trit_t glyph_unpacked[7];
    m4t_trit_t m4t_unpacked[7];

    glyph_pack_trits_1d(glyph_packed, src, 7);
    m4t_pack_trits_1d(m4t_packed, src, 7);
    ASSERT_EQ_I32(memcmp(glyph_packed, m4t_packed, sizeof(glyph_packed)), 0, "pack alias");

    glyph_unpack_trits_1d(glyph_unpacked, glyph_packed, 7);
    m4t_unpack_trits_1d(m4t_unpacked, m4t_packed, 7);
    for (int i = 0; i < 7; i++) {
        ASSERT_EQ_I32(glyph_unpacked[i], m4t_unpacked[i], "unpack alias");
    }

    {
        uint8_t mask[GLYPH_TRIT_PACKED_BYTES(7)];
        memset(mask, 0xFF, sizeof(mask));
        ASSERT_EQ_I32(glyph_popcount_dist(glyph_packed, m4t_packed, mask, (int)sizeof(mask)),
                      m4t_popcount_dist(glyph_packed, m4t_packed, mask, (int)sizeof(mask)),
                      "popcount alias");
    }
    return 0;
}

static int test_ternary_matmul_alias(void) {
    enum { M = 1, K = 4, N = 2 };
    glyph_mtfp_t X[K] = { 10, 20, 30, 40 };
    glyph_trit_t W_trits[N * K] = {
         1,  1,  1,  1,
         1, -1,  0,  1
    };
    uint8_t W_packed[N * GLYPH_TRIT_PACKED_BYTES(K)];
    glyph_mtfp_t Y_glyph[N];
    m4t_mtfp_t Y_m4t[N];

    glyph_pack_trits_rowmajor(W_packed, W_trits, N, K);
    glyph_mtfp_ternary_matmul_bt(Y_glyph, X, W_packed, M, K, N);
    m4t_mtfp_ternary_matmul_bt(Y_m4t, X, W_packed, M, K, N);

    for (int j = 0; j < N; j++) {
        ASSERT_EQ_I32(Y_glyph[j], Y_m4t[j], "ternary matmul alias");
    }
    return 0;
}

static int test_route_aliases(void) {
    int64_t values[5] = { 7, 0, -9, 4, -4 };
    uint8_t glyph_packed[GLYPH_TRIT_PACKED_BYTES(5)];
    uint8_t m4t_packed[M4T_TRIT_PACKED_BYTES(5)];
    int32_t scores[4] = { 3, -7, 0, 5 };
    glyph_route_decision_t glyph_decisions[3];
    m4t_route_decision_t m4t_decisions[3];

    glyph_route_threshold_extract(glyph_packed, values, 4, 5);
    m4t_route_threshold_extract(m4t_packed, values, 4, 5);
    ASSERT_EQ_I32(memcmp(glyph_packed, m4t_packed, sizeof(glyph_packed)), 0, "threshold alias");

    glyph_route_topk_abs(glyph_decisions, scores, 4, 3);
    m4t_route_topk_abs(m4t_decisions, scores, 4, 3);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ_I32(glyph_decisions[i].tile_idx, m4t_decisions[i].tile_idx, "topk idx alias");
        ASSERT_EQ_I32(glyph_decisions[i].sign, m4t_decisions[i].sign, "topk sign alias");
    }

    {
        const int D = 4;
        m4t_mtfp_t tile_outs[8] = { 10, 20, 30, 40, 1, 2, 3, 4 };
        m4t_mtfp_t glyph_result[4] = { 0, 0, 0, 0 };
        m4t_mtfp_t m4t_result[4] = { 0, 0, 0, 0 };
        glyph_route_apply_signed(glyph_result, tile_outs, glyph_decisions, 3, D);
        m4t_route_apply_signed(m4t_result, tile_outs, m4t_decisions, 3, D);
        for (int i = 0; i < D; i++) {
            ASSERT_EQ_I32(glyph_result[i], m4t_result[i], "apply alias");
        }
    }
    return 0;
}

int main(void) {
    if (test_types()) return 1;
    if (test_trit_pack_aliases()) return 1;
    if (test_ternary_matmul_alias()) return 1;
    if (test_route_aliases()) return 1;
    printf("glyph_wrapper: all tests passed\n");
    return 0;
}
