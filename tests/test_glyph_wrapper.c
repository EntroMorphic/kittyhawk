/*
 * test_glyph_wrapper.c — verify glyph wrapper headers alias M4T correctly.
 *
 * Calls every glyph_* alias and confirms it produces the same result as
 * the corresponding m4t_* function. If the aliases are wrong, the test
 * either won't compile (wrong type) or will produce wrong values (wrong
 * function).
 */

#include "glyph_types.h"
#include "glyph_mtfp.h"
#include "glyph_trit_pack.h"
#include "glyph_ternary_matmul.h"
#include "glyph_route.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_EQ(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %d, expected %d (line %d)\n", \
                (msg), (int)(actual), (int)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

static int test_types(void) {
    /* Verify type aliases are the right size. */
    ASSERT_EQ((int)sizeof(glyph_mtfp4_t), 1, "mtfp4 size");
    ASSERT_EQ((int)sizeof(glyph_mtfp9_t), 2, "mtfp9 size");
    ASSERT_EQ((int)sizeof(glyph_mtfp_t), 4, "mtfp size");
    ASSERT_EQ((int)sizeof(glyph_mtfp_w_t), 8, "mtfp_w size");
    ASSERT_EQ((int)sizeof(glyph_trit_t), 1, "trit size");
    ASSERT_EQ((int)sizeof(glyph_route_decision_t), (int)sizeof(m4t_route_decision_t), "decision size");
    ASSERT_EQ(GLYPH_MTFP_SCALE, M4T_MTFP_SCALE, "scale");
    ASSERT_EQ(GLYPH_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, "max_val");
    return 0;
}

static int test_mtfp_aliases(void) {
    glyph_mtfp_t a = GLYPH_MTFP_SCALE;
    glyph_mtfp_t b = GLYPH_MTFP_SCALE;

    ASSERT_EQ(glyph_mtfp_add(a, b), m4t_mtfp_add(a, b), "add alias");
    ASSERT_EQ(glyph_mtfp_sub(a, b), m4t_mtfp_sub(a, b), "sub alias");
    ASSERT_EQ(glyph_mtfp_mul(a, b), m4t_mtfp_mul(a, b), "mul alias");
    ASSERT_EQ(glyph_mtfp_neg(a), m4t_mtfp_neg(a), "neg alias");

    glyph_mtfp_t va[4] = {1, 2, 3, 4};
    glyph_mtfp_t vb[4] = {10, 20, 30, 40};
    glyph_mtfp_t vc[4], vd[4];

    glyph_mtfp_vec_add(vc, va, vb, 4);
    m4t_mtfp_vec_add(vd, va, vb, 4);
    for (int i = 0; i < 4; i++) ASSERT_EQ(vc[i], vd[i], "vec_add alias");

    return 0;
}

static int test_trit_aliases(void) {
    glyph_trit_t trits[4] = { 1, -1, 0, 1 };
    uint8_t gp[1], mp[1];
    glyph_pack_trits_1d(gp, trits, 4);
    m4t_pack_trits_1d(mp, trits, 4);
    ASSERT_EQ(gp[0], mp[0], "pack alias");

    glyph_trit_t gunp[4], munp[4];
    glyph_unpack_trits_1d(gunp, gp, 4);
    m4t_unpack_trits_1d(munp, mp, 4);
    for (int i = 0; i < 4; i++) ASSERT_EQ(gunp[i], munp[i], "unpack alias");

    return 0;
}

static int test_ternary_matmul_alias(void) {
    glyph_mtfp_t X[4] = { 59049, 59049, 59049, 59049 };
    glyph_trit_t W[4] = { 1, 1, 1, 1 };
    uint8_t Wp[1];
    glyph_pack_trits_1d(Wp, W, 4);

    glyph_mtfp_t Yg[1], Ym[1];
    glyph_mtfp_ternary_matmul_bt(Yg, X, Wp, 1, 4, 1);
    m4t_mtfp_ternary_matmul_bt(Ym, X, Wp, 1, 4, 1);
    ASSERT_EQ(Yg[0], Ym[0], "ternary matmul alias");

    return 0;
}

static int test_route_aliases(void) {
    int32_t scores[3] = { 5, -10, 3 };
    glyph_route_decision_t gd[1];
    m4t_route_decision_t md[1];

    glyph_route_topk_abs(gd, scores, 3, 1);
    m4t_route_topk_abs(md, scores, 3, 1);

    ASSERT_EQ(gd[0].tile_idx, md[0].tile_idx, "route topk idx alias");
    ASSERT_EQ(gd[0].sign, md[0].sign, "route topk sign alias");
    return 0;
}

int main(void) {
    if (test_types())               return 1;
    if (test_mtfp_aliases())        return 1;
    if (test_trit_aliases())        return 1;
    if (test_ternary_matmul_alias()) return 1;
    if (test_route_aliases())       return 1;
    printf("glyph_wrapper: all tests passed\n");
    return 0;
}
