/*
 * test_m4t_ops.c — round-trip tests for function-pointer opcode tables.
 *
 * Verifies that every table entry is non-NULL and that dispatching
 * through the table produces the same result as a direct call.
 */

#include "m4t_ops.h"

#include <stdio.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
        return 1; \
    } \
} while (0)

/* ── Trit ops table ────────────────────────────────────────────────────── */

static int test_trit_ops_table(void) {
    /* Verify all entries are non-NULL. */
    for (int i = 0; i < M4T_TOP_COUNT; i++) {
        ASSERT(m4t_trit_ops[i] != NULL, "trit op NULL");
    }

    /* Dispatch mul through table, compare to direct call. */
    m4t_trit_t a_trits[4] = { 1, -1, 0, 1 };
    m4t_trit_t b_trits[4] = { 1,  1, 1, -1 };
    uint8_t pa[1], pb[1], direct[1], table_out[1];

    m4t_pack_trits_1d(pa, a_trits, 4);
    m4t_pack_trits_1d(pb, b_trits, 4);

    m4t_trit_mul(direct, pa, pb, 4);
    m4t_trit_ops[M4T_TOP_MUL](table_out, pa, pb, 4);

    ASSERT(memcmp(direct, table_out, 1) == 0, "trit table mul mismatch");
    return 0;
}

/* ── MTFP ops table ────────────────────────────────────────────────────── */

static int test_mtfp_ops_table(void) {
    /* Verify all entries are non-NULL with valid shapes. */
    for (int i = 0; i < M4T_MOP_COUNT; i++) {
        ASSERT(m4t_mtfp_ops[i].fn != NULL, "mtfp op NULL");
    }

    /* Dispatch vec_add through table, compare to direct call. */
    m4t_mtfp_t a[4] = { 100, 200, 300, 400 };
    m4t_mtfp_t b[4] = { 10, 20, 30, 40 };
    m4t_mtfp_t direct[4], table_out[4];

    m4t_mtfp_vec_add(direct, a, b, 4);

    typedef void (*vec_add_fn)(m4t_mtfp_t*, const m4t_mtfp_t*, const m4t_mtfp_t*, int);
    vec_add_fn fn = (vec_add_fn)m4t_mtfp_ops[M4T_MOP_VEC_ADD].fn;
    fn(table_out, a, b, 4);

    ASSERT(memcmp(direct, table_out, sizeof(direct)) == 0, "mtfp table vec_add mismatch");

    /* Verify shape tag. */
    ASSERT(m4t_mtfp_ops[M4T_MOP_VEC_ADD].shape == M4T_MOP_SHAPE_VEC_BINARY, "vec_add shape");
    ASSERT(m4t_mtfp_ops[M4T_MOP_MATMUL_BT].shape == M4T_MOP_SHAPE_MATMUL, "matmul_bt shape");
    ASSERT(m4t_mtfp_ops[M4T_MOP_LAYERNORM].shape == M4T_MOP_SHAPE_LAYERNORM, "layernorm shape");
    ASSERT(m4t_mtfp_ops[M4T_MOP_TERNARY_MATMUL_BT].shape == M4T_MOP_SHAPE_TERNARY_MATMUL, "ternary shape");
    return 0;
}

/* ── Route ops table ───────────────────────────────────────────────────── */

static int test_route_ops_table(void) {
    for (int i = 0; i < M4T_ROP_COUNT; i++) {
        ASSERT(m4t_route_ops[i].fn != NULL, "route op NULL");
        ASSERT(m4t_route_ops[i].name != NULL, "route op name NULL");
    }

    /* Dispatch topk_abs through table, compare to direct call. */
    int32_t scores[3] = { 5, -10, 3 };
    m4t_route_decision_t direct[1], table_out[1];

    m4t_route_topk_abs(direct, scores, 3, 1);

    typedef void (*topk_fn)(m4t_route_decision_t*, const int32_t*, int, int);
    topk_fn fn = (topk_fn)m4t_route_ops[M4T_ROP_TOPK_ABS].fn;
    fn(table_out, scores, 3, 1);

    ASSERT(direct[0].tile_idx == table_out[0].tile_idx, "route table topk idx");
    ASSERT(direct[0].sign == table_out[0].sign, "route table topk sign");

    /* Verify names. */
    ASSERT(strcmp(m4t_route_ops[M4T_ROP_SIGN_EXTRACT].name, "sign_extract") == 0, "name sign_extract");
    ASSERT(strcmp(m4t_route_ops[M4T_ROP_SIGNATURE_UPDATE].name, "signature_update") == 0, "name sig_update");
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    if (test_trit_ops_table())   return 1;
    if (test_mtfp_ops_table())   return 1;
    if (test_route_ops_table())  return 1;
    printf("m4t_ops: all tests passed\n");
    return 0;
}
