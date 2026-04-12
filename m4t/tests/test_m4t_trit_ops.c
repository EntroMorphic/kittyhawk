/*
 * test_m4t_trit_ops.c — exhaustive tests for TBL-based trit operations.
 *
 * Each test packs all 9 valid trit-pair combinations, runs the op,
 * unpacks, and checks against hand-derived expected values from the
 * truth tables in m4t/tools/m4t_trit_golden.c.
 *
 * Also tests a 65-trit vector (exercises NEON block + scalar tail).
 */

#include "m4t_types.h"
#include "m4t_trit_pack.h"
#include "m4t_trit_ops.h"

#include <stdio.h>
#include <string.h>

#define FAIL(msg, line) do { \
    fprintf(stderr, "FAIL: %s (line %d)\n", (msg), (line)); \
    return 1; \
} while (0)

/* All 9 input pairs: a × b for a,b ∈ {-1, 0, +1}. */
static const m4t_trit_t ALL_A[9] = { -1, -1, -1,  0,  0,  0,  1,  1,  1 };
static const m4t_trit_t ALL_B[9] = { -1,  0,  1, -1,  0,  1, -1,  0,  1 };

typedef void (*trit_op_fn)(uint8_t*, const uint8_t*, const uint8_t*, int);

static int test_op_9(const char* name, trit_op_fn op,
                     const m4t_trit_t expected[9])
{
    uint8_t pa[M4T_TRIT_PACKED_BYTES(9)];
    uint8_t pb[M4T_TRIT_PACKED_BYTES(9)];
    uint8_t pc[M4T_TRIT_PACKED_BYTES(9)];
    m4t_trit_t result[9];

    m4t_pack_trits_1d(pa, ALL_A, 9);
    m4t_pack_trits_1d(pb, ALL_B, 9);
    memset(pc, 0, sizeof(pc));

    op(pc, pa, pb, 9);
    m4t_unpack_trits_1d(result, pc, 9);

    for (int i = 0; i < 9; i++) {
        if (result[i] != expected[i]) {
            fprintf(stderr,
                "FAIL: %s — pair (%+d, %+d): got %+d, expected %+d (index %d)\n",
                name, ALL_A[i], ALL_B[i], result[i], expected[i], i);
            return 1;
        }
    }
    return 0;
}

/* ── Expected outputs from truth tables ─────────────────────────────────── */

/*           (-1,-1) (-1,0) (-1,+1) (0,-1) (0,0) (0,+1) (+1,-1) (+1,0) (+1,+1) */
static const m4t_trit_t EXP_MUL[9]     = {  1,  0, -1,  0,  0,  0, -1,  0,  1 };
static const m4t_trit_t EXP_SAT_ADD[9] = { -1, -1,  0, -1,  0,  1,  0,  1,  1 };
static const m4t_trit_t EXP_MAX[9]     = { -1,  0,  1,  0,  0,  1,  1,  1,  1 };
static const m4t_trit_t EXP_MIN[9]     = { -1, -1, -1, -1,  0,  0, -1,  0,  1 };
static const m4t_trit_t EXP_EQ[9]      = {  1,  0,  0,  0,  1,  0,  0,  0,  1 };
static const m4t_trit_t EXP_NEG[9]     = {  1,  1,  1,  0,  0,  0, -1, -1, -1 };
/* neg ignores b, so expected = -a for each row. */

/* ── 65-trit vector test (NEON block + scalar tail) ───────────────────── */

static int test_op_65(const char* name, trit_op_fn op) {
    /* 65 trits = 16 bytes NEON + 1 byte scalar tail (4 trits) + 1 partial.
     * Use all-+1 for a and all--1 for b. */
    const int N = 65;
    m4t_trit_t ta[65], tb[65], expected[65], result[65];

    for (int i = 0; i < N; i++) { ta[i] = 1; tb[i] = -1; }

    uint8_t pa[M4T_TRIT_PACKED_BYTES(65)];
    uint8_t pb[M4T_TRIT_PACKED_BYTES(65)];
    uint8_t pc[M4T_TRIT_PACKED_BYTES(65)];

    m4t_pack_trits_1d(pa, ta, N);
    m4t_pack_trits_1d(pb, tb, N);
    memset(pc, 0, sizeof(pc));

    op(pc, pa, pb, N);
    m4t_unpack_trits_1d(result, pc, N);

    /* Compute expected by scalar reference. */
    if (op == m4t_trit_mul) {
        for (int i = 0; i < N; i++) expected[i] = (m4t_trit_t)(ta[i] * tb[i]);
    } else if (op == m4t_trit_sat_add) {
        for (int i = 0; i < N; i++) {
            int s = ta[i] + tb[i];
            expected[i] = (m4t_trit_t)(s > 1 ? 1 : s < -1 ? -1 : s);
        }
    } else if (op == m4t_trit_max) {
        for (int i = 0; i < N; i++) expected[i] = (ta[i] > tb[i]) ? ta[i] : tb[i];
    } else if (op == m4t_trit_min) {
        for (int i = 0; i < N; i++) expected[i] = (ta[i] < tb[i]) ? ta[i] : tb[i];
    } else if (op == m4t_trit_eq) {
        for (int i = 0; i < N; i++) expected[i] = (ta[i] == tb[i]) ? 1 : 0;
    } else if (op == m4t_trit_neg) {
        for (int i = 0; i < N; i++) expected[i] = (m4t_trit_t)(-ta[i]);
    } else {
        FAIL("unknown op", __LINE__);
    }

    for (int i = 0; i < N; i++) {
        if (result[i] != expected[i]) {
            fprintf(stderr,
                "FAIL: %s (n=65) — index %d: got %+d, expected %+d\n",
                name, i, result[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

/* ── Varied-data NEON test (T1) ─────────────────────────────────────────── */

static int test_op_varied_65(void) {
    /* 65 trits with cycling data: ta[i] = {-1, 0, +1, -1, 0, +1, ...}
     * and tb[i] = {+1, +1, 0, -1, -1, 0, ...}. Exercises per-lane
     * variation within each NEON register. */
    const int N = 65;
    m4t_trit_t ta[65], tb[65], result[65];
    static const m4t_trit_t cycle_a[3] = { -1, 0, 1 };
    static const m4t_trit_t cycle_b[6] = { 1, 1, 0, -1, -1, 0 };

    for (int i = 0; i < N; i++) { ta[i] = cycle_a[i % 3]; tb[i] = cycle_b[i % 6]; }

    uint8_t pa[M4T_TRIT_PACKED_BYTES(65)];
    uint8_t pb[M4T_TRIT_PACKED_BYTES(65)];
    uint8_t pc[M4T_TRIT_PACKED_BYTES(65)];

    m4t_pack_trits_1d(pa, ta, N);
    m4t_pack_trits_1d(pb, tb, N);

    /* Test mul with varied data. */
    memset(pc, 0, sizeof(pc));
    m4t_trit_mul(pc, pa, pb, N);
    m4t_unpack_trits_1d(result, pc, N);
    for (int i = 0; i < N; i++) {
        m4t_trit_t expected = (m4t_trit_t)(ta[i] * tb[i]);
        if (result[i] != expected) {
            fprintf(stderr, "FAIL: varied mul — index %d: got %+d, expected %+d\n",
                    i, result[i], expected);
            return 1;
        }
    }

    /* Test sat_add with varied data. */
    memset(pc, 0, sizeof(pc));
    m4t_trit_sat_add(pc, pa, pb, N);
    m4t_unpack_trits_1d(result, pc, N);
    for (int i = 0; i < N; i++) {
        int s = ta[i] + tb[i];
        m4t_trit_t expected = (m4t_trit_t)(s > 1 ? 1 : s < -1 ? -1 : s);
        if (result[i] != expected) {
            fprintf(stderr, "FAIL: varied sat_add — index %d: got %+d, expected %+d\n",
                    i, result[i], expected);
            return 1;
        }
    }
    return 0;
}

/* ── In-place aliasing test ─────────────────────────────────────────────── */

static int test_op_inplace(void) {
    /* dst aliasing a: mul(a, b) with dst == a. */
    m4t_trit_t ta[9], tb[9];
    for (int i = 0; i < 9; i++) { ta[i] = ALL_A[i]; tb[i] = ALL_B[i]; }

    uint8_t pa[M4T_TRIT_PACKED_BYTES(9)];
    uint8_t pb[M4T_TRIT_PACKED_BYTES(9)];
    m4t_pack_trits_1d(pa, ta, 9);
    m4t_pack_trits_1d(pb, tb, 9);

    m4t_trit_mul(pa, pa, pb, 9);  /* in-place */

    m4t_trit_t result[9];
    m4t_unpack_trits_1d(result, pa, 9);
    for (int i = 0; i < 9; i++) {
        if (result[i] != EXP_MUL[i]) {
            fprintf(stderr, "FAIL: inplace mul — index %d: got %+d, expected %+d\n",
                    i, result[i], EXP_MUL[i]);
            return 1;
        }
    }
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    /* Exhaustive 3×3 tests. */
    if (test_op_9("mul",     m4t_trit_mul,     EXP_MUL))     return 1;
    if (test_op_9("sat_add", m4t_trit_sat_add, EXP_SAT_ADD)) return 1;
    if (test_op_9("max",     m4t_trit_max,     EXP_MAX))     return 1;
    if (test_op_9("min",     m4t_trit_min,     EXP_MIN))     return 1;
    if (test_op_9("eq",      m4t_trit_eq,      EXP_EQ))      return 1;
    if (test_op_9("neg",     m4t_trit_neg,     EXP_NEG))     return 1;

    /* 65-trit NEON + tail tests. */
    if (test_op_65("mul",     m4t_trit_mul))     return 1;
    if (test_op_65("sat_add", m4t_trit_sat_add)) return 1;
    if (test_op_65("max",     m4t_trit_max))     return 1;
    if (test_op_65("min",     m4t_trit_min))     return 1;
    if (test_op_65("eq",      m4t_trit_eq))      return 1;
    if (test_op_65("neg",     m4t_trit_neg))     return 1;

    /* Varied-data NEON per-lane test. */
    if (test_op_varied_65()) return 1;

    /* In-place aliasing. */
    if (test_op_inplace()) return 1;

    /* In-place neg aliasing (T2 — different kernel path from binary ops). */
    {
        m4t_trit_t ta_neg[9];
        for (int i = 0; i < 9; i++) ta_neg[i] = ALL_A[i];
        uint8_t pa_neg[M4T_TRIT_PACKED_BYTES(9)];
        m4t_pack_trits_1d(pa_neg, ta_neg, 9);
        m4t_trit_neg(pa_neg, pa_neg, NULL, 9);
        m4t_trit_t res_neg[9];
        m4t_unpack_trits_1d(res_neg, pa_neg, 9);
        for (int i = 0; i < 9; i++) {
            if (res_neg[i] != EXP_NEG[i]) {
                fprintf(stderr, "FAIL: inplace neg — index %d: got %+d, expected %+d\n",
                        i, res_neg[i], EXP_NEG[i]);
                return 1;
            }
        }
    }

    printf("m4t_trit_ops: all tests passed\n");
    return 0;
}
