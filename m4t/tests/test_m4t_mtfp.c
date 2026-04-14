/*
 * test_m4t_mtfp.c — direct tests for mantissa-layer MTFP19 primitives.
 *
 * Covers: clamp64 boundaries, vec_zero edge cases, block_add/block_sub
 * saturation in both directions, vec_add_inplace/vec_sub_inplace over
 * NEON-only / scalar-only / NEON+tail paths, aliasing, and NEON/scalar
 * boundary equivalence.
 *
 * No float anywhere. Hand-derived integer golden values.
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Compile-time invariants (M1: make block constants load-bearing). */
_Static_assert(M4T_MTFP_CELLS_PER_BLOCK == 4,
               "MTFP19 block is exactly 4 cells");
_Static_assert(M4T_BLOCK_BYTES == 16,
               "Substrate block is 16 bytes (one NEON vector)");

static int failures = 0;

#define FAIL(fmt, ...) do { \
    fprintf(stderr, "FAIL [%s:%d] " fmt "\n", __func__, __LINE__, __VA_ARGS__); \
    failures++; \
} while (0)

#define CHECK_EQ(actual, expected) do { \
    int64_t _a = (int64_t)(actual), _e = (int64_t)(expected); \
    if (_a != _e) FAIL("got %lld, expected %lld", (long long)_a, (long long)_e); \
} while (0)

#define CHECK_VEC(actual, expected, n) do { \
    for (int _i = 0; _i < (n); _i++) { \
        if ((actual)[_i] != (expected)[_i]) { \
            FAIL("mismatch at [%d]: got %d, expected %d", \
                 _i, (int)(actual)[_i], (int)(expected)[_i]); \
            break; \
        } \
    } \
} while (0)

/* ── clamp64 ─────────────────────────────────────────────────────────── */

static void test_clamp64(void) {
    CHECK_EQ(m4t_mtfp_clamp64(0), 0);
    CHECK_EQ(m4t_mtfp_clamp64(1), 1);
    CHECK_EQ(m4t_mtfp_clamp64(-1), -1);
    CHECK_EQ(m4t_mtfp_clamp64(M4T_MTFP_MAX_VAL), M4T_MTFP_MAX_VAL);
    CHECK_EQ(m4t_mtfp_clamp64(-(int64_t)M4T_MTFP_MAX_VAL), -M4T_MTFP_MAX_VAL);
    CHECK_EQ(m4t_mtfp_clamp64((int64_t)M4T_MTFP_MAX_VAL + 1), M4T_MTFP_MAX_VAL);
    CHECK_EQ(m4t_mtfp_clamp64(-(int64_t)M4T_MTFP_MAX_VAL - 1), -M4T_MTFP_MAX_VAL);
    CHECK_EQ(m4t_mtfp_clamp64(INT64_MAX), M4T_MTFP_MAX_VAL);
    CHECK_EQ(m4t_mtfp_clamp64(INT64_MIN), -M4T_MTFP_MAX_VAL);
}

/* ── vec_zero ────────────────────────────────────────────────────────── */

static void test_vec_zero_empty(void) {
    m4t_mtfp_t buf[4] = { 1, 2, 3, 4 };
    m4t_mtfp_t expected[4] = { 1, 2, 3, 4 };
    m4t_mtfp_vec_zero(buf, 0);
    CHECK_VEC(buf, expected, 4);
}

static void test_vec_zero_sizes(void) {
    for (int n = 1; n <= 1024; n *= 2) {
        m4t_mtfp_t* buf = calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) buf[i] = 12345;
        m4t_mtfp_vec_zero(buf, n);
        for (int i = 0; i < n; i++) {
            if (buf[i] != 0) { FAIL("n=%d, buf[%d]=%d", n, i, (int)buf[i]); break; }
        }
        free(buf);
    }
    /* Irregular sizes that exercise misaligned scalar tail. */
    for (int n = 1; n < 20; n++) {
        m4t_mtfp_t buf[32];
        for (int i = 0; i < 32; i++) buf[i] = 999;
        m4t_mtfp_vec_zero(buf, n);
        for (int i = 0; i < n; i++)
            if (buf[i] != 0) { FAIL("n=%d, buf[%d]=%d", n, i, (int)buf[i]); break; }
        for (int i = n; i < 32; i++)
            if (buf[i] != 999) { FAIL("n=%d trashed tail at [%d]", n, i); break; }
    }
}

/* ── block_add ────────────────────────────────────────────────────────── */

static void test_block_add_identity(void) {
    m4t_mtfp_t dst[4] = { 0, 0, 0, 0 };
    m4t_mtfp_t a[4]   = { 0, 0, 0, 0 };
    m4t_mtfp_block_add(dst, a);
    m4t_mtfp_t expected[4] = { 0, 0, 0, 0 };
    CHECK_VEC(dst, expected, 4);
}

static void test_block_add_mixed_sign(void) {
    m4t_mtfp_t dst[4] = {  10, -20,  30, -40 };
    m4t_mtfp_t a[4]   = {   5,  15, -25,  35 };
    m4t_mtfp_block_add(dst, a);
    m4t_mtfp_t expected[4] = { 15, -5, 5, -5 };
    CHECK_VEC(dst, expected, 4);
}

static void test_block_add_positive_saturation(void) {
    m4t_mtfp_t dst[4] = { M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL - 10, M4T_MTFP_MAX_VAL / 2, 0 };
    m4t_mtfp_t a[4]   = { 1,                100,                   M4T_MTFP_MAX_VAL,     M4T_MTFP_MAX_VAL };
    m4t_mtfp_block_add(dst, a);
    /* [0]: MAX+1 → sat MAX. [1]: (MAX-10)+100 = MAX+90 → sat MAX.
     * [2]: MAX/2 + MAX = 1.5*MAX → sat MAX. [3]: 0 + MAX = MAX (exact). */
    m4t_mtfp_t expected[4] = { M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL };
    CHECK_VEC(dst, expected, 4);
}

static void test_block_add_negative_saturation(void) {
    m4t_mtfp_t dst[4] = { -M4T_MTFP_MAX_VAL, -(M4T_MTFP_MAX_VAL - 10), -M4T_MTFP_MAX_VAL / 2, 0 };
    m4t_mtfp_t a[4]   = { -1,                -100,                      -M4T_MTFP_MAX_VAL,     -M4T_MTFP_MAX_VAL };
    m4t_mtfp_block_add(dst, a);
    m4t_mtfp_t expected[4] = { -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL };
    CHECK_VEC(dst, expected, 4);
}

static void test_block_add_aliasing(void) {
    m4t_mtfp_t dst[4] = { 10, -20, 30, -40 };
    m4t_mtfp_block_add(dst, dst);
    m4t_mtfp_t expected[4] = { 20, -40, 60, -80 };
    CHECK_VEC(dst, expected, 4);
}

static void test_block_add_aliasing_saturates(void) {
    /* dst = dst + dst should saturate when |dst| > MAX/2. */
    m4t_mtfp_t dst[4] = { M4T_MTFP_MAX_VAL / 2 + 1, -(M4T_MTFP_MAX_VAL / 2 + 1),
                          M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL };
    m4t_mtfp_block_add(dst, dst);
    m4t_mtfp_t expected[4] = { M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL,
                               M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL };
    CHECK_VEC(dst, expected, 4);
}

/* ── block_sub ────────────────────────────────────────────────────────── */

static void test_block_sub_identity_to_zero(void) {
    m4t_mtfp_t dst[4] = { 100, -200, 300, -400 };
    m4t_mtfp_t a[4]   = { 100, -200, 300, -400 };
    m4t_mtfp_block_sub(dst, a);
    m4t_mtfp_t expected[4] = { 0, 0, 0, 0 };
    CHECK_VEC(dst, expected, 4);
}

static void test_block_sub_saturation(void) {
    m4t_mtfp_t dst[4] = { -M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, 0, 0 };
    m4t_mtfp_t a[4]   = {  1,                -1,               M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL };
    m4t_mtfp_block_sub(dst, a);
    /* [0]: -MAX - 1 → sat -MAX. [1]: MAX - (-1) = MAX+1 → sat MAX.
     * [2]: 0 - MAX = -MAX (exact). [3]: 0 - (-MAX) = MAX (exact). */
    m4t_mtfp_t expected[4] = { -M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL };
    CHECK_VEC(dst, expected, 4);
}

static void test_block_sub_aliasing_to_zero(void) {
    m4t_mtfp_t dst[4] = { 123, -456, 789, -1000 };
    m4t_mtfp_block_sub(dst, dst);
    m4t_mtfp_t expected[4] = { 0, 0, 0, 0 };
    CHECK_VEC(dst, expected, 4);
}

/* ── vec_add_inplace ──────────────────────────────────────────────────── */

static void test_vec_add_empty(void) {
    m4t_mtfp_t dst[4] = { 1, 2, 3, 4 };
    m4t_mtfp_t a[4]   = { 10, 20, 30, 40 };
    m4t_mtfp_vec_add_inplace(dst, a, 0);
    m4t_mtfp_t expected[4] = { 1, 2, 3, 4 };
    CHECK_VEC(dst, expected, 4);
}

static void test_vec_add_scalar_only(void) {
    /* n=3: all cells go through scalar path. */
    m4t_mtfp_t dst[3] = { 1, 2, 3 };
    m4t_mtfp_t a[3]   = { 10, 20, 30 };
    m4t_mtfp_vec_add_inplace(dst, a, 3);
    m4t_mtfp_t expected[3] = { 11, 22, 33 };
    CHECK_VEC(dst, expected, 3);
}

static void test_vec_add_neon_only(void) {
    /* n=4: exactly one block, no tail. */
    m4t_mtfp_t dst[4] = { 1, 2, 3, 4 };
    m4t_mtfp_t a[4]   = { 10, 20, 30, 40 };
    m4t_mtfp_vec_add_inplace(dst, a, 4);
    m4t_mtfp_t expected[4] = { 11, 22, 33, 44 };
    CHECK_VEC(dst, expected, 4);
}

static void test_vec_add_neon_plus_tail(void) {
    /* n=7: one block + 3-cell tail. */
    m4t_mtfp_t dst[7] = { 1, 2, 3, 4, 5, 6, 7 };
    m4t_mtfp_t a[7]   = { 10, 20, 30, 40, 50, 60, 70 };
    m4t_mtfp_vec_add_inplace(dst, a, 7);
    m4t_mtfp_t expected[7] = { 11, 22, 33, 44, 55, 66, 77 };
    CHECK_VEC(dst, expected, 7);
}

static void test_vec_add_neon_and_tail_both_saturate(void) {
    /* L3: NEON + scalar tail must agree on saturation semantics.
     * n=5 crafted so lanes 0..3 go via NEON and lane 4 via scalar;
     * both hit the positive saturation boundary. */
    m4t_mtfp_t dst[5] = { M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL };
    m4t_mtfp_t a[5]   = { 1, 1, 1, 1, 1 };
    m4t_mtfp_vec_add_inplace(dst, a, 5);
    for (int i = 0; i < 5; i++)
        if (dst[i] != M4T_MTFP_MAX_VAL)
            FAIL("NEON/scalar saturation disagree at [%d]: got %d", i, (int)dst[i]);
}

static void test_vec_add_large(void) {
    const int N = 1024;
    m4t_mtfp_t* dst = calloc((size_t)N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* a   = calloc((size_t)N, sizeof(m4t_mtfp_t));
    for (int i = 0; i < N; i++) { dst[i] = i; a[i] = 2 * i; }
    m4t_mtfp_vec_add_inplace(dst, a, N);
    for (int i = 0; i < N; i++)
        if (dst[i] != 3 * i) { FAIL("large add [%d]: %d", i, (int)dst[i]); break; }
    free(dst); free(a);
}

static void test_vec_add_aliasing(void) {
    /* dst + dst = 2*dst, saturated. */
    m4t_mtfp_t dst[7] = { 1, -2, 3, -4, M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, 0 };
    m4t_mtfp_vec_add_inplace(dst, dst, 7);
    m4t_mtfp_t expected[7] = { 2, -4, 6, -8, M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, 0 };
    CHECK_VEC(dst, expected, 7);
}

/* ── vec_sub_inplace ──────────────────────────────────────────────────── */

static void test_vec_sub_aliasing_to_zero(void) {
    m4t_mtfp_t dst[7] = { 1, 2, 3, 4, 5, 6, 7 };
    m4t_mtfp_vec_sub_inplace(dst, dst, 7);
    m4t_mtfp_t expected[7] = { 0, 0, 0, 0, 0, 0, 0 };
    CHECK_VEC(dst, expected, 7);
}

static void test_vec_sub_neon_and_tail(void) {
    m4t_mtfp_t dst[7] = { 10, 20, 30, 40, 50, 60, 70 };
    m4t_mtfp_t a[7]   = {  1,  2,  3,  4,  5,  6,  7 };
    m4t_mtfp_vec_sub_inplace(dst, a, 7);
    m4t_mtfp_t expected[7] = { 9, 18, 27, 36, 45, 54, 63 };
    CHECK_VEC(dst, expected, 7);
}

static void test_vec_sub_saturation(void) {
    m4t_mtfp_t dst[5] = { -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL };
    m4t_mtfp_t a[5]   = { 1, 1, 1, 1, 1 };
    m4t_mtfp_vec_sub_inplace(dst, a, 5);
    for (int i = 0; i < 5; i++)
        if (dst[i] != -M4T_MTFP_MAX_VAL)
            FAIL("NEON/scalar sub-saturation disagree at [%d]: got %d", i, (int)dst[i]);
}

int main(void) {
    test_clamp64();

    test_vec_zero_empty();
    test_vec_zero_sizes();

    test_block_add_identity();
    test_block_add_mixed_sign();
    test_block_add_positive_saturation();
    test_block_add_negative_saturation();
    test_block_add_aliasing();
    test_block_add_aliasing_saturates();

    test_block_sub_identity_to_zero();
    test_block_sub_saturation();
    test_block_sub_aliasing_to_zero();

    test_vec_add_empty();
    test_vec_add_scalar_only();
    test_vec_add_neon_only();
    test_vec_add_neon_plus_tail();
    test_vec_add_neon_and_tail_both_saturate();
    test_vec_add_large();
    test_vec_add_aliasing();

    test_vec_sub_aliasing_to_zero();
    test_vec_sub_neon_and_tail();
    test_vec_sub_saturation();

    if (failures) {
        fprintf(stderr, "\n%d assertion(s) failed\n", failures);
        return 1;
    }
    printf("m4t_mtfp: all tests passed\n");
    return 0;
}
