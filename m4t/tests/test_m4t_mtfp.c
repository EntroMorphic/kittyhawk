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

/* V13.B: bit-exact NEON vs scalar_ref for the int32×int32 → int64 dot.
 * Coverage: aligned (n%4=0), unaligned tail (n%4 ∈ {1,2,3}), n=0,
 * n=1, large n typical of BitNet usage (HIDDEN_SIZE), edge magnitudes. */
static void test_vec_dot_i64_empty(void) {
    int64_t a = m4t_mtfp_vec_dot_i64(NULL, NULL, 0);
    if (a != 0) FAIL("dot_i64 n=0 should be 0, got %lld", (long long)a);
}

static void test_vec_dot_i64_aligned(void) {
    /* Use small distinct values; verify against scalar_ref.
     * Cover: 4, 8, 12, 16, 80, 256, 2560 — covers n%4=0 cases. */
    int sizes[] = { 4, 8, 12, 16, 80, 256, 2560 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            x[i] = (m4t_mtfp_t)((i * 37) % 2000 - 1000);
            y[i] = (m4t_mtfp_t)((i * 23) % 2000 - 1000);
        }
        int64_t neon = m4t_mtfp_vec_dot_i64(x, y, n);
        int64_t ref  = m4t_mtfp_vec_dot_i64_scalar_ref(x, y, n);
        if (neon != ref) {
            FAIL("dot_i64 aligned n=%d: NEON=%lld ref=%lld",
                 n, (long long)neon, (long long)ref);
        }
        free(x); free(y);
    }
}

static void test_vec_dot_i64_unaligned(void) {
    /* n%4 ∈ {1, 2, 3} cases — boundary tile fires. */
    int sizes[] = { 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 81, 257, 2561, 2563 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            x[i] = (m4t_mtfp_t)((i * 41) % 2000 - 1000);
            y[i] = (m4t_mtfp_t)((i * 19) % 2000 - 1000);
        }
        int64_t neon = m4t_mtfp_vec_dot_i64(x, y, n);
        int64_t ref  = m4t_mtfp_vec_dot_i64_scalar_ref(x, y, n);
        if (neon != ref) {
            FAIL("dot_i64 unaligned n=%d: NEON=%lld ref=%lld",
                 n, (long long)neon, (long long)ref);
        }
        free(x); free(y);
    }
}

static void test_vec_dot_i64_extreme(void) {
    /* High-magnitude inputs to exercise the int64 widening. */
    int n = 2560;
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) {
        x[i] = (i & 1) ? 1000000 : -1000000;
        y[i] = (i & 2) ? 1000000 : -1000000;
    }
    int64_t neon = m4t_mtfp_vec_dot_i64(x, y, n);
    int64_t ref  = m4t_mtfp_vec_dot_i64_scalar_ref(x, y, n);
    if (neon != ref) {
        FAIL("dot_i64 extreme: NEON=%lld ref=%lld", (long long)neon, (long long)ref);
    }
    free(x); free(y);
}

/* ── m4t_mtfp_relu2_inplace_bx (V14.D) ─────────────────────────────────── */

static void test_relu2_bx_aligned(void) {
    int sizes[]  = { 4, 16, 64, 256, 6912 };
    /* Cover the full assertion range [0, 39], not just BitNet practical values. */
    int shifts[] = { 0, 1, 2, 3, 4, 8, 16, 19, 20, 25, 30, 35, 39 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        for (size_t k = 0; k < sizeof(shifts)/sizeof(shifts[0]); k++) {
            int shift_exp = shifts[k];
            int x_bx = (shift_exp + 1) / 2 + 1;
            int target_bx = 2 * x_bx - shift_exp;
            m4t_mtfp_t* x  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* xr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            for (int i = 0; i < n; i++) {
                int v = (i * 547) % 60000 - 30000;
                x[i] = (m4t_mtfp_t)v;
                xr[i] = (m4t_mtfp_t)v;
            }
            m4t_mtfp_relu2_inplace_bx          (x,  x_bx, target_bx, n);
            m4t_mtfp_relu2_inplace_bx_scalar_ref(xr, x_bx, target_bx, n);
            for (int i = 0; i < n; i++) {
                if (x[i] != xr[i]) {
                    FAIL("relu2_bx aligned shift_exp=%d n=%d i=%d: NEON=%d ref=%d",
                         shift_exp, n, i, (int)x[i], (int)xr[i]);
                    break;
                }
            }
            free(x); free(xr);
        }
    }
}

static void test_relu2_bx_unaligned(void) {
    int sizes[]  = { 1, 2, 3, 5, 7, 11, 17, 33, 65, 129 };
    int shifts[] = { 0, 1, 3, 7, 19, 25, 39 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        for (size_t k = 0; k < sizeof(shifts)/sizeof(shifts[0]); k++) {
            int shift_exp = shifts[k];
            int x_bx = (shift_exp + 1) / 2 + 1;
            int target_bx = 2 * x_bx - shift_exp;
            m4t_mtfp_t* x  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* xr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            for (int i = 0; i < n; i++) {
                int v = (i * 191) % 50000 - 25000;
                x[i] = (m4t_mtfp_t)v;
                xr[i] = (m4t_mtfp_t)v;
            }
            m4t_mtfp_relu2_inplace_bx          (x,  x_bx, target_bx, n);
            m4t_mtfp_relu2_inplace_bx_scalar_ref(xr, x_bx, target_bx, n);
            for (int i = 0; i < n; i++) {
                if (x[i] != xr[i]) {
                    FAIL("relu2_bx unaligned shift_exp=%d n=%d i=%d: NEON=%d ref=%d",
                         shift_exp, n, i, (int)x[i], (int)xr[i]);
                    break;
                }
            }
            free(x); free(xr);
        }
    }
}

static void test_relu2_bx_extreme(void) {
    int n = 256;
    /* MAX_VAL squared = 2^58.3 — exercise iterated /3 across full assert range. */
    int shifts[] = { 1, 2, 5, 10, 19, 25, 30, 39 };
    for (size_t k = 0; k < sizeof(shifts)/sizeof(shifts[0]); k++) {
        int shift_exp = shifts[k];
        int xb = (shift_exp + 1) / 2 + 5;
        int tb = 2 * xb - shift_exp;
        m4t_mtfp_t* x  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* xr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            m4t_mtfp_t v = (i & 1) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
            x[i] = v; xr[i] = v;
        }
        m4t_mtfp_relu2_inplace_bx          (x,  xb, tb, n);
        m4t_mtfp_relu2_inplace_bx_scalar_ref(xr, xb, tb, n);
        for (int i = 0; i < n; i++) {
            if (x[i] != xr[i]) {
                FAIL("relu2_bx extreme shift_exp=%d i=%d: NEON=%d ref=%d",
                     shift_exp, i, (int)x[i], (int)xr[i]);
                break;
            }
        }
        free(x); free(xr);
    }
}

/* ── m4t_mtfp_elementwise_mul_bx (V14.E) ───────────────────────────────── */

static void test_elementwise_mul_bx_aligned(void) {
    int sizes[]  = { 4, 16, 64, 256, 6912 };
    int shifts[] = { 0, 1, 2, 3, 5, 10, 19, 25, 30, 35, 39 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        for (size_t k = 0; k < sizeof(shifts)/sizeof(shifts[0]); k++) {
            int shift_exp = shifts[k];
            int a_bx = (shift_exp + 1) / 2 + 1;
            int b_bx = (shift_exp / 2) + 1;
            int target_bx = a_bx + b_bx - shift_exp;
            m4t_mtfp_t* a = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* b = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            for (int i = 0; i < n; i++) {
                a[i] = (m4t_mtfp_t)((i * 311) % 60000 - 30000);
                b[i] = (m4t_mtfp_t)((i * 103) % 50000 - 25000);
            }
            m4t_mtfp_elementwise_mul_bx          (y,  a, a_bx, b, b_bx, target_bx, n);
            m4t_mtfp_elementwise_mul_bx_scalar_ref(yr, a, a_bx, b, b_bx, target_bx, n);
            for (int i = 0; i < n; i++) {
                if (y[i] != yr[i]) {
                    FAIL("ew_mul_bx aligned shift_exp=%d n=%d i=%d: NEON=%d ref=%d",
                         shift_exp, n, i, (int)y[i], (int)yr[i]);
                    break;
                }
            }
            free(a); free(b); free(y); free(yr);
        }
    }
}

static void test_elementwise_mul_bx_unaligned(void) {
    int sizes[]  = { 1, 2, 3, 5, 7, 11, 17, 33, 129 };
    int shifts[] = { 0, 1, 3, 8, 19, 25, 39 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        for (size_t k = 0; k < sizeof(shifts)/sizeof(shifts[0]); k++) {
            int shift_exp = shifts[k];
            int a_bx = (shift_exp + 1) / 2 + 1;
            int b_bx = (shift_exp / 2) + 1;
            int target_bx = a_bx + b_bx - shift_exp;
            m4t_mtfp_t* a = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* b = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            for (int i = 0; i < n; i++) {
                a[i] = (m4t_mtfp_t)((i * 401) % 30000 - 15000);
                b[i] = (m4t_mtfp_t)((i * 79)  % 30000 - 15000);
            }
            m4t_mtfp_elementwise_mul_bx          (y,  a, a_bx, b, b_bx, target_bx, n);
            m4t_mtfp_elementwise_mul_bx_scalar_ref(yr, a, a_bx, b, b_bx, target_bx, n);
            for (int i = 0; i < n; i++) {
                if (y[i] != yr[i]) {
                    FAIL("ew_mul_bx unaligned shift_exp=%d n=%d i=%d: NEON=%d ref=%d",
                         shift_exp, n, i, (int)y[i], (int)yr[i]);
                    break;
                }
            }
            free(a); free(b); free(y); free(yr);
        }
    }
}

static void test_elementwise_mul_bx_extreme(void) {
    int n = 256;
    int shifts[] = { 1, 5, 10, 19, 25, 30, 39 };
    for (size_t k = 0; k < sizeof(shifts)/sizeof(shifts[0]); k++) {
        int shift_exp = shifts[k];
        int a_bx = (shift_exp + 1) / 2 + 5;
        int b_bx = (shift_exp / 2) + 5;
        int target_bx = a_bx + b_bx - shift_exp;
        m4t_mtfp_t* a = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* b = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            a[i] = (i & 1) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
            b[i] = (i & 2) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
        }
        m4t_mtfp_elementwise_mul_bx          (y,  a, a_bx, b, b_bx, target_bx, n);
        m4t_mtfp_elementwise_mul_bx_scalar_ref(yr, a, a_bx, b, b_bx, target_bx, n);
        for (int i = 0; i < n; i++) {
            if (y[i] != yr[i]) {
                FAIL("ew_mul_bx extreme shift_exp=%d i=%d: NEON=%d ref=%d",
                     shift_exp, i, (int)y[i], (int)yr[i]);
                break;
            }
        }
        free(a); free(b); free(y); free(yr);
    }
}

/* ── m4t_mtfp_rmsnorm_bx (V14.C) ───────────────────────────────────────── */

static void test_rmsnorm_bx_aligned(void) {
    /* Cover several n × bx combinations. */
    int sizes[]  = { 4, 16, 64, 256, 2560 };
    struct { int x_bx, g_bx, t_bx; } cases[] = {
        { 4, 4, 4 },     /* gamma_bx == target_bx, no rescale */
        { 6, 8, 8 },     /* same */
        { 4, 4, 6 },     /* gamma_bx != target_bx, triggers rescale */
        { 8, 12, 14 },
    };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
            m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* g = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            for (int i = 0; i < n; i++) {
                x[i] = (m4t_mtfp_t)((i * 547) % 30000 - 15000);
                g[i] = (m4t_mtfp_t)((i * 311) % 4000 + 100);
            }
            m4t_mtfp_rmsnorm_bx          (y,  x, g, cases[c].x_bx, cases[c].g_bx,
                                          cases[c].t_bx, 1, n);
            m4t_mtfp_rmsnorm_bx_scalar_ref(yr, x, g, cases[c].x_bx, cases[c].g_bx,
                                          cases[c].t_bx, 1, n);
            for (int i = 0; i < n; i++) {
                if (y[i] != yr[i]) {
                    FAIL("rmsnorm_bx aligned case=%zu n=%d i=%d: NEON=%d ref=%d",
                         c, n, i, (int)y[i], (int)yr[i]);
                    break;
                }
            }
            free(x); free(g); free(y); free(yr);
        }
    }
}

static void test_rmsnorm_bx_unaligned(void) {
    int sizes[] = { 1, 2, 3, 5, 7, 11, 17, 33, 129, 6911 };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* g = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            x[i] = (m4t_mtfp_t)((i * 191) % 25000 - 12500);
            g[i] = (m4t_mtfp_t)((i * 79)  %  3000 + 50);
        }
        m4t_mtfp_rmsnorm_bx          (y,  x, g, 4, 4, 4, 1, n);
        m4t_mtfp_rmsnorm_bx_scalar_ref(yr, x, g, 4, 4, 4, 1, n);
        for (int i = 0; i < n; i++) {
            if (y[i] != yr[i]) {
                FAIL("rmsnorm_bx unaligned n=%d i=%d: NEON=%d ref=%d",
                     n, i, (int)y[i], (int)yr[i]);
                break;
            }
        }
        free(x); free(g); free(y); free(yr);
    }
}

static void test_rmsnorm_bx_extreme(void) {
    /* MAX_VAL inputs — exercises the SoS overflow tracking and the
     * total_shift varying due to large mean. */
    int n = 256;
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* g = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) {
        x[i] = (i & 1) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
        g[i] = (i & 2) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
    }
    m4t_mtfp_rmsnorm_bx          (y,  x, g, 4, 4, 4, 1, n);
    m4t_mtfp_rmsnorm_bx_scalar_ref(yr, x, g, 4, 4, 4, 1, n);
    for (int i = 0; i < n; i++) {
        if (y[i] != yr[i]) {
            FAIL("rmsnorm_bx extreme i=%d: NEON=%d ref=%d", i, (int)y[i], (int)yr[i]);
            break;
        }
    }
    free(x); free(g); free(y); free(yr);
}

/* ── m4t_mtfp_bitlinear_scale_bx (V14.F) ───────────────────────────────── */

static void test_bitlinear_scale_bx_aligned(void) {
    /* (alpha_bx, x_bx, target_bx, alpha_m, absmax_m) parameter sweep.
     * Cover both num signs, several shift_exp ∈ [0, 25] (BitNet practical). */
    int sizes[] = { 4, 16, 64, 256, 2560 };
    struct { int a_bx, x_bx, t_bx; int alpha_m; int absmax_m; } cases[] = {
        { 0, 0, 0,  100,  100 },     /* shift_exp=0; small */
        { 1, 1, 1,  500, -500 },     /* shift_exp=1; mixed sign */
        { 4, 5, 5, 1234,  4567 },    /* shift_exp=4 */
        { 8, 10, 6, -7777, 8888 },   /* shift_exp=12 */
        { 12, 13, 5, 30000, -30000 },/* shift_exp=20 */
        { 14, 17, 6, -50000, 50000 },/* shift_exp=25 */
    };
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
            m4t_mtfp_t alpha = (m4t_mtfp_t)cases[c].alpha_m;
            m4t_mtfp_t* yraw = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* y    = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* yr   = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
            for (int i = 0; i < n; i++) {
                yraw[i] = (m4t_mtfp_t)((i * 311) % 60000 - 30000);
            }
            m4t_mtfp_bitlinear_scale_bx          (y,  yraw, &alpha,
                cases[c].a_bx, (m4t_mtfp_t)cases[c].absmax_m,
                cases[c].x_bx, cases[c].t_bx, n);
            m4t_mtfp_bitlinear_scale_bx_scalar_ref(yr, yraw, &alpha,
                cases[c].a_bx, (m4t_mtfp_t)cases[c].absmax_m,
                cases[c].x_bx, cases[c].t_bx, n);
            for (int i = 0; i < n; i++) {
                if (y[i] != yr[i]) {
                    FAIL("bitlinear_scale aligned case=%zu n=%d i=%d alpha=%d am=%d shifts=(%d,%d,%d): NEON=%d ref=%d",
                         c, n, i, (int)alpha, cases[c].absmax_m,
                         cases[c].a_bx, cases[c].x_bx, cases[c].t_bx,
                         (int)y[i], (int)yr[i]);
                    break;
                }
            }
            free(yraw); free(y); free(yr);
        }
    }
}

static void test_bitlinear_scale_bx_unaligned(void) {
    int sizes[] = { 1, 2, 3, 5, 7, 11, 17, 33, 129 };
    int alpha_m = 12345, absmax_m = -6789;
    int a_bx = 4, x_bx = 6, t_bx = 5;  /* shift_exp = 5 */
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        m4t_mtfp_t alpha = (m4t_mtfp_t)alpha_m;
        m4t_mtfp_t* yraw = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y    = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* yr   = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            yraw[i] = (m4t_mtfp_t)((i * 191) % 50000 - 25000);
        }
        m4t_mtfp_bitlinear_scale_bx          (y,  yraw, &alpha, a_bx,
            (m4t_mtfp_t)absmax_m, x_bx, t_bx, n);
        m4t_mtfp_bitlinear_scale_bx_scalar_ref(yr, yraw, &alpha, a_bx,
            (m4t_mtfp_t)absmax_m, x_bx, t_bx, n);
        for (int i = 0; i < n; i++) {
            if (y[i] != yr[i]) {
                FAIL("bitlinear_scale unaligned n=%d i=%d: NEON=%d ref=%d",
                     n, i, (int)y[i], (int)yr[i]);
                break;
            }
        }
        free(yraw); free(y); free(yr);
    }
}

static void test_bitlinear_scale_bx_extreme(void) {
    /* MAX_VAL inputs to exercise the uint96 multiply at full scale. */
    int n = 256;
    int shift_exps[] = { 0, 1, 5, 10, 25, 35 };
    for (size_t k = 0; k < sizeof(shift_exps)/sizeof(shift_exps[0]); k++) {
        int sh = shift_exps[k];
        int a_bx = sh, x_bx = 0, t_bx = 0;
        if (a_bx > 18) { a_bx = 18; x_bx = sh - 18; }
        m4t_mtfp_t alpha = M4T_MTFP_MAX_VAL;
        m4t_mtfp_t absmax = M4T_MTFP_MAX_VAL;
        m4t_mtfp_t* yraw = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y    = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* yr   = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            yraw[i] = (i & 1) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
        }
        m4t_mtfp_bitlinear_scale_bx          (y,  yraw, &alpha, a_bx,
            absmax, x_bx, t_bx, n);
        m4t_mtfp_bitlinear_scale_bx_scalar_ref(yr, yraw, &alpha, a_bx,
            absmax, x_bx, t_bx, n);
        for (int i = 0; i < n; i++) {
            if (y[i] != yr[i]) {
                FAIL("bitlinear_scale extreme sh=%d i=%d: NEON=%d ref=%d",
                     sh, i, (int)y[i], (int)yr[i]);
                break;
            }
        }
        free(yraw); free(y); free(yr);
    }
}

static void test_bitlinear_scale_bx_alpha_zero(void) {
    /* alpha = 0 → fall through to m4t_mtfp_rescale_bx; NEON and scalar_ref
     * should match. */
    int n = 16;
    m4t_mtfp_t alpha = 0;
    m4t_mtfp_t* yraw = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y    = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* yr   = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) yraw[i] = (m4t_mtfp_t)((i * 41) % 1000 - 500);
    m4t_mtfp_bitlinear_scale_bx          (y,  yraw, &alpha, 4, 100, 4, 4, n);
    m4t_mtfp_bitlinear_scale_bx_scalar_ref(yr, yraw, &alpha, 4, 100, 4, 4, n);
    for (int i = 0; i < n; i++) {
        if (y[i] != yr[i]) FAIL("bitlinear_scale alpha=0 i=%d: NEON=%d ref=%d", i, (int)y[i], (int)yr[i]);
    }
    free(yraw); free(y); free(yr);
}

/* ── m4t_mtfp_attn_v_combine (V14.B) ───────────────────────────────────── */

static void test_attn_v_combine_aligned(void) {
    /* head_dim%4=0 cases: NEON inner loop, no boundary tile. */
    int seq_k_sizes[] = { 1, 4, 32, 256 };
    int hd_sizes[]    = { 4, 8, 16, 64, 128 };
    for (size_t a = 0; a < sizeof(seq_k_sizes)/sizeof(seq_k_sizes[0]); a++) {
        int seq_k = seq_k_sizes[a];
        for (size_t b = 0; b < sizeof(hd_sizes)/sizeof(hd_sizes[0]); b++) {
            int hd = hd_sizes[b];
            size_t v_stride = (size_t)hd + 7;  /* not contiguous between rows */
            m4t_mtfp_t* w  = (m4t_mtfp_t*)calloc((size_t)seq_k, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* V  = (m4t_mtfp_t*)calloc(v_stride * (size_t)seq_k, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)hd, sizeof(m4t_mtfp_t));
            m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)hd, sizeof(m4t_mtfp_t));
            for (int t = 0; t < seq_k; t++) w[t] = (m4t_mtfp_t)((t * 311) % 5000 - 2500);
            for (size_t i = 0; i < v_stride * (size_t)seq_k; i++)
                V[i] = (m4t_mtfp_t)(((int)i * 137) % 6000 - 3000);
            m4t_mtfp_attn_v_combine          (y,  10, w, V, v_stride, seq_k, hd);
            m4t_mtfp_attn_v_combine_scalar_ref(yr, 10, w, V, v_stride, seq_k, hd);
            for (int d = 0; d < hd; d++) {
                if (y[d] != yr[d]) {
                    FAIL("attn_v_combine aligned seq_k=%d hd=%d d=%d: NEON=%d ref=%d",
                         seq_k, hd, d, (int)y[d], (int)yr[d]);
                }
            }
            free(w); free(V); free(y); free(yr);
        }
    }
}

static void test_attn_v_combine_unaligned(void) {
    /* head_dim%4 ∈ {1,2,3}: shift+clamp boundary tile and inner-loop tile fire. */
    int hd_sizes[] = { 1, 2, 3, 5, 6, 7, 9, 13, 15, 17, 33, 65, 127 };
    int seq_k = 17;
    for (size_t b = 0; b < sizeof(hd_sizes)/sizeof(hd_sizes[0]); b++) {
        int hd = hd_sizes[b];
        size_t v_stride = (size_t)hd + 3;
        m4t_mtfp_t* w  = (m4t_mtfp_t*)calloc((size_t)seq_k, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* V  = (m4t_mtfp_t*)calloc(v_stride * (size_t)seq_k, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)hd, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)hd, sizeof(m4t_mtfp_t));
        for (int t = 0; t < seq_k; t++) w[t] = (m4t_mtfp_t)((t * 41) % 4000 - 2000);
        for (size_t i = 0; i < v_stride * (size_t)seq_k; i++)
            V[i] = (m4t_mtfp_t)(((int)i * 19) % 4000 - 2000);
        m4t_mtfp_attn_v_combine          (y,  8, w, V, v_stride, seq_k, hd);
        m4t_mtfp_attn_v_combine_scalar_ref(yr, 8, w, V, v_stride, seq_k, hd);
        for (int d = 0; d < hd; d++) {
            if (y[d] != yr[d]) {
                FAIL("attn_v_combine unaligned seq_k=%d hd=%d d=%d: NEON=%d ref=%d",
                     seq_k, hd, d, (int)y[d], (int)yr[d]);
            }
        }
        free(w); free(V); free(y); free(yr);
    }
}

static void test_attn_v_combine_clamps(void) {
    /* High-magnitude weights and values to exercise the int64 saturating
     * narrow + MTFP19 clamp at the tail of the helper. */
    int seq_k = 64, hd = 32;
    size_t v_stride = (size_t)hd;
    m4t_mtfp_t* w  = (m4t_mtfp_t*)calloc((size_t)seq_k, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* V  = (m4t_mtfp_t*)calloc(v_stride * (size_t)seq_k, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y  = (m4t_mtfp_t*)calloc((size_t)hd, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* yr = (m4t_mtfp_t*)calloc((size_t)hd, sizeof(m4t_mtfp_t));
    for (int t = 0; t < seq_k; t++) w[t] = (t & 1) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
    for (size_t i = 0; i < v_stride * (size_t)seq_k; i++)
        V[i] = ((int)i & 1) ? M4T_MTFP_MAX_VAL : -M4T_MTFP_MAX_VAL;
    m4t_mtfp_attn_v_combine          (y,  30, w, V, v_stride, seq_k, hd);
    m4t_mtfp_attn_v_combine_scalar_ref(yr, 30, w, V, v_stride, seq_k, hd);
    for (int d = 0; d < hd; d++) {
        if (y[d] != yr[d]) {
            FAIL("attn_v_combine clamps d=%d: NEON=%d ref=%d", d, (int)y[d], (int)yr[d]);
        }
    }
    free(w); free(V); free(y); free(yr);
}

static void test_attn_v_combine_zero(void) {
    /* seq_k=0 leaves output untouched (caller-zeroed semantics: helper
     * returns early without writing). */
    int hd = 8;
    m4t_mtfp_t y[8];
    for (int i = 0; i < hd; i++) y[i] = 0xABCDEF;
    m4t_mtfp_attn_v_combine(y, 30, NULL, NULL, 0, 0, hd);
    for (int i = 0; i < hd; i++) {
        if (y[i] != 0xABCDEF) FAIL("attn_v_combine seq_k=0 wrote y[%d]", i);
    }
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

    test_vec_dot_i64_empty();
    test_vec_dot_i64_aligned();
    test_vec_dot_i64_unaligned();
    test_vec_dot_i64_extreme();

    test_attn_v_combine_aligned();
    test_attn_v_combine_unaligned();
    test_attn_v_combine_clamps();
    test_attn_v_combine_zero();

    test_relu2_bx_aligned();
    test_relu2_bx_unaligned();
    test_relu2_bx_extreme();

    test_elementwise_mul_bx_aligned();
    test_elementwise_mul_bx_unaligned();
    test_elementwise_mul_bx_extreme();

    test_bitlinear_scale_bx_aligned();
    test_bitlinear_scale_bx_unaligned();
    test_bitlinear_scale_bx_extreme();
    test_bitlinear_scale_bx_alpha_zero();

    test_rmsnorm_bx_aligned();
    test_rmsnorm_bx_unaligned();
    test_rmsnorm_bx_extreme();

    if (failures) {
        fprintf(stderr, "\n%d assertion(s) failed\n", failures);
        return 1;
    }
    printf("m4t_mtfp: all tests passed\n");
    return 0;
}
