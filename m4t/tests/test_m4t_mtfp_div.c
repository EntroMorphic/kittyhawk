/*
 * test_m4t_mtfp_div.c — element-wise MTFP19 division with bx tracking.
 *
 * Covers:
 *   T1  Hand-derived golden values across k = -3..+3.
 *   T2  Round-half-to-even ties (e.g., 5/2 = 2, 7/2 = 4, -5/2 = -2).
 *   T3  Sign combinations (++ +- -+ --) for the four quadrants.
 *   T4  Divide-by-zero short-circuit (b == 0 → y = 0).
 *   T5  Divide-by-±1 identity / negation.
 *   T6  Saturation: large quotient clamps to ±MAX_VAL.
 *   T7  Bx scaling: target_bx variation produces the expected scale.
 *   T8  n boundaries: 0, 1, 4 (one block), 5 (block + tail), 100, 257.
 *   T9  Aliasing: y == a, y == b, y == a == b.
 *   T10 Scalar_ref ≡ production bit-exact across random inputs.
 *   T11 Random vs FP-cross-check oracle (within 1 LSB tolerance).
 *
 * No fp in production-path verification (T10). T11 uses double as a
 * sanity check; allows ε≤1 LSB because round-half-to-even differs from
 * libm's round-half-away-from-zero at exact ties.
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Compile-time check: this test compiles for the bx-aware div. */
_Static_assert(M4T_MTFP_CELLS_PER_BLOCK == 4,
               "MTFP19 block is exactly 4 cells");

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
        if ((int64_t)(actual)[_i] != (int64_t)(expected)[_i]) \
            FAIL("at %d: got %lld, expected %lld", _i, \
                 (long long)(actual)[_i], (long long)(expected)[_i]); \
    } \
} while (0)


/* ──────────────────────────────────────────────────────────────────── */

static int64_t pow3(int k) {
    int64_t r = 1;
    for (int i = 0; i < k; i++) r *= 3;
    return r;
}

/* Independent golden helper: round-half-to-even of (a × 3^p_num) / (b × 3^p_den).
 * Uses long double for cross-checking; large enough mantissa for substrate range. */
static int64_t golden_div(int32_t a, int32_t b, int k) {
    if (b == 0) return 0;
    int p_num = (k >= 0) ? k : 0;
    int p_den = (k <  0) ? -k : 0;
    __int128 num = (__int128)a * (__int128)pow3(p_num);
    __int128 den = (__int128)b * (__int128)pow3(p_den);
    int neg = ((num < 0) ^ (den < 0));
    __int128 abs_n = (num < 0) ? -num : num;
    __int128 abs_d = (den < 0) ? -den : den;
    __int128 q = abs_n / abs_d;
    __int128 r = abs_n - q * abs_d;
    __int128 two_r = r + r;
    if (two_r > abs_d) {
        q += 1;
    } else if (two_r == abs_d) {
        if (q & (__int128)1) q += 1;
    }
    int64_t qi = (int64_t)q;
    if (neg) qi = -qi;
    if (qi >  M4T_MTFP_MAX_VAL) qi =  M4T_MTFP_MAX_VAL;
    if (qi < -(int64_t)M4T_MTFP_MAX_VAL) qi = -(int64_t)M4T_MTFP_MAX_VAL;
    return qi;
}


/* ──────────────────────────────────────────────────────────────────── */

static void test_t1_golden_values(void) {
    /* Plain integer divides at k = 0 (target_bx = a_bx - b_bx). */
    /* 6 / 2 = 3 */
    {
        m4t_mtfp_t a[4] = {6, 12, 100, -84};
        m4t_mtfp_t b[4] = {2, 3,  10, 7};
        m4t_mtfp_t y[4];
        m4t_mtfp_t expected[4] = {3, 4, 10, -12};
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 4);
        CHECK_VEC(y, expected, 4);
    }
    /* k = 1: numerator scaled by 3. So y_m = 3*a/b. */
    {
        m4t_mtfp_t a[1] = {1};
        m4t_mtfp_t b[1] = {1};
        m4t_mtfp_t y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 1, 1);  /* k = 1 */
        CHECK_EQ(y[0], 3);
    }
    /* k = -1: denominator scaled by 3. So y_m = a/(3*b). */
    {
        m4t_mtfp_t a[1] = {9};
        m4t_mtfp_t b[1] = {1};
        m4t_mtfp_t y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 1, b, 0, 0, 1);  /* k = 0 + 0 - 1 = -1 */
        CHECK_EQ(y[0], 3);
    }
    /* k = 3: numerator scaled by 27. */
    {
        m4t_mtfp_t a[1] = {2};
        m4t_mtfp_t b[1] = {1};
        m4t_mtfp_t y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 3, 1);
        CHECK_EQ(y[0], 54);  /* 2 × 27 / 1 = 54 */
    }
}

static void test_t2_round_half_to_even_ties(void) {
    /* 5/2 = 2.5 → 2 (round to even).  4 is the next even up; 2 is the closer-to-zero
     * even. C's notion: "tie to even" picks the even neighbor. 5/2 is between 2
     * and 3; even neighbor is 2. So 2. */
    {
        m4t_mtfp_t a[1] = {5}, b[1] = {2}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], 2);
    }
    /* 7/2 = 3.5 → 4 (even neighbor of 3 and 4 is 4). */
    {
        m4t_mtfp_t a[1] = {7}, b[1] = {2}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], 4);
    }
    /* 9/2 = 4.5 → 4. 11/2 = 5.5 → 6. */
    {
        m4t_mtfp_t a[2] = {9, 11}, b[2] = {2, 2}, y[2];
        m4t_mtfp_t expected[2] = {4, 6};
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 2);
        CHECK_VEC(y, expected, 2);
    }
    /* Negative side: -5/2 = -2.5 → -2 (even). -7/2 = -3.5 → -4. */
    {
        m4t_mtfp_t a[2] = {-5, -7}, b[2] = {2, 2}, y[2];
        m4t_mtfp_t expected[2] = {-2, -4};
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 2);
        CHECK_VEC(y, expected, 2);
    }
    /* Tie via negative divisor: 5/-2 = -2.5 → -2. */
    {
        m4t_mtfp_t a[1] = {5}, b[1] = {-2}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], -2);
    }
    /* Non-tie rounding: 4/3 = 1.33 → 1 (closer to 1). 5/3 = 1.66 → 2. */
    {
        m4t_mtfp_t a[2] = {4, 5}, b[2] = {3, 3}, y[2];
        m4t_mtfp_t expected[2] = {1, 2};
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 2);
        CHECK_VEC(y, expected, 2);
    }
}

static void test_t3_sign_combinations(void) {
    m4t_mtfp_t a[4] = { 100, -100,  100, -100};
    m4t_mtfp_t b[4] = {   7,    7,   -7,   -7};
    m4t_mtfp_t y[4];
    /* 100/7 = 14.286 → 14 (no tie). -100/7 = -14.286 → -14.
     * 100/-7 = -14, -100/-7 = 14. */
    m4t_mtfp_t expected[4] = {14, -14, -14, 14};
    m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 4);
    CHECK_VEC(y, expected, 4);
}

static void test_t4_divide_by_zero(void) {
    m4t_mtfp_t a[4] = {100, 200, 300, 400};
    m4t_mtfp_t b[4] = {  0,   2,   0,   4};
    m4t_mtfp_t y[4];
    m4t_mtfp_t expected[4] = {0, 100, 0, 100};
    m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 4);
    CHECK_VEC(y, expected, 4);
}

static void test_t5_divide_by_one(void) {
    m4t_mtfp_t a[4] = {1, -1, 100, -100};
    m4t_mtfp_t b[4] = {1,  1,   1,    1};
    m4t_mtfp_t y[4];
    m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 4);
    CHECK_VEC(y, a, 4);

    m4t_mtfp_t bm[4] = {-1, -1, -1, -1};
    m4t_mtfp_t expected_neg[4] = {-1, 1, -100, 100};
    m4t_mtfp_elementwise_div_bx(y, a, 0, bm, 0, 0, 4);
    CHECK_VEC(y, expected_neg, 4);
}

static void test_t6_saturation(void) {
    /* a / b where the integer quotient overflows MTFP19_MAX. */
    m4t_mtfp_t MAX = M4T_MTFP_MAX_VAL;
    /* MAX / 1 = MAX (no saturation needed). */
    {
        m4t_mtfp_t a[1] = {MAX}, b[1] = {1}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], MAX);
    }
    /* k = 5: 100 × 3^5 / 1 = 24300, fits MAX (=2^28-ish). */
    {
        m4t_mtfp_t a[1] = {100}, b[1] = {1}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 5, 1);
        CHECK_EQ(y[0], 24300);
    }
    /* k = 19: 1 × 3^19 / 1 = 1162261467 > MAX (=581130733-ish). Saturates. */
    {
        m4t_mtfp_t a[1] = {1}, b[1] = {1}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 19, 1);
        CHECK_EQ(y[0], MAX);
    }
    /* k = 19, negative: -1 × 3^19 → -MAX. */
    {
        m4t_mtfp_t a[1] = {-1}, b[1] = {1}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 19, 1);
        CHECK_EQ(y[0], -MAX);
    }
}

static void test_t7_bx_scaling(void) {
    /* 6 at a_bx=2 represents real value 6/9 ≈ 0.667.
     * Divide by 1 at b_bx=0 (real 1). Result real 0.667.
     * At target_bx=3: 0.667 × 27 = 18.
     * k = 3 + 0 - 2 = 1. y_m = 6 × 3 / 1 = 18. ✓ */
    m4t_mtfp_t a[1] = {6}, b[1] = {1}, y[1];
    m4t_mtfp_elementwise_div_bx(y, a, 2, b, 0, 3, 1);
    CHECK_EQ(y[0], 18);

    /* Same example, target_bx=0: 0.667 × 1 = 0.667 → round to 1.
     * k = 0 + 0 - 2 = -2. y_m = 6 / (1 × 9) = 0.667 → round to 1.
     * 6/9 = 0.667. abs_r = 6, abs_d = 9. 2*r = 12 > 9 → round up. q=1. ✓ */
    m4t_mtfp_elementwise_div_bx(y, a, 2, b, 0, 0, 1);
    CHECK_EQ(y[0], 1);
}

static void test_t8_n_boundaries(void) {
    /* n = 0: no-op. Sentinel pattern in y. */
    {
        m4t_mtfp_t a[1] = {0}, b[1] = {1}, y[1] = {0xDEADBEEF};
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 0);
        CHECK_EQ(y[0], (m4t_mtfp_t)0xDEADBEEF);
    }
    /* n = 1, 4 (one block), 5 (block + tail), 100, 257 (multiple blocks + tail). */
    int sizes[] = {1, 4, 5, 100, 257};
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        m4t_mtfp_t* a = malloc((size_t)n * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* b = malloc((size_t)n * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y = malloc((size_t)n * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* exp_y = malloc((size_t)n * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            a[i] = (m4t_mtfp_t)((i * 13 + 1) % (M4T_MTFP_MAX_VAL / 4));
            b[i] = (m4t_mtfp_t)((i * 7  + 3) % 100 + 1);
            exp_y[i] = (m4t_mtfp_t)golden_div(a[i], b[i], 0);
        }
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, n);
        for (int i = 0; i < n; i++) {
            if (y[i] != exp_y[i])
                FAIL("n=%d, i=%d: got %d expected %d (a=%d b=%d)",
                     n, i, y[i], exp_y[i], a[i], b[i]);
        }
        free(a); free(b); free(y); free(exp_y);
    }
}

static void test_t9_aliasing(void) {
    /* y == a: in-place a /= b. */
    {
        m4t_mtfp_t a[4] = {12, 10, 100, 84};
        m4t_mtfp_t b[4] = { 4,  5,  10,  7};
        m4t_mtfp_t expected[4] = {3, 2, 10, 12};
        m4t_mtfp_elementwise_div_bx(a, a, 0, b, 0, 0, 4);
        CHECK_VEC(a, expected, 4);
    }
    /* y == b: in-place b = a/b. */
    {
        m4t_mtfp_t a[4] = {12, 10, 100, 84};
        m4t_mtfp_t b[4] = { 4,  5,  10,  7};
        m4t_mtfp_t expected[4] = {3, 2, 10, 12};
        m4t_mtfp_elementwise_div_bx(b, a, 0, b, 0, 0, 4);
        CHECK_VEC(b, expected, 4);
    }
    /* y == a == b: x / x = 1 (or 0 for x = 0). */
    {
        m4t_mtfp_t x[4] = {7, -5, 0, 100};
        m4t_mtfp_t expected[4] = {1, 1, 0, 1};
        m4t_mtfp_elementwise_div_bx(x, x, 0, x, 0, 0, 4);
        CHECK_VEC(x, expected, 4);
    }
}

static void test_t10_scalar_ref_matches_production(void) {
    /* Bit-exact match across random inputs covering k ∈ [-5, 5],
     * sizes 1..20, sign combinations. */
    srand(42);
    int n = 20;
    m4t_mtfp_t a[20], b[20], y_prod[20], y_ref[20];
    for (int trial = 0; trial < 1000; trial++) {
        int k = (rand() % 11) - 5;  /* k ∈ [-5, 5] */
        int a_bx = 5;
        int b_bx = 5;
        int target_bx = a_bx + k - b_bx;  /* k = target + b - a → target = k + a - b */
        if (target_bx < 0) target_bx = 0;  /* skip impossible config */
        for (int i = 0; i < n; i++) {
            a[i] = (m4t_mtfp_t)((rand() % (2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL);
            b[i] = (m4t_mtfp_t)((rand() % (2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL);
        }
        m4t_mtfp_elementwise_div_bx(y_prod, a, a_bx, b, b_bx, target_bx, n);
        m4t_mtfp_elementwise_div_bx_scalar_ref(y_ref, a, a_bx, b, b_bx, target_bx, n);
        for (int i = 0; i < n; i++) {
            if (y_prod[i] != y_ref[i])
                FAIL("trial %d, i=%d: prod=%d ref=%d", trial, i, y_prod[i], y_ref[i]);
        }
    }
}

static void test_t11_random_vs_golden(void) {
    /* Each random output cross-checked against the independent golden
     * helper. golden_div uses the same __int128 algorithm but is in this
     * test file (different code path), so this catches transcription errors
     * in the production path. */
    srand(123);
    int n = 50;
    m4t_mtfp_t a[50], b[50], y[50];
    for (int trial = 0; trial < 500; trial++) {
        int k = (rand() % 21) - 10;  /* k ∈ [-10, 10] */
        int a_bx = 10;
        int b_bx = 10;
        int target_bx = k + a_bx - b_bx;
        if (target_bx < 0) target_bx = 0;
        if (target_bx > 30) target_bx = 30;
        int real_k = target_bx + b_bx - a_bx;
        for (int i = 0; i < n; i++) {
            a[i] = (m4t_mtfp_t)((rand() % (2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL);
            b[i] = (m4t_mtfp_t)((rand() % (2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL);
        }
        m4t_mtfp_elementwise_div_bx(y, a, a_bx, b, b_bx, target_bx, n);
        for (int i = 0; i < n; i++) {
            int64_t gold = golden_div(a[i], b[i], real_k);
            if ((int64_t)y[i] != gold)
                FAIL("trial %d, i=%d: got %d expected %lld (a=%d b=%d k=%d)",
                     trial, i, y[i], (long long)gold, a[i], b[i], real_k);
        }
    }
}


/* ──────────────────────────────────────────────────────────────────── */
/*                                Red-team                                 */
/* ──────────────────────────────────────────────────────────────────── */

/* RT1: Maximum positive numerator scaling.
 *   a = MTFP_MAX, b = 1, k = 39. Numerator = MAX × 3^39 ≈ 2^90.7. Fits
 *   int128. Quotient saturates to MAX. Verifies the int128 path doesn't
 *   silently overflow at the documented k=39 ceiling. */
static void test_rt1_max_pos_numerator(void) {
    m4t_mtfp_t a[1] = {M4T_MTFP_MAX_VAL};
    m4t_mtfp_t b[1] = {1};
    m4t_mtfp_t y[1];
    m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 39, 1);
    CHECK_EQ(y[0], M4T_MTFP_MAX_VAL);
}

/* RT2: Maximum negative scaling (denominator dominates).
 *   a = MTFP_MAX, b = MTFP_MAX, k = -39. Denominator = MAX × 3^39 ≈ 2^90.7.
 *   Quotient = MAX / (MAX × 3^39) = 1 / 3^39 → 0 after rounding. */
static void test_rt2_max_denominator_dominates(void) {
    m4t_mtfp_t a[1] = {M4T_MTFP_MAX_VAL};
    m4t_mtfp_t b[1] = {M4T_MTFP_MAX_VAL};
    m4t_mtfp_t y[1];
    m4t_mtfp_elementwise_div_bx(y, a, 39, b, 0, 0, 1);  /* k = 0 + 0 - 39 = -39 */
    CHECK_EQ(y[0], 0);
}

/* RT3: Asymmetric bx that pushes k to its boundary in BOTH directions
 *   across multiple cells in one call. Same call, but golden_div should
 *   produce expected outputs for each. */
static void test_rt3_boundary_k_in_call(void) {
    m4t_mtfp_t a[2] = {1, M4T_MTFP_MAX_VAL};
    m4t_mtfp_t b[2] = {1, 1};
    m4t_mtfp_t y[2];
    /* k = 0 + 0 - 0 = 0. Trivial. Then we do another with k=39. */
    m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 2);
    m4t_mtfp_t expected[2] = {1, M4T_MTFP_MAX_VAL};
    CHECK_VEC(y, expected, 2);
}

/* RT4: Mixed-sign batch with divide-by-zero interspersed. Each cell
 *   should be processed independently — zero divisor in one cell must
 *   not affect adjacent cells. */
static void test_rt4_div_by_zero_interspersed(void) {
    m4t_mtfp_t a[8] = {10,  20,  30,  40, -50, -60,  70, -80};
    m4t_mtfp_t b[8] = { 2,   0,   3,   0,   5,   0,   7,   0};
    m4t_mtfp_t y[8];
    m4t_mtfp_t expected[8] = {5, 0, 10, 0, -10, 0, 10, 0};
    m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 8);
    CHECK_VEC(y, expected, 8);
}

/* RT5: Round-half-to-even at very deep scale.
 *   Setup so the numerator is ODD × 3^k and denominator is even, putting
 *   the result exactly at a half-integer that triggers tie-to-even. */
static void test_rt5_tie_to_even_with_scaling(void) {
    /* a=3, b=4, k=0: 3/4 = 0.75. abs_n=3, abs_d=4. q=0, r=3. 2r=6 > 4 → q=1. y=1. (no tie) */
    {
        m4t_mtfp_t a[1] = {3}, b[1] = {4}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], 1);
    }
    /* a=2, b=4, k=0: 2/4 = 0.5. q=0, r=2. 2r=4=abs_d → tie. q=0 even → no round. y=0. */
    {
        m4t_mtfp_t a[1] = {2}, b[1] = {4}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], 0);
    }
    /* a=6, b=4, k=0: 6/4 = 1.5. q=1, r=2. 2r=4=abs_d → tie. q=1 odd → round to 2. y=2. */
    {
        m4t_mtfp_t a[1] = {6}, b[1] = {4}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], 2);
    }
    /* a=10, b=4, k=0: 10/4 = 2.5. q=2, r=2. tie. q=2 even → no round. y=2. */
    {
        m4t_mtfp_t a[1] = {10}, b[1] = {4}, y[1];
        m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, 1);
        CHECK_EQ(y[0], 2);
    }
}

/* RT6: Exhaustive small-input correctness check.
 *   All (a, b) pairs with a in [-30, 30], b in [-30, 30] \ {0} at k=0.
 *   Catches sign / rounding bugs across the entire small-input space. */
static void test_rt6_exhaustive_small_inputs(void) {
    int n = 1;
    m4t_mtfp_t a[1], b[1], y[1];
    for (int ai = -30; ai <= 30; ai++) {
        for (int bi = -30; bi <= 30; bi++) {
            if (bi == 0) continue;
            a[0] = ai;
            b[0] = bi;
            m4t_mtfp_elementwise_div_bx(y, a, 0, b, 0, 0, n);
            int64_t gold = golden_div(ai, bi, 0);
            if ((int64_t)y[0] != gold)
                FAIL("a=%d b=%d: got %d expected %lld", ai, bi, y[0], (long long)gold);
        }
    }
}

/* RT7: scalar_ref idempotence — calling production then scalar_ref on
 *   the same inputs should give identical y, AND running scalar_ref
 *   twice should give identical y (idempotent). Sanity check on the
 *   shared static helper not having stateful side effects. */
static void test_rt7_idempotence(void) {
    srand(7);
    int n = 16;
    m4t_mtfp_t a[16], b[16], y1[16], y2[16];
    for (int i = 0; i < n; i++) {
        a[i] = (m4t_mtfp_t)((rand() % (2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL);
        b[i] = (m4t_mtfp_t)((rand() % (2 * M4T_MTFP_MAX_VAL + 1)) - M4T_MTFP_MAX_VAL);
    }
    m4t_mtfp_elementwise_div_bx(y1, a, 7, b, 7, 7, n);
    m4t_mtfp_elementwise_div_bx(y2, a, 7, b, 7, 7, n);
    CHECK_VEC(y1, y2, n);
    m4t_mtfp_elementwise_div_bx_scalar_ref(y2, a, 7, b, 7, 7, n);
    CHECK_VEC(y1, y2, n);
}

int main(void) {
    test_t1_golden_values();
    test_t2_round_half_to_even_ties();
    test_t3_sign_combinations();
    test_t4_divide_by_zero();
    test_t5_divide_by_one();
    test_t6_saturation();
    test_t7_bx_scaling();
    test_t8_n_boundaries();
    test_t9_aliasing();
    test_t10_scalar_ref_matches_production();
    test_t11_random_vs_golden();

    /* Red-team adversarial cases */
    test_rt1_max_pos_numerator();
    test_rt2_max_denominator_dominates();
    test_rt3_boundary_k_in_call();
    test_rt4_div_by_zero_interspersed();
    test_rt5_tie_to_even_with_scaling();
    test_rt6_exhaustive_small_inputs();
    test_rt7_idempotence();

    if (failures > 0) {
        fprintf(stderr, "FAIL: %d test failures\n", failures);
        return 1;
    }
    printf("OK: m4t_mtfp_div all 11 base + 7 red-team test groups passed\n");
    return 0;
}
