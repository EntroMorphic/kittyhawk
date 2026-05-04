/*
 * gen_pow3_magic.c — derive magic constants for 64-bit-intermediate
 * NEON divide-by-3^k. Prototype tool for the proposed
 * `m4t_shift3_div_neon` kernel.
 *
 * Strategy (after the vqrdmulh + vrshl two-stage approach failed
 * bit-exactness due to compound rounding): use vmull_s32 to compute
 * an int64 intermediate, add a constant bias for round-half-up, then
 * shift right and narrow to int32. Pipeline:
 *
 *   prod = (int64_t)x * M + bias        // int64
 *   result = (int32_t)(prod >> N)        // arithmetic shift
 *
 * Where M = round(2^N / d) and bias = 2^(N-1) implements round-to-nearest
 * (round-half-up; for odd d the substrate has no ties so this matches
 * m4t_pow3_round_div bit-exact).
 *
 * For each k ∈ [1, 19], we search (M, N) for one where:
 *   M ∈ [1, 2^31 - 1] (fits in int32 broadcast for vmull)
 *   N ∈ [32, 62] (intermediate stays within int64)
 *   |x*M + bias| < 2^63 for all x in substrate input range (no overflow)
 *   For all x in [-MAX_VAL, +MAX_VAL]: result == m4t_pow3_round_div(x, d)
 *
 * Build: cc -O2 m4t/tools/gen_pow3_magic.c -o /tmp/gen_pow3_magic
 * Run:   /tmp/gen_pow3_magic
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>

#define MAX_VAL 581130733  /* (3^19 - 1) / 2 */

static const int64_t POW3[20] = {
    1LL, 3LL, 9LL, 27LL, 81LL, 243LL, 729LL, 2187LL, 6561LL, 19683LL,
    59049LL, 177147LL, 531441LL, 1594323LL, 4782969LL, 14348907LL,
    43046721LL, 129140163LL, 387420489LL, 1162261467LL,
};

/* Reference: substrate's m4t_pow3_round_div verbatim. */
static int64_t pow3_round_div_ref(int64_t M, int64_t s) {
    int64_t q = M / s;
    int64_t rem = M - q * s;
    if (rem > 0) {
        if (2 * rem > s) q += 1;
    } else if (rem < 0) {
        if (2 * (-rem) > s) q -= 1;
    }
    return q;
}

/* Magic-multiply div emulator. result = (x*M + bias) >> N (arith shift). */
static int32_t magic_div_emul(int32_t x, int32_t M, int N) {
    int64_t prod = (int64_t)x * (int64_t)M;
    int64_t bias = (int64_t)1 << (N - 1);
    int64_t adj  = prod + bias;
    return (int32_t)(adj >> N);  /* arithmetic right shift */
}

/* Smart-set verification: 100K base points × ±3 deltas + corners. */
static int64_t verify_smart_set(int32_t M, int N, int64_t d) {
    int64_t mismatches = 0;
    int64_t step = (2 * (int64_t)MAX_VAL) / 100000;
    if (step < 1) step = 1;

    for (int64_t base = -MAX_VAL; base <= MAX_VAL; base += step) {
        for (int delta = -3; delta <= 3; delta++) {
            int64_t x64 = base + delta;
            if (x64 < -MAX_VAL || x64 > MAX_VAL) continue;
            int32_t x = (int32_t)x64;
            int32_t neon = magic_div_emul(x, M, N);
            int64_t ref  = pow3_round_div_ref(x, d);
            if ((int64_t)neon != ref) mismatches++;
        }
    }
    int32_t corners[5] = {0, 1, -1, MAX_VAL, -MAX_VAL};
    for (int i = 0; i < 5; i++) {
        int32_t x = corners[i];
        int32_t neon = magic_div_emul(x, M, N);
        int64_t ref  = pow3_round_div_ref(x, d);
        if ((int64_t)neon != ref) mismatches++;
    }
    return mismatches;
}

/* Exhaustive bit-exact verification across [-MAX_VAL, +MAX_VAL]. */
static int64_t verify_exhaustive(int32_t M, int N, int64_t d) {
    int64_t mismatches = 0;
    for (int64_t x64 = -MAX_VAL; x64 <= MAX_VAL; x64++) {
        int32_t x = (int32_t)x64;
        int32_t neon = magic_div_emul(x, M, N);
        int64_t ref  = pow3_round_div_ref(x, d);
        if ((int64_t)neon != ref) mismatches++;
    }
    return mismatches;
}

/* Search for (M, N) bit-exact for divide-by-d.
 *
 * For each N starting from largest (where M = 2^N/d still fits in int32),
 * try the few M values around floor and ceil of 2^N/d. Prefer the largest
 * N (highest precision) to reduce the chance of off-by-one at boundaries.
 * Two-stage: smart-set winnow → exhaustive verify on smart-set winners.
 */
static int derive_and_verify(int k, int32_t* out_M, int* out_N) {
    int64_t d = POW3[k];

    /* N_max = largest N such that round(2^N / d) ≤ INT32_MAX.
     * Bisect: try increasing N until 2^N / d (computed in int64) overflows. */
    int N_max = 32;
    while (N_max <= 62) {
        int64_t pow_2 = (int64_t)1 << N_max;
        int64_t M_test = pow_2 / d;
        if (M_test > INT32_MAX) break;
        N_max++;
    }
    N_max -= 1;  /* last N that fit */
    if (N_max > 62) N_max = 62;  /* keep x*M + bias within int64 */

    for (int N = N_max; N >= 32; N--) {
        int64_t pow_2 = (int64_t)1 << N;
        int64_t M_floor = pow_2 / d;
        for (int delta = 0; delta <= 8; delta++) {
            for (int sign = +1; sign >= -1; sign -= 2) {
                int64_t M_try = M_floor + (int64_t)sign * delta;
                if (M_try <= 0 || M_try > INT32_MAX) continue;
                if (verify_smart_set((int32_t)M_try, N, d) != 0) continue;
                if (verify_exhaustive((int32_t)M_try, N, d) == 0) {
                    *out_M = (int32_t)M_try;
                    *out_N = N;
                    return 0;
                }
                if (delta == 0) break;  /* sign=+1 and sign=-1 give same M */
            }
        }
    }
    fprintf(stderr,
            "k=%d d=%" PRId64 " : NO bit-exact (M, N) found; N_max=%d\n",
            k, d, N_max);
    return 1;
}

int main(void) {
    printf("/* Generated by m4t/tools/gen_pow3_magic.c — DO NOT EDIT.\n");
    printf(" * Magic-multiply constants for round-to-nearest divide-by-3^k.\n");
    printf(" * Pipeline: result = (int32_t)((int64_t)x * M_table[k] +\n");
    printf(" *                              (1LL << (N_table[k] - 1))) >> N_table[k]\n");
    printf(" * Verified bit-exact vs m4t_pow3_round_div across full input range. */\n\n");

    int32_t M_table[20] = {0};
    int     N_table[20] = {0};
    int     fail = 0;

    for (int k = 1; k <= 19; k++) {
        int32_t M;
        int N;
        if (derive_and_verify(k, &M, &N) != 0) {
            fail = 1;
            continue;
        }
        M_table[k] = M;
        N_table[k] = N;
    }

    if (fail) {
        fprintf(stderr, "FAIL: at least one k did not verify bit-exact.\n");
        return 1;
    }

    printf("static const int32_t M_table[20] = {\n");
    printf("    0,  /* k=0: identity copy, no magic needed */\n");
    for (int k = 1; k <= 19; k++) {
        printf("    %10" PRId32 ",  /* k=%2d  d=3^%-2d=%-13" PRId64 "  N=%2d */\n",
               M_table[k], k, k, POW3[k], N_table[k]);
    }
    printf("};\n\n");

    printf("static const int8_t N_table[20] = {\n");
    printf("    0,  /* k=0 */\n");
    for (int k = 1; k <= 19; k++) {
        printf("    %2d,  /* k=%2d */\n", N_table[k], k);
    }
    printf("};\n\n");

    printf("/* PASS: all k in [1, 19] verified bit-exact across full input range. */\n");
    fprintf(stderr, "PASS: 19/19 k values verified bit-exact.\n");
    return 0;
}
