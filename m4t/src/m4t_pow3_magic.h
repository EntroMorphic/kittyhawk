/*
 * m4t_pow3_magic.h — magic-multiply constants for divide-by-3^k.
 *
 * Single source of truth for the divide-by-3^k NEON kernel and any future
 * consumer that needs to round-divide by 3^k via magic-multiply.
 *
 * Generation: m4t/tools/gen_pow3_magic.c. The generator does:
 *   - For each k ∈ [1, 19], finds (M, N) such that for all
 *     x ∈ [-MAX_VAL, +MAX_VAL]:
 *       (int32_t)(((int64_t)x * M + (1LL << (N-1))) >> N)
 *         == m4t_pow3_round_div(x, 3^k)
 *   - Verifies bit-exact across the FULL range (1.16 × 10⁹ × 19 = 2.2 × 10¹⁰
 *     test points). Generator runtime ~25 seconds. PASS verdict per
 *     shift3_neon synthesize cycle's G1.
 *
 * Saturation argument (per G2):
 *   |x|         ≤ MAX_VAL = (3^19 - 1)/2 ≈ 2^29.1
 *   M           ≤ INT32_MAX < 2^31
 *   |x*M|       ≤ 2^60.1
 *   |x*M+bias|  ≤ 2^61.0    (verified empirically; 2-bit headroom under INT64_MAX)
 *   |result|    ≤ MAX_VAL/3 ≈ 2^27.5  (3.47-bit headroom under INT32_MAX)
 *
 * Regenerating: re-run m4t/tools/gen_pow3_magic.c, copy the output below.
 * NEVER EDIT BY HAND — values are the result of an exhaustive search and
 * must match the generator's output bit-exactly. */

#ifndef M4T_POW3_MAGIC_H
#define M4T_POW3_MAGIC_H

#include <stdint.h>

/* M_table[k] for k ∈ [0, 19]. M_table[0] is unused (k=0 is identity copy). */
static const int32_t M4T_POW3_DIV_M[20] = {
             0,  /* k= 0: identity */
    1431655765,  /* k= 1  d=3 */
    1908874353,  /* k= 2  d=9 */
    1272582902,  /* k= 3  d=27 */
    1696777203,  /* k= 4  d=81 */
    1131184802,  /* k= 5  d=243 */
    1508246402,  /* k= 6  d=729 */
    2010995203,  /* k= 7  d=2187 */
    1340663469,  /* k= 8  d=6561 */
    1787551292,  /* k= 9  d=19683 */
    1191700861,  /* k=10  d=59049 */
    1588934482,  /* k=11  d=177147 */
    2118579309,  /* k=12  d=531441 */
    1412386206,  /* k=13  d=1594323 */
    1883181608,  /* k=14  d=4782969 */
    1255454405,  /* k=15  d=14348907 */
    1673939207,  /* k=16  d=43046721 */
    1115959471,  /* k=17  d=129140163 */
    1487945962,  /* k=18  d=387420489 */
    1983927949,  /* k=19  d=1162261467 */
};

/* Right-shift count N per k. */
static const int8_t M4T_POW3_DIV_N[20] = {
     0,  /* k= 0: unused */
    32, 34, 35, 37, 38, 40, 42, 43, 45, 46,
    48, 50, 51, 53, 54, 56, 57, 59, 61
};

#endif /* M4T_POW3_MAGIC_H */
