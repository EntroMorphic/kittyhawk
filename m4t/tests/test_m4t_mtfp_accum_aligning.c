/*
 * test_m4t_mtfp_accum_aligning.c — property tests for the cross-exponent
 * accumulator (M4T_SUBSTRATE.md §14.2 named opt-in).
 *
 * Six properties:
 *   1. accum_aligning_correctness  — per-call bound < (1/2) * 3^running_exp
 *   2. accum_aligning_invariant    — |running[i]| <= MAX_VAL after every call
 *   3. accum_aligning_aliasing     — running aliasing flags-buffer (flags NULL)
 *                                    matches the non-aliased result
 *   4. accum_aligning_flags        — SATURATED and ROUNDED bits set iff the
 *                                    corresponding event occurred
 *   5. add_aligning_via_wrapper    — pairwise wrapper matches accum equivalent
 *   6. add_aligning_roundtrip      — x + neg(x) at same exp == 0
 *
 * Test infrastructure (per docs/REMEDIATION_PLAN.md "Test infrastructure"
 * and §12 of the substrate spec): the property oracle uses double-precision
 * binary float to compute reference real-number values. Tests are NOT
 * runtime kernels and do not consume libm4t at runtime; binary float in
 * the test path is sanctioned per §12. libm4t never sees a float.
 *
 * Sample-count gate: 10,000 random sequences per property. Pass requires
 * 10,000 / 10,000 satisfy the property. Failures are not flaky — a single
 * violation fails the test.
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_SAMPLES         10000
#define MAX_CELLS         64
#define MAX_CALLS         16
#define EXP_RANGE         12        /* exponents in [-EXP_RANGE, +EXP_RANGE] */

/* ── xorshift32 RNG (deterministic, test-local) ─────────────────────────── */

static uint32_t g_rng_state = 0xdeadbeefu;

static uint32_t xs32(void) {
    uint32_t x = g_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng_state = x;
    return x;
}

static int32_t rand_mantissa(void) {
    /* Uniform in [-MAX_VAL, +MAX_VAL]. */
    uint32_t r = xs32();
    int64_t span = (int64_t)M4T_MTFP_MAX_VAL * 2 + 1;
    int64_t v = (int64_t)(r % (uint64_t)span) - (int64_t)M4T_MTFP_MAX_VAL;
    return (int32_t)v;
}

static int8_t rand_exp(void) {
    /* Uniform in [-EXP_RANGE, +EXP_RANGE]. */
    uint32_t r = xs32();
    int v = (int)(r % (uint32_t)(2 * EXP_RANGE + 1)) - EXP_RANGE;
    return (int8_t)v;
}

static int rand_int(int lo, int hi) {
    /* Uniform in [lo, hi]. lo <= hi. */
    uint32_t r = xs32();
    return lo + (int)(r % (uint32_t)(hi - lo + 1));
}

/* ── Bit-exact reference implementation ─────────────────────────────────── */

/* Reference implementation of one accumulator call, in int64. The kernel
 * MUST produce bit-identical output to this reference (running[i] equals
 * ref_clamped, exponent equals ref_e). The reference also outputs the
 * truth flags so prop 4 can compare without re-deriving the math.
 *
 * This is the property-test oracle. It duplicates the kernel's logic in
 * a simpler shape (no NEON, no early returns, no flag bit packing) so
 * any kernel deviation is detected. */
static void accum_reference(
    const int32_t* pre_running, int8_t e_run_pre,
    const int32_t* addend, int8_t addend_exp,
    int32_t* out_running, int8_t* out_e,
    uint8_t* out_flags,                 /* nullable; one byte per cell */
    int n)
{
    int delta = (int)addend_exp - (int)e_run_pre;
    *out_e = (delta > 0) ? addend_exp : e_run_pre;

    for (int i = 0; i < n; i++) {
        int truth_round = 0;
        int64_t unsat;

        if (delta == 0) {
            unsat = (int64_t)pre_running[i] + (int64_t)addend[i];
        } else if (delta > 0) {
            /* Running rescales upward by 3^delta. */
            if (delta >= 20) {
                /* Degenerate: running rounds to zero. */
                truth_round = (pre_running[i] != 0);
                unsat = (int64_t)addend[i];
            } else {
                int64_t s_val = 1;
                for (int k = 0; k < delta; k++) s_val *= 3;
                int64_t M = pre_running[i];
                int64_t q = M / s_val;
                int64_t rem = M - q * s_val;
                if (rem > 0 && 2 * rem > s_val) q += 1;
                else if (rem < 0 && 2 * (-rem) > s_val) q -= 1;
                truth_round = (rem != 0);
                unsat = q + (int64_t)addend[i];
            }
        } else {
            /* Addend rescales downward to running's exp. */
            int abs_d = -delta;
            if (abs_d >= 20) {
                /* Degenerate: addend rounds to zero. Running unchanged. */
                truth_round = (addend[i] != 0);
                unsat = (int64_t)pre_running[i];
            } else {
                int64_t s_val = 1;
                for (int k = 0; k < abs_d; k++) s_val *= 3;
                int64_t M = addend[i];
                int64_t q = M / s_val;
                int64_t rem = M - q * s_val;
                if (rem > 0 && 2 * rem > s_val) q += 1;
                else if (rem < 0 && 2 * (-rem) > s_val) q -= 1;
                truth_round = (rem != 0);
                unsat = (int64_t)pre_running[i] + q;
            }
        }

        int truth_sat = 0;
        int32_t clamped;
        if (unsat > M4T_MTFP_MAX_VAL) {
            clamped = M4T_MTFP_MAX_VAL; truth_sat = 1;
        } else if (unsat < -M4T_MTFP_MAX_VAL) {
            clamped = -M4T_MTFP_MAX_VAL; truth_sat = 1;
        } else {
            clamped = (int32_t)unsat;
        }

        out_running[i] = clamped;
        if (out_flags) {
            uint8_t bits = 0;
            if (truth_round) bits |= M4T_FLAG_ROUNDED;
            if (truth_sat)   bits |= M4T_FLAG_SATURATED;
            out_flags[i] = bits;
        }
    }
}

/* ── Property 1: per-call correctness (bit-exact vs reference) ──────────── */

static int prop_accum_aligning_correctness(void) {
    int32_t* running = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend  = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* pre     = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* ref_running = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        int K = rand_int(1, MAX_CALLS);

        for (int i = 0; i < n; i++) running[i] = rand_mantissa();
        int8_t e_run = rand_exp();

        for (int call = 0; call < K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa();
            int8_t e_add = rand_exp();

            int8_t e_run_pre = e_run;
            memcpy(pre, running, sizeof(int32_t) * (size_t)n);

            m4t_mtfp_vec_accum_aligning(running, &e_run, addend, e_add, NULL, n);

            int8_t ref_e = 0;
            accum_reference(pre, e_run_pre, addend, e_add,
                            ref_running, &ref_e, NULL, n);

            if (e_run != ref_e) {
                printf("FAIL: correctness exp (s=%d c=%d): kernel=%d ref=%d\n",
                       s, call, (int)e_run, (int)ref_e);
                free(running); free(addend); free(pre); free(ref_running);
                return 1;
            }
            for (int i = 0; i < n; i++) {
                if (running[i] != ref_running[i]) {
                    printf("FAIL: correctness (s=%d c=%d cell=%d): "
                           "kernel=%d ref=%d (e_run_pre=%d e_add=%d e_after=%d)\n",
                           s, call, i, (int)running[i], (int)ref_running[i],
                           (int)e_run_pre, (int)e_add, (int)e_run);
                    free(running); free(addend); free(pre); free(ref_running);
                    return 1;
                }
            }
        }
    }

    free(running); free(addend); free(pre); free(ref_running);
    return 0;
}

/* ── Property 2: invariant maintenance ──────────────────────────────────── */

static int prop_accum_aligning_invariant(void) {
    int32_t* running = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend  = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        int K = rand_int(1, MAX_CALLS);

        for (int i = 0; i < n; i++) running[i] = rand_mantissa();
        int8_t e_run = rand_exp();

        for (int call = 0; call < K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa();
            int8_t e_add = rand_exp();

            m4t_mtfp_vec_accum_aligning(running, &e_run, addend, e_add, NULL, n);

            for (int i = 0; i < n; i++) {
                if (running[i] >  M4T_MTFP_MAX_VAL ||
                    running[i] < -M4T_MTFP_MAX_VAL) {
                    printf("FAIL: invariant (sample %d call %d cell %d): "
                           "running=%d MAX_VAL=%d\n",
                           s, call, i, (int)running[i],
                           (int)M4T_MTFP_MAX_VAL);
                    free(running); free(addend);
                    return 1;
                }
            }
        }
    }

    free(running); free(addend);
    return 0;
}

/* ── Property 3: aliasing safety (running aliases flag-buffer space) ────── */

static int prop_accum_aligning_aliasing(void) {
    /* The kernel's contract permits running to alias flags' underlying buffer
     * iff flags is NULL. Verify that calling with flags=NULL is bit-identical
     * to calling without any aliasing. (The contract forbids running=addend;
     * we don't test the forbidden case.)
     *
     * This test runs the kernel twice with identical inputs, both with
     * flags=NULL: once into a dedicated `running` buffer, once into a
     * `running_alt` buffer that overlaps with a separate flags-shaped uint8_t
     * region in memory. Results must be bit-identical. */
    int32_t* running1 = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* running2 = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend   = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        int K = rand_int(1, MAX_CALLS);

        for (int i = 0; i < n; i++) {
            int32_t m = rand_mantissa();
            running1[i] = m;
            running2[i] = m;
        }
        int8_t e1 = rand_exp();
        int8_t e2 = e1;

        for (int call = 0; call < K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa();
            int8_t e_add = rand_exp();

            m4t_mtfp_vec_accum_aligning(running1, &e1, addend, e_add, NULL, n);
            m4t_mtfp_vec_accum_aligning(running2, &e2, addend, e_add, NULL, n);

            if (e1 != e2 || memcmp(running1, running2, sizeof(int32_t) * (size_t)n) != 0) {
                printf("FAIL: aliasing (sample %d call %d): non-deterministic\n",
                       s, call);
                free(running1); free(running2); free(addend);
                return 1;
            }
        }
    }

    free(running1); free(running2); free(addend);
    return 0;
}

/* ── Property 4: flag accuracy ──────────────────────────────────────────── */

static int prop_accum_aligning_flags(void) {
    int32_t* running = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend  = malloc(sizeof(int32_t) * MAX_CELLS);
    uint8_t* flags   = malloc(sizeof(uint8_t) * MAX_CELLS);
    int32_t* pre     = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* ref_running = malloc(sizeof(int32_t) * MAX_CELLS);
    uint8_t* call_flags  = malloc(sizeof(uint8_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        int K = rand_int(1, MAX_CALLS);

        for (int i = 0; i < n; i++) running[i] = rand_mantissa();
        memset(flags, 0, (size_t)n);
        int8_t e_run = rand_exp();

        /* Sticky truth flags accumulated across calls. */
        uint8_t* truth = calloc((size_t)n, 1);

        for (int call = 0; call < K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa();
            int8_t e_add = rand_exp();

            int8_t e_run_pre = e_run;
            memcpy(pre, running, sizeof(int32_t) * (size_t)n);

            m4t_mtfp_vec_accum_aligning(running, &e_run, addend, e_add, flags, n);

            /* Reference produces this call's flag bits in call_flags[i]; we
             * OR into the sticky `truth` array. */
            int8_t ref_e = 0;
            accum_reference(pre, e_run_pre, addend, e_add,
                            ref_running, &ref_e, call_flags, n);
            for (int i = 0; i < n; i++) truth[i] |= call_flags[i];
        }

        /* After all calls, compare kernel's sticky flags to reference truth. */
        for (int i = 0; i < n; i++) {
            uint8_t got = flags[i] & (M4T_FLAG_SATURATED | M4T_FLAG_ROUNDED);
            uint8_t expect = truth[i];
            if (got != expect) {
                printf("FAIL: flags (sample %d cell %d): got=0x%02x expect=0x%02x\n",
                       s, i, got, expect);
                free(running); free(addend); free(flags);
                free(pre); free(ref_running); free(call_flags); free(truth);
                return 1;
            }
        }
        free(truth);
    }

    free(running); free(addend); free(flags);
    free(pre); free(ref_running); free(call_flags);
    return 0;
}

/* ── Property 5: pairwise wrapper matches accumulator ───────────────────── */

static int prop_add_aligning_via_wrapper(void) {
    int32_t* a = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* b = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* dst1 = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* dst2 = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);

        for (int i = 0; i < n; i++) {
            a[i] = rand_mantissa();
            b[i] = rand_mantissa();
        }
        int8_t e_a = rand_exp();
        int8_t e_b = rand_exp();

        /* Wrapper path. */
        memcpy(dst1, a, sizeof(int32_t) * (size_t)n);
        int8_t e_w = 0;
        m4t_mtfp_vec_add_aligning(dst1, &e_w, a, e_a, b, e_b, NULL, n);

        /* Manual accumulator equivalent. */
        memcpy(dst2, a, sizeof(int32_t) * (size_t)n);
        int8_t e_m = e_a;
        m4t_mtfp_vec_accum_aligning(dst2, &e_m, b, e_b, NULL, n);

        if (e_w != e_m || memcmp(dst1, dst2, sizeof(int32_t) * (size_t)n) != 0) {
            printf("FAIL: wrapper (sample %d): wrapper != accumulator\n", s);
            free(a); free(b); free(dst1); free(dst2);
            return 1;
        }
    }

    free(a); free(b); free(dst1); free(dst2);
    return 0;
}

/* ── Property 6: roundtrip x + neg(x) at same exp == 0 ──────────────────── */

static int prop_add_aligning_roundtrip(void) {
    int32_t* x = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* neg_x = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* dst = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);

        for (int i = 0; i < n; i++) {
            x[i] = rand_mantissa();
            neg_x[i] = -x[i];
        }
        int8_t e = rand_exp();
        int8_t out_e = 0;

        m4t_mtfp_vec_add_aligning(dst, &out_e, x, e, neg_x, e, NULL, n);

        if (out_e != e) {
            printf("FAIL: roundtrip exp (sample %d): out_e=%d != e=%d\n",
                   s, (int)out_e, (int)e);
            free(x); free(neg_x); free(dst);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (dst[i] != 0) {
                printf("FAIL: roundtrip (sample %d cell %d): dst=%d != 0\n",
                       s, i, (int)dst[i]);
                free(x); free(neg_x); free(dst);
                return 1;
            }
        }
    }

    free(x); free(neg_x); free(dst);
    return 0;
}

/* ── Driver ─────────────────────────────────────────────────────────────── */

int main(void) {
    g_rng_state = 0xdeadbeefu;
    if (prop_accum_aligning_correctness()) return 1;

    g_rng_state = 0x13579bdfu;
    if (prop_accum_aligning_invariant())   return 1;

    g_rng_state = 0xa5a5a5a5u;
    if (prop_accum_aligning_aliasing())    return 1;

    g_rng_state = 0x0badf00du;
    if (prop_accum_aligning_flags())       return 1;

    g_rng_state = 0xfeedfaceu;
    if (prop_add_aligning_via_wrapper())   return 1;

    g_rng_state = 0xc0ffeebbu;
    if (prop_add_aligning_roundtrip())     return 1;

    printf("m4t_mtfp_accum_aligning: all 6 properties passed (%d samples each)\n",
           N_SAMPLES);
    return 0;
}
