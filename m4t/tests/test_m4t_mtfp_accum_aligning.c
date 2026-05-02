/*
 * test_m4t_mtfp_accum_aligning.c — property tests for the cross-exponent
 * accumulator (M4T_SUBSTRATE.md §14.2 named opt-in).
 *
 * Properties:
 *   1.  accum_correctness          — bit-exact vs int64 reference
 *   2.  accum_invariant            — |running[i]| <= MAX_VAL after every call
 *   3.  accum_determinism          — two parallel kernel invocations agree
 *   4.  accum_flags                — per-block SATURATED + ROUNDED bits exact
 *   5.  accum_partial_block        — trailing-block bits past n stay zero
 *   6.  accum_long_sequence        — invariant + correctness across K=256
 *   7.  accum_boundary             — curated edge cases (MAX_VAL, 0, Δ=0/1/19/20, n=0)
 *   8.  accum_n_zero               — n=0 is a clean no-op
 *   9.  add_via_wrapper            — pairwise wrapper matches accumulator
 *  10.  add_roundtrip              — x + neg(x) at same exp == 0
 *  11.  add_dst_alias_a            — wrapper dst==a result identical to dst!=a
 *  12.  add_out_e_nullable         — wrapper accepts out_e == NULL
 *  13.  sub_via_negation           — sub(a, b) == add(a, neg(b)) at storage layer
 *  14.  sub_self                   — sub(x, x) at same exp == 0
 *
 * Test infrastructure: the bit-exact int64 reference implementation is the
 * oracle. No fp in the test path. The reference duplicates the kernel logic
 * in a structurally simpler form (no early returns, no inlined flag
 * accessor) so most kernel/reference bug correlations are unlikely; this
 * is a known limitation of single-implementation testing in C.
 *
 * Sample-count gate: 10,000 random sequences per property where applicable.
 * Boundary properties enumerate their cases exhaustively. A single
 * violation fails the property (no flake tolerance).
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_SAMPLES         10000
#define LONG_SEQ_K        256
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
    uint32_t r = xs32();
    int64_t span = (int64_t)M4T_MTFP_MAX_VAL * 2 + 1;
    int64_t v = (int64_t)(r % (uint64_t)span) - (int64_t)M4T_MTFP_MAX_VAL;
    return (int32_t)v;
}

static int32_t rand_mantissa_near_max(void) {
    /* Saturation-targeted: 60% chance of mantissa in the top 10% of range. */
    if ((xs32() % 100) < 60) {
        int sign = (xs32() & 1) ? 1 : -1;
        int32_t span = M4T_MTFP_MAX_VAL / 10;
        int32_t base = M4T_MTFP_MAX_VAL - (int32_t)(xs32() % (uint32_t)span);
        return sign * base;
    }
    return rand_mantissa();
}

static int8_t rand_exp(void) {
    uint32_t r = xs32();
    int v = (int)(r % (uint32_t)(2 * EXP_RANGE + 1)) - EXP_RANGE;
    return (int8_t)v;
}

static int rand_int(int lo, int hi) {
    uint32_t r = xs32();
    return lo + (int)(r % (uint32_t)(hi - lo + 1));
}

/* ── Bit-exact reference implementation ─────────────────────────────────── */

/* Reference performs one accumulator call into `out_running` and (optionally)
 * sets the per-block flag bits the kernel should set.
 *
 * Structurally simpler than the kernel: no early returns for degenerate
 * cases, no inlined flag accessor, no separate-loop saturation-only fast
 * paths. This independence reduces the chance of correlated bugs between
 * the kernel and the reference. */
static void accum_reference(
    const int32_t* pre_running, int8_t e_run_pre,
    const int32_t* addend, int8_t addend_exp,
    int32_t* out_running, int8_t* out_e,
    uint8_t* out_flags,           /* nullable; M4T_FLAG_BYTES(n) bytes */
    int n)
{
    int delta = (int)addend_exp - (int)e_run_pre;
    *out_e = (delta > 0) ? addend_exp : e_run_pre;

    int n_blocks = (int)M4T_FLAG_BYTES(n);
    if (out_flags) memset(out_flags, 0, (size_t)n_blocks);

    for (int i = 0; i < n; i++) {
        int truth_round = 0;
        int64_t unsat;

        if (delta == 0) {
            unsat = (int64_t)pre_running[i] + (int64_t)addend[i];
        } else if (delta > 0) {
            if (delta >= 20) {
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
            int abs_d = -delta;
            if (abs_d >= 20) {
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
            int block = i / M4T_MTFP_CELLS_PER_BLOCK;
            int slot  = i % M4T_MTFP_CELLS_PER_BLOCK;
            out_flags[block] |= (uint8_t)(bits << (slot * 2));
        }
    }
}

/* ── Property 1: per-call correctness (bit-exact vs reference) ──────────── */

static int prop_accum_correctness(void) {
    int32_t* running = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend  = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* pre     = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* ref     = malloc(sizeof(int32_t) * MAX_CELLS);

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
            accum_reference(pre, e_run_pre, addend, e_add, ref, &ref_e, NULL, n);

            if (e_run != ref_e) {
                printf("FAIL: correctness exp (s=%d c=%d): kernel=%d ref=%d\n",
                       s, call, (int)e_run, (int)ref_e);
                free(running); free(addend); free(pre); free(ref); return 1;
            }
            for (int i = 0; i < n; i++) {
                if (running[i] != ref[i]) {
                    printf("FAIL: correctness (s=%d c=%d cell=%d): "
                           "kernel=%d ref=%d (e_pre=%d e_add=%d e_after=%d)\n",
                           s, call, i, (int)running[i], (int)ref[i],
                           (int)e_run_pre, (int)e_add, (int)e_run);
                    free(running); free(addend); free(pre); free(ref); return 1;
                }
            }
        }
    }

    free(running); free(addend); free(pre); free(ref);
    return 0;
}

/* ── Property 2: invariant maintenance ──────────────────────────────────── */

static int prop_accum_invariant(void) {
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
                    printf("FAIL: invariant (s=%d c=%d cell=%d): running=%d\n",
                           s, call, i, (int)running[i]);
                    free(running); free(addend); return 1;
                }
            }
        }
    }
    free(running); free(addend);
    return 0;
}

/* ── Property 3: determinism (two invocations agree bit-exactly) ────────── */

static int prop_accum_determinism(void) {
    int32_t* r1 = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* r2 = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        int K = rand_int(1, MAX_CALLS);
        for (int i = 0; i < n; i++) {
            int32_t m = rand_mantissa();
            r1[i] = m; r2[i] = m;
        }
        int8_t e1 = rand_exp(), e2 = e1;

        for (int call = 0; call < K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa();
            int8_t e_add = rand_exp();
            m4t_mtfp_vec_accum_aligning(r1, &e1, addend, e_add, NULL, n);
            m4t_mtfp_vec_accum_aligning(r2, &e2, addend, e_add, NULL, n);
            if (e1 != e2 || memcmp(r1, r2, sizeof(int32_t) * (size_t)n) != 0) {
                printf("FAIL: determinism (s=%d c=%d): nondeterministic\n", s, call);
                free(r1); free(r2); free(addend); return 1;
            }
        }
    }
    free(r1); free(r2); free(addend);
    return 0;
}

/* ── Property 4: per-block flag bits exact ──────────────────────────────── */

static int prop_accum_flags(void) {
    int32_t* running    = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend     = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* pre        = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* ref_run    = malloc(sizeof(int32_t) * MAX_CELLS);
    size_t   n_flag_bytes = M4T_FLAG_BYTES(MAX_CELLS);
    uint8_t* flags      = malloc(n_flag_bytes);
    uint8_t* call_flags = malloc(n_flag_bytes);
    uint8_t* truth      = malloc(n_flag_bytes);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        int K = rand_int(1, MAX_CALLS);
        size_t fb = M4T_FLAG_BYTES(n);

        /* Saturation-targeted operands so flags fire often. */
        for (int i = 0; i < n; i++) running[i] = rand_mantissa_near_max();
        memset(flags, 0, fb);
        memset(truth, 0, fb);
        int8_t e_run = rand_exp();

        for (int call = 0; call < K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa_near_max();
            int8_t e_add = rand_exp();
            int8_t e_run_pre = e_run;
            memcpy(pre, running, sizeof(int32_t) * (size_t)n);

            m4t_mtfp_vec_accum_aligning(running, &e_run, addend, e_add, flags, n);

            int8_t ref_e = 0;
            accum_reference(pre, e_run_pre, addend, e_add,
                            ref_run, &ref_e, call_flags, n);
            for (size_t b = 0; b < fb; b++) truth[b] |= call_flags[b];
        }

        if (memcmp(flags, truth, fb) != 0) {
            printf("FAIL: flags (s=%d): per-block bytes diverge\n", s);
            for (size_t b = 0; b < fb; b++) {
                if (flags[b] != truth[b]) {
                    printf("  block %zu: kernel=0x%02x truth=0x%02x\n",
                           b, flags[b], truth[b]);
                }
            }
            free(running); free(addend); free(pre); free(ref_run);
            free(flags); free(call_flags); free(truth);
            return 1;
        }
    }

    free(running); free(addend); free(pre); free(ref_run);
    free(flags); free(call_flags); free(truth);
    return 0;
}

/* ── Property 5: trailing partial-block bits stay zero ──────────────────── */

/* For n not a multiple of 4, the last byte of the flags array has bits
 * for cell positions [n_blocks*4 - (4 - n%4) ... n_blocks*4), which are
 * outside the tensor. The kernel must not touch those bits. */
static int prop_accum_partial_block(void) {
    int32_t* running = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend  = malloc(sizeof(int32_t) * MAX_CELLS);
    size_t   n_flag_bytes = M4T_FLAG_BYTES(MAX_CELLS);
    uint8_t* flags   = malloc(n_flag_bytes);

    for (int s = 0; s < N_SAMPLES; s++) {
        /* Force partial trailing block: n in {1, 2, 3, 5, 6, 7, ...}. */
        int n;
        do { n = rand_int(1, MAX_CELLS); }
        while (n % M4T_MTFP_CELLS_PER_BLOCK == 0);

        size_t fb = M4T_FLAG_BYTES(n);
        int last_block_used_cells = n - (int)((fb - 1) * M4T_MTFP_CELLS_PER_BLOCK);
        /* Bits we care about in the last byte: the lower 2*used cells. */
        uint8_t used_mask = (uint8_t)((1u << (last_block_used_cells * 2)) - 1u);
        uint8_t unused_mask = (uint8_t)(~used_mask);

        for (int i = 0; i < n; i++) running[i] = rand_mantissa_near_max();
        memset(flags, 0, fb);
        int8_t e_run = rand_exp();

        int K = rand_int(1, MAX_CALLS);
        for (int call = 0; call < K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa_near_max();
            int8_t e_add = rand_exp();
            m4t_mtfp_vec_accum_aligning(running, &e_run, addend, e_add, flags, n);
        }

        if ((flags[fb - 1] & unused_mask) != 0) {
            printf("FAIL: partial_block (s=%d n=%d): unused bits set "
                   "(byte=0x%02x mask=0x%02x)\n",
                   s, n, flags[fb - 1], unused_mask);
            free(running); free(addend); free(flags); return 1;
        }
    }

    free(running); free(addend); free(flags);
    return 0;
}

/* ── Property 6: long-sequence stress (K=256) ───────────────────────────── */

static int prop_accum_long_sequence(void) {
    int32_t* running = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* addend  = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* pre     = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* ref     = malloc(sizeof(int32_t) * MAX_CELLS);

    /* Lower sample count since each is K=256 calls. 200 sequences × 256
     * calls × 32 average cells ≈ 1.6M cell-call ops. */
    int n_long_samples = 200;

    for (int s = 0; s < n_long_samples; s++) {
        int n = rand_int(1, MAX_CELLS);
        for (int i = 0; i < n; i++) running[i] = rand_mantissa();
        int8_t e_run = rand_exp();

        for (int call = 0; call < LONG_SEQ_K; call++) {
            for (int i = 0; i < n; i++) addend[i] = rand_mantissa();
            int8_t e_add = rand_exp();
            int8_t e_run_pre = e_run;
            memcpy(pre, running, sizeof(int32_t) * (size_t)n);

            m4t_mtfp_vec_accum_aligning(running, &e_run, addend, e_add, NULL, n);

            int8_t ref_e = 0;
            accum_reference(pre, e_run_pre, addend, e_add, ref, &ref_e, NULL, n);

            if (e_run != ref_e) {
                printf("FAIL: long_sequence exp (s=%d c=%d)\n", s, call);
                free(running); free(addend); free(pre); free(ref); return 1;
            }
            for (int i = 0; i < n; i++) {
                if (running[i] != ref[i]) {
                    printf("FAIL: long_sequence (s=%d c=%d i=%d)\n", s, call, i);
                    free(running); free(addend); free(pre); free(ref); return 1;
                }
                if (running[i] >  M4T_MTFP_MAX_VAL ||
                    running[i] < -M4T_MTFP_MAX_VAL) {
                    printf("FAIL: long_sequence invariant violated\n");
                    free(running); free(addend); free(pre); free(ref); return 1;
                }
            }
        }
    }

    free(running); free(addend); free(pre); free(ref);
    return 0;
}

/* ── Property 7: curated boundary cases ─────────────────────────────────── */

static int run_boundary_case(
    const char* name,
    const int32_t* pre_running, int8_t e_run_pre,
    const int32_t* addend, int8_t e_add,
    int n)
{
    int32_t kernel_run[8], ref_run[8];
    uint8_t kernel_flags[2] = {0, 0};
    uint8_t ref_flags[2] = {0, 0};
    int8_t e_kernel = e_run_pre, e_ref = 0;

    if (n > 8) { printf("internal: boundary n>8\n"); return 1; }
    memcpy(kernel_run, pre_running, sizeof(int32_t) * (size_t)n);

    m4t_mtfp_vec_accum_aligning(kernel_run, &e_kernel, addend, e_add, kernel_flags, n);
    accum_reference(pre_running, e_run_pre, addend, e_add,
                    ref_run, &e_ref, ref_flags, n);

    if (e_kernel != e_ref) {
        printf("FAIL boundary[%s] exp: kernel=%d ref=%d\n",
               name, (int)e_kernel, (int)e_ref);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (kernel_run[i] != ref_run[i]) {
            printf("FAIL boundary[%s] cell %d: kernel=%d ref=%d\n",
                   name, i, kernel_run[i], ref_run[i]);
            return 1;
        }
    }
    size_t fb = M4T_FLAG_BYTES(n);
    if (memcmp(kernel_flags, ref_flags, fb) != 0) {
        printf("FAIL boundary[%s] flags diverge\n", name);
        return 1;
    }
    return 0;
}

static int prop_accum_boundary(void) {
    /* Case A: all-zero mantissas, same exp. */
    {
        int32_t pre[4] = {0,0,0,0}, add[4] = {0,0,0,0};
        if (run_boundary_case("zero", pre, 0, add, 0, 4)) return 1;
    }
    /* Case B: mantissa = +MAX_VAL on both sides, same exp → saturation. */
    {
        int32_t pre[4] = {M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL};
        int32_t add[4] = {M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL};
        if (run_boundary_case("max+max same exp", pre, 0, add, 0, 4)) return 1;
    }
    /* Case C: mantissa = -MAX_VAL on both sides, same exp → -saturation. */
    {
        int32_t pre[4] = {-M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL};
        int32_t add[4] = {-M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL, -M4T_MTFP_MAX_VAL};
        if (run_boundary_case("-max + -max", pre, 0, add, 0, 4)) return 1;
    }
    /* Case D: x + (-x) at same exp → 0, no rounding, no saturation. */
    {
        int32_t pre[4] = {12345, 67890, M4T_MTFP_MAX_VAL/2, -54321};
        int32_t add[4] = {-12345, -67890, -(M4T_MTFP_MAX_VAL/2), 54321};
        if (run_boundary_case("x + -x same exp", pre, 0, add, 0, 4)) return 1;
    }
    /* Case E: Δ = 1 (smallest non-trivial rescale). */
    {
        int32_t pre[4] = {3, 4, 5, 6};
        int32_t add[4] = {1, 1, 1, 1};
        if (run_boundary_case("delta 1", pre, 0, add, 1, 4)) return 1;
    }
    /* Case F: Δ = 19 (largest non-degenerate). */
    {
        int32_t pre[4] = {M4T_MTFP_MAX_VAL, 0, -M4T_MTFP_MAX_VAL, 1};
        int32_t add[4] = {1, 1, 1, 1};
        if (run_boundary_case("delta 19", pre, 0, add, 19, 4)) return 1;
    }
    /* Case G: Δ = 20 (degenerate edge). */
    {
        int32_t pre[4] = {M4T_MTFP_MAX_VAL, 0, -M4T_MTFP_MAX_VAL, 1};
        int32_t add[4] = {1, 1, 1, 1};
        if (run_boundary_case("delta 20 deg", pre, 0, add, 20, 4)) return 1;
    }
    /* Case H: n = 1, single cell. */
    {
        int32_t pre[1] = {100};
        int32_t add[1] = {200};
        if (run_boundary_case("n=1", pre, 0, add, 0, 1)) return 1;
    }
    /* Case I: addend much smaller exp → addend rescales to zero. */
    {
        int32_t pre[4] = {1, 2, 3, 4};
        int32_t add[4] = {M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL};
        if (run_boundary_case("addend tiny", pre, 5, add, -25, 4)) return 1;
    }
    /* Case J: running much smaller exp → running rescales to zero. */
    {
        int32_t pre[4] = {M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL};
        int32_t add[4] = {1, 2, 3, 4};
        if (run_boundary_case("running tiny", pre, -25, add, 5, 4)) return 1;
    }
    return 0;
}

/* ── Property 8: n=0 is a clean no-op ───────────────────────────────────── */

static int prop_accum_n_zero(void) {
    int8_t e = 5;
    int32_t* dummy = NULL;
    /* n==0 with NULL pointers is permitted by the precondition. */
    m4t_mtfp_vec_accum_aligning(dummy, &e, dummy, 7, NULL, 0);
    if (e != 5) {
        printf("FAIL: n=0 modified e (got %d, expected 5)\n", (int)e);
        return 1;
    }
    return 0;
}

/* ── Property 9: pairwise wrapper matches accumulator ───────────────────── */

static int prop_add_via_wrapper(void) {
    int32_t* a = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* b = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* d1 = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* d2 = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        for (int i = 0; i < n; i++) {
            a[i] = rand_mantissa();
            b[i] = rand_mantissa();
        }
        int8_t e_a = rand_exp(), e_b = rand_exp();

        memcpy(d1, a, sizeof(int32_t) * (size_t)n);
        int8_t e_w = 0;
        m4t_mtfp_vec_add_aligning(d1, &e_w, a, e_a, b, e_b, NULL, n);

        memcpy(d2, a, sizeof(int32_t) * (size_t)n);
        int8_t e_m = e_a;
        m4t_mtfp_vec_accum_aligning(d2, &e_m, b, e_b, NULL, n);

        if (e_w != e_m || memcmp(d1, d2, sizeof(int32_t) * (size_t)n) != 0) {
            printf("FAIL: wrapper s=%d\n", s);
            free(a); free(b); free(d1); free(d2); return 1;
        }
    }

    free(a); free(b); free(d1); free(d2);
    return 0;
}

/* ── Property 10: roundtrip x + neg(x) at same exp == 0 ─────────────────── */

static int prop_add_roundtrip(void) {
    int32_t* x = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* neg_x = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* dst = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        for (int i = 0; i < n; i++) {
            x[i] = rand_mantissa();
            neg_x[i] = -x[i];
        }
        int8_t e = rand_exp(), out_e = 0;
        m4t_mtfp_vec_add_aligning(dst, &out_e, x, e, neg_x, e, NULL, n);
        if (out_e != e) {
            printf("FAIL: roundtrip exp s=%d\n", s);
            free(x); free(neg_x); free(dst); return 1;
        }
        for (int i = 0; i < n; i++) {
            if (dst[i] != 0) {
                printf("FAIL: roundtrip s=%d i=%d dst=%d\n", s, i, dst[i]);
                free(x); free(neg_x); free(dst); return 1;
            }
        }
    }
    free(x); free(neg_x); free(dst);
    return 0;
}

/* ── Property 11: wrapper aliasing dst==a matches non-aliased ───────────── */

static int prop_add_dst_alias_a(void) {
    int32_t* a_buf  = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* b_buf  = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* d_sep  = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        for (int i = 0; i < n; i++) {
            a_buf[i] = rand_mantissa();
            b_buf[i] = rand_mantissa();
        }
        int8_t e_a = rand_exp(), e_b = rand_exp();

        /* Non-aliased path: dst != a, dst != b. */
        int8_t e_sep = 0;
        m4t_mtfp_vec_add_aligning(d_sep, &e_sep, a_buf, e_a, b_buf, e_b, NULL, n);

        /* Aliased path: dst == a. The wrapper must produce identical output
         * by skipping the internal copy and writing into a_buf in place. */
        int32_t* a_alias = malloc(sizeof(int32_t) * (size_t)n);
        memcpy(a_alias, a_buf, sizeof(int32_t) * (size_t)n);
        int8_t e_alias = 0;
        m4t_mtfp_vec_add_aligning(a_alias, &e_alias, a_alias, e_a, b_buf, e_b, NULL, n);

        if (e_alias != e_sep || memcmp(a_alias, d_sep, sizeof(int32_t) * (size_t)n) != 0) {
            printf("FAIL: dst_alias_a s=%d\n", s);
            free(a_buf); free(b_buf); free(d_sep); free(a_alias);
            return 1;
        }
        free(a_alias);
    }

    free(a_buf); free(b_buf); free(d_sep);
    return 0;
}

/* ── Property 12: wrapper accepts out_e == NULL ─────────────────────────── */

static int prop_add_out_e_nullable(void) {
    int32_t a[4] = {1, 2, 3, 4};
    int32_t b[4] = {10, 20, 30, 40};
    int32_t dst[4];

    /* Should not crash and should produce the same dst as with out_e set. */
    m4t_mtfp_vec_add_aligning(dst, NULL, a, 0, b, 0, NULL, 4);

    int32_t expected[4] = {11, 22, 33, 44};
    for (int i = 0; i < 4; i++) {
        if (dst[i] != expected[i]) {
            printf("FAIL: out_e_nullable dst[%d] = %d (expected %d)\n",
                   i, dst[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

/* ── Property 13: sub_aligning matches add_aligning(a, neg(b)) ──────────── */

static int prop_sub_via_negation(void) {
    int32_t* a    = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* b    = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* nb   = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* d_sub = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* d_add = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        for (int i = 0; i < n; i++) {
            a[i] = rand_mantissa();
            b[i] = rand_mantissa();
            nb[i] = -b[i];
        }
        int8_t e_a = rand_exp(), e_b = rand_exp();
        int8_t e_sub = 0, e_add = 0;

        m4t_mtfp_vec_sub_aligning(d_sub, &e_sub, a, e_a, b, e_b, NULL, n);
        m4t_mtfp_vec_add_aligning(d_add, &e_add, a, e_a, nb, e_b, NULL, n);

        if (e_sub != e_add || memcmp(d_sub, d_add, sizeof(int32_t) * (size_t)n) != 0) {
            printf("FAIL: sub_via_negation s=%d\n", s);
            free(a); free(b); free(nb); free(d_sub); free(d_add);
            return 1;
        }
    }

    free(a); free(b); free(nb); free(d_sub); free(d_add);
    return 0;
}

/* ── Property 14: sub(x, x) at same exp == 0 ────────────────────────────── */

static int prop_sub_self(void) {
    int32_t* x = malloc(sizeof(int32_t) * MAX_CELLS);
    int32_t* dst = malloc(sizeof(int32_t) * MAX_CELLS);

    for (int s = 0; s < N_SAMPLES; s++) {
        int n = rand_int(1, MAX_CELLS);
        for (int i = 0; i < n; i++) x[i] = rand_mantissa();
        int8_t e = rand_exp(), out_e = 0;
        m4t_mtfp_vec_sub_aligning(dst, &out_e, x, e, x, e, NULL, n);
        if (out_e != e) {
            printf("FAIL: sub_self exp s=%d\n", s);
            free(x); free(dst); return 1;
        }
        for (int i = 0; i < n; i++) {
            if (dst[i] != 0) {
                printf("FAIL: sub_self s=%d i=%d dst=%d\n", s, i, dst[i]);
                free(x); free(dst); return 1;
            }
        }
    }
    free(x); free(dst);
    return 0;
}

/* ── Driver ─────────────────────────────────────────────────────────────── */

int main(void) {
    g_rng_state = 0xdeadbeefu; if (prop_accum_correctness())     return 1;
    g_rng_state = 0x13579bdfu; if (prop_accum_invariant())       return 1;
    g_rng_state = 0xa5a5a5a5u; if (prop_accum_determinism())     return 1;
    g_rng_state = 0x0badf00du; if (prop_accum_flags())           return 1;
    g_rng_state = 0xcafebabeu; if (prop_accum_partial_block())   return 1;
    g_rng_state = 0xc001d00du; if (prop_accum_long_sequence())   return 1;
                                if (prop_accum_boundary())       return 1;
                                if (prop_accum_n_zero())         return 1;
    g_rng_state = 0xfeedfaceu; if (prop_add_via_wrapper())       return 1;
    g_rng_state = 0xc0ffeebbu; if (prop_add_roundtrip())         return 1;
    g_rng_state = 0xb16b00b5u; if (prop_add_dst_alias_a())       return 1;
                                if (prop_add_out_e_nullable())   return 1;
    g_rng_state = 0xdeadc0deu; if (prop_sub_via_negation())      return 1;
    g_rng_state = 0xfacef00du; if (prop_sub_self())              return 1;

    printf("m4t_mtfp_accum_aligning: all 14 properties passed\n");
    return 0;
}
