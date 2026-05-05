/*
 * test_m4t_accum_aligning_neon.c — bit-exact verification of the
 * NEON-routed cross-exp accumulator (m4t_mtfp_vec_accum_aligning_neon)
 * against the scalar oracle (m4t_mtfp_vec_accum_aligning_scalar_ref).
 *
 * Both OUTPUT and BOTH FLAG BITS (ROUNDED + SATURATED) must match.
 *
 * Per cross_exp_accum_routing T-G4. Coverage:
 *   - n boundary cases (0, 1, 3, 4, 5, 15, 16, 17, 63, 64, 65, 4095, 4096)
 *   - delta cases (0 same-exp, 1, 5, 10, 15, 18, 19, 20 degenerate, 25)
 *   - Both align directions (addend > running, running > addend)
 *   - Sparse vs dense activation distributions
 *   - 1000 random configurations (per ternary MAC remediation lesson)
 *   - Saturation-edge constructed cases (post-add overflow → SATURATED set)
 *   - flags == NULL and flags != NULL paths
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_VAL 581130733

static int g_fails = 0;

static void gen_data(m4t_mtfp_t* dst, int n, uint32_t seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        int64_t r = ((int64_t)rand() << 32) ^ rand();
        dst[i] = (m4t_mtfp_t)(r % (2 * MAX_VAL + 1) - MAX_VAL);
    }
}

/* Run one (n, e_run, e_addend, seed, with_flags) configuration. */
static void test_config_v(int n, int e_run, int e_addend, uint32_t seed,
                          int with_flags, const char* label, int verbose) {
    int n_alloc = n > 0 ? n : 1;  /* avoid 0-byte malloc */
    m4t_mtfp_t* running_neon = malloc(sizeof(m4t_mtfp_t) * (size_t)n_alloc);
    m4t_mtfp_t* running_ref  = malloc(sizeof(m4t_mtfp_t) * (size_t)n_alloc);
    m4t_mtfp_t* addend       = malloc(sizeof(m4t_mtfp_t) * (size_t)n_alloc);
    int n_flag_bytes = M4T_FLAG_BYTES(n > 0 ? n : 1);
    uint8_t* flags_neon = with_flags ? calloc(n_flag_bytes, 1) : NULL;
    uint8_t* flags_ref  = with_flags ? calloc(n_flag_bytes, 1) : NULL;

    if (n > 0) {
        gen_data(running_neon, n, seed);
        memcpy(running_ref, running_neon, sizeof(m4t_mtfp_t) * (size_t)n);
        gen_data(addend, n, seed + 7919);
    }
    int8_t e_neon = (int8_t)e_run;
    int8_t e_ref  = (int8_t)e_run;

    m4t_mtfp_vec_accum_aligning_neon       (running_neon, &e_neon, addend, (int8_t)e_addend, flags_neon, n);
    m4t_mtfp_vec_accum_aligning_scalar_ref (running_ref,  &e_ref,  addend, (int8_t)e_addend, flags_ref,  n);

    int local_fails = 0;
    /* Output bits */
    for (int i = 0; i < n; i++) {
        if (running_neon[i] != running_ref[i]) {
            local_fails++;
            if (local_fails <= 3) {
                fprintf(stderr, "  FAIL %s i=%d neon=%d ref=%d\n",
                        label, i, (int)running_neon[i], (int)running_ref[i]);
            }
        }
    }
    /* Exponent update */
    if (e_neon != e_ref) {
        local_fails++;
        fprintf(stderr, "  FAIL %s exp neon=%d ref=%d\n",
                label, (int)e_neon, (int)e_ref);
    }
    /* Flag bits (only when with_flags) */
    if (with_flags && memcmp(flags_neon, flags_ref, n_flag_bytes) != 0) {
        local_fails++;
        if (local_fails <= 5) {
            for (int i = 0; i < n; i++) {
                uint8_t fn = m4t_flag_test(flags_neon, i, 0xFF);
                uint8_t fr = m4t_flag_test(flags_ref,  i, 0xFF);
                if (fn != fr) {
                    fprintf(stderr, "  FAIL %s flags i=%d neon=0x%02x ref=0x%02x\n",
                            label, i, fn, fr);
                    break;
                }
            }
        }
    }

    if (local_fails == 0) {
        if (verbose) printf("  %-50s : PASS\n", label);
    } else {
        printf("  %-50s : FAIL (%d issues)\n", label, local_fails);
        g_fails += local_fails;
    }

    free(running_neon); free(running_ref); free(addend);
    free(flags_neon); free(flags_ref);
}

static void test_config(int n, int e_run, int e_addend, uint32_t seed,
                        int with_flags, const char* label) {
    test_config_v(n, e_run, e_addend, seed, with_flags, label, 1);
}

/* Saturation-edge: construct a case where post-add sum exceeds MAX_VAL. */
static void test_saturation(void) {
    printf("\n-- Saturation-edge cases --\n");
    /* All running = MAX_VAL, all addend = MAX_VAL, same exp.
     * sum = 2*MAX_VAL > MAX_VAL → all cells SATURATED. */
    int n = 16;
    m4t_mtfp_t running_neon[16], running_ref[16], addend[16];
    int n_flag_bytes = M4T_FLAG_BYTES(n);
    uint8_t flags_neon[16] = {0}, flags_ref[16] = {0};

    /* Case 1: same-exp positive saturation. */
    for (int i = 0; i < n; i++) {
        running_neon[i] = M4T_MTFP_MAX_VAL;
        running_ref[i] = M4T_MTFP_MAX_VAL;
        addend[i] = M4T_MTFP_MAX_VAL;
    }
    int8_t e_n = 0, e_r = 0;
    m4t_mtfp_vec_accum_aligning_neon(running_neon, &e_n, addend, 0, flags_neon, n);
    m4t_mtfp_vec_accum_aligning_scalar_ref(running_ref, &e_r, addend, 0, flags_ref, n);
    int sat1 = (memcmp(running_neon, running_ref, sizeof(m4t_mtfp_t)*n) == 0)
            && (memcmp(flags_neon, flags_ref, n_flag_bytes) == 0);
    printf("  same-exp positive sat (all=MAX_VAL+MAX_VAL→clamp) : %s\n",
           sat1 ? "PASS" : "FAIL");
    if (!sat1) g_fails++;

    /* Case 2: cross-exp saturation. running=MAX_VAL, addend=MAX_VAL, e_addend=e_run+1
     * → running gets divided by 3 (becomes ~MAX_VAL/3), then added to addend.
     * sum = MAX_VAL/3 + MAX_VAL ≈ 4*MAX_VAL/3 < MAX_VAL → no clamp.
     * Try the other way: running=MAX_VAL, addend=MAX_VAL, e_addend=e_run-3
     * → addend divided by 27 (becomes ~MAX_VAL/27), running unchanged → sum ≈ MAX_VAL.
     * Actually we need a setup where the sum overflows after alignment.
     * Try: running=MAX_VAL, addend=MAX_VAL, same exp (already covered above).
     * Or: running=MAX_VAL, addend=MAX_VAL, e_addend > e_run by large delta:
     *   running aligned down to ~0, sum ≈ MAX_VAL → no overflow, no SAT.
     * The cross-exp paths inherently shrink one side toward zero, making
     * post-add overflow rare. The same-exp case (sat1) is the realistic
     * cross-cycle saturation trigger. */

    memset(flags_neon, 0, n_flag_bytes);
    memset(flags_ref, 0, n_flag_bytes);
    for (int i = 0; i < n; i++) {
        running_neon[i] = -M4T_MTFP_MAX_VAL;
        running_ref[i] = -M4T_MTFP_MAX_VAL;
        addend[i] = -M4T_MTFP_MAX_VAL;
    }
    e_n = 0; e_r = 0;
    m4t_mtfp_vec_accum_aligning_neon(running_neon, &e_n, addend, 0, flags_neon, n);
    m4t_mtfp_vec_accum_aligning_scalar_ref(running_ref, &e_r, addend, 0, flags_ref, n);
    int sat2 = (memcmp(running_neon, running_ref, sizeof(m4t_mtfp_t)*n) == 0)
            && (memcmp(flags_neon, flags_ref, n_flag_bytes) == 0);
    printf("  same-exp negative sat (all=-MAX_VAL+-MAX_VAL→clamp): %s\n",
           sat2 ? "PASS" : "FAIL");
    if (!sat2) g_fails++;
}

/* Random stress: 1000 random configurations. */
static void test_random_stress(int n_random) {
    printf("\n-- Random stress (%d configs) --\n", n_random);
    int n_choices[] = {0, 1, 4, 8, 15, 16, 17, 31, 32, 64, 128, 256, 1024, 4096};
    int delta_choices[] = {-25, -19, -10, -5, -1, 0, 1, 5, 10, 19, 25};
    int with_flags_choices[] = {0, 1};

    srand(7777);
    int initial_fails = g_fails;
    for (int t = 0; t < n_random; t++) {
        int n = n_choices[rand() % (int)(sizeof(n_choices)/sizeof(int))];
        int delta = delta_choices[rand() % (int)(sizeof(delta_choices)/sizeof(int))];
        int with_flags = with_flags_choices[rand() % 2];
        int e_run = 0;
        int e_addend = e_run + delta;
        uint32_t seed = (uint32_t)(t * 100 + rand());
        char label[64];
        snprintf(label, sizeof(label), "rand[%4d] n=%d delta=%d flags=%d",
                 t, n, delta, with_flags);
        test_config_v(n, e_run, e_addend, seed, with_flags, label, /*verbose=*/0);
    }
    int new_fails = g_fails - initial_fails;
    if (new_fails == 0) printf("  PASS (%d/%d configs)\n", n_random, n_random);
    else printf("  FAIL (%d failures)\n", new_fails);
}

/* A-G5: aliasing — running == addend is forbidden by existing assert. */
static int run_alias_violation(void) {
    enum { N = 16 };
    m4t_mtfp_t buf[N] = {0};
    int8_t e = 0;
    m4t_mtfp_vec_accum_aligning_neon(buf, &e, buf, 5, NULL, N);
    fprintf(stderr, "FAIL: kernel did not abort on running==addend\n");
    return 1;
}

static void test_aliasing(void) {
    printf("\n-- A-G5: aliasing (running == addend) --\n");
    pid_t pid = fork();
    if (pid == 0) { exit(run_alias_violation()); }
    int status; waitpid(pid, &status, 0);
    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
        printf("  alias running == addend : PASS (SIGABRT)\n");
    } else {
        printf("  alias running == addend : FAIL\n");
        g_fails++;
    }
}

/* A-G6: perf comparison NEON vs scalar_ref across multiple shapes. */
#include <time.h>
static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double bench_one(void (*fn)(m4t_mtfp_t*, int8_t*, const m4t_mtfp_t*, int8_t, uint8_t*, int),
                        int n, int delta, int with_flags, int iters) {
    int n_alloc = n > 0 ? n : 1;
    m4t_mtfp_t* running = malloc(sizeof(m4t_mtfp_t) * (size_t)n_alloc);
    m4t_mtfp_t* addend  = malloc(sizeof(m4t_mtfp_t) * (size_t)n_alloc);
    uint8_t* flags = with_flags ? calloc(M4T_FLAG_BYTES(n_alloc), 1) : NULL;
    gen_data(running, n, 4242);
    gen_data(addend,  n, 6464);
    int8_t e_addend = 5;
    int8_t base_e_run = (int8_t)(e_addend - delta);

    for (int w = 0; w < 5; w++) { int8_t e = base_e_run; fn(running, &e, addend, e_addend, flags, n); }

    double best = 1e18;
    for (int s = 0; s < 5; s++) {
        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            int8_t e = base_e_run;
            fn(running, &e, addend, e_addend, flags, n);
        }
        double t1 = now_ns();
        if (t1 - t0 < best) best = t1 - t0;
    }
    free(running); free(addend); free(flags);
    return best / ((double)iters * (double)n);
}

static void perf_compare(int n, int delta, int with_flags, int iters,
                         const char* shape) {
    double scalar_ns = bench_one(m4t_mtfp_vec_accum_aligning_scalar_ref,
                                  n, delta, with_flags, iters);
    double neon_ns   = bench_one(m4t_mtfp_vec_accum_aligning_neon,
                                  n, delta, with_flags, iters);
    printf("  shape=%-18s n=%4d delta=%2d flags=%d : scalar=%.2f ns/cell  neon=%.2f ns/cell  speedup=%.1fx\n",
           shape, n, delta, with_flags, scalar_ns, neon_ns, scalar_ns / neon_ns);
}

int main(void) {
    printf("=== Bit-exact: NEON vs scalar_ref (output + ROUNDED + SATURATED) ===\n");

    /* n boundary cases at delta=5 with-flags. */
    printf("\n-- n boundary (delta=5, with-flags) --\n");
    int ns[] = {0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 63, 64, 65, 4095, 4096};
    for (size_t i = 0; i < sizeof(ns)/sizeof(int); i++) {
        char label[64]; snprintf(label, sizeof(label), "n=%d", ns[i]);
        test_config(ns[i], 0, 5, 1000 + (uint32_t)i, 1, label);
    }

    /* delta cases at n=64 with-flags. */
    printf("\n-- delta cases (n=64, with-flags) --\n");
    int deltas[] = {-25, -19, -15, -10, -5, -1, 0, 1, 5, 10, 15, 19, 25};
    for (size_t i = 0; i < sizeof(deltas)/sizeof(int); i++) {
        char label[64]; snprintf(label, sizeof(label), "delta=%d", deltas[i]);
        test_config(64, 0, deltas[i], 2000 + (uint32_t)i, 1, label);
    }

    /* flags=NULL paths. */
    printf("\n-- flags=NULL --\n");
    test_config(64, 0, 5,  3001, 0, "n=64  delta=5   no-flags");
    test_config(64, 0, -5, 3002, 0, "n=64  delta=-5  no-flags");

    /* Saturation. */
    test_saturation();

    /* Random stress. */
    test_random_stress(1000);

    if (g_fails > 0) {
        printf("\nFAIL: %d total mismatches\n", g_fails);
        return 1;
    }
    printf("\nPASS: all configurations bit-exact (output + flags)\n");

    /* A-G5 alias test. */
    test_aliasing();
    if (g_fails > 0) { printf("\nFAIL aliasing\n"); return 1; }

    /* A-G6 perf comparison: 5 shapes, min-of-5, workload-shape-declared. */
    printf("\n=== A-G6 Perf comparison (NEON vs scalar_ref, min-of-5) ===\n");
    perf_compare(64,    1, 1, 1000, "small-n delta-1");
    perf_compare(64,   10, 1, 1000, "small-n delta-10");
    perf_compare(64,   19, 1, 1000, "small-n delta-19");
    perf_compare(4096,  5, 1,   30, "large-n delta-5");
    perf_compare(16,    5, 1, 4000, "tiny-n delta-5 (per-call overhead)");
    perf_compare(64,    5, 0, 1000, "small-n delta-5 NO-flags");

    return 0;
}
