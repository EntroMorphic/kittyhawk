/*
 * test_m4t_shift3_neon.c — bit-exact + perf test for m4t_mtfp_shift3's
 * NEON divide path.
 *
 * Compares the production m4t_mtfp_shift3 (which dispatches to NEON
 * for abs_k ∈ [1, 19] when M4T_HAS_NEON) against m4t_mtfp_shift3_scalar_ref
 * (always-scalar, exposed for tests). Both are public substrate APIs;
 * neither is duplicated here. Per remediation R-G3/R-G4/R-G6.
 *
 * Sample mode (default ctest): boundary + random sample per k. Fast.
 * Exhaustive mode: pass `x` as argv[1]. Sweeps every x in
 * [-MAX_VAL, +MAX_VAL] for every k. ~25s. Run by hand or via a
 * dedicated ctest target (`m4t_shift3_neon_exhaustive`).
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_VAL 581130733  /* (3^19 - 1) / 2 */

/* ── Tests ─────────────────────────────────────────────────────────────── */

static int g_fails = 0;

/* Sample bit-exact: production NEON path vs scalar reference. */
static void test_bit_exact_per_k(int abs_k) {
    enum { N_RAND = 50000 };
    int n = N_RAND + 12 + 4;
    m4t_mtfp_t* in_buf  = malloc(n * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* out_neon = malloc(n * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* out_ref  = malloc(n * sizeof(m4t_mtfp_t));
    if (!in_buf || !out_neon || !out_ref) {
        fprintf(stderr, "malloc fail\n"); exit(1);
    }

    int idx = 0;
    in_buf[idx++] = 0;
    in_buf[idx++] = 1;
    in_buf[idx++] = -1;
    in_buf[idx++] = (m4t_mtfp_t)MAX_VAL;
    in_buf[idx++] = (m4t_mtfp_t)(-MAX_VAL);
    in_buf[idx++] = (m4t_mtfp_t)(MAX_VAL - 1);
    in_buf[idx++] = (m4t_mtfp_t)(-MAX_VAL + 1);
    in_buf[idx++] = (m4t_mtfp_t)(MAX_VAL / 2);
    in_buf[idx++] = (m4t_mtfp_t)(-MAX_VAL / 2);
    in_buf[idx++] = 1000000;
    in_buf[idx++] = -1000000;
    in_buf[idx++] = 12345;

    srand(42 + abs_k);
    for (int j = 0; j < N_RAND; j++) {
        int64_t r = ((int64_t)rand() << 32) ^ rand();
        r = r % (2 * MAX_VAL + 1) - MAX_VAL;
        in_buf[idx++] = (m4t_mtfp_t)r;
    }
    for (; idx < n; idx++) in_buf[idx] = 0;

    /* Production path (NEON via m4t_mtfp_shift3 when M4T_HAS_NEON). */
    m4t_mtfp_shift3(out_neon, in_buf, -abs_k, n);
    /* Scalar oracle (always-scalar, never NEON). */
    m4t_mtfp_shift3_scalar_ref(out_ref, in_buf, -abs_k, n);

    int per_k_fails = 0;
    for (int i = 0; i < n; i++) {
        if (out_neon[i] != out_ref[i]) {
            per_k_fails++;
            if (per_k_fails <= 3) {
                fprintf(stderr,
                    "  k=%d FAIL i=%d in=%d production=%d scalar_ref=%d\n",
                    abs_k, i, (int)in_buf[i], (int)out_neon[i], (int)out_ref[i]);
            }
        }
    }
    if (per_k_fails == 0) {
        printf("  k=%2d PASS  (%d cases bit-exact vs scalar_ref)\n", abs_k, n);
    } else {
        printf("  k=%2d FAIL  (%d / %d mismatches)\n", abs_k, per_k_fails, n);
        g_fails += per_k_fails;
    }

    free(in_buf); free(out_neon); free(out_ref);
}

/* Aliasing test: dst == src must produce the same result as separate buffers. */
static void test_aliasing(void) {
    const int ks[] = {1, 10, 19};
    const int ns[] = {4, 5, 64, 65};
    int alias_fails = 0;
    for (size_t ki = 0; ki < sizeof(ks)/sizeof(ks[0]); ki++) {
        for (size_t ni = 0; ni < sizeof(ns)/sizeof(ns[0]); ni++) {
            int abs_k = ks[ki];
            int n = ns[ni];
            m4t_mtfp_t* src = malloc(n * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* expected = malloc(n * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* inplace = malloc(n * sizeof(m4t_mtfp_t));
            srand(7 * abs_k + n);
            for (int i = 0; i < n; i++) {
                int64_t r = ((int64_t)rand() << 32) ^ rand();
                src[i] = (m4t_mtfp_t)(r % (2 * MAX_VAL + 1) - MAX_VAL);
                inplace[i] = src[i];
            }
            m4t_mtfp_shift3(expected, src, -abs_k, n);
            m4t_mtfp_shift3(inplace, inplace, -abs_k, n);
            int local_fails = 0;
            for (int i = 0; i < n; i++) {
                if (expected[i] != inplace[i]) {
                    alias_fails++;
                    local_fails++;
                    if (alias_fails <= 3) {
                        fprintf(stderr,
                            "  ALIAS FAIL k=%d n=%d i=%d in=%d expected=%d inplace=%d\n",
                            abs_k, n, i, (int)src[i],
                            (int)expected[i], (int)inplace[i]);
                    }
                }
            }
            printf("  alias k=%2d n=%2d : %s\n", abs_k, n,
                   local_fails == 0 ? "PASS" : "FAIL");
            free(src); free(expected); free(inplace);
        }
    }
    if (alias_fails > 0) g_fails += alias_fails;
}

/* Exhaustive bit-exact: every x in [-MAX_VAL, MAX_VAL] for every k. */
static int exhaustive_verify_per_k(int abs_k) {
    enum { CHUNK = 65536 };
    m4t_mtfp_t* in_buf  = malloc(CHUNK * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* out_neon = malloc(CHUNK * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* out_ref  = malloc(CHUNK * sizeof(m4t_mtfp_t));
    if (!in_buf || !out_neon || !out_ref) {
        fprintf(stderr, "malloc fail\n"); exit(1);
    }
    int64_t per_k_fails = 0;
    int64_t total = 0;

    for (int64_t base = -MAX_VAL; base <= MAX_VAL; base += CHUNK) {
        int n = (int)(base + CHUNK <= MAX_VAL + 1
                          ? CHUNK
                          : (MAX_VAL + 1 - base));
        for (int j = 0; j < n; j++) in_buf[j] = (m4t_mtfp_t)(base + j);
        m4t_mtfp_shift3(out_neon, in_buf, -abs_k, n);
        m4t_mtfp_shift3_scalar_ref(out_ref, in_buf, -abs_k, n);
        for (int j = 0; j < n; j++) {
            if (out_neon[j] != out_ref[j]) {
                per_k_fails++;
                if (per_k_fails <= 3) {
                    fprintf(stderr,
                        "  k=%d EXHAUSTIVE FAIL x=%d production=%d scalar_ref=%d\n",
                        abs_k, (int)in_buf[j], (int)out_neon[j], (int)out_ref[j]);
                }
            }
        }
        total += n;
    }
    free(in_buf); free(out_neon); free(out_ref);
    if (per_k_fails == 0) {
        printf("  k=%2d EXHAUSTIVE PASS (%lld bit-exact)\n",
               abs_k, (long long)total);
        return 0;
    }
    printf("  k=%2d EXHAUSTIVE FAIL (%lld / %lld mismatches)\n",
           abs_k, (long long)per_k_fails, (long long)total);
    return 1;
}

/* ── Perf comparison ────────────────────────────────────────────────────── */

static double sample_neon(m4t_mtfp_t* out, const m4t_mtfp_t* in,
                          int abs_k, int n, int call_count) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int c = 0; c < call_count; c++)
        m4t_mtfp_shift3(out, in, -abs_k, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = ((t1.tv_sec - t0.tv_sec) * 1e9
                  + (t1.tv_nsec - t0.tv_nsec));
    return ns / ((double)call_count * n);
}
static double sample_scalar(m4t_mtfp_t* out, const m4t_mtfp_t* in,
                            int abs_k, int n, int call_count) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int c = 0; c < call_count; c++)
        m4t_mtfp_shift3_scalar_ref(out, in, -abs_k, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = ((t1.tv_sec - t0.tv_sec) * 1e9
                  + (t1.tv_nsec - t0.tv_nsec));
    return ns / ((double)call_count * n);
}

static void perf_compare(int abs_k, int n, int call_count, const char* shape) {
    m4t_mtfp_t* in_buf  = malloc(n * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* out_buf = malloc(n * sizeof(m4t_mtfp_t));
    if (!in_buf || !out_buf) { fprintf(stderr, "malloc fail\n"); exit(1); }
    srand(123 + abs_k * 17);
    for (int i = 0; i < n; i++) {
        int64_t r = ((int64_t)rand() << 32) ^ rand();
        in_buf[i] = (m4t_mtfp_t)(r % (2 * MAX_VAL + 1) - MAX_VAL);
    }

    /* Warm up. */
    for (int t = 0; t < 5; t++) {
        m4t_mtfp_shift3(out_buf, in_buf, -abs_k, n);
        m4t_mtfp_shift3_scalar_ref(out_buf, in_buf, -abs_k, n);
    }

    /* Min-of-5 each path. */
    double scalar_min = 1e18, neon_min = 1e18;
    for (int s = 0; s < 5; s++) {
        double sc = sample_scalar(out_buf, in_buf, abs_k, n, call_count);
        double ne = sample_neon  (out_buf, in_buf, abs_k, n, call_count);
        if (sc < scalar_min) scalar_min = sc;
        if (ne < neon_min)   neon_min   = ne;
    }
    printf("  shape=%-13s k=%2d n=%4d calls=%d : scalar_ref=%.3f ns/elem  "
           "production=%.3f ns/elem  speedup=%.1fx\n",
           shape, abs_k, n, call_count, scalar_min, neon_min,
           scalar_min / neon_min);
    free(in_buf); free(out_buf);
}

int main(int argc, char** argv) {
    int do_exhaustive = (argc > 1 && argv[1][0] == 'x');

    printf("=== Bit-exact (sample): production m4t_mtfp_shift3 vs m4t_mtfp_shift3_scalar_ref (DIVIDE k<0) ===\n");
    for (int k = 1; k <= 19; k++) test_bit_exact_per_k(k);

    /* k>0 multiply path bit-exact (no-scalar audit remediation: this path
     * was scalar in production until 2026-05-06; now NEON-routed). Reuse
     * test_bit_exact_per_k's mix of boundary + random inputs but with
     * positive k. */
    printf("\n=== Bit-exact (sample): production m4t_mtfp_shift3 vs m4t_mtfp_shift3_scalar_ref (MULTIPLY k>0) ===\n");
    for (int k = 1; k <= 19; k++) {
        enum { N_RAND = 50000 };
        int n = N_RAND + 12 + 4;
        m4t_mtfp_t* in_buf  = malloc(n * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* out_neon = malloc(n * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* out_ref  = malloc(n * sizeof(m4t_mtfp_t));
        if (!in_buf || !out_neon || !out_ref) { fprintf(stderr, "malloc fail\n"); exit(1); }

        int idx = 0;
        in_buf[idx++] = 0;
        in_buf[idx++] = 1;
        in_buf[idx++] = -1;
        in_buf[idx++] = (m4t_mtfp_t)MAX_VAL;
        in_buf[idx++] = (m4t_mtfp_t)(-MAX_VAL);
        in_buf[idx++] = (m4t_mtfp_t)(MAX_VAL - 1);
        in_buf[idx++] = (m4t_mtfp_t)(-MAX_VAL + 1);
        in_buf[idx++] = (m4t_mtfp_t)(MAX_VAL / 2);
        in_buf[idx++] = (m4t_mtfp_t)(-MAX_VAL / 2);
        in_buf[idx++] = 1000000;
        in_buf[idx++] = -1000000;
        in_buf[idx++] = 12345;
        srand(42 + k);
        for (int j = 0; j < N_RAND; j++) {
            int64_t r = ((int64_t)rand() << 32) ^ rand();
            r = r % (2 * MAX_VAL + 1) - MAX_VAL;
            in_buf[idx++] = (m4t_mtfp_t)r;
        }
        for (; idx < n; idx++) in_buf[idx] = 0;
        m4t_mtfp_shift3(out_neon, in_buf, k, n);
        m4t_mtfp_shift3_scalar_ref(out_ref, in_buf, k, n);
        int fails = 0;
        for (int i = 0; i < n; i++) {
            if (out_neon[i] != out_ref[i]) {
                fails++;
                if (fails <= 3) {
                    fprintf(stderr,
                        "  k=+%d FAIL i=%d in=%d production=%d scalar_ref=%d\n",
                        k, i, (int)in_buf[i], (int)out_neon[i], (int)out_ref[i]);
                }
            }
        }
        if (fails == 0) {
            printf("  k=+%2d PASS  (%d cases bit-exact vs scalar_ref)\n", k, n);
        } else {
            printf("  k=+%2d FAIL  (%d / %d mismatches)\n", k, fails, n);
            g_fails += fails;
        }
        free(in_buf); free(out_neon); free(out_ref);
    }

    /* k>=20 saturation collapse path. */
    printf("\n=== Bit-exact: k>=20 saturation collapse path ===\n");
    {
        const int kvals[] = {20, 25, 31};
        for (size_t kk = 0; kk < sizeof(kvals)/sizeof(kvals[0]); kk++) {
            int k = kvals[kk];
            int n = 4096 + 13;
            m4t_mtfp_t* in_buf  = malloc(n * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* out_neon = malloc(n * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* out_ref  = malloc(n * sizeof(m4t_mtfp_t));
            srand(99 + k);
            for (int i = 0; i < n; i++) {
                int64_t r = ((int64_t)rand() << 32) ^ rand();
                in_buf[i] = (m4t_mtfp_t)(r % (2 * MAX_VAL + 1) - MAX_VAL);
            }
            m4t_mtfp_shift3(out_neon, in_buf, k, n);
            m4t_mtfp_shift3_scalar_ref(out_ref, in_buf, k, n);
            int fails = 0;
            for (int i = 0; i < n; i++) if (out_neon[i] != out_ref[i]) fails++;
            if (fails == 0) printf("  k=+%d (saturate) PASS  (%d cases)\n", k, n);
            else { printf("  k=+%d FAIL %d\n", k, fails); g_fails += fails; }
            free(in_buf); free(out_neon); free(out_ref);
        }
    }
    if (g_fails > 0) {
        printf("\nFAIL: %d total mismatches across all k (sample)\n", g_fails);
        return 1;
    }
    printf("\nPASS: 19/19 k values bit-exact (sample)\n");

    printf("\n=== Aliasing test (dst == src) ===\n");
    test_aliasing();
    if (g_fails > 0) {
        printf("\nFAIL: aliasing test\n");
        return 1;
    }

    if (do_exhaustive) {
        printf("\n=== Exhaustive bit-exact (1.16e9 per k × 19 k) ===\n");
        int total_fails = 0;
        for (int k = 1; k <= 19; k++) total_fails += exhaustive_verify_per_k(k);
        if (total_fails > 0) {
            printf("\nFAIL: %d k values had exhaustive mismatches\n", total_fails);
            return 1;
        }
        printf("\nPASS: 19/19 k values exhaustively bit-exact\n");
    } else {
        printf("\n(pass 'x' as arg to also run exhaustive verify; ~25s)\n");
    }

    printf("\n=== Perf comparison: production NEON vs scalar_ref (min-of-5) ===\n");
    printf("Shape A (BATCHED): one call, n=4096 — bulk divide.\n");
    for (int k = 1; k <= 19; k += 6)
        perf_compare(k, 4096, 200, "BATCHED");
    printf("\n");
    printf("Shape B (TIGHT-LOOP): n=4 per call, many calls — per-call overhead bound.\n");
    for (int k = 1; k <= 19; k += 6)
        perf_compare(k, 4, 200000, "TIGHT-LOOP");

    return 0;
}
