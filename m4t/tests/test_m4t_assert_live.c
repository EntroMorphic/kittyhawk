/*
 * test_m4t_assert_live.c — V4-G2 + V4-residual-1 parameterized meta-test.
 *
 * Forks one child per case; each child deliberately violates a substrate
 * precondition in a DIFFERENT source file. Verifies SIGABRT in every child.
 *
 * Coverage rationale: V4-G2 originally exercised ONE assert in m4t_route.c.
 * That proved the mechanism, but didn't prove every substrate source's
 * asserts are actually compiled into libm4t_test. V4-residual-1 closure
 * walks through every substrate .c with asserts:
 *
 *   m4t_route.c           — m4t_route_topk_abs (T > M4T_ROUTE_MAX_T)
 *   m4t_mtfp.c            — m4t_mtfp_vec_zero (n < 0)
 *   m4t_mtfp4.c           — m4t_mtfp4_sdot_matmul_bt (M < 0)
 *   m4t_ternary_matmul.c  — m4t_mtfp_ternary_matmul_bt (Y aliases X)
 *   m4t_trit_pack.c       — m4t_pack_trits_1d (trit value out of {-1,0,1})
 *
 * Two substrate sources have NO asserts (m4t_trit_ops.c, m4t_trit_reducers.c).
 * Confirmed via grep at the time of writing this test; documented in
 * journal/v4_residual_1_assert_live_closeout.md.
 *
 * Each case follows the same fork-and-verify pattern as the original V4
 * meta-test. Distinguishes "assert silenced" (child exits cleanly with
 * sentinel code 42) from "child crashed for unrelated reason" (different
 * signal/exit), so a regression points to the actual cause.
 */

#include "m4t_route.h"
#include "m4t_mtfp.h"
#include "m4t_mtfp4.h"
#include "m4t_ternary_matmul.h"
#include "m4t_ternary_routed16.h"
#include "m4t_ternary_rowskip.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* Sentinel exit code: child returned without aborting. */
#define EXIT_ASSERT_SILENCED 42

/* ── Per-source-file violation functions ───────────────────────────────── */

static void violate_route(void) {
    /* m4t_route.c:320 — assert(T <= M4T_ROUTE_MAX_T). */
    enum { T_BAD = 200, K_REQ = 4 };
    int32_t scores[T_BAD];
    for (int i = 0; i < T_BAD; i++) scores[i] = i;
    m4t_route_decision_t decisions[K_REQ];
    m4t_route_topk_abs(decisions, scores, T_BAD, K_REQ);
}

static void violate_mtfp(void) {
    /* m4t_mtfp.c:68 — assert(n >= 0) inside m4t_mtfp_vec_zero. */
    m4t_mtfp_t dst[16];
    m4t_mtfp_vec_zero(dst, -1);
}

static void violate_mtfp4(void) {
    /* m4t_mtfp4.c:36 — assert(M >= 0 && K >= 0 && N >= 0). */
    m4t_mtfp_t Y[1] = {0};
    m4t_mtfp4_t X[1] = {0};
    m4t_trit_t W[1] = {0};
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, -1, 1, 1);
}

static void violate_ternary_matmul(void) {
    /* m4t_ternary_matmul.c:213 — assert((const void*)Y != (const void*)X).
     * Pass the same buffer for Y and X to trip the aliasing assert. */
    m4t_mtfp_t buf[4] = {0};
    uint8_t W_packed[1] = {0};
    m4t_mtfp_ternary_matmul_bt(buf, buf, W_packed, NULL, 1, 1, 1);
}

static void violate_trit_pack(void) {
    /* m4t_trit_pack.c:43 — assert(t >= -1 && t <= 1) inside trit_to_code,
     * called from m4t_pack_trits_1d. Pass an out-of-range trit value. */
    uint8_t dst[1] = {0};
    m4t_trit_t src[1] = {5};   /* invalid: outside {-1, 0, +1} */
    m4t_pack_trits_1d(dst, src, 1);
}

static void violate_routed16(void) {
    /* m4t_ternary_routed16.c:74 — assert(K >= 0 && N >= 0) at the start
     * of m4t_ternary_routed16_pack. Pass K=-1 to trip. */
    (void)m4t_ternary_routed16_pack(NULL, -1, 1);
}

static void violate_rowskip(void) {
    /* m4t_ternary_rowskip.c:93 — assert(K >= 0 && N >= 0) at the start
     * of m4t_ternary_rowskip_pack. Pass K=-1 to trip. */
    (void)m4t_ternary_rowskip_pack(NULL, -1, 1);
}

/* ── Parameterized harness ─────────────────────────────────────────────── */

typedef struct {
    const char* source_file;
    const char* label;
    void      (*violate)(void);
} assert_case_t;

static const assert_case_t cases[] = {
    { "m4t_route.c",            "m4t_route_topk_abs(T > MAX_T)",
                                violate_route },
    { "m4t_mtfp.c",             "m4t_mtfp_vec_zero(n < 0)",
                                violate_mtfp },
    { "m4t_mtfp4.c",            "m4t_mtfp4_sdot_matmul_bt(M < 0)",
                                violate_mtfp4 },
    { "m4t_ternary_matmul.c",   "m4t_mtfp_ternary_matmul_bt(Y aliases X)",
                                violate_ternary_matmul },
    { "m4t_trit_pack.c",        "m4t_pack_trits_1d(trit out of {-1,0,1})",
                                violate_trit_pack },
    { "m4t_ternary_routed16.c", "m4t_ternary_routed16_pack(K < 0)",
                                violate_routed16 },
    { "m4t_ternary_rowskip.c",  "m4t_ternary_rowskip_pack(K < 0)",
                                violate_rowskip },
};
static const int N_CASES = (int)(sizeof(cases) / sizeof(cases[0]));

/* Returns 1 on PASS (child aborted via SIGABRT), 0 on FAIL. */
static int run_case(const assert_case_t* tc) {
    /* RT-1: flush before fork so the child does NOT inherit our buffered
     * stdout. Without this, every line we printf'd before the fork gets
     * re-emitted by every child when it exits, producing output like
     * "test_m4t_assert_live: 5 cases..." repeated N times. */
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 0;
    }

    if (pid == 0) {
        /* Child: violate precondition; should die via SIGABRT. */
        tc->violate();
        /* If we get here, the assert did NOT fire. */
        fprintf(stderr, "  child[%s]: returned without aborting\n",
                tc->source_file);
        exit(EXIT_ASSERT_SILENCED);
    }

    int status = 0;
    pid_t w = waitpid(pid, &status, 0);
    if (w != pid) {
        perror("waitpid");
        return 0;
    }

    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        if (sig == SIGABRT) {
            printf("  PASS [%s] %s — SIGABRT\n",
                   tc->source_file, tc->label);
            return 1;
        }
        fprintf(stderr,
                "  FAIL [%s] %s — terminated by signal %d (expected SIGABRT %d)\n",
                tc->source_file, tc->label, sig, SIGABRT);
        return 0;
    }

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        if (code == EXIT_ASSERT_SILENCED) {
            fprintf(stderr,
                    "  FAIL [%s] %s — assert SILENCED (child returned cleanly)\n",
                    tc->source_file, tc->label);
        } else {
            fprintf(stderr,
                    "  FAIL [%s] %s — child exited with code %d\n",
                    tc->source_file, tc->label, code);
        }
        return 0;
    }

    fprintf(stderr, "  FAIL [%s] %s — unknown wait status %d\n",
            tc->source_file, tc->label, status);
    return 0;
}

int main(void) {
    printf("test_m4t_assert_live: %d cases, one per substrate source with asserts\n",
           N_CASES);

    int n_pass = 0;
    for (int i = 0; i < N_CASES; i++) {
        n_pass += run_case(&cases[i]);
    }

    printf("\n%d/%d cases PASS\n", n_pass, N_CASES);
    if (n_pass != N_CASES) {
        fprintf(stderr,
                "FAIL: substrate asserts are NOT uniformly live in libm4t_test\n");
        return 1;
    }
    printf("PASS: substrate asserts are live across every source file with asserts\n");
    return 0;
}
