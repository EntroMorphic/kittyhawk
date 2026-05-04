/*
 * test_m4t_assert_live.c — V4-G2 meta-test.
 *
 * Forks a child process that deliberately violates a substrate precondition
 * (m4t_route_topk_abs called with T > M4T_ROUTE_MAX_T) and asserts that the
 * child exits via SIGABRT.
 *
 * This proves that:
 *   1. Substrate-internal asserts are actually compiled into libm4t_test.
 *   2. They actually trigger when their precondition is violated.
 *   3. The triggering propagates from a substrate function (compiled with
 *      -UNDEBUG via m4t_test) called from test code (also -UNDEBUG via the
 *      gesh_test_undebug helper).
 *
 * Under the prior structure (libm4t built with -DNDEBUG, only the test
 * binary's own .o files getting -UNDEBUG), this child would NOT abort:
 * the assert in m4t_route_topk_abs would have been compiled away as
 * `((void)0)` and the function would proceed to write past its uint64_t
 * bitmask (silent memory corruption).
 */

#include "m4t_route.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static int run_child_violating_precondition(void) {
    /* T = 200 > M4T_ROUTE_MAX_T (64). Assert at m4t_route.c:320 must fire. */
    enum { T_BAD = 200, K_REQ = 4 };
    int32_t scores[T_BAD];
    for (int i = 0; i < T_BAD; i++) scores[i] = i;

    m4t_route_decision_t decisions[K_REQ];
    m4t_route_topk_abs(decisions, scores, T_BAD, K_REQ);

    /* If we get here, the assert did NOT fire. That's a test failure —
     * use a distinctive exit code so the parent can distinguish "assert
     * silenced" from "child crashed for an unrelated reason". */
    fprintf(stderr, "FAIL: m4t_route_topk_abs(T=200) returned without "
                    "tripping assert (substrate asserts are SILENCED)\n");
    return 42;
}

int main(void) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        /* Child: violate precondition; should die via SIGABRT. */
        exit(run_child_violating_precondition());
    }

    /* Parent: wait for child, inspect exit status. */
    int status = 0;
    pid_t w = waitpid(pid, &status, 0);
    if (w != pid) {
        perror("waitpid");
        return 1;
    }

    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        if (sig == SIGABRT) {
            printf("  PASS test_assert_live: child aborted via SIGABRT "
                   "(substrate asserts ARE live in m4t_test)\n");
            return 0;
        }
        fprintf(stderr, "FAIL: child terminated via signal %d, expected "
                        "SIGABRT (%d)\n", sig, SIGABRT);
        return 1;
    }

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        if (code == 42) {
            fprintf(stderr, "FAIL: substrate assert was SILENCED — child "
                            "ran to completion without aborting. "
                            "m4t_test is not actually compiling asserts in.\n");
        } else {
            fprintf(stderr, "FAIL: child exited cleanly with code %d "
                            "(expected SIGABRT)\n", code);
        }
        return 1;
    }

    fprintf(stderr, "FAIL: child exited via unknown status %d\n", status);
    return 1;
}
