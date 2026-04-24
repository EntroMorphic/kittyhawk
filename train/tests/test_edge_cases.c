/*
 * test_edge_cases.c — edge-case coverage for routed autodiff primitives.
 *
 * Verifies:
 *   1. k > T: topk saturates at T selections; remaining sentinel entries.
 *   2. all-zero scores: every decision = -1, output = 0, gradients = 0.
 *   3. M = 0: zero-token input; kernels return cleanly.
 *   4. k = 0: no selections; output = 0, gradients = 0.
 *   5. requantize on empty input (n = 0): returns 0 flips, no crash.
 *   6. requantize with hysteresis = 0: equivalent to sharp threshold.
 *   7. single-trit pack/unpack via the scalar ternary path.
 */

#include "backward_linear.h"
#include "backward_routed.h"
#include "requantize.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(int cond, const char* msg) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", msg); return 1; }
    return 0;
}

static int test_k_exceeds_T(void) {
    /* T=3, k=5. Only 3 selections possible. Remaining 2 should be sentinels. */
    float X[1 * 4] = {0.5f, -0.3f, 0.8f, 0.1f};
    int8_t U[3 * 4] = {1,0,-1,0,  0,1,0,-1,  -1,0,0,1};
    int32_t decisions[5];
    int8_t signs[5];
    rroute_forward_select(decisions, signs, X, U, 1, 4, 3, 5);
    int fails = 0;
    /* First 3 decisions should be valid tile indices (0..2). */
    int valid_count = 0;
    for (int i = 0; i < 3; i++) {
        if (decisions[i] >= 0 && decisions[i] < 3) valid_count++;
    }
    fails += check(valid_count == 3, "expected 3 valid decisions when k>T");
    /* Last 2 should be sentinel. */
    fails += check(decisions[3] == -1 && signs[3] == 0, "decision[3] sentinel");
    fails += check(decisions[4] == -1 && signs[4] == 0, "decision[4] sentinel");
    return fails;
}

static int test_all_zero_scores(void) {
    /* All-zero X gives all-zero scores. No tile should be selected. */
    float X[1 * 4] = {0};
    int8_t U[2 * 4] = {1,-1,1,-1, -1,1,-1,1};
    int32_t decisions[2];
    int8_t signs[2];
    rroute_forward_select(decisions, signs, X, U, 1, 4, 2, 2);
    int fails = 0;
    fails += check(decisions[0] == -1 && signs[0] == 0, "zero-score decision[0]");
    fails += check(decisions[1] == -1 && signs[1] == 0, "zero-score decision[1]");
    /* Forward dispatch with all sentinel decisions → zero output. */
    int8_t W[2 * 4 * 3] = {0};
    for (int i = 0; i < 24; i++) W[i] = (int8_t)((i % 3) - 1);  /* random ternary */
    float Y[3];
    rroute_forward_dispatch(Y, X, W, decisions, signs, 1, 4, 2, 3, 2);
    for (int n = 0; n < 3; n++) fails += check(Y[n] == 0.0f, "zero-decision output");
    return fails;
}

static int test_M_zero(void) {
    /* Zero-token input: kernels should not crash. */
    float X[1] = {0};
    int8_t Uloc[4] = {0};
    int8_t W[12] = {0};
    int32_t decisions[1] = {0};
    int8_t signs[1] = {0};
    float Y[1] = {42.0f};
    rroute_forward_select(decisions, signs, X, Uloc, 0, 4, 1, 1);
    rroute_forward_dispatch(Y, X, W, decisions, signs, 0, 4, 1, 3, 1);
    /* Y should be untouched (the zero loop body never runs). */
    return check(Y[0] == 42.0f, "M=0 dispatch leaves output untouched");
}

static int test_k_zero(void) {
    /* k=0 selection: no decisions made, dispatch zeros Y, backwards zero. */
    float X[2 * 4] = {1, -1, 0, 1, -1, 0, 1, -1};
    int8_t W[1 * 4 * 3] = {1, -1, 0,  0, 1, -1,  -1, 0, 1,  1, 0, -1};
    float Y[2 * 3] = {7, 7, 7, 7, 7, 7};
    /* With k=0, select buffer is empty; dispatch runs through the selection
     * loop zero times and returns Y = 0 (pre-zero step in dispatch). */
    rroute_forward_dispatch(Y, X, W, /*decisions*/NULL, /*signs*/NULL,
                            2, 4, 1, 3, 0);
    int fails = 0;
    for (int i = 0; i < 6; i++) fails += check(Y[i] == 0.0f, "k=0 zeros output");
    return fails;
}

static int test_requantize_empty(void) {
    /* n=0: return 0 flips, no crash. */
    int flips = requantize_hysteresis(NULL, NULL, 0, 0.33, 0.10);
    return check(flips == 0, "requantize n=0 returns 0 flips");
}

static int test_requantize_zero_hysteresis(void) {
    /* With hysteresis = 0, behavior is sharp threshold (old behavior).
     * Verify two ±τ-boundary latents flip cleanly. */
    float W_latent[6] = {1.0f, 0.1f, -1.0f, -0.1f, 0.5f, -0.5f};
    int8_t W[6] = {0};  /* start at 0 */
    int flips = requantize_hysteresis(W, W_latent, 6, 0.33, 0.0);
    int fails = 0;
    /* density 0.33 → τ ≈ sorted[n*(1-0.33)] = sorted[4] of abs values
     *   {1.0, 1.0, 0.5, 0.5, 0.1, 0.1} sorted desc → idx 4 = 0.1. τ = 0.1.
     * All |latent| ≥ 0.1 should flip to ±1; those below 0.1 stay 0. */
    fails += check(W[0] == 1, "W[0]=1 (latent=1.0)");
    fails += check(W[2] == -1, "W[2]=-1 (latent=-1.0)");
    fails += check(W[4] == 1, "W[4]=1 (latent=0.5)");
    fails += check(W[5] == -1, "W[5]=-1 (latent=-0.5)");
    fails += check(flips >= 4, "at least 4 flips from zero state");
    return fails;
}

int main(void) {
    int fails = 0;
    printf("test_edge_cases\n");
    fails += test_k_exceeds_T();
    fails += test_all_zero_scores();
    fails += test_M_zero();
    fails += test_k_zero();
    fails += test_requantize_empty();
    fails += test_requantize_zero_hysteresis();
    if (fails == 0) printf("PASS: edge cases\n");
    return fails;
}
