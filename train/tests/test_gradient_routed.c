/*
 * test_gradient_routed.c — numerical gradient check for routed dispatch.
 *
 * WHAT THIS TEST VALIDATES (and what it doesn't):
 *
 *   • Validates: the linear portion of the routed backward is correct
 *     *given fixed routing decisions*. For any fixed (decisions, signs)
 *     tuple, dX and dW match finite differences of forward-with-same-
 *     selection within 1e-3 relative error.
 *
 *   • Does NOT validate: the straight-through estimator (STE) of the
 *     top-k selection itself. STE is definitionally a violation of the
 *     local-linearization that finite differences assume — the forward
 *     discretely jumps between tile subsets as scores cross each other,
 *     and the "gradient through selection" is a policy decision, not a
 *     derivation. Testing STE correctness requires behavioral criteria
 *     (does training actually change selections? does the loss decrease?),
 *     which live in test_toy_convergence and test_toy_10class, not here.
 *
 *   • Why the fixed-decisions matters: if the finite-difference perturbs
 *     an X or W entry large enough to flip a selection, the two forward
 *     passes use different tile subsets and the numerical gradient
 *     becomes meaningless. Holding decisions/signs constant across the
 *     two forward evaluations is the only way to compare to the backward
 *     under STE.
 *
 * Two checks:
 *
 *   1. dX validation. Fix X, U, W, k; run rroute_forward_select to freeze
 *      decisions; compute analytical dX via rroute_backward_dX; finite-
 *      difference numerical dX using the SAME fixed decisions.
 *
 *   2. dW_latent validation. Same fixed decisions; promote W to float;
 *      analytical dW_latent via rroute_backward_dW; finite-difference
 *      against a float W on SELECTED tile slots only.
 *
 * Critical subtlety: the gradient check must use FIXED decisions (STE
 * pretends they're identity). If we re-select on each forward pass during
 * finite difference, a large-enough perturbation will flip a selection and
 * break the gradient check trivially. Using fixed decisions matches how
 * STE is defined — gradient flows through the selection as if it were
 * held constant.
 */

#include "backward_routed.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 2
#define H 5
#define T 6
#define N 4
#define K_SEL 3

static uint32_t xorshift_state = 0x13579bdfu;
static uint32_t xorshift32(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xorshift_state = x; return x;
}
static float rand_float(void) {
    return (float)((int32_t)xorshift32()) / 2147483648.0f;
}

static double loss_sqsum(const float* Y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += 0.5 * (double)Y[i] * (double)Y[i];
    return s;
}

/* Forward with a float W tile bank; used for finite-diff dW check.
 * Mirrors the MVP selection-only semantics of rroute_forward_dispatch:
 * contribute with +1 weight regardless of score sign. */
static void rroute_forward_dispatch_float_W(
    float* Y, const float* X, const float* Wf,
    const int32_t* decisions, const int8_t* signs,
    int M_, int H_, int T_, int N_, int k_)
{
    (void)T_; (void)signs;
    for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) Y[(size_t)m * N_ + n] = 0.0f;
        for (int sel = 0; sel < k_; sel++) {
            int32_t t = decisions[(size_t)m * k_ + sel];
            if (t < 0) continue;
            const float* x_row = X + (size_t)m * H_;
            const float* w_tile = Wf + ((size_t)t * H_) * N_;
            for (int n = 0; n < N_; n++) {
                float acc = 0.0f;
                for (int h = 0; h < H_; h++) {
                    acc += x_row[h] * w_tile[(size_t)h * N_ + n];
                }
                Y[(size_t)m * N_ + n] += acc;
            }
        }
    }
}

static int check_dX(void) {
    float X[M * H], Y[M * N];
    int8_t U[T * H], W[T * H * N];
    int32_t decisions[M * K_SEL];
    int8_t signs[M * K_SEL];
    float dY[M * N], dX[M * H];

    for (int i = 0; i < M * H; i++) X[i] = rand_float();
    for (int i = 0; i < T * H; i++) {
        uint32_t r = xorshift32() % 3u;
        U[i] = (int8_t)((r == 0) ? -1 : (r == 1) ? 0 : 1);
    }
    for (int i = 0; i < T * H * N; i++) {
        uint32_t r = xorshift32() % 3u;
        W[i] = (int8_t)((r == 0) ? -1 : (r == 1) ? 0 : 1);
    }

    rroute_forward_select(decisions, signs, X, U, M, H, T, K_SEL);
    rroute_forward_dispatch(Y, X, W, decisions, signs, M, H, T, N, K_SEL);
    memcpy(dY, Y, sizeof(Y));
    rroute_backward_dX(dX, dY, W, decisions, signs, M, H, T, N, K_SEL);

    const double eps = 1e-2;
    double max_err = 0.0;
    int max_idx = -1;
    for (int i = 0; i < M * H; i++) {
        float Xsave = X[i];
        X[i] = (float)(Xsave + eps);
        /* IMPORTANT: keep decisions fixed (STE). Do NOT re-select. */
        rroute_forward_dispatch(Y, X, W, decisions, signs, M, H, T, N, K_SEL);
        double lp = loss_sqsum(Y, M * N);
        X[i] = (float)(Xsave - eps);
        rroute_forward_dispatch(Y, X, W, decisions, signs, M, H, T, N, K_SEL);
        double lm = loss_sqsum(Y, M * N);
        X[i] = Xsave;
        double num = (lp - lm) / (2.0 * eps);
        double ana = dX[i];
        double abs_err = fabs(num - ana);
        double denom = fmax(fabs(num), fabs(ana));
        double rel_err = (denom > 1e-6) ? abs_err / denom : abs_err;
        double err = fmin(rel_err, abs_err / 1e-1);
        if (err > max_err) { max_err = err; max_idx = i; }
    }

    printf("  dX combined err = %.3e @ idx %d\n", max_err, max_idx);
    return (max_err < 1e-3) ? 0 : 1;
}

static int check_dW(void) {
    float X[M * H], Y[M * N];
    int8_t U[T * H], W[T * H * N];
    float Wf[T * H * N];
    int32_t decisions[M * K_SEL];
    int8_t signs[M * K_SEL];
    float dY[M * N];
    float dW_latent[T * H * N];

    for (int i = 0; i < M * H; i++) X[i] = rand_float();
    for (int i = 0; i < T * H; i++) {
        uint32_t r = xorshift32() % 3u;
        U[i] = (int8_t)((r == 0) ? -1 : (r == 1) ? 0 : 1);
    }
    for (int i = 0; i < T * H * N; i++) {
        uint32_t r = xorshift32() % 3u;
        W[i] = (int8_t)((r == 0) ? -1 : (r == 1) ? 0 : 1);
        Wf[i] = (float)W[i];
    }

    rroute_forward_select(decisions, signs, X, U, M, H, T, K_SEL);
    rroute_forward_dispatch(Y, X, W, decisions, signs, M, H, T, N, K_SEL);
    memcpy(dY, Y, sizeof(Y));
    memset(dW_latent, 0, sizeof(dW_latent));
    rroute_backward_dW(dW_latent, dY, X, decisions, signs, M, H, T, N, K_SEL);

    const double eps = 1e-2;
    double max_err = 0.0;
    int max_idx = -1;
    int checked = 0;
    for (int i = 0; i < T * H * N; i++) {
        /* Only check W slots for tiles that were actually selected
         * (STE: unselected tiles get zero gradient by construction). */
        int t = i / (H * N);
        int selected_for_any = 0;
        for (int m = 0; m < M && !selected_for_any; m++) {
            for (int sel = 0; sel < K_SEL; sel++) {
                if (decisions[m * K_SEL + sel] == t && signs[m * K_SEL + sel] != 0) {
                    selected_for_any = 1; break;
                }
            }
        }
        if (!selected_for_any) continue;
        checked++;

        float Wsave = Wf[i];
        Wf[i] = (float)(Wsave + eps);
        rroute_forward_dispatch_float_W(Y, X, Wf, decisions, signs, M, H, T, N, K_SEL);
        double lp = loss_sqsum(Y, M * N);
        Wf[i] = (float)(Wsave - eps);
        rroute_forward_dispatch_float_W(Y, X, Wf, decisions, signs, M, H, T, N, K_SEL);
        double lm = loss_sqsum(Y, M * N);
        Wf[i] = Wsave;
        double num = (lp - lm) / (2.0 * eps);
        double ana = dW_latent[i];
        double abs_err = fabs(num - ana);
        double denom = fmax(fabs(num), fabs(ana));
        double rel_err = (denom > 1e-6) ? abs_err / denom : abs_err;
        double err = fmin(rel_err, abs_err / 1e-1);
        if (err > max_err) { max_err = err; max_idx = i; }
    }

    printf("  dW_latent combined err = %.3e @ idx %d  (checked %d selected slots of %d)\n",
           max_err, max_idx, checked, T * H * N);
    return (max_err < 1e-3) ? 0 : 1;
}

int main(void) {
    printf("test_gradient_routed (M=%d, H=%d, T=%d, N=%d, k=%d)\n",
           M, H, T, N, K_SEL);
    int fails = 0;
    if (check_dX()) { fprintf(stderr, "FAIL: dX gradient check\n"); fails++; }
    if (check_dW()) { fprintf(stderr, "FAIL: dW gradient check\n"); fails++; }
    if (fails == 0) printf("PASS: rroute gradients match finite differences on selected slots\n");
    return fails;
}
