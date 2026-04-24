/*
 * test_gradient_linear.c — numerical gradient check for tlinear backward.
 *
 * Two checks:
 *
 *   1. dX validation: fix ternary W, float X, compute loss L(X) = ½||Y||²
 *      against forward output Y. Analytical dX via tlinear_backward_dX.
 *      Numerical dX via central finite difference on X[m,k]. Compare.
 *
 *   2. dW_latent validation: promote W to float (identical numerical values
 *      since W is already {-1, 0, +1}). Compute loss L(W_float). Analytical
 *      dW_latent via tlinear_backward_dW (which is the float outer product
 *      of X and dY — the exact thing we'd accumulate into a latent).
 *      Numerical dW via central finite difference on W_float[n,k]. Compare.
 *
 * Both checks use identical X, W, and dY values so the analytical gradient
 * comes from one backward call and the numerical gradient comes from
 * repeated forward calls.
 *
 * Relative error tolerance: 1e-3. Strict enough to catch index-swap bugs
 * and transposes; loose enough to tolerate float32 roundoff at these
 * matrix dimensions.
 */

#include "backward_linear.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 3
#define K 7
#define N 5

static uint32_t xorshift_state = 0x13579bdfu;
static uint32_t xorshift32(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}
static float rand_float(void) {
    /* (-1, +1) uniform, reproducible. */
    return (float)((int32_t)xorshift32()) / 2147483648.0f;
}

/* Loss accumulation in double precision. The finite-difference gradient
 * subtracts two losses that differ by O(ε × gradient); when losses are
 * computed in float32 at magnitude O(N×K), their subtraction leaves
 * roundoff on the order of 1e-5 absolute, which at ε=1e-3 dominates
 * relative error. Double-precision accumulation drops that to ~1e-13,
 * well below any analytical gradient magnitude we care about. */
static double loss_sqsum(const float* Y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += 0.5 * (double)Y[i] * (double)Y[i];
    return s;
}

/* Forward with a float W (used for dW check where W is promoted to float). */
static void tlinear_forward_float(
    float* Y, const float* X, const float* Wf, int M_, int K_, int N_)
{
    for (int m = 0; m < M_; m++) {
        for (int n = 0; n < N_; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K_; k++) {
                acc += X[(size_t)m * K_ + k] * Wf[(size_t)n * K_ + k];
            }
            Y[(size_t)m * N_ + n] = acc;
        }
    }
}

static int check_dX(void) {
    float X[M * K], Y[M * N];
    int8_t W[N * K];
    float dY[M * N];
    float dX[M * K];

    for (int i = 0; i < M * K; i++) X[i] = rand_float();
    for (int i = 0; i < N * K; i++) {
        uint32_t r = xorshift32() % 3u;
        W[i] = (int8_t)((r == 0) ? -1 : (r == 1) ? 0 : 1);
    }

    tlinear_forward(Y, X, W, M, K, N);
    /* For loss L = ½||Y||², dL/dY = Y. */
    memcpy(dY, Y, sizeof(Y));
    tlinear_backward_dX(dX, dY, W, M, K, N);

    /* ε chosen so finite-difference amplitude (2ε × gradient) dominates
     * float32 analytical noise (~7·ε_f · N·K). At ε=1e-2 and matrix sizes
     * here, that's O(1e-4) noise vs O(1e-2) signal — 100× margin. */
    const double eps = 1e-2;
    double max_err = 0.0;
    int max_idx = -1;
    for (int i = 0; i < M * K; i++) {
        float Xsave = X[i];
        X[i] = (float)(Xsave + eps);
        tlinear_forward(Y, X, W, M, K, N);
        double lp = loss_sqsum(Y, M * N);
        X[i] = (float)(Xsave - eps);
        tlinear_forward(Y, X, W, M, K, N);
        double lm = loss_sqsum(Y, M * N);
        X[i] = Xsave;
        double num = (lp - lm) / (2.0 * eps);
        double ana = dX[i];
        /* Combined tolerance: accept if relative error < 1e-3 OR absolute
         * error < 1e-4. The absolute fallback handles positions where the
         * true gradient is near zero (relative error has no useful meaning). */
        double abs_err = fabs(num - ana);
        double denom = fmax(fabs(num), fabs(ana));
        double rel_err = (denom > 1e-6) ? abs_err / denom : abs_err;
        double err = fmin(rel_err, abs_err / 1e-1);  /* 1e-4 abs == 1e-3 "rel-like" */
        if (err > max_err) { max_err = err; max_idx = i; }
    }

    printf("  dX combined err = %.3e @ idx %d\n", max_err, max_idx);
    return (max_err < 1e-3) ? 0 : 1;
}

static int check_dW(void) {
    float X[M * K], Y[M * N];
    int8_t W[N * K];
    float Wf[N * K];
    float dY[M * N];
    float dW_latent[N * K];

    for (int i = 0; i < M * K; i++) X[i] = rand_float();
    for (int i = 0; i < N * K; i++) {
        uint32_t r = xorshift32() % 3u;
        W[i] = (int8_t)((r == 0) ? -1 : (r == 1) ? 0 : 1);
        Wf[i] = (float)W[i];
    }

    tlinear_forward(Y, X, W, M, K, N);
    memcpy(dY, Y, sizeof(Y));
    memset(dW_latent, 0, sizeof(dW_latent));
    tlinear_backward_dW(dW_latent, dY, X, M, K, N);

    const double eps = 1e-2;
    double max_err = 0.0;
    int max_idx = -1;
    for (int i = 0; i < N * K; i++) {
        float Wsave = Wf[i];
        Wf[i] = (float)(Wsave + eps);
        tlinear_forward_float(Y, X, Wf, M, K, N);
        double lp = loss_sqsum(Y, M * N);
        Wf[i] = (float)(Wsave - eps);
        tlinear_forward_float(Y, X, Wf, M, K, N);
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

    printf("  dW_latent combined err = %.3e @ idx %d\n", max_err, max_idx);
    return (max_err < 1e-3) ? 0 : 1;
}

int main(void) {
    printf("test_gradient_linear (M=%d, K=%d, N=%d)\n", M, K, N);
    int fails = 0;
    if (check_dX()) { fprintf(stderr, "FAIL: dX gradient check\n"); fails++; }
    if (check_dW()) { fprintf(stderr, "FAIL: dW gradient check\n"); fails++; }
    if (fails == 0) printf("PASS: tlinear gradients match finite differences\n");
    return fails;
}
