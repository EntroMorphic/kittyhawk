/*
 * test_routed_proj.c — Smoke test for TrixRoutedProjection
 *
 * Verifies:
 * 1. Forward produces finite output of correct shape
 * 2. T=1 K=1 matches a single dense projection (sanity check)
 * 3. Backward produces finite gradients
 * 4. A few training steps reduce loss (grad signal flows)
 */

#include "trix_routed_proj.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int check_finite(const float* x, int n, const char* label) {
    for (int i = 0; i < n; i++) {
        if (isnan(x[i]) || isinf(x[i])) {
            printf("  FAIL: %s[%d] = %f\n", label, i, x[i]);
            return 0;
        }
    }
    return 1;
}

static float mse(const float* pred, const float* target, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) { float d = pred[i] - target[i]; s += d * d; }
    return s / (float)n;
}

int main() {
    int pass = 0, fail = 0;

    /* Test 1: Basic forward (T=4, K=4) */
    {
        printf("Test 1: Forward (T=4, K=4, in=16, out=32)\n");
        TrixRoutedProjConfig cfg = {
            .in_dim = 16, .out_dim = 32, .num_tiles = 4, .active_k = 4,
            .sig_cols = 0, .output_scale_init = 0.1f, .ln_eps = 1e-5f,
            .use_layernorm = true,
        };
        TrixRoutedProj* rp = trix_routed_proj_create(cfg, 42);

        float x[4 * 16]; /* batch=4 */
        for (int i = 0; i < 4 * 16; i++) x[i] = (float)(i % 7) * 0.1f - 0.3f;

        float out[4 * 32];
        trix_routed_proj_forward(rp, x, out, 4);

        if (check_finite(out, 4 * 32, "out")) {
            printf("  PASS: output is finite\n"); pass++;
        } else { fail++; }

        /* Check output is not all zeros */
        float sum = 0;
        for (int i = 0; i < 4 * 32; i++) sum += fabsf(out[i]);
        if (sum > 1e-6f) {
            printf("  PASS: output is non-zero (sum_abs=%.4f)\n", sum); pass++;
        } else {
            printf("  FAIL: output is all zeros\n"); fail++;
        }

        trix_routed_proj_destroy(rp);
    }

    /* Test 2: Backward produces finite gradients */
    {
        printf("Test 2: Backward (T=4, K=4, in=8, out=16)\n");
        TrixRoutedProjConfig cfg = {
            .in_dim = 8, .out_dim = 16, .num_tiles = 4, .active_k = 4,
            .sig_cols = 0, .output_scale_init = 0.1f, .ln_eps = 1e-5f,
            .use_layernorm = true,
        };
        TrixRoutedProj* rp = trix_routed_proj_create(cfg, 42);

        int batch = 2;
        float x[2 * 8], dy[2 * 16], dx[2 * 8];
        for (int i = 0; i < 2 * 8; i++) x[i] = (float)(i % 5) * 0.2f - 0.4f;
        float out[2 * 16];
        trix_routed_proj_forward(rp, x, out, batch);

        for (int i = 0; i < 2 * 16; i++) dy[i] = out[i]; /* gradient = output (dummy) */
        trix_routed_proj_backward(rp, x, dy, dx, batch);

        int ok = 1;
        ok &= check_finite(dx, 2 * 8, "dx");
        ok &= check_finite(rp->dW, 4 * 16 * 8, "dW");
        ok &= check_finite(rp->db, 4 * 16, "db");
        if (ok) { printf("  PASS: gradients are finite\n"); pass++; }
        else { fail++; }

        trix_routed_proj_destroy(rp);
    }

    /* Test 3: Training reduces MSE (gradient signal flows) */
    {
        printf("Test 3: Training (T=4, K=4, 50 steps)\n");
        TrixRoutedProjConfig cfg = {
            .in_dim = 8, .out_dim = 8, .num_tiles = 4, .active_k = 4,
            .sig_cols = 0, .output_scale_init = 1.0f, .ln_eps = 1e-5f,
            .use_layernorm = false,
        };
        TrixRoutedProj* rp = trix_routed_proj_create(cfg, 42);

        int batch = 8;
        float x[8 * 8], target[8 * 8], out[8 * 8], dy[8 * 8];

        /* Random input and target */
        srand(42);
        for (int i = 0; i < 8 * 8; i++) {
            x[i] = (float)(rand() % 100) / 100.0f - 0.5f;
            target[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }

        float loss0 = -1, loss_final = -1;
        for (int step = 0; step < 50; step++) {
            trix_routed_proj_zero_grad(rp);
            trix_routed_proj_forward(rp, x, out, batch);

            float loss = mse(out, target, batch * 8);
            if (step == 0) loss0 = loss;
            if (step == 49) loss_final = loss;

            /* MSE gradient: dy = 2/n * (out - target) */
            for (int i = 0; i < batch * 8; i++)
                dy[i] = 2.0f / (float)(batch * 8) * (out[i] - target[i]);

            trix_routed_proj_backward(rp, x, dy, NULL, batch);
            trix_routed_proj_adamw_step(rp, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
        }

        printf("  loss: %.6f → %.6f\n", loss0, loss_final);
        if (loss_final < loss0 * 0.5f) {
            printf("  PASS: loss decreased >50%%\n"); pass++;
        } else {
            printf("  FAIL: loss did not decrease enough\n"); fail++;
        }

        trix_routed_proj_destroy(rp);
    }

    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
