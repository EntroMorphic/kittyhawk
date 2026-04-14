/*
 * test_mtfp.c — Smoke test for MTFP arithmetic
 */

#include "trix_mtfp.h"
#include <stdio.h>
#include <math.h>

int main() {
    int pass = 0, fail = 0;

    /* Test 1: Conversion roundtrip */
    {
        float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.123f, 3.14159f, -6.0f};
        int n = sizeof(vals) / sizeof(vals[0]);
        printf("Test 1: Float → MTFP → Float roundtrip\n");
        int ok = 1;
        for (int i = 0; i < n; i++) {
            mtfp_t m = mtfp_from_float(vals[i]);
            float back = mtfp_to_float(m);
            float err = fabsf(back - vals[i]);
            if (err > 2e-5f) { printf("  FAIL: %.6f → %d → %.6f (err=%.2e)\n", vals[i], m, back, err); ok = 0; }
        }
        if (ok) { printf("  PASS\n"); pass++; } else fail++;
    }

    /* Test 2: Ternary matmul */
    {
        printf("Test 2: MTFP ternary matmul_bt\n");
        /* X[2,3] @ W[4,3]^T = Y[2,4] */
        float xf[] = {1.0f, 2.0f, 3.0f,   -1.0f, 0.5f, -0.5f};
        int8_t w[] = {1, 0, -1,   0, 1, 1,   -1, -1, 0,   1, 1, 1};  /* [4,3] */
        mtfp_t x[6], y[8];
        mtfp_from_float_batch(x, xf, 6);
        mtfp_ternary_matmul_bt(y, x, w, 2, 3, 4);

        /* Expected: row 0: 1*1+2*0+3*(-1)=-2, 1*0+2*1+3*1=5, 1*(-1)+2*(-1)+3*0=-3, 1*1+2*1+3*1=6
         *           row 1: -1*1+.5*0+(-.5)*(-1)=-.5, -1*0+.5*1+(-.5)*1=0, etc. */
        float yf[8];
        mtfp_to_float_batch(yf, y, 8);
        /* row1: -1*1+.5*0+(-.5)*(-1)=-.5, -1*0+.5*1+(-.5)*1=0, -1*(-1)+.5*(-1)+(-.5)*0=.5, -1*1+.5*1+(-.5)*1=-1 */
        float expected[] = {-2.0f, 5.0f, -3.0f, 6.0f,  -0.5f, 0.0f, 0.5f, -1.0f};
        int ok = 1;
        for (int i = 0; i < 8; i++) {
            if (fabsf(yf[i] - expected[i]) > 1e-4f) {
                printf("  FAIL: y[%d]=%.4f expected %.4f\n", i, yf[i], expected[i]);
                ok = 0;
            }
        }
        if (ok) { printf("  PASS\n"); pass++; } else fail++;
    }

    /* Test 3: GELU lookup */
    {
        printf("Test 3: MTFP GELU via lookup table\n");
        mtfp_gelu_init();
        float test_vals[] = {-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
        int n = sizeof(test_vals) / sizeof(test_vals[0]);
        int ok = 1;
        for (int i = 0; i < n; i++) {
            mtfp_t in = mtfp_from_float(test_vals[i]);
            mtfp_t out;
            mtfp_gelu(&out, &in, 1);
            float result = mtfp_to_float(out);
            /* Reference GELU */
            float x = test_vals[i];
            float ref = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f*x*x*x)));
            float err = fabsf(result - ref);
            if (err > 5e-4f) {
                printf("  FAIL: GELU(%.2f) = %.6f, expected %.6f (err=%.2e)\n", x, result, ref, err);
                ok = 0;
            }
        }
        if (ok) { printf("  PASS\n"); pass++; } else fail++;
    }

    /* Test 4: LayerNorm */
    {
        printf("Test 4: MTFP LayerNorm\n");
        float xf[] = {1.0f, 2.0f, 3.0f, 4.0f};  /* 1 row, 4 cols */
        float wf[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float bf[] = {0.0f, 0.0f, 0.0f, 0.0f};
        mtfp_t x[4], w[4], b[4], y[4];
        mtfp_from_float_batch(x, xf, 4);
        mtfp_from_float_batch(w, wf, 4);
        mtfp_from_float_batch(b, bf, 4);
        mtfp_layernorm(y, x, w, b, 1, 4);
        float yf[4];
        mtfp_to_float_batch(yf, y, 4);
        /* mean=2.5, var=1.25, std≈1.118. normalized: (-1.342, -0.447, 0.447, 1.342) */
        float expected[] = {-1.3416f, -0.4472f, 0.4472f, 1.3416f};
        int ok = 1;
        for (int i = 0; i < 4; i++) {
            if (fabsf(yf[i] - expected[i]) > 0.01f) {
                printf("  FAIL: y[%d]=%.4f expected %.4f\n", i, yf[i], expected[i]);
                ok = 0;
            }
        }
        if (ok) { printf("  PASS\n"); pass++; } else fail++;
    }

    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
