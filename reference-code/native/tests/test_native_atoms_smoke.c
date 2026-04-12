#include "trix_atoms.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_near(float actual, float expected, float tol, const char* label) {
    if (fabsf(actual - expected) > tol) {
        fprintf(stderr, "%s mismatch: got %.6f expected %.6f\n", label, actual, expected);
        return 0;
    }
    return 1;
}

static int check_finite_buffer(const float* x, int n, const char* label) {
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) {
            fprintf(stderr, "%s[%d] is not finite\n", label, i);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {7, 8, 9, 10, 11, 12};
    float C[4] = {0};
    float CBT[4] = {0};
    float CM[4] = {0};
    float db[2] = {123.0f, -456.0f};
    float dy[] = {1, 2, 3, 4, 5, 6};

    trix_matmul_bt(CBT, A, B, 2, 3, 2);
    if (!check_near(CBT[0], 50.0f, 1e-4f, "matmul_bt[0,0]")) return 1;
    if (!check_near(CBT[1], 68.0f, 1e-4f, "matmul_bt[0,1]")) return 1;
    if (!check_near(CBT[2], 122.0f, 1e-4f, "matmul_bt[1,0]")) return 1;
    if (!check_near(CBT[3], 167.0f, 1e-4f, "matmul_bt[1,1]")) return 1;

    trix_matmul(CM, A, B, 2, 3, 2);
    if (!check_near(CM[0], 58.0f, 1e-4f, "matmul[0,0]")) return 1;
    if (!check_near(CM[1], 64.0f, 1e-4f, "matmul[0,1]")) return 1;
    if (!check_near(CM[2], 139.0f, 1e-4f, "matmul[1,0]")) return 1;
    if (!check_near(CM[3], 154.0f, 1e-4f, "matmul[1,1]")) return 1;

    trix_matmul_at(C, A, B, 3, 2, 2);
    if (!check_near(C[0], 89.0f, 1e-4f, "matmul_at[0,0]")) return 1;
    if (!check_near(C[1], 98.0f, 1e-4f, "matmul_at[0,1]")) return 1;
    if (!check_near(C[2], 116.0f, 1e-4f, "matmul_at[1,0]")) return 1;
    if (!check_near(C[3], 128.0f, 1e-4f, "matmul_at[1,1]")) return 1;

    trix_bias_grad(db, dy, 3, 2);
    if (!check_near(db[0], 9.0f, 1e-6f, "bias_grad[0]")) return 1;
    if (!check_near(db[1], 12.0f, 1e-6f, "bias_grad[1]")) return 1;

    TrixAtomFFN* ffn = trix_atom_ffn_create(2, 4, 3, 42);
    if (!ffn) {
        fprintf(stderr, "failed to create atom ffn\n");
        return 1;
    }

    float x[] = {0.25f, -0.5f, 0.75f, 1.25f};
    float out[6] = {0};
    float dout[] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f};
    float dx[4] = {0};

    trix_atom_ffn_forward(ffn, x, out, 2);
    trix_atom_ffn_backward(ffn, x, dout, dx, 2);

    if (!check_finite_buffer(ffn->W1, 8, "W1")) return 1;
    if (!check_finite_buffer(out, 6, "out")) return 1;
    if (!check_finite_buffer(dx, 4, "dx")) return 1;
    if (!check_finite_buffer(ffn->dW1, 8, "dW1")) return 1;
    if (!check_finite_buffer(ffn->dW2, 12, "dW2")) return 1;

    trix_atom_ffn_destroy(ffn);
    return 0;
}
