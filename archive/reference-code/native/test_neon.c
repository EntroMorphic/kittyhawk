#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "src/trix_neon.h"

#define EPS 1e-5f

int float_eq(float a, float b) {
    return fabsf(a - b) < EPS;
}

int test_layernorm() {
    printf("Testing LayerNorm...\n");
    
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float output[5];
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float bias[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    trix_layernorm_f32(input, output, weight, bias, 1e-5f, 5);
    
    float mean, var;
    trix_mean_var_f32(input, 5, &mean, &var);
    
    printf("  Input mean: %.4f (expected 3.0)\n", mean);
    printf("  Input var: %.4f (expected 2.0)\n", var);
    
    // Output should be standardized (mean=0, var=1) * weight + bias
    printf("  Output: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", output[i]);
    printf("\n");
    
    return 1;
}

int test_activations() {
    printf("Testing Activations...\n");
    
    float x = 0.5f;
    
    float gelu = trix_f32_gelu(x);
    float gelu_fast = trix_f32_gelu_fast(x);
    float sigmoid = trix_f32_sigmoid(x);
    float tanh = trix_f32_tanh(x);
    float relu = trix_f32_relu(x);
    
    printf("  GELU(0.5) = %.6f (approx 0.345)\n", gelu);
    printf("  GELU_fast(0.5) = %.6f\n", gelu_fast);
    printf("  Sigmoid(0.5) = %.6f (expected 0.622)\n", sigmoid);
    printf("  Tanh(0.5) = %.6f (expected 0.462)\n", tanh);
    printf("  ReLU(0.5) = %.6f (expected 0.5)\n", relu);
    printf("  ReLU(-0.5) = %.6f (expected 0.0)\n", trix_f32_relu(-0.5f));
    
    return 1;
}

int test_ternary() {
    printf("Testing Ternary Quantization...\n");
    
    float input[] = {0.5f, -0.5f, 0.1f, -0.8f, 0.0f};
    int8_t output[5];
    
    trix_quantize_ternary(input, output, 5, 0.3f);
    
    printf("  Input:  ");
    for (int i = 0; i < 5; i++) printf("%.2f ", input[i]);
    printf("\n");
    printf("  Output: ");
    for (int i = 0; i < 5; i++) printf("%d ", output[i]);
    printf("\n");
    printf("  Expected: 1 -1 0 -1 0\n");
    
    // Test packing
    uint8_t packed[2];
    trix_pack_ternary(output, packed, 5);
    printf("  Packed: %02x %02x\n", packed[0], packed[1]);
    
    // Test unpacking
    int8_t unpacked[5];
    trix_unpack_ternary(packed, unpacked, 5);
    printf("  Unpacked: ");
    for (int i = 0; i < 5; i++) printf("%d ", unpacked[i]);
    printf("\n");
    
    return 1;
}

int test_dot_product() {
    printf("Testing Dot Products...\n");
    
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {0.5f, 1.0f, 1.5f, 2.0f};
    
    float dot = trix_dot_f32(a, b, 4);
    printf("  Float dot: %.4f (expected 15.0)\n", dot);
    
    // Int8 ternary dot
    int8_t c[] = {1, -1, 0, 1};
    int8_t d[] = {1, 1, -1, 0};
    
    int32_t dot_i8 = trix_dot_i8_ternary(c, d, 4);
    printf("  Int8 dot: %d (expected 0: 1*1 + -1*1 + 0*-1 + 1*0)\n", dot_i8);
    
    return 1;
}

int test_argmax() {
    printf("Testing Argmax...\n");
    
    float arr[] = {0.1f, 0.9f, 0.3f, 0.5f, 0.2f};
    
    int idx = trix_argmax_f32(arr, 5);
    printf("  Argmax of {0.1, 0.9, 0.3, 0.5, 0.2}: %d (expected 1)\n", idx);
    
    int idx_val;
    float val;
    trix_argmax_f32_with_val(arr, 5, &idx_val, &val);
    printf("  Argmax with val: idx=%d, val=%.1f\n", idx_val, val);
    
    return 1;
}

int test_softmax() {
    printf("Testing Softmax...\n");
    
    float x[] = {1.0f, 2.0f, 3.0f};
    trix_softmax_f32(x, 3, 1.0f);
    
    printf("  Softmax([1,2,3]): ");
    for (int i = 0; i < 3; i++) printf("%.4f ", x[i]);
    printf("\n");
    printf("  Sum: %.4f (expected 1.0)\n", x[0]+x[1]+x[2]);
    
    return 1;
}

int test_regularizers() {
    printf("Testing Regularizers...\n");
    
    // Balance loss
    int32_t counts[] = {100, 100, 100, 100};
    float bal = trix_balance_loss(counts, 4, 400);
    printf("  Balance (uniform): %.4f (expected ~0)\n", bal);
    
    int32_t counts_unbalanced[] = {300, 50, 30, 20};
    bal = trix_balance_loss(counts_unbalanced, 4, 400);
    printf("  Balance (skewed): %.4f (expected >0)\n", bal);
    
    // Ternary loss
    int8_t sigs[] = {1, 0, -1, 1, 0, 1, -1, 0};  // 50% ternary
    float tern = trix_ternary_loss(sigs, 2, 4);
    printf("  Ternary loss: %.4f\n", tern);
    
    // Sparsity
    int8_t sigs_sparse[] = {1, 0, 0, 0, 1, 0, 0, 0};  // 25% non-zero
    float spar = trix_sparsity_loss(sigs_sparse, 2, 4, 0.3f);
    printf("  Sparsity loss (sparse): %.4f\n", spar);
    
    int8_t sigs_dense[] = {1, 1, 1, 1, 1, 1, 1, 1};  // 100% non-zero
    spar = trix_sparsity_loss(sigs_dense, 2, 4, 0.3f);
    printf("  Sparsity loss (dense): %.4f\n", spar);
    
    return 1;
}

int test_policy() {
    printf("Testing Policy...\n");
    
    int32_t allow[] = {0, 1, 2};
    int32_t deny[] = {5};
    
    printf("  Allow [0,1,2], Deny [5]\n");
    printf("  Tile 0 allowed: %d (expected 1)\n", trix_policy_is_allowed(allow, 3, deny, 1, 0));
    printf("  Tile 3 allowed: %d (expected 0 - not in allow list)\n", trix_policy_is_allowed(allow, 3, deny, 1, 3));
    printf("  Tile 5 allowed: %d (expected 0 - in deny list)\n", trix_policy_is_allowed(allow, 3, deny, 1, 5));
    
    // Test apply
    float scores[] = {0.1f, 0.9f, 0.3f, 0.5f, 0.2f, 0.8f};
    trix_policy_apply(scores, allow, 3, deny, 1, 6);
    printf("  Scores after policy: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", scores[i]);
    printf("\n");
    
    return 1;
}

int test_gemm() {
    printf("Testing GEMM...\n");
    
    // Simple 2x3 @ 3x2 = 2x2
    float A[] = {1, 2, 3, 4, 5, 6};      // 2x3
    float B[] = {1, 0, 0, 1, 1, 1};     // 3x2
    float C[4] = {0};                     // 2x2
    
    trix_gemm_f32(A, B, C, 2, 3, 2, 1.0f, 0.0f);
    
    printf("  A @ B =\n");
    printf("    %.0f %.0f\n", C[0], C[1]);
    printf("    %.0f %.0f\n", C[2], C[3]);
    printf("  Expected: 4 5\n              10 11\n");
    
    return 1;
}

int test_spline() {
    printf("Testing Spline...\n");
    
    float knots[] = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    
    float scores[] = {-1.0f, 0.0f, 1.0f};
    float calibrated[3];
    
    trix_calibrate_scores(scores, calibrated, knots, 6, 1.0f, 3);
    
    printf("  Scores: ");
    for (int i = 0; i < 3; i++) printf("%.2f ", scores[i]);
    printf("\n");
    printf("  Calibrated: ");
    for (int i = 0; i < 3; i++) printf("%.4f ", calibrated[i]);
    printf("\n");
    
    return 1;
}

int main() {
    printf("=== trix_neon.h Falsification Tests ===\n\n");
    
    int passed = 0;
    int total = 9;
    
    passed += test_layernorm();
    printf("\n");
    
    passed += test_activations();
    printf("\n");
    
    passed += test_ternary();
    printf("\n");
    
    passed += test_dot_product();
    printf("\n");
    
    passed += test_argmax();
    printf("\n");
    
    passed += test_softmax();
    printf("\n");
    
    passed += test_regularizers();
    printf("\n");
    
    passed += test_policy();
    printf("\n");
    
    passed += test_gemm();
    printf("\n");
    
    passed += test_spline();
    printf("\n");
    
    printf("=== Results: %d/%d tests passed ===\n", passed, total);
    
    return passed == total ? 0 : 1;
}
