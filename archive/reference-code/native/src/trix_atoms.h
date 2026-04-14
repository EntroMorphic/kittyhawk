#ifndef TRIX_ATOMS_H
#define TRIX_ATOMS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -- Core Types -- */

typedef struct {
    int in_dim;
    int hidden_dim;
    int out_dim;
    float* W1; float* b1;
    float* W2; float* b2;
    float* z1; float* h1; float* z2;
    float* dz2; float* dh1; float* dz1;
    float* dW1; float* db1; float* dW2; float* db2;
    int batch_cap;
} TrixAtomFFN;

/* -- Vector -- */
void trix_vec_add(float* dst, const float* a, const float* b, int n);
void trix_vec_sub(float* dst, const float* a, const float* b, int n);
void trix_vec_mul(float* dst, const float* a, const float* b, int n);
void trix_vec_scale(float* dst, const float* a, float s, int n);
void trix_vec_fma(float* dst, const float* a, float s, int n);
void trix_vec_zero(float* x, int n);
void trix_vec_add_inplace(float* dst, const float* a, int n);

/* -- Reduce -- */
float trix_dot(const float* a, const float* b, int n);
float trix_sum_sq(const float* x, int n);

/* -- MatMul -- */
void trix_matmul(float* C, const float* A, const float* B, int M, int N, int K);
void trix_matmul_bt(float* C, const float* A, const float* B, int M, int N, int K);
void trix_matmul_at(float* C, const float* A, const float* B, int M, int N, int K);

/* -- Bias -- */
void trix_bias_add(float* x, const float* b, int batch, int dim);
void trix_bias_grad(float* db, const float* dy, int batch, int dim);

/* -- Activate -- */
void trix_gelu(float* dst, const float* src, int n);
void trix_gelu_grad(float* dx, const float* dy, const float* src, int n);
void trix_softmax(float* dst, const float* src, int rows, int cols);
void trix_argmax(int* dst, const float* src, int rows, int cols);

/* -- Normalization -- */
void trix_layernorm_forward_save(float* y, float* mean_out, float* rstd_out, const float* x, const float* weight, const float* bias, int rows, int cols, float eps);
void trix_layernorm_backward(float* dx, float* dw, float* db, const float* dy, const float* x, const float* weight, const float* mean, const float* rstd, int rows, int cols);

/* -- Optimizer -- */
void trix_adamw_update(float* w, const float* grad, float* m, float* v, float lr, float b1, float b2, float eps, float wd, int step, int n);
void trix_sgd_update(float* w, const float* grad, float lr, int n);

/* -- Bitwise / Ternary -- */
void trix_mtfp21_quantize(float* dst, const float* src, int n);
void trix_pack_ternary(const int8_t* src, uint8_t* dst, int dim);
int32_t trix_popcount_dist_neon(const uint8_t* a, const uint8_t* b, const uint8_t* mask, int packed_dim);
void trix_ternary_pack_weights_i8(uint8_t* packed, const int8_t* weights, int M, int K);
void trix_ternary_matvec_i8(int32_t* y, const int8_t* act, const uint8_t* W_packed, int M, int K);

/* -- FFN Atoms -- */
TrixAtomFFN* trix_atom_ffn_create(int in_dim, int hidden_dim, int out_dim, uint64_t seed);
void trix_atom_ffn_destroy(TrixAtomFFN* ffn);
void trix_atom_ffn_forward(TrixAtomFFN* ffn, const float* x, float* out, int batch);
void trix_atom_ffn_backward(TrixAtomFFN* ffn, const float* x, const float* dy, float* dx, int batch);
void trix_atom_ffn_zero_grad(TrixAtomFFN* ffn);
void trix_atom_ffn_sgd_step(TrixAtomFFN* ffn, float lr);
float trix_atom_ffn_accuracy(TrixAtomFFN* ffn, const float* x, const int* labels, int batch);

/* -- Loss -- */
float trix_cross_entropy_loss(const float* logits, const int* targets, int batch, int vocab);
void trix_cross_entropy_grad(float* d_logits, const float* logits, const int* targets, int batch, int vocab);

#ifdef __cplusplus
}
#endif

#endif
