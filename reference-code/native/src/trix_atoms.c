#include "trix_atoms.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdio.h>

#ifdef APPLE
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#define TRIX_HAS_ACCELERATE 1
#else
#define TRIX_HAS_ACCELERATE 0
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TRIX_HAS_NEON 1
#else
#define TRIX_HAS_NEON 0
#endif

/* -- PRNG (xoshiro128+) -- */
static inline uint32_t at_rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
typedef struct { uint32_t s[4]; } at_rng_t;
static uint32_t at_rng_next(at_rng_t* r) {
    uint32_t result = r->s[0] + r->s[3], t = r->s[1] << 9;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1]; r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = at_rotl(r->s[3], 11); return result;
}
static float at_rng_uniform(at_rng_t* r) { return (float)(at_rng_next(r) >> 8) / 16777216.0f; }
static at_rng_t at_rng_seed(uint64_t seed) {
    at_rng_t r; r.s[0]=(uint32_t)seed; r.s[1]=(uint32_t)(seed>>32); r.s[2]=(uint32_t)(seed*2654435761ULL); r.s[3]=(uint32_t)((seed*2654435761ULL)>>32);
    for(int i=0; i<16; i++) at_rng_next(&r); return r;
}
static void at_xavier_init(float* w, int fan_in, int fan_out, int n, at_rng_t* rng) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i=0; i<n; i++) w[i] = (2.0f * at_rng_uniform(rng) - 1.0f) * limit;
}

/* -- Basic Vector Atoms -- */
void trix_vec_add(float* dst, const float* a, const float* b, int n) {
    int i = 0;
#if TRIX_HAS_NEON
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
    for (; i < n; i++) dst[i] = a[i] + b[i];
}
void trix_vec_sub(float* dst, const float* a, const float* b, int n) {
    int i = 0;
#if TRIX_HAS_NEON
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
    for (; i < n; i++) dst[i] = a[i] - b[i];
}
void trix_vec_mul(float* dst, const float* a, const float* b, int n) {
    int i = 0;
#if TRIX_HAS_NEON
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
    for (; i < n; i++) dst[i] = a[i] * b[i];
}
void trix_vec_scale(float* dst, const float* a, float s, int n) {
    int i = 0;
#if TRIX_HAS_NEON
    float32x4_t vs = vdupq_n_f32(s);
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vs));
#endif
    for (; i < n; i++) dst[i] = a[i] * s;
}
void trix_vec_fma(float* dst, const float* a, float s, int n) {
    int i = 0;
#if TRIX_HAS_NEON
    float32x4_t vs = vdupq_n_f32(s);
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vmlaq_f32(vld1q_f32(dst + i), vld1q_f32(a + i), vs));
#endif
    for (; i < n; i++) dst[i] += a[i] * s;
}
void trix_vec_zero(float* x, int n) { memset(x, 0, n * sizeof(float)); }
void trix_vec_add_inplace(float* dst, const float* a, int n) { trix_vec_add(dst, dst, a, n); }

float trix_dot(const float* a, const float* b, int n) {
    float sum = 0; int i = 0;
#if TRIX_HAS_NEON
    float32x4_t vacc = vdupq_n_f32(0);
    for (; i + 4 <= n; i += 4) vacc = vfmaq_f32(vacc, vld1q_f32(a + i), vld1q_f32(b + i));
    sum = vaddvq_f32(vacc);
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
float trix_sum_sq(const float* x, int n) { return trix_dot(x, x, n); }

/* -- Matrix Atoms -- */

/* NEON small-matrix threshold: below this, direct NEON beats Accelerate's dispatch overhead */
#define TRIX_SMALL_GEMM_THRESH 16

#if TRIX_HAS_NEON
/* C[M,N] = A[M,K] @ B[K,N] — direct NEON, no library overhead
 * Strategy: for each row of A, accumulate A[i,k] * B[k,:] into C[i,:].
 * B[k,:] is a contiguous row — perfect for NEON. */
static void trix_matmul_neon(float* C, const float* A, const float* B, int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float* ci = C + i*N;
        const float* ai = A + i*K;
        for (int k = 0; k < K; k++) {
            float32x4_t va = vdupq_n_f32(ai[k]);
            const float* bk = B + k*N;
            int j = 0;
            for (; j + 4 <= N; j += 4)
                vst1q_f32(ci + j, vfmaq_f32(vld1q_f32(ci + j), va, vld1q_f32(bk + j)));
            for (; j < N; j++) ci[j] += ai[k] * bk[j];
        }
    }
}

/* C[M,N] = A[M,K] @ B^T[K,N]  where B is [N,K] row-major
 * B^T column j = B row j — contiguous access, perfect for NEON */
static void trix_matmul_bt_neon(float* C, const float* A, const float* B, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        const float* ai = A + i*K;
        for (int j = 0; j < N; j++) {
            const float* bj = B + j*K;
            float32x4_t vacc = vdupq_n_f32(0);
            int k = 0;
            for (; k + 4 <= K; k += 4)
                vacc = vfmaq_f32(vacc, vld1q_f32(ai + k), vld1q_f32(bj + k));
            float sum = vaddvq_f32(vacc);
            for (; k < K; k++) sum += ai[k] * bj[k];
            C[i*N + j] = sum;
        }
    }
}

/* C[K,N] = A^T[K,M] @ B[M,N]  where A is [M,K] row-major
 * Accumulate outer products: for each row m, C += A[m,:].T @ B[m,:] */
static void trix_matmul_at_neon(float* C, const float* A, const float* B, int M, int K, int N) {
    memset(C, 0, (size_t)K * N * sizeof(float));
    for (int m = 0; m < M; m++) {
        const float* am = A + m*K;
        const float* bm = B + m*N;
        for (int i = 0; i < K; i++) {
            float32x4_t va = vdupq_n_f32(am[i]);
            float* ci = C + i*N;
            int j = 0;
            for (; j + 4 <= N; j += 4)
                vst1q_f32(ci + j, vfmaq_f32(vld1q_f32(ci + j), va, vld1q_f32(bm + j)));
            for (; j < N; j++) ci[j] += am[i] * bm[j];
        }
    }
}
#endif

void trix_matmul(float* C, const float* A, const float* B, int M, int K, int N) {
#if TRIX_HAS_NEON && TRIX_HAS_ACCELERATE
    if (M * K * N < TRIX_SMALL_GEMM_THRESH * TRIX_SMALL_GEMM_THRESH * TRIX_SMALL_GEMM_THRESH)
        trix_matmul_neon(C, A, B, M, K, N);
    else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#elif TRIX_HAS_NEON
    trix_matmul_neon(C, A, B, M, K, N);
#elif TRIX_HAS_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    for (int i = 0; i < M; i++) { for (int j = 0; j < N; j++) { float sum = 0; for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j]; C[i * N + j] = sum; } }
#endif
}
void trix_matmul_bt(float* C, const float* A, const float* B, int M, int K, int N) {
#if TRIX_HAS_NEON && TRIX_HAS_ACCELERATE
    if (M * K * N < TRIX_SMALL_GEMM_THRESH * TRIX_SMALL_GEMM_THRESH * TRIX_SMALL_GEMM_THRESH)
        trix_matmul_bt_neon(C, A, B, M, K, N);
    else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#elif TRIX_HAS_NEON
    trix_matmul_bt_neon(C, A, B, M, K, N);
#elif TRIX_HAS_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int i = 0; i < M; i++) { for (int j = 0; j < N; j++) { float sum = 0; for (int k = 0; k < K; k++) sum += A[i * K + k] * B[j * K + k]; C[i * N + j] = sum; } }
#endif
}
void trix_matmul_at(float* C, const float* A, const float* B, int M, int K, int N) {
#if TRIX_HAS_NEON && TRIX_HAS_ACCELERATE
    if (M * K * N < TRIX_SMALL_GEMM_THRESH * TRIX_SMALL_GEMM_THRESH * TRIX_SMALL_GEMM_THRESH)
        trix_matmul_at_neon(C, A, B, M, K, N);
    else
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0f, A, K, B, N, 0.0f, C, N);
#elif TRIX_HAS_NEON
    trix_matmul_at_neon(C, A, B, M, K, N);
#elif TRIX_HAS_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int m = 0; m < M; m++) sum += A[m * K + i] * B[m * N + j];
            C[i * N + j] = sum;
        }
    }
#endif
}
void trix_bias_add(float* x, const float* b, int batch, int dim) { for (int i = 0; i < batch; i++) trix_vec_add_inplace(x + i * dim, b, dim); }
void trix_bias_grad(float* db, const float* dy, int batch, int dim) {
    trix_vec_zero(db, dim);
    for (int i = 0; i < batch; i++) trix_vec_add_inplace(db, dy + i * dim, dim);
}

/* -- Activation -- */
static inline float gelu_scalar(float x) { return x * 0.5f * (1.0f + erff(x * 0.70710678118f)); }
void trix_gelu(float* dst, const float* src, int n) { for (int i = 0; i < n; i++) dst[i] = gelu_scalar(src[i]); }
void trix_gelu_grad(float* dx, const float* dy, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        float x = src[i], c = 0.70710678118f, s = 0.79788456f;
        float e = expf(-0.5f * x * x); float phi = 0.5f * (1.0f + erff(x * c));
        dx[i] = dy[i] * (phi + x * s * e * 0.5f);
    }
}
void trix_softmax(float* dst, const float* src, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* s = src + r * cols; float* d = dst + r * cols;
        float max_val = -FLT_MAX; for (int c = 0; c < cols; c++) if (s[c] > max_val) max_val = s[c];
        float sum = 0; for (int c = 0; c < cols; c++) { d[c] = expf(s[c] - max_val); sum += d[c]; }
        float inv_sum = 1.0f / sum; for (int c = 0; c < cols; c++) d[c] *= inv_sum;
    }
}
void trix_argmax(int* dst, const float* src, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* s = src + r * cols; int best = 0; float best_val = s[0];
        for (int c = 1; c < cols; c++) if (s[c] > best_val) { best_val = s[c]; best = c; }
        dst[r] = best;
    }
}

/* -- Normalization -- */
void trix_layernorm_forward_save(float* y, float* mean_out, float* rstd_out, const float* x, const float* weight, const float* bias, int rows, int cols, float eps) {
#ifdef APPLE
    int chunks = rows < 64 ? rows : 64;
    dispatch_apply((size_t)chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk) {
        int start = (int)((chunk * (size_t)rows) / (size_t)chunks);
        int end = (int)((((chunk + 1) * (size_t)rows) / (size_t)chunks));
        for (int i = start; i < end; i++) {
#else
    for (int i=0; i<rows; i++) {
#endif
        const float* xi = x + i*cols; float* yi = y + i*cols;
        float sum = 0; for (int j=0; j<cols; j++) sum += xi[j];
        float mean = sum/(float)cols, var = 0;
        for (int j=0; j<cols; j++) { float d = xi[j]-mean; var += d*d; }
        float rstd = 1.0f / sqrtf(var/(float)cols + eps);
        mean_out[i] = mean; rstd_out[i] = rstd;
        for (int j=0; j<cols; j++) yi[j] = (xi[j]-mean)*rstd*weight[j] + bias[j];
#ifdef APPLE
        }
    });
#else
    }
#endif
}
void trix_layernorm_backward(float* dx, float* dw, float* db, const float* dy, const float* x, const float* weight, const float* mean, const float* rstd, int rows, int cols) {
    float inv_cols = 1.0f/(float)cols;
#ifdef APPLE
    int chunks = rows < 64 ? rows : 64;
    dispatch_apply((size_t)chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk) {
        int start = (int)((chunk * (size_t)rows) / (size_t)chunks);
        int end = (int)((((chunk + 1) * (size_t)rows) / (size_t)chunks));
        for (int i = start; i < end; i++) {
#else
    for (int i=0; i<rows; i++) {
#endif
        const float* dyi=dy+i*cols, *xi=x+i*cols; float* dxi=dx?dx+i*cols:NULL, m=mean[i], rs=rstd[i];
        float sum_dy_w=0, sum_dy_w_xh=0;
        for (int j=0; j<cols; j++) {
            float xh=(xi[j]-m)*rs, dy_w=dyi[j]*weight[j];
            sum_dy_w+=dy_w; sum_dy_w_xh+=dy_w*xh;
        }
        if(dxi) for(int j=0; j<cols; j++) dxi[j]=rs*(dyi[j]*weight[j] - inv_cols*(sum_dy_w + (xi[j]-m)*rs*sum_dy_w_xh));
#ifdef APPLE
        }
    });
#else
    }
#endif
    for (int i=0; i<rows; i++) {
        const float* dyi=dy+i*cols, *xi=x+i*cols; float m=mean[i], rs=rstd[i];
        for (int j=0; j<cols; j++) { dw[j]+=dyi[j]*(xi[j]-m)*rs; db[j]+=dyi[j]; }
    }
}

/* -- Bitwise Kernels -- */
#define MTFP21_SCALE  59049.0f
#define MTFP21_MAX    5230176601.0f
static inline float mtfp21_quantize_scalar(float x) {
    float scaled = x * MTFP21_SCALE; float rounded = __builtin_roundf(scaled);
    if (rounded > MTFP21_MAX) rounded = MTFP21_MAX; if (rounded < -MTFP21_MAX) rounded = -MTFP21_MAX;
    return rounded / MTFP21_SCALE;
}
void trix_mtfp21_quantize(float* dst, const float* src, int n) {
#if TRIX_HAS_NEON
    float32x4_t v_scale=vdupq_n_f32(MTFP21_SCALE), v_inv=vdupq_n_f32(1.0f/MTFP21_SCALE), v_max=vdupq_n_f32(MTFP21_MAX), v_neg=vdupq_n_f32(-MTFP21_MAX);
    int i=0; for(; i+4<=n; i+=4) {
        float32x4_t rounded = vrndnq_f32(vmulq_f32(vld1q_f32(src+i), v_scale));
        vst1q_f32(dst+i, vmulq_f32(vminq_f32(vmaxq_f32(rounded, v_neg), v_max), v_inv));
    }
    for(; i<n; i++) dst[i] = mtfp21_quantize_scalar(src[i]);
#else
    for (int i=0; i<n; i++) dst[i] = mtfp21_quantize_scalar(src[i]);
#endif
}
void trix_pack_ternary(const int8_t* src, uint8_t* dst, int dim) {
    int packed_dim = (dim + 3) / 4; memset(dst, 0, (size_t)packed_dim);
    for (int i = 0; i < dim; i++) { uint8_t code = (src[i] == 1) ? 0x01 : (src[i] == -1) ? 0x02 : 0x00; dst[i / 4] |= (code << ((i % 4) * 2)); }
}
int32_t trix_popcount_dist_neon(const uint8_t* a, const uint8_t* b, const uint8_t* mask, int packed_dim) {
#if TRIX_HAS_NEON
    int i = 0; uint32x4_t acc = vdupq_n_u32(0);
    for (; i + 16 <= packed_dim; i += 16) {
        uint8x16_t xored = vandq_u8(veorq_u8(vld1q_u8(a+i), vld1q_u8(b+i)), vld1q_u8(mask+i));
        acc = vaddq_u32(acc, vpaddlq_u16(vpaddlq_u8(vcntq_u8(xored))));
    }
    int32_t total = (int32_t)vaddvq_u32(acc);
    for (; i < packed_dim; i++) { uint8_t x = (a[i] ^ b[i]) & mask[i]; while (x) { total++; x &= x - 1; } }
    return total;
#else
    int32_t total = 0; for (int i = 0; i < packed_dim; i++) { uint8_t x = (a[i] ^ b[i]) & mask[i]; while (x) { total++; x &= x - 1; } }
    return total;
#endif
}
void trix_ternary_pack_weights_i8(uint8_t* packed, const int8_t* weights, int M, int K) {
    int Kp = K / 4;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t byte = 0;
            for (int i = 0; i < 4; i++) { int8_t w = weights[m * K + k + i]; uint8_t trit = (w == 1) ? 1 : (w == -1) ? 2 : 0; byte |= (trit << (i * 2)); }
            packed[m * Kp + k / 4] = byte;
        }
    }
}
void trix_ternary_matvec_i8(int32_t* y, const int8_t* act, const uint8_t* W_packed, int M, int K) {
    int Kp = K / 4;
#if TRIX_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    static const int8_t TRIT_DECODE[16] __attribute__((aligned(16))) = {0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0};
    int8x16_t lut = vld1q_s8(TRIT_DECODE); uint8x16_t mask_03 = vdupq_n_u8(0x03);
    for (int m = 0; m < M; m++) {
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0), acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
        const int8_t* a_ptr = act; const uint8_t* w_ptr = W_packed + m * Kp;
        int k = 0;
        for (; k + 64 <= K; k += 64) {
            int8x16x4_t a4 = vld4q_s8(a_ptr); a_ptr += 64; uint8x16_t wp = vld1q_u8(w_ptr); w_ptr += 16;
            acc0 = vdotq_s32(acc0, vqtbl1q_s8(lut, vandq_u8(wp, mask_03)), a4.val[0]);
            acc1 = vdotq_s32(acc1, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(wp, 2), mask_03)), a4.val[1]);
            acc2 = vdotq_s32(acc2, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(wp, 4), mask_03)), a4.val[2]);
            acc3 = vdotq_s32(acc3, vqtbl1q_s8(lut, vshrq_n_u8(wp, 6)), a4.val[3]);
        }
        y[m] = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
        for (; k < K; k += 4) {
            uint8_t p = W_packed[m * Kp + k / 4];
            for (int i = 0; i < 4 && (k + i) < K; i++) {
                int t = (p >> (i * 2)) & 0x3;
                if (t == 1) y[m] += act[k + i];
                else if (t == 2) y[m] -= act[k + i];
            }
        }
    }
#else
    for (int m = 0; m < M; m++) { y[m] = 0; for (int k = 0; k < K; k += 4) { uint8_t p = W_packed[m * Kp + k / 4]; for (int i = 0; i < 4 && (k + i) < K; i++) { int t = (p >> (i * 2)) & 0x3; if (t == 1) y[m] += act[k + i]; else if (t == 2) y[m] -= act[k + i]; } } }
#endif
}

/* -- FFN Atoms -- */
TrixAtomFFN* trix_atom_ffn_create(int in_dim, int hidden_dim, int out_dim, uint64_t seed) {
    TrixAtomFFN* f = calloc(1, sizeof(TrixAtomFFN));
    at_rng_t rng = at_rng_seed(seed);
    f->in_dim = in_dim; f->hidden_dim = hidden_dim; f->out_dim = out_dim;
    f->W1 = malloc(hidden_dim * in_dim * sizeof(float)); f->b1 = calloc(hidden_dim, sizeof(float));
    f->W2 = malloc(out_dim * hidden_dim * sizeof(float)); f->b2 = calloc(out_dim, sizeof(float));
    f->dW1 = calloc(hidden_dim * in_dim, sizeof(float)); f->db1 = calloc(hidden_dim, sizeof(float));
    f->dW2 = calloc(out_dim * hidden_dim, sizeof(float)); f->db2 = calloc(out_dim, sizeof(float));
    at_xavier_init(f->W1, in_dim, hidden_dim, hidden_dim * in_dim, &rng);
    at_xavier_init(f->W2, hidden_dim, out_dim, out_dim * hidden_dim, &rng);
    f->batch_cap = 0; return f;
}
void trix_atom_ffn_destroy(TrixAtomFFN* ffn) { if (!ffn) return; free(ffn->W1); free(ffn->b1); free(ffn->W2); free(ffn->b2); free(ffn->dW1); free(ffn->db1); free(ffn->dW2); free(ffn->db2); free(ffn->z1); free(ffn->h1); free(ffn->z2); free(ffn->dz2); free(ffn->dh1); free(ffn->dz1); free(ffn); }
void trix_atom_ffn_forward(TrixAtomFFN* ffn, const float* x, float* out, int batch) {
    if (batch > ffn->batch_cap) {
        ffn->z1 = realloc(ffn->z1, batch * ffn->hidden_dim * sizeof(float));
        ffn->h1 = realloc(ffn->h1, batch * ffn->hidden_dim * sizeof(float));
        ffn->dh1 = realloc(ffn->dh1, batch * ffn->hidden_dim * sizeof(float));
        ffn->dz1 = realloc(ffn->dz1, batch * ffn->hidden_dim * sizeof(float));
        ffn->z2 = realloc(ffn->z2, batch * ffn->out_dim * sizeof(float));
        ffn->dz2 = realloc(ffn->dz2, batch * ffn->out_dim * sizeof(float));
        ffn->batch_cap = batch;
    }
    trix_matmul_bt(ffn->z1, x, ffn->W1, batch, ffn->in_dim, ffn->hidden_dim); trix_bias_add(ffn->z1, ffn->b1, batch, ffn->hidden_dim); trix_gelu(ffn->h1, ffn->z1, batch * ffn->hidden_dim);
    trix_matmul_bt(out, ffn->h1, ffn->W2, batch, ffn->hidden_dim, ffn->out_dim); trix_bias_add(out, ffn->b2, batch, ffn->out_dim);
}
void trix_atom_ffn_backward(TrixAtomFFN* ffn, const float* x, const float* dy, float* dx, int batch) {
    trix_matmul_at(ffn->dW2, dy, ffn->h1, batch, ffn->out_dim, ffn->hidden_dim); trix_bias_grad(ffn->db2, dy, batch, ffn->out_dim);
    trix_matmul(ffn->dh1, dy, ffn->W2, batch, ffn->out_dim, ffn->hidden_dim); trix_gelu_grad(ffn->dz1, ffn->dh1, ffn->z1, batch * ffn->hidden_dim);
    trix_matmul_at(ffn->dW1, ffn->dz1, x, batch, ffn->hidden_dim, ffn->in_dim); trix_bias_grad(ffn->db1, ffn->dz1, batch, ffn->hidden_dim);
    if (dx) trix_matmul(dx, ffn->dz1, ffn->W1, batch, ffn->hidden_dim, ffn->in_dim);
}
void trix_atom_ffn_zero_grad(TrixAtomFFN* ffn) { trix_vec_zero(ffn->dW1, ffn->hidden_dim * ffn->in_dim); trix_vec_zero(ffn->db1, ffn->hidden_dim); trix_vec_zero(ffn->dW2, ffn->out_dim * ffn->hidden_dim); trix_vec_zero(ffn->db2, ffn->out_dim); }
void trix_atom_ffn_sgd_step(TrixAtomFFN* ffn, float lr) { trix_sgd_update(ffn->W1, ffn->dW1, lr, ffn->hidden_dim * ffn->in_dim); trix_sgd_update(ffn->b1, ffn->db1, lr, ffn->hidden_dim); trix_sgd_update(ffn->W2, ffn->dW2, lr, ffn->out_dim * ffn->hidden_dim); trix_sgd_update(ffn->b2, ffn->db2, lr, ffn->out_dim); }
void trix_sgd_update(float* w, const float* grad, float lr, int n) { for (int i = 0; i < n; i++) w[i] -= lr * grad[i]; }
float trix_atom_ffn_accuracy(TrixAtomFFN* ffn, const float* x, const int* labels, int batch) {
    float* out = malloc(batch * ffn->out_dim * sizeof(float)); trix_atom_ffn_forward(ffn, x, out, batch);
    int correct = 0; for (int i = 0; i < batch; i++) {
        const float* p = out + i * ffn->out_dim; int best = 0; float bv = p[0];
        for (int j = 1; j < ffn->out_dim; j++) if (p[j] > bv) { bv = p[j]; best = j; }
        if (best == labels[i]) correct++;
    }
    free(out); return (float)correct / (float)batch;
}

/* -- Loss -- */
float trix_cross_entropy_loss(const float* logits, const int* targets, int batch, int vocab) {
    float total = 0; for (int i = 0; i < batch; i++) {
        const float* l = logits + i * vocab; float max_l = -FLT_MAX;
        for (int j=0; j<vocab; j++) if (l[j]>max_l) max_l = l[j];
        float sum_exp = 0; for (int j=0; j<vocab; j++) sum_exp += expf(l[j] - max_l);
        total += logf(sum_exp) + max_l - l[targets[i]];
    }
    return total / (float)batch;
}
void trix_cross_entropy_grad(float* d_logits, const float* logits, const int* targets, int batch, int vocab) {
    for (int i = 0; i < batch; i++) {
        float* dl = d_logits + i * vocab; const float* l = logits + i * vocab;
        float max_l = -FLT_MAX; for (int j=0; j<vocab; j++) if (l[j]>max_l) max_l = l[j];
        float sum_exp = 0; for (int j=0; j<vocab; j++) { dl[j] = expf(l[j] - max_l); sum_exp += dl[j]; }
        float inv_sum = 1.0f / sum_exp; for (int j=0; j<vocab; j++) dl[j] = (dl[j] * inv_sum) / (float)batch;
        dl[targets[i]] -= 1.0f / (float)batch;
    }
}

/* -- AdamW Update -- */
void trix_adamw_update(float* w, const float* grad, float* m, float* v, float lr, float b1, float b2, float eps, float wd, int step, int n) {
    float bc1=1.0f-powf(b1,(float)step), bc2=1.0f-powf(b2,(float)step);
    for (int i=0; i<n; i++) {
        w[i] -= lr*wd*w[i]; m[i] = b1*m[i] + (1.0f-b1)*grad[i]; v[i] = b2*v[i] + (1.0f-b2)*grad[i]*grad[i];
        w[i] -= lr*(m[i]/bc1) / (sqrtf(v[i]/bc2) + eps);
    }
}
