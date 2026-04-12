/*
 * trix_ternary_matvec.c — NEON multiply-free ternary matvec
 *
 * Adapted from yinsen/neon/neon_ternary.c.
 * SDOT kernel for ARM with dotprod, scalar fallback otherwise.
 */

#include "trix_ternary_matvec.h"
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TRIX_HAS_NEON 1
#else
#define TRIX_HAS_NEON 0
#endif

/* Decode table: 2-bit trit index → signed int8
 * 0b00 (0) → 0, 0b01 (1) → +1, 0b10 (2) → -1, 0b11 (3) → 0 */
#if TRIX_HAS_NEON
static const int8_t TRIT_DECODE[16] __attribute__((aligned(16))) = {
    0, 1, -1, 0,
    0, 1, -1, 0,
    0, 1, -1, 0,
    0, 1, -1, 0,
};
#endif

/* ══════════════════════════════════════════════════════════════════════
 * Weight packing
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_pack_weights(
    uint8_t* packed, const float* weights, int M, int K)
{
    int Kp = K / 4;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t byte = 0;
            for (int i = 0; i < 4; i++) {
                float w = weights[m * K + k + i];
                uint8_t trit = 0;
                if (w > 0.5f) trit = 1;       /* +1 */
                else if (w < -0.5f) trit = 2;  /* -1 */
                byte |= (trit << (i * 2));
            }
            packed[m * Kp + k / 4] = byte;
        }
    }
}

void trix_ternary_pack_weights_i8(
    uint8_t* packed, const int8_t* weights, int M, int K)
{
    int Kp = K / 4;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t byte = 0;
            for (int i = 0; i < 4; i++) {
                int8_t w = weights[m * K + k + i];
                uint8_t trit = 0;
                if (w == 1) trit = 1;
                else if (w == -1) trit = 2;
                byte |= (trit << (i * 2));
            }
            packed[m * Kp + k / 4] = byte;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Int8 matvec: NEON SDOT with VLD4 deinterleaving
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_matvec_i8(
    int32_t* y, const int8_t* act, const uint8_t* W_packed,
    int M, int K)
{
    int Kp = K / 4;

#if TRIX_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    int8x16_t lut = vld1q_s8(TRIT_DECODE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);

    for (int m = 0; m < M; m++) {
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);

        const int8_t* a_ptr = act;
        const uint8_t* w_ptr = W_packed + m * Kp;

        /* Process 64 K-elements per iteration */
        int k = 0;
        for (; k + 64 <= K; k += 64) {
            /* VLD4 deinterleaves 64 activations into 4 streams of 16 */
            int8x16x4_t a4 = vld4q_s8(a_ptr);
            a_ptr += 64;

            /* Load 16 packed weight bytes = 64 trits */
            uint8x16_t wp = vld1q_u8(w_ptr);
            w_ptr += 16;

            /* Stream 0: bits 1:0 */
            int8x16_t w0 = vqtbl1q_s8(lut, vandq_u8(wp, mask_03));
            acc0 = vdotq_s32(acc0, w0, a4.val[0]);

            /* Stream 1: bits 3:2 */
            int8x16_t w1 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(wp, 2), mask_03));
            acc1 = vdotq_s32(acc1, w1, a4.val[1]);

            /* Stream 2: bits 5:4 */
            int8x16_t w2 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(wp, 4), mask_03));
            acc2 = vdotq_s32(acc2, w2, a4.val[2]);

            /* Stream 3: bits 7:6 */
            int8x16_t w3 = vqtbl1q_s8(lut, vshrq_n_u8(wp, 6));
            acc3 = vdotq_s32(acc3, w3, a4.val[3]);
        }

        /* Merge accumulators */
        int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        int32_t sum = vaddvq_s32(acc);

        /* Scalar tail for remaining elements */
        for (; k < K; k += 4) {
            uint8_t packed = W_packed[m * Kp + k / 4];
            for (int i = 0; i < 4 && (k + i) < K; i++) {
                int trit = (packed >> (i * 2)) & 0x3;
                if (trit == 1) sum += act[k + i];
                else if (trit == 2) sum -= act[k + i];
            }
        }

        y[m] = sum;
    }

#else
    /* Scalar fallback */
    for (int m = 0; m < M; m++) {
        int32_t sum = 0;
        for (int k = 0; k < K; k += 4) {
            uint8_t packed = W_packed[m * Kp + k / 4];
            for (int i = 0; i < 4 && (k + i) < K; i++) {
                int trit = (packed >> (i * 2)) & 0x3;
                if (trit == 1) sum += act[k + i];
                else if (trit == 2) sum -= act[k + i];
            }
        }
        y[m] = sum;
    }
#endif
}

/* ══════════════════════════════════════════════════════════════════════
 * Float interface: quantize → ternary matvec → dequantize
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_matvec_f32(
    float* y, const float* W_ternary, const float* x,
    int M, int K)
{
    /* Find scale for int8 quantization of x */
    float absmax = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = fabsf(x[k]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-10f) {
        memset(y, 0, M * sizeof(float));
        return;
    }

    float scale = 127.0f / absmax;
    float inv_scale = absmax / 127.0f;

    /* Quantize x to int8 */
    int8_t* x_i8 = (int8_t*)__builtin_alloca(((K + 63) / 64 * 64) * sizeof(int8_t));
    for (int k = 0; k < K; k++) {
        float v = x[k] * scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        x_i8[k] = (int8_t)__builtin_roundf(v);
    }
    /* Pad to multiple of 64 for SDOT */
    for (int k = K; k < (K + 63) / 64 * 64; k++) x_i8[k] = 0;

    /* Pack ternary weights */
    int Kp = (K + 3) / 4;
    uint8_t* W_packed = (uint8_t*)__builtin_alloca(M * Kp);
    trix_ternary_pack_weights(W_packed, W_ternary, M, (K / 4) * 4);

    /* Run int8 ternary matvec */
    int K_aligned = (K / 4) * 4;
    int32_t* y_i32 = (int32_t*)__builtin_alloca(M * sizeof(int32_t));
    trix_ternary_matvec_i8(y_i32, x_i8, W_packed, M, K_aligned);

    /* Convert back to float */
    for (int m = 0; m < M; m++) {
        y[m] = (float)y_i32[m] * inv_scale;
    }

    /* Handle remaining K elements (if K not multiple of 4) */
    for (int m = 0; m < M; m++) {
        for (int k = K_aligned; k < K; k++) {
            float w = W_ternary[m * K + k];
            if (w > 0.5f) y[m] += x[k];
            else if (w < -0.5f) y[m] -= x[k];
        }
    }
}
