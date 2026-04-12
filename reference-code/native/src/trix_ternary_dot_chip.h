/*
 * Imported and adapted from yinsen/include/chips/ternary_dot_chip.h
 * Purpose: frozen ternary dot primitive for routed signature scoring.
 */

#ifndef TRIX_TERNARY_DOT_CHIP_H
#define TRIX_TERNARY_DOT_CHIP_H

#include <stdint.h>
#include <stddef.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TRIX_TERNARY_CHIP_HAS_NEON 1
#else
#define TRIX_TERNARY_CHIP_HAS_NEON 0
#endif

static inline int trix_ternary_coeff_from_code(uint8_t code) {
    return (int)(code & 1u) - (int)((code >> 1) & 1u);
}

static inline float trix_ternary_dot_chip_f32_scalar(
    const uint8_t* w_packed,
    const float* x,
    int n
) {
    float sum = 0.0f;
    int full_bytes = n / 4;
    int remainder = n & 3;

    for (int b = 0; b < full_bytes; b++) {
        uint8_t packed = w_packed[b];
        const float* xp = x + b * 4;

        uint8_t t0 = packed & 0x03u;
        uint8_t t1 = (packed >> 2) & 0x03u;
        uint8_t t2 = (packed >> 4) & 0x03u;
        uint8_t t3 = (packed >> 6) & 0x03u;

        int c0 = trix_ternary_coeff_from_code(t0);
        int c1 = trix_ternary_coeff_from_code(t1);
        int c2 = trix_ternary_coeff_from_code(t2);
        int c3 = trix_ternary_coeff_from_code(t3);

        sum += (float)c0 * xp[0]
            +  (float)c1 * xp[1]
            +  (float)c2 * xp[2]
            +  (float)c3 * xp[3];
    }

    if (remainder > 0) {
        uint8_t packed = w_packed[full_bytes];
        const float* xp = x + full_bytes * 4;
        for (int i = 0; i < remainder; i++) {
            uint8_t t = (packed >> (i * 2)) & 0x03u;
            int c = trix_ternary_coeff_from_code(t);
            sum += (float)c * xp[i];
        }
    }

    return sum;
}

#if TRIX_TERNARY_CHIP_HAS_NEON
static inline float trix_ternary_dot_chip_f32_neon(
    const uint8_t* w_packed,
    const float* x,
    int n
) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int d = 0;
    int pb = 0;

    for (; d + 16 <= n; d += 16, pb += 4) {
        int8_t coeffs[16];
        for (int j = 0; j < 4; j++) {
            uint8_t packed = w_packed[pb + j];
            coeffs[j * 4 + 0] = (int8_t)trix_ternary_coeff_from_code((uint8_t)(packed & 0x03u));
            coeffs[j * 4 + 1] = (int8_t)trix_ternary_coeff_from_code((uint8_t)((packed >> 2) & 0x03u));
            coeffs[j * 4 + 2] = (int8_t)trix_ternary_coeff_from_code((uint8_t)((packed >> 4) & 0x03u));
            coeffs[j * 4 + 3] = (int8_t)trix_ternary_coeff_from_code((uint8_t)((packed >> 6) & 0x03u));
        }

        int8x16_t vc = vld1q_s8(coeffs);
        int16x8_t vc_lo_s16 = vmovl_s8(vget_low_s8(vc));
        int16x8_t vc_hi_s16 = vmovl_s8(vget_high_s8(vc));

        int32x4_t vc0_s32 = vmovl_s16(vget_low_s16(vc_lo_s16));
        int32x4_t vc1_s32 = vmovl_s16(vget_high_s16(vc_lo_s16));
        int32x4_t vc2_s32 = vmovl_s16(vget_low_s16(vc_hi_s16));
        int32x4_t vc3_s32 = vmovl_s16(vget_high_s16(vc_hi_s16));

        float32x4_t c0 = vcvtq_f32_s32(vc0_s32);
        float32x4_t c1 = vcvtq_f32_s32(vc1_s32);
        float32x4_t c2 = vcvtq_f32_s32(vc2_s32);
        float32x4_t c3 = vcvtq_f32_s32(vc3_s32);

        float32x4_t x0 = vld1q_f32(x + d + 0);
        float32x4_t x1 = vld1q_f32(x + d + 4);
        float32x4_t x2 = vld1q_f32(x + d + 8);
        float32x4_t x3 = vld1q_f32(x + d + 12);

        acc = vfmaq_f32(acc, x0, c0);
        acc = vfmaq_f32(acc, x1, c1);
        acc = vfmaq_f32(acc, x2, c2);
        acc = vfmaq_f32(acc, x3, c3);
    }

    float sum = vaddvq_f32(acc);

    if (d < n) {
        int rem = n - d;
        int rem_bytes = (rem + 3) / 4;
        const uint8_t* wp = w_packed + pb;
        const float* xp = x + d;
        for (int b = 0; b < rem_bytes; b++) {
            uint8_t packed = wp[b];
            for (int j = 0; j < 4; j++) {
                int idx = b * 4 + j;
                if (idx >= rem) break;
                uint8_t t = (uint8_t)((packed >> (j * 2)) & 0x03u);
                int c = trix_ternary_coeff_from_code(t);
                sum += (float)c * xp[idx];
            }
        }
    }

    return sum;
}
#endif

static inline float trix_ternary_dot_chip_f32(
    const uint8_t* w_packed,
    const float* x,
    int n
) {
#if TRIX_TERNARY_CHIP_HAS_NEON
    return trix_ternary_dot_chip_f32_neon(w_packed, x, n);
#else
    return trix_ternary_dot_chip_f32_scalar(w_packed, x, n);
#endif
}

static inline float trix_ternary_dot_i8_f32_scalar(
    const int8_t* coeff,
    const float* x,
    int n
) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += (float)coeff[i] * x[i];
    }
    return sum;
}

#if TRIX_TERNARY_CHIP_HAS_NEON
static inline float trix_ternary_dot_i8_f32_neon(
    const int8_t* coeff,
    const float* x,
    int n
) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 16 <= n; i += 16) {
        int8x16_t vc = vld1q_s8(coeff + i);
        int16x8_t vc_lo_s16 = vmovl_s8(vget_low_s8(vc));
        int16x8_t vc_hi_s16 = vmovl_s8(vget_high_s8(vc));

        int32x4_t vc0_s32 = vmovl_s16(vget_low_s16(vc_lo_s16));
        int32x4_t vc1_s32 = vmovl_s16(vget_high_s16(vc_lo_s16));
        int32x4_t vc2_s32 = vmovl_s16(vget_low_s16(vc_hi_s16));
        int32x4_t vc3_s32 = vmovl_s16(vget_high_s16(vc_hi_s16));

        float32x4_t c0 = vcvtq_f32_s32(vc0_s32);
        float32x4_t c1 = vcvtq_f32_s32(vc1_s32);
        float32x4_t c2 = vcvtq_f32_s32(vc2_s32);
        float32x4_t c3 = vcvtq_f32_s32(vc3_s32);

        float32x4_t x0 = vld1q_f32(x + i + 0);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);

        acc = vfmaq_f32(acc, x0, c0);
        acc = vfmaq_f32(acc, x1, c1);
        acc = vfmaq_f32(acc, x2, c2);
        acc = vfmaq_f32(acc, x3, c3);
    }

    float sum = vaddvq_f32(acc);
    for (; i < n; i++) {
        sum += (float)coeff[i] * x[i];
    }
    return sum;
}
#endif

static inline float trix_ternary_dot_i8_f32(
    const int8_t* coeff,
    const float* x,
    int n
) {
#if TRIX_TERNARY_CHIP_HAS_NEON
    return trix_ternary_dot_i8_f32_neon(coeff, x, n);
#else
    return trix_ternary_dot_i8_f32_scalar(coeff, x, n);
#endif
}

static inline void trix_ternary_score4_i8_f32(
    float* scores,
    const int8_t* sig0,
    const int8_t* sig1,
    const int8_t* sig2,
    const int8_t* sig3,
    const float* x,
    int n
) {
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    for (int i = 0; i < n; i++) {
        float xv = x[i];
        s0 += (float)sig0[i] * xv;
        s1 += (float)sig1[i] * xv;
        s2 += (float)sig2[i] * xv;
        s3 += (float)sig3[i] * xv;
    }
    scores[0] = s0;
    scores[1] = s1;
    scores[2] = s2;
    scores[3] = s3;
}

static inline void trix_ternary_score_tiles_i8_f32(
    float* scores,
    const int8_t* signatures_i8,
    int num_tiles,
    int dim,
    const float* x
) {
    if (num_tiles == 4) {
        const int8_t* s0 = signatures_i8 + (size_t)0 * (size_t)dim;
        const int8_t* s1 = signatures_i8 + (size_t)1 * (size_t)dim;
        const int8_t* s2 = signatures_i8 + (size_t)2 * (size_t)dim;
        const int8_t* s3 = signatures_i8 + (size_t)3 * (size_t)dim;
        trix_ternary_score4_i8_f32(scores, s0, s1, s2, s3, x, dim);
        return;
    }
    for (int t = 0; t < num_tiles; t++) {
        const int8_t* sig = signatures_i8 + (size_t)t * (size_t)dim;
        scores[t] = trix_ternary_dot_i8_f32(sig, x, dim);
    }
}

#endif /* TRIX_TERNARY_DOT_CHIP_H */
