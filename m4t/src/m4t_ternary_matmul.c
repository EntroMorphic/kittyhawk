/*
 * m4t_ternary_matmul.c — MTFP × packed-trit matmul
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 *   Y[M,N] = X[M,K] @ W^T     where W is [N,K] packed ternary
 *
 * Activations X are m4t_mtfp_t (int32 MTFP cells).
 * Weights W_packed are 2-bit packed trits, LSB-first, Kp = (K+3)/4 bytes/row.
 * Output Y is m4t_mtfp_t (Case S — saturating clamp on store).
 *
 * Inner product per (i, j): decode K trits from W_packed[j], conditionally
 * add or subtract X[i, k] into an int64 accumulator. Zero multiplies.
 *
 * The trit-decode idiom (vld1q_u8 → shift/mask → vqtbl1q_s8) is borrowed
 * from trix-z's trix_ternary_matvec_i8. The accumulator pattern is new:
 * trix-z had no kernel that combines packed-trit weights with MTFP
 * activations.
 *
 * Bound analysis: |X[k]| ≤ M4T_MTFP_MAX_VAL = 581,130,733 (MTFP19).
 * Summing K contributions with int64 accumulator: for K ≤ ~1.59e10 the
 * total fits int64 (9.22e18 / 5.81e8 ≈ 1.59e10). Plain int64 accumulators
 * are safe; no periodic narrowing required.
 *
 * Saturation: per §8.5 Case S. If |acc| > MAX_VAL_MTFP19 on store, the
 * output is clamped to ±MAX_VAL and (when flags is non-NULL) the cell's
 * SATURATED bit is set per §14.4 layout.
 */

#include "m4t_ternary_matmul.h"
#include "m4t_mtfp4.h"
#include "m4t_trit_pack.h"
#include "m4t_mtfp.h"
#include "m4t_internal.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

/* NEON 16-trit decode constants. Static to file scope so they are loaded
 * once into constant pools, not re-materialized inside the inner loop. */
#if M4T_HAS_NEON
static const uint8_t M4T_TM_DUP_IDX[16] = {
    0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3
};
static const uint8_t M4T_TM_SHIFT_LANE[16] = {
    0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6
};
#endif

/* ── Forward decl for the prototype vmlal path (T-G3 below) ───────────── */
#if M4T_HAS_NEON
static int64_t ternary_dot_vmlal(
    const m4t_mtfp_t* xi,
    const uint8_t* wj,
    int K);
#endif

/* ── Inner product for a single output cell ───────────────────────────── */

/* Scalar-only: pure C, no NEON regardless of M4T_HAS_NEON. The reference
 * implementation. Used ONLY by m4t_mtfp_ternary_matmul_bt_scalar_ref
 * (test oracle). The "fallback inside ternary_dot" framing in earlier
 * comments was stale — production ternary_dot now dispatches directly
 * to ternary_dot_vmlal (NEON) per the no-scalar-in-production rule. */
static int64_t ternary_dot_scalar(
    const m4t_mtfp_t* xi,
    const uint8_t* wj,
    int K)
{
    int64_t acc = 0;
    for (int k = 0; k < K; k++) {
        uint8_t code = (uint8_t)((wj[k >> 2] >> ((k & 3) * 2)) & 0x3u);
        if      (code == 0x01u) acc += (int64_t)xi[k];
        else if (code == 0x02u) acc -= (int64_t)xi[k];
    }
    return acc;
}

/* T-G9 productionization: dispatch to vmlal_s32 path when M4T_HAS_NEON.
 *
 * History: the prior NEON path was a bit-select + conditional-negate
 * pipeline (~57 NEON ops per 16-trit block — mask-widening dominated).
 * The vmlal path uses the multiply-by-trit shortcut: trit ∈ {-1, 0, +1}
 * means multiplying by sign IS the operation, with the int64 widening
 * absorbing the accumulator semantics. Per the ternary_mac_routing LMM
 * cycle and T-G8 measurements:
 *   BATCHED  M=64 K=4096 N=64 : bsl-NEON 766 ns/cell → vmlal 657 (1.17×)
 *   TIGHT-LOOP M=4 K=64  N=4  : bsl-NEON 12.25 → vmlal 5.00 (2.45×)
 * vmlal beats bsl in both shapes; productionized as the default NEON
 * path.
 *
 * The prior bsl-NEON ternary_dot implementation is preserved in git
 * history per project rule "DELETE = never". To recover the bsl path
 * for reference (e.g., evaluating a different cell width that doesn't
 * fit vmlal's int32×int32→int64 shape):
 *   git show 35e5b58~1:m4t/src/m4t_ternary_matmul.c
 * The bsl approach is structurally important even though vmlal beat
 * it for ternary; its multi-stage decode + bsl pattern would generalize
 * to other "small-set value × wide-cell" workloads where multiplication
 * isn't naturally available. */
static int64_t ternary_dot(
    const m4t_mtfp_t* xi,    /* [K] MTFP activations */
    const uint8_t* wj,         /* [Kp] packed-trit weights */
    int K)
{
    /* NEON-only production dispatch. ternary_dot_scalar remains as
     * the implementation behind m4t_mtfp_ternary_matmul_bt_scalar_ref
     * (test oracle). Per project rule (feedback_function_over_speed_no_scalar). */
    return ternary_dot_vmlal(xi, wj, K);
}

#if M4T_HAS_NEON
/* T-G3: vmlal_s32-based ternary dot product. Routes through vmlal_s32
 * (signed multiply-accumulate long, int32×int32→int64 widening, 2 lanes
 * per instruction) — the closest M4/NEON hardware analog to a "ternary
 * MAC at int32 width." Per ternary_mac_routing_synthesize.md.
 *
 * T-G5 saturation argument:
 *   |x[k]|   ≤ M4T_MTFP_MAX_VAL ≈ 2^29.1
 *   |sign|   ≤ 1                = 2^0
 *   |x*sign| ≤ MAX_VAL          ≈ 2^29.1   (single-element MAC product)
 *   per-block sum across 16 elements: ≤ 16 × MAX_VAL ≈ 2^33.1
 *   The acc0+acc1 pair is int64x2, so each lane is int64.
 *   Worst-case |acc| over the entire dot product (K elements):
 *     |acc| ≤ K × MAX_VAL = K × 5.81e8
 *     For K up to ≈ 1.59 × 10^10, |acc| < INT64_MAX = 9.22 × 10^18
 *   The substrate's documented K bound (per file header) is ~1.59e10,
 *   matching the existing bsl-NEON path's bound. No int64 overflow
 *   for any K within that bound.
 *
 *   Final clamp via m4t_mtfp_clamp64 in the outer loop handles the
 *   MTFP19 store saturation (Case S, §8.5).
 *
 * Insight: trit ∈ {-1, 0, +1}. Multiplying by trit subsumes both the
 * conditional-negate AND the zero-gate of the existing bsl pattern.
 *   trit ==  0 → 0 × x = 0
 *   trit == +1 → 1 × x = x
 *   trit == -1 → -1 × x = -x
 * No mask widening, no vbsl, no vneg. Just decode → sign-extend → vmlal.
 *
 * Pipeline per 16-trit block:
 *   Decode 16 packed trits → 16 int8 signs (~6 ops, reuses TBL pipeline)
 *   Sign-extend int8 → int32 (~4 ops, vmovl chains)
 *   8× vmlal_s32 (2 lanes each = 16 elements; 4 calls into acc0,
 *     4 into acc1 — matches T-G1 measured throughput pattern C)
 *
 * T-G1 measured pattern C at 0.84 vmlal/cycle ≈ 9.5 cycles per 16 trits
 * for the vmlal phase alone. Full block ~17 cycles vs current bsl-NEON
 * ~30 cycles → ~1.8× expected speedup. */
static int64_t ternary_dot_vmlal(
    const m4t_mtfp_t* xi,
    const uint8_t* wj,
    int K)
{
    int64_t acc = 0;
    int k = 0;

    /* Decode constants (same as ternary_dot's NEON path; constants reused). */
    const uint8x16_t dup_idx  = vld1q_u8(M4T_TM_DUP_IDX);
    const int8x16_t  shift_s  = vreinterpretq_s8_u8(vld1q_u8(M4T_TM_SHIFT_LANE));
    const uint8x16_t mask_03  = vdupq_n_u8(0x03u);
    const int8x16_t  lut_sign = vld1q_s8(M4T_TRIT_DECODE_LUT);

    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);

    while (k + 16 <= K) {
        /* Decode 16 trits → 16 int8 signs in {-1, 0, +1}. */
        uint32_t w32;
        memcpy(&w32, wj + (k >> 2), 4);
        uint8x16_t packed  = vreinterpretq_u8_u32(vdupq_n_u32(w32));
        uint8x16_t dup     = vqtbl1q_u8(packed, dup_idx);
        uint8x16_t shifted = vshlq_u8(dup, vnegq_s8(shift_s));
        uint8x16_t codes   = vandq_u8(shifted, mask_03);
        int8x16_t  signs   = vqtbl1q_s8(lut_sign, codes);

        /* Sign-extend int8 → int16 → int32, two halves of 8 lanes each. */
        int16x8_t s16_lo = vmovl_s8(vget_low_s8(signs));
        int16x8_t s16_hi = vmovl_s8(vget_high_s8(signs));
        int32x4_t s32_0 = vmovl_s16(vget_low_s16(s16_lo));   /* signs[ 0.. 3] */
        int32x4_t s32_1 = vmovl_s16(vget_high_s16(s16_lo));  /* signs[ 4.. 7] */
        int32x4_t s32_2 = vmovl_s16(vget_low_s16(s16_hi));   /* signs[ 8..11] */
        int32x4_t s32_3 = vmovl_s16(vget_high_s16(s16_hi));  /* signs[12..15] */

        /* Load 16 int32 activations. */
        int32x4_t a0 = vld1q_s32(xi + k);
        int32x4_t a1 = vld1q_s32(xi + k + 4);
        int32x4_t a2 = vld1q_s32(xi + k + 8);
        int32x4_t a3 = vld1q_s32(xi + k + 12);

        /* MAC: acc += sign × activation, widening to int64.
         * 8 vmlal_s32 calls, split 4 into acc0 (low halves) and 4 into acc1
         * (high halves) — matches T-G1 measured pattern C. */
        acc0 = vmlal_s32(acc0, vget_low_s32(s32_0),  vget_low_s32(a0));
        acc1 = vmlal_s32(acc1, vget_high_s32(s32_0), vget_high_s32(a0));
        acc0 = vmlal_s32(acc0, vget_low_s32(s32_1),  vget_low_s32(a1));
        acc1 = vmlal_s32(acc1, vget_high_s32(s32_1), vget_high_s32(a1));
        acc0 = vmlal_s32(acc0, vget_low_s32(s32_2),  vget_low_s32(a2));
        acc1 = vmlal_s32(acc1, vget_high_s32(s32_2), vget_high_s32(a2));
        acc0 = vmlal_s32(acc0, vget_low_s32(s32_3),  vget_low_s32(a3));
        acc1 = vmlal_s32(acc1, vget_high_s32(s32_3), vget_high_s32(a3));

        k += 16;
    }
    acc = vgetq_lane_s64(acc0, 0) + vgetq_lane_s64(acc0, 1)
        + vgetq_lane_s64(acc1, 0) + vgetq_lane_s64(acc1, 1);

    /* Scalar tail. */
    for (; k < K; k++) {
        uint8_t code = (uint8_t)((wj[k >> 2] >> ((k & 3) * 2)) & 0x3u);
        if      (code == 0x01u) acc += (int64_t)xi[k];
        else if (code == 0x02u) acc -= (int64_t)xi[k];
    }
    return acc;
}

/* Per journal/m4t_matmul_tile_synthesize.md: tile-by-4 j cells.
 * 4 parallel acc-pair chains (8 acc registers total), shared X load across
 * 4 j cells per K iteration. Reduces vmlal-acc dependency latency-bound
 * throughput. Audit demonstrated ~1.8× wall-clock gain at apples-to-apples
 * comparison.
 *
 * out[0..3] receives the 4 dot products. wj0..wj3 are 4 packed-trit weight
 * rows. */
static void ternary_dot_vmlal_x4(
    int64_t out[4],
    const m4t_mtfp_t* xi,
    const uint8_t* wj0,
    const uint8_t* wj1,
    const uint8_t* wj2,
    const uint8_t* wj3,
    int K)
{
    int k = 0;

    /* Decode constants (shared across all 4 j cells). */
    const uint8x16_t dup_idx  = vld1q_u8(M4T_TM_DUP_IDX);
    const int8x16_t  shift_s  = vreinterpretq_s8_u8(vld1q_u8(M4T_TM_SHIFT_LANE));
    const uint8x16_t mask_03  = vdupq_n_u8(0x03u);
    const int8x16_t  lut_sign = vld1q_s8(M4T_TRIT_DECODE_LUT);

    /* 4 acc pairs, one per j cell. 8 acc registers total — within NEON's
     * 32 V regs comfortably alongside decode constants + X loads + scratch. */
    int64x2_t acc0_lo = vdupq_n_s64(0), acc0_hi = vdupq_n_s64(0);
    int64x2_t acc1_lo = vdupq_n_s64(0), acc1_hi = vdupq_n_s64(0);
    int64x2_t acc2_lo = vdupq_n_s64(0), acc2_hi = vdupq_n_s64(0);
    int64x2_t acc3_lo = vdupq_n_s64(0), acc3_hi = vdupq_n_s64(0);

    while (k + 16 <= K) {
        /* Load 16 int32 X activations (shared across 4 j cells). */
        int32x4_t a0 = vld1q_s32(xi + k);
        int32x4_t a1 = vld1q_s32(xi + k + 4);
        int32x4_t a2 = vld1q_s32(xi + k + 8);
        int32x4_t a3 = vld1q_s32(xi + k + 12);

        /* Per-j-cell macro: decode 16 trits + 8 vmlal_s32 into ACC pair.
         * Same shape as ternary_dot_vmlal's body, using shared a0..a3. */
        #define DECODE_AND_VMLAL_J(WJ, ACC_LO, ACC_HI) do {                    \
            uint32_t w32;                                                      \
            memcpy(&w32, (WJ) + (k >> 2), 4);                                  \
            uint8x16_t packed  = vreinterpretq_u8_u32(vdupq_n_u32(w32));       \
            uint8x16_t dup     = vqtbl1q_u8(packed, dup_idx);                  \
            uint8x16_t shifted = vshlq_u8(dup, vnegq_s8(shift_s));             \
            uint8x16_t codes   = vandq_u8(shifted, mask_03);                   \
            int8x16_t  signs   = vqtbl1q_s8(lut_sign, codes);                  \
            int16x8_t  s16_lo  = vmovl_s8(vget_low_s8(signs));                 \
            int16x8_t  s16_hi  = vmovl_s8(vget_high_s8(signs));                \
            int32x4_t  s32_0   = vmovl_s16(vget_low_s16(s16_lo));              \
            int32x4_t  s32_1   = vmovl_s16(vget_high_s16(s16_lo));             \
            int32x4_t  s32_2   = vmovl_s16(vget_low_s16(s16_hi));              \
            int32x4_t  s32_3   = vmovl_s16(vget_high_s16(s16_hi));             \
            (ACC_LO) = vmlal_s32((ACC_LO), vget_low_s32(s32_0),  vget_low_s32(a0)); \
            (ACC_HI) = vmlal_s32((ACC_HI), vget_high_s32(s32_0), vget_high_s32(a0)); \
            (ACC_LO) = vmlal_s32((ACC_LO), vget_low_s32(s32_1),  vget_low_s32(a1)); \
            (ACC_HI) = vmlal_s32((ACC_HI), vget_high_s32(s32_1), vget_high_s32(a1)); \
            (ACC_LO) = vmlal_s32((ACC_LO), vget_low_s32(s32_2),  vget_low_s32(a2)); \
            (ACC_HI) = vmlal_s32((ACC_HI), vget_high_s32(s32_2), vget_high_s32(a2)); \
            (ACC_LO) = vmlal_s32((ACC_LO), vget_low_s32(s32_3),  vget_low_s32(a3)); \
            (ACC_HI) = vmlal_s32((ACC_HI), vget_high_s32(s32_3), vget_high_s32(a3)); \
        } while (0)

        DECODE_AND_VMLAL_J(wj0, acc0_lo, acc0_hi);
        DECODE_AND_VMLAL_J(wj1, acc1_lo, acc1_hi);
        DECODE_AND_VMLAL_J(wj2, acc2_lo, acc2_hi);
        DECODE_AND_VMLAL_J(wj3, acc3_lo, acc3_hi);

        #undef DECODE_AND_VMLAL_J

        k += 16;
    }

    /* Reduce each acc pair to scalar. */
    int64_t r0 = vgetq_lane_s64(acc0_lo, 0) + vgetq_lane_s64(acc0_lo, 1)
               + vgetq_lane_s64(acc0_hi, 0) + vgetq_lane_s64(acc0_hi, 1);
    int64_t r1 = vgetq_lane_s64(acc1_lo, 0) + vgetq_lane_s64(acc1_lo, 1)
               + vgetq_lane_s64(acc1_hi, 0) + vgetq_lane_s64(acc1_hi, 1);
    int64_t r2 = vgetq_lane_s64(acc2_lo, 0) + vgetq_lane_s64(acc2_lo, 1)
               + vgetq_lane_s64(acc2_hi, 0) + vgetq_lane_s64(acc2_hi, 1);
    int64_t r3 = vgetq_lane_s64(acc3_lo, 0) + vgetq_lane_s64(acc3_lo, 1)
               + vgetq_lane_s64(acc3_hi, 0) + vgetq_lane_s64(acc3_hi, 1);

    /* Scalar geometric tail (k not multiple of 16). 4 j cells in lockstep. */
    for (; k < K; k++) {
        int64_t x_k = (int64_t)xi[k];
        uint8_t c0 = (uint8_t)((wj0[k >> 2] >> ((k & 3) * 2)) & 0x3u);
        uint8_t c1 = (uint8_t)((wj1[k >> 2] >> ((k & 3) * 2)) & 0x3u);
        uint8_t c2 = (uint8_t)((wj2[k >> 2] >> ((k & 3) * 2)) & 0x3u);
        uint8_t c3 = (uint8_t)((wj3[k >> 2] >> ((k & 3) * 2)) & 0x3u);
        if      (c0 == 0x01u) r0 += x_k;
        else if (c0 == 0x02u) r0 -= x_k;
        if      (c1 == 0x01u) r1 += x_k;
        else if (c1 == 0x02u) r1 -= x_k;
        if      (c2 == 0x01u) r2 += x_k;
        else if (c2 == 0x02u) r2 -= x_k;
        if      (c3 == 0x01u) r3 += x_k;
        else if (c3 == 0x02u) r3 -= x_k;
    }

    out[0] = r0; out[1] = r1; out[2] = r2; out[3] = r3;
}
#endif

/* ── Public entry ──────────────────────────────────────────────────────── */

/* Ternary × ternary → MTFP19 via SDOT.
 *
 * Bit-compatibility: m4t_trit_t and m4t_mtfp4_t are both int8_t
 * underneath. Ternary values {-1, 0, +1} are a strict subset of MTFP4's
 * mantissa range ±40, so the SDOT kernel computes the correct dot
 * product without overflow or scope reach.
 *
 * The static_assert here documents the deliberate type-pun: if the
 * underlying types ever diverge, the cast becomes unsafe and the build
 * fails. This is the substrate-discipline hook that catches a future
 * type drift.
 */
void m4t_ternary_dot_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_trit_t* W,
    int M, int K, int N)
{
    _Static_assert(sizeof(m4t_trit_t) == sizeof(m4t_mtfp4_t),
                   "m4t_trit_t and m4t_mtfp4_t must share underlying width "
                   "for the SDOT delegation to be bit-safe");
    /* Both inputs are int8 at the bit level; SDOT applies. The cast
     * is identity at runtime; the kernel's int8 × int8 → int32 path
     * is bit-exact for values in {-1, 0, +1}. */
    m4t_mtfp4_sdot_matmul_bt(Y, (const m4t_mtfp4_t*)X, W, M, K, N);
}

void m4t_mtfp_ternary_matmul_bt(
    m4t_mtfp_t* Y, const m4t_mtfp_t* X, const uint8_t* W_packed,
    uint8_t* flags,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_packed)));
    /* Aliasing: Y must not alias X or W_packed. */
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_packed);

    int Kp = M4T_TRIT_PACKED_BYTES(K);

    /* Per journal/m4t_matmul_tile_synthesize.md: register-tile by 4 j cells.
     * 4 parallel acc-pair chains (8 acc registers) pipeline better on
     * M-series. N%4 tail handled by single-j-cell ternary_dot path
     * (NEON, geometric tail rule). Without NEON, ternary_dot_vmlal_x4
     * doesn't exist; the entire j range falls through to the tail
     * (which dispatches to ternary_dot, also gated by NEON). */
#if M4T_HAS_NEON
    int j_tile_end = N - (N % 4);
#else
    int j_tile_end = 0;
#endif

    for (int i = 0; i < M; i++) {
        const m4t_mtfp_t* X_row = X + (size_t)i * K;
        m4t_mtfp_t*       Y_row = Y + (size_t)i * N;

#if M4T_HAS_NEON
        /* Tiled body: 4 j cells per outer iter. */
        for (int j = 0; j < j_tile_end; j += 4) {
            int64_t accs[4];
            ternary_dot_vmlal_x4(accs, X_row,
                W_packed + (size_t)(j + 0) * Kp,
                W_packed + (size_t)(j + 1) * Kp,
                W_packed + (size_t)(j + 2) * Kp,
                W_packed + (size_t)(j + 3) * Kp,
                K);
            for (int dj = 0; dj < 4; dj++) {
                m4t_mtfp_t out = m4t_mtfp_clamp64(accs[dj]);
                if (flags && accs[dj] != (int64_t)out) {
                    int cell_index = i * N + j + dj;
                    m4t_flag_or(flags, cell_index, M4T_FLAG_SATURATED);
                }
                Y_row[j + dj] = out;
            }
        }
#endif

        /* N%4 tail: 1-3 remaining j cells, single-j-cell NEON path. */
        for (int j = j_tile_end; j < N; j++) {
            int64_t acc = ternary_dot(X_row, W_packed + (size_t)j * Kp, K);
            m4t_mtfp_t out = m4t_mtfp_clamp64(acc);
            if (flags && acc != (int64_t)out) {
                int cell_index = i * N + j;
                m4t_flag_or(flags, cell_index, M4T_FLAG_SATURATED);
            }
            Y_row[j] = out;
        }
    }
}

/* Scalar-only reference. Same M·N outer loop; uses ternary_dot_scalar
 * for every cell regardless of M4T_HAS_NEON. Test-only oracle for
 * bit-exact verification gates. Per ternary_mac_routing T-G2. */
void m4t_mtfp_ternary_matmul_bt_scalar_ref(
    m4t_mtfp_t* Y, const m4t_mtfp_t* X, const uint8_t* W_packed,
    uint8_t* flags,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_packed)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_packed);

    int Kp = M4T_TRIT_PACKED_BYTES(K);

    for (int i = 0; i < M; i++) {
        const m4t_mtfp_t* X_row = X + (size_t)i * K;
        m4t_mtfp_t*       Y_row = Y + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            int64_t acc = ternary_dot_scalar(X_row, W_packed + (size_t)j * Kp, K);
            m4t_mtfp_t out = m4t_mtfp_clamp64(acc);
            if (flags && acc != (int64_t)out) {
                int cell_index = i * N + j;
                m4t_flag_or(flags, cell_index, M4T_FLAG_SATURATED);
            }
            Y_row[j] = out;
        }
    }
}

/* ── §20 5-in-8 base-3 packed matmul ──────────────────────────────────────
 *
 * Per M4T_SUBSTRATE.md §20 + journal/m4t_5in8_synthesize.md.
 * Ternary X (int8, unpacked) × 5-in-8 packed W → MTFP19 Y.
 *
 * Implementation ported from audit Path D (`base3_5in8_matmul_neon`) with
 * the same split-LUT decode, pre-permuted X, register-tile-by-4 pattern.
 * Bit-exact verified against scalar reference per ctest.
 */

#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)

/* §20 split-LUT decode tables. trit_value(u) = {0→0, 1→+1, 2→-1}. */
static const int8_t M4T_5IN8_LUT_LOW_D0[16] __attribute__((aligned(16))) = {
     0,  1, -1,  0,  1, -1,  0,  1, -1,    /* low % 3, low ∈ [0, 8] */
};
static const int8_t M4T_5IN8_LUT_LOW_D1[16] __attribute__((aligned(16))) = {
     0,  0,  0,  1,  1,  1, -1, -1, -1,    /* low / 3 */
};
static const int8_t M4T_5IN8_LUT_HIGH_D2[32] __attribute__((aligned(16))) = {
     0,  1, -1,  0,  1, -1,  0,  1, -1,    /* high % 3, high ∈ [0, 26] */
     0,  1, -1,  0,  1, -1,  0,  1, -1,
     0,  1, -1,  0,  1, -1,  0,  1, -1,
};
static const int8_t M4T_5IN8_LUT_HIGH_D3[32] __attribute__((aligned(16))) = {
     0,  0,  0,  1,  1,  1, -1, -1, -1,    /* (high/3) % 3 */
     0,  0,  0,  1,  1,  1, -1, -1, -1,
     0,  0,  0,  1,  1,  1, -1, -1, -1,
};
static const int8_t M4T_5IN8_LUT_HIGH_D4[32] __attribute__((aligned(16))) = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,    /* high / 9 */
     1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

#endif

void m4t_ternary_5in8_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_packed)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_packed);

    /* K=0 degenerate case: dot product over zero terms is 0. Handle
     * up front so the rest of the kernel never sees K=0 (avoids
     * NULL+0 pointer arithmetic UB on X, W_packed). */
    if (K == 0) {
        memset(Y, 0, (size_t)M * (size_t)N * sizeof(m4t_mtfp_t));
        return;
    }

#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    /* Per journal/k80_fix_lmm.md: the inner tile body processes 80-trit
     * chunks via 5 SDOTs each. Previously, K%80 trailing trits used a
     * geometric scalar tail (per-trit divide-modulo decode + multiply).
     * That tail dominated when K%80 was large — e.g., K=6912 (K%80=32)
     * spent ~18% of the kernel in scalar.
     *
     * The new path runs the NEON tile body to K_padded (next multiple
     * of 80 ≥ K). Trits at positions [K, K_padded) are zero on both
     * sides:
     *   - X side: pre-permute loop already zero-fills past K (trit_idx
     *     < K check). Just need the X_strided buffer sized for K_padded.
     *   - W side: at the boundary tile (k = K_aligned), W_packed has
     *     fewer than 16 valid bytes. We use a 16-byte stack-local buffer
     *     per j cell, copying the available real bytes and zeroing the
     *     rest. Zero W trits contribute 0 to the dot product.
     *
     * Result: bit-exact, no scalar tail, single NEON path covers all K.
     *
     * Performance characteristics (measured, K=N=2560 unless noted):
     *   K%80 == 0:        unchanged (fast path; conditional skips boundary).
     *   K%80 ∈ [4..79]:   collapses to ~K%80=0 baseline (former scalar
     *                     tail eliminated; up to +2.9× on K%80=79).
     *   K%80 ∈ [1..3]:    boundary tile fires for 1-3 real trits + many
     *                     zero-padded — pays full 80-trit SDOT setup.
     *                     Slight regression (~5%) at K%80=1 vs same
     *                     K%80=0 baseline. In BitNet's actual shapes
     *                     (K ∈ {2560, 6912, 640}, K%80 ∈ {0, 32}) this
     *                     case never fires.
     *   K < ~10:          boundary tile pays full SDOT setup for 1-9
     *                     real trits; slower in absolute terms (~µs)
     *                     than the old scalar-only path. Not a realistic
     *                     BitLinear shape; not optimized. */
    int Kp = (K + 4) / 5;            /* packed bytes per row */
    int K_aligned = K - (K % 80);    /* last full-80 boundary (≤ K) */
    int K_padded = ((K + 79) / 80) * 80;  /* next multiple of 80 ≥ K */
    int K5_padded = K_padded / 5;    /* X_d[d] array length per digit
                                      * (K_padded is always a multiple
                                      * of 80 = 16*5, hence of 5) */
    int j_tile_end = N - (N % 4);

    const uint8x16_t nine_v = vdupq_n_u8(9);
    const int8x16_t  lut_d0 = vld1q_s8(M4T_5IN8_LUT_LOW_D0);
    const int8x16_t  lut_d1 = vld1q_s8(M4T_5IN8_LUT_LOW_D1);
    const int8x16x2_t lut_d2 = { {
        vld1q_s8(M4T_5IN8_LUT_HIGH_D2 +  0),
        vld1q_s8(M4T_5IN8_LUT_HIGH_D2 + 16),
    } };
    const int8x16x2_t lut_d3 = { {
        vld1q_s8(M4T_5IN8_LUT_HIGH_D3 +  0),
        vld1q_s8(M4T_5IN8_LUT_HIGH_D3 + 16),
    } };
    const int8x16x2_t lut_d4 = { {
        vld1q_s8(M4T_5IN8_LUT_HIGH_D4 +  0),
        vld1q_s8(M4T_5IN8_LUT_HIGH_D4 + 16),
    } };

    /* Strided X buffer: K5_padded * 5 ≥ K_padded bytes. Trailing slots
     * (positions ≥ K) zero-padded by the pre-permute loop. */
    int alloc_size = K5_padded * 5;
    int8_t* X_strided = (alloc_size > 0) ? (int8_t*)malloc((size_t)alloc_size) : NULL;
    if (alloc_size > 0 && !X_strided) return;
    int8_t* X_d[5];
    for (int d = 0; d < 5; d++) {
        X_d[d] = X_strided + (size_t)d * K5_padded;
    }

    for (int i = 0; i < M; i++) {
        const m4t_trit_t* xi = X + (size_t)i * K;

        /* Permute X[i, :] into 5 stride-aligned arrays. Trit indices >= K
         * are zero-padded; the tile body now reads up to K5_padded slots
         * per X_d[d], which covers the boundary tile at k=K_aligned. */
        for (int n = 0; n < K5_padded; n++) {
            for (int d = 0; d < 5; d++) {
                int trit_idx = 5 * n + d;
                X_d[d][n] = (trit_idx < K) ? xi[trit_idx] : 0;
            }
        }

        /* Tile body: 4 j cells × full 80-trit chunks. */
        for (int j = 0; j < j_tile_end; j += 4) {
            const uint8_t* wj0 = W_packed + (size_t)(j + 0) * Kp;
            const uint8_t* wj1 = W_packed + (size_t)(j + 1) * Kp;
            const uint8_t* wj2 = W_packed + (size_t)(j + 2) * Kp;
            const uint8_t* wj3 = W_packed + (size_t)(j + 3) * Kp;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            #define M4T_5IN8_DECODE_AND_SDOT_BUF(BUF, ACC) do {            \
                uint8x16_t b = vld1q_u8(BUF);                              \
                uint16x8_t lo16 = vshrq_n_u16(                             \
                    vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 57), 9);         \
                uint16x8_t hi16 = vshrq_n_u16(                             \
                    vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 57), 9);        \
                uint8x16_t high = vcombine_u8(                             \
                    vmovn_u16(lo16), vmovn_u16(hi16));                     \
                uint8x16_t low  = vsubq_u8(b, vmulq_u8(high, nine_v));     \
                int8x16_t d0 = vqtbl1q_s8(lut_d0, low);                    \
                int8x16_t d1 = vqtbl1q_s8(lut_d1, low);                    \
                int8x16_t d2 = vqtbl2q_s8(lut_d2, high);                   \
                int8x16_t d3 = vqtbl2q_s8(lut_d3, high);                   \
                int8x16_t d4 = vqtbl2q_s8(lut_d4, high);                   \
                (ACC) = vdotq_s32((ACC), xv0, d0);                         \
                (ACC) = vdotq_s32((ACC), xv1, d1);                         \
                (ACC) = vdotq_s32((ACC), xv2, d2);                         \
                (ACC) = vdotq_s32((ACC), xv3, d3);                         \
                (ACC) = vdotq_s32((ACC), xv4, d4);                         \
            } while (0)

            /* Main NEON tile body: full 80-trit chunks within [0, K_aligned). */
            for (int k = 0; k < K_aligned; k += 80) {
                int x_idx = k / 5;
                int8x16_t xv0 = vld1q_s8(X_d[0] + x_idx);
                int8x16_t xv1 = vld1q_s8(X_d[1] + x_idx);
                int8x16_t xv2 = vld1q_s8(X_d[2] + x_idx);
                int8x16_t xv3 = vld1q_s8(X_d[3] + x_idx);
                int8x16_t xv4 = vld1q_s8(X_d[4] + x_idx);

                M4T_5IN8_DECODE_AND_SDOT_BUF(wj0 + k / 5, acc0);
                M4T_5IN8_DECODE_AND_SDOT_BUF(wj1 + k / 5, acc1);
                M4T_5IN8_DECODE_AND_SDOT_BUF(wj2 + k / 5, acc2);
                M4T_5IN8_DECODE_AND_SDOT_BUF(wj3 + k / 5, acc3);
            }

            /* Boundary tile: covers [K_aligned, K_padded) when K%80 != 0.
             * W_packed may have fewer than 16 valid bytes here; copy
             * what's available into a 16-byte stack-local buffer and
             * zero the rest. Zero W trits contribute 0 to the dot. */
            if (K_padded > K_aligned) {
                int k = K_aligned;
                int x_idx = k / 5;
                int8x16_t xv0 = vld1q_s8(X_d[0] + x_idx);
                int8x16_t xv1 = vld1q_s8(X_d[1] + x_idx);
                int8x16_t xv2 = vld1q_s8(X_d[2] + x_idx);
                int8x16_t xv3 = vld1q_s8(X_d[3] + x_idx);
                int8x16_t xv4 = vld1q_s8(X_d[4] + x_idx);

                int byte_off = k / 5;
                int avail = Kp - byte_off;
                /* Math: avail ∈ [1, 16] when boundary tile fires.
                 *   K = 80q + r, r ∈ [1, 79], byte_off = 16q,
                 *   Kp = 16q + ceil(r/5), so avail = ceil(r/5) ∈ [1, 16]. */
                assert(avail >= 1 && avail <= 16);
                uint8_t bb0[16] = {0}, bb1[16] = {0}, bb2[16] = {0}, bb3[16] = {0};
                memcpy(bb0, wj0 + byte_off, (size_t)avail);
                memcpy(bb1, wj1 + byte_off, (size_t)avail);
                memcpy(bb2, wj2 + byte_off, (size_t)avail);
                memcpy(bb3, wj3 + byte_off, (size_t)avail);
                M4T_5IN8_DECODE_AND_SDOT_BUF(bb0, acc0);
                M4T_5IN8_DECODE_AND_SDOT_BUF(bb1, acc1);
                M4T_5IN8_DECODE_AND_SDOT_BUF(bb2, acc2);
                M4T_5IN8_DECODE_AND_SDOT_BUF(bb3, acc3);
            }

            #undef M4T_5IN8_DECODE_AND_SDOT_BUF

            int32_t s0 = vaddvq_s32(acc0);
            int32_t s1 = vaddvq_s32(acc1);
            int32_t s2 = vaddvq_s32(acc2);
            int32_t s3 = vaddvq_s32(acc3);

            Y[(size_t)i * N + j + 0] = (m4t_mtfp_t)s0;
            Y[(size_t)i * N + j + 1] = (m4t_mtfp_t)s1;
            Y[(size_t)i * N + j + 2] = (m4t_mtfp_t)s2;
            Y[(size_t)i * N + j + 3] = (m4t_mtfp_t)s3;
        }

        /* N-tail: 1-3 trailing j cells. Single-acc NEON path covering
         * full 80-trit chunks plus boundary tile (no scalar tail). */
        for (int j = j_tile_end; j < N; j++) {
            const uint8_t* wj = W_packed + (size_t)j * Kp;
            int32x4_t acc = vdupq_n_s32(0);

            #define M4T_5IN8_JTAIL_SDOT(BUF) do {                          \
                uint8x16_t b = vld1q_u8(BUF);                              \
                uint16x8_t lo16 = vshrq_n_u16(                             \
                    vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 57), 9);         \
                uint16x8_t hi16 = vshrq_n_u16(                             \
                    vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 57), 9);        \
                uint8x16_t high = vcombine_u8(                             \
                    vmovn_u16(lo16), vmovn_u16(hi16));                     \
                uint8x16_t low  = vsubq_u8(b, vmulq_u8(high, nine_v));     \
                int8x16_t d0 = vqtbl1q_s8(lut_d0, low);                    \
                int8x16_t d1 = vqtbl1q_s8(lut_d1, low);                    \
                int8x16_t d2 = vqtbl2q_s8(lut_d2, high);                   \
                int8x16_t d3 = vqtbl2q_s8(lut_d3, high);                   \
                int8x16_t d4 = vqtbl2q_s8(lut_d4, high);                   \
                acc = vdotq_s32(acc, xv0, d0);                             \
                acc = vdotq_s32(acc, xv1, d1);                             \
                acc = vdotq_s32(acc, xv2, d2);                             \
                acc = vdotq_s32(acc, xv3, d3);                             \
                acc = vdotq_s32(acc, xv4, d4);                             \
            } while (0)

            for (int k = 0; k < K_aligned; k += 80) {
                int x_idx = k / 5;
                int8x16_t xv0 = vld1q_s8(X_d[0] + x_idx);
                int8x16_t xv1 = vld1q_s8(X_d[1] + x_idx);
                int8x16_t xv2 = vld1q_s8(X_d[2] + x_idx);
                int8x16_t xv3 = vld1q_s8(X_d[3] + x_idx);
                int8x16_t xv4 = vld1q_s8(X_d[4] + x_idx);
                M4T_5IN8_JTAIL_SDOT(wj + k / 5);
            }

            if (K_padded > K_aligned) {
                int k = K_aligned;
                int x_idx = k / 5;
                int8x16_t xv0 = vld1q_s8(X_d[0] + x_idx);
                int8x16_t xv1 = vld1q_s8(X_d[1] + x_idx);
                int8x16_t xv2 = vld1q_s8(X_d[2] + x_idx);
                int8x16_t xv3 = vld1q_s8(X_d[3] + x_idx);
                int8x16_t xv4 = vld1q_s8(X_d[4] + x_idx);

                int byte_off = k / 5;
                int avail = Kp - byte_off;
                assert(avail >= 1 && avail <= 16);
                uint8_t bb[16] = {0};
                memcpy(bb, wj + byte_off, (size_t)avail);
                M4T_5IN8_JTAIL_SDOT(bb);
            }

            #undef M4T_5IN8_JTAIL_SDOT

            Y[(size_t)i * N + j] = (m4t_mtfp_t)vaddvq_s32(acc);
        }
    }

    if (X_strided) free(X_strided);
#else
    /* Per project no-scalar-in-production rule (CONTRIBUTING.md +
     * feedback_function_over_speed_no_scalar memory): production
     * dispatchers are NEON-only. Calling scalar_ref from production
     * would violate the rule. Hard-fail at compile time instead. */
#error "m4t_ternary_5in8_matmul_bt requires NEON + ARM_FEATURE_DOTPROD; \
no scalar fallback per project rule. See CONTRIBUTING.md no-scalar audit."
#endif
}

/* §20 scalar reference oracle. Per-cell decoded via the spec formula
 * (u_i = (byte / 3^i) mod 3); never dispatches to NEON.
 * Test-only; production code MUST NOT call this. */
/* Sparse-routed reference oracle. See m4t_ternary_matmul.h for semantics.
 *
 * Walk packed bytes; for each byte, decode 5 trits; for trits != 0,
 * conditionally add ±X[i, k] to the accumulator. Zero trits skip the
 * X load entirely. Bit-exact vs the dense scalar_ref.
 *
 * **Test/measurement oracle only — production code MUST NOT call this.**
 *
 * On the existing 5-in-8 packed layout, "routing" cannot beat NEON SDOT
 * on this representation: in vectorized arithmetic multiply-by-zero is
 * free (one int8 lane in 16 contributes 0 to the dot — no cycle saved
 * by skipping it). To realize routing as a *speed* primitive requires
 * a different representation (e.g. per-output sparse index list with
 * NEON-friendly chunking). This oracle exists so that future NEON
 * sparse-routed primitive has a stable bit-exact gate, and so we can
 * measure actual zero-trit density in real weight matrices.
 *
 * Per project no-scalar-in-production rule + the
 * "math as signatures via routing" foundation. */
void m4t_ternary_5in8_matmul_bt_routed_ref(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const uint8_t* W_packed,
    int M, int K, int N,
    int64_t* skipped_zeros)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_packed)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_packed);

    int Kp = (K + 4) / 5;
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };

    int64_t local_skipped = 0;

    for (int i = 0; i < M; i++) {
        const m4t_trit_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_packed + (size_t)j * Kp;
            int32_t acc = 0;

            for (int b = 0; b < Kp; b++) {
                uint8_t byte = wj[b];
                int k_base = b * 5;

                /* Fast path: byte == 0 means all 5 trits are 0 (route absent). */
                if (byte == 0u) {
                    /* Count the in-range zero trits skipped via this byte. */
                    int n_in_range = (k_base + 5 <= K) ? 5 : (K - k_base);
                    if (n_in_range < 0) n_in_range = 0;
                    local_skipped += n_in_range;
                    continue;
                }

                /* Per-trit decode + route. */
                for (int d = 0; d < 5; d++) {
                    int k = k_base + d;
                    if (k >= K) break;
                    uint8_t u = (uint8_t)((byte / POW3[d]) % 3u);
                    if (u == 0u) {
                        /* trit = 0 → route absent; no X load, no add. */
                        local_skipped++;
                    } else if (u == 1u) {
                        /* trit = +1 → forward X[i, k] with positive sign. */
                        acc += (int32_t)xi[k];
                    } else {
                        /* trit = -1 (u == 2) → forward X[i, k] with negation. */
                        acc -= (int32_t)xi[k];
                    }
                }
            }

            Y[(size_t)i * N + j] = (m4t_mtfp_t)acc;
        }
    }

    if (skipped_zeros) *skipped_zeros = local_skipped;
}

void m4t_ternary_5in8_matmul_bt_scalar_ref(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X && W_packed)));
    assert((const void*)Y != (const void*)X);
    assert((const void*)Y != (const void*)W_packed);

    int Kp = (K + 4) / 5;
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };

    for (int i = 0; i < M; i++) {
        const m4t_trit_t* xi = X + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_packed + (size_t)j * Kp;
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int byte_idx  = k / 5;
                int digit_pos = k % 5;
                uint8_t u = (uint8_t)((wj[byte_idx] / POW3[digit_pos]) % 3u);
                int w = (u == 1u) ? 1 : (u == 2u) ? -1 : 0;
                acc += (int32_t)xi[k] * (int32_t)w;
            }
            Y[(size_t)i * N + j] = (m4t_mtfp_t)acc;
        }
    }
}

/* Per TD-7: §20 sibling with X also packed 5-in-8.
 * Implementation: per i, decode X_packed[i, :] into 5 stride-aligned int8
 * arrays via the same split-LUT pattern used for W. Then run the §20 tile
 * body verbatim. Same arbitrary-(K,N) support as §20 (TD-1). */
void m4t_ternary_5in8_matmul_xpacked_bt(
    m4t_mtfp_t* Y,
    const uint8_t* X_packed,
    const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X_packed && W_packed)));
    assert((const void*)Y != (const void*)X_packed);
    assert((const void*)Y != (const void*)W_packed);

#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    int Kp = (K + 4) / 5;
    int K5 = (K + 4) / 5;
    int k_tile_end = K - (K % 80);
    int j_tile_end = N - (N % 4);
    int kp_tile = Kp - (Kp % 16);    /* multiple of 16 for X-decode chunking */

    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };

    const uint8x16_t nine_v = vdupq_n_u8(9);
    const int8x16_t  lut_d0 = vld1q_s8(M4T_5IN8_LUT_LOW_D0);
    const int8x16_t  lut_d1 = vld1q_s8(M4T_5IN8_LUT_LOW_D1);
    const int8x16x2_t lut_d2 = { {
        vld1q_s8(M4T_5IN8_LUT_HIGH_D2 +  0),
        vld1q_s8(M4T_5IN8_LUT_HIGH_D2 + 16),
    } };
    const int8x16x2_t lut_d3 = { {
        vld1q_s8(M4T_5IN8_LUT_HIGH_D3 +  0),
        vld1q_s8(M4T_5IN8_LUT_HIGH_D3 + 16),
    } };
    const int8x16x2_t lut_d4 = { {
        vld1q_s8(M4T_5IN8_LUT_HIGH_D4 +  0),
        vld1q_s8(M4T_5IN8_LUT_HIGH_D4 + 16),
    } };

    int alloc_size = K5 * 5;
    int8_t* X_strided = (alloc_size > 0) ? (int8_t*)malloc((size_t)alloc_size) : NULL;
    if (alloc_size > 0 && !X_strided) return;
    int8_t* X_d[5];
    for (int d = 0; d < 5; d++) {
        X_d[d] = X_strided + (size_t)d * K5;
    }

    /* Scratch xi buffer for per-trit scalar K-tail (decoded once per row). */
    int8_t* xi_scratch = (K % 80 != 0)
        ? (int8_t*)malloc((size_t)(K % 80))
        : NULL;
    if ((K % 80 != 0) && !xi_scratch) {
        if (X_strided) free(X_strided);
        return;
    }

    for (int i = 0; i < M; i++) {
        const uint8_t* xi_p = X_packed + (size_t)i * Kp;

        /* Decode X_packed[i, :] into 5 stride-aligned arrays via split-LUT
         * (full 16-byte chunks NEON, trailing bytes scalar — geometric tail). */
        for (int b = 0; b < kp_tile; b += 16) {
            uint8x16_t bv = vld1q_u8(xi_p + b);
            uint16x8_t lo16 = vshrq_n_u16(
                vmulq_n_u16(vmovl_u8(vget_low_u8(bv)), 57), 9);
            uint16x8_t hi16 = vshrq_n_u16(
                vmulq_n_u16(vmovl_u8(vget_high_u8(bv)), 57), 9);
            uint8x16_t high = vcombine_u8(vmovn_u16(lo16), vmovn_u16(hi16));
            uint8x16_t low  = vsubq_u8(bv, vmulq_u8(high, nine_v));
            vst1q_s8(X_d[0] + b, vqtbl1q_s8(lut_d0, low));
            vst1q_s8(X_d[1] + b, vqtbl1q_s8(lut_d1, low));
            vst1q_s8(X_d[2] + b, vqtbl2q_s8(lut_d2, high));
            vst1q_s8(X_d[3] + b, vqtbl2q_s8(lut_d3, high));
            vst1q_s8(X_d[4] + b, vqtbl2q_s8(lut_d4, high));
        }
        for (int b = kp_tile; b < Kp; b++) {
            uint8_t byte = xi_p[b];
            for (int d = 0; d < 5; d++) {
                uint8_t u = (uint8_t)((byte / POW3[d]) % 3u);
                X_d[d][b] = (u == 1u) ? 1 : (u == 2u) ? -1 : 0;
            }
        }

        /* Materialize the K-tail xi values once (raw int8 trits at k=k_tile_end..K-1)
         * by decoding from X_packed. Used by both K-tail paths below. */
        if (xi_scratch) {
            for (int k = k_tile_end; k < K; k++) {
                int b = k / 5, dp = k % 5;
                uint8_t u = (uint8_t)((xi_p[b] / POW3[dp]) % 3u);
                xi_scratch[k - k_tile_end] = (u == 1u) ? 1 : (u == 2u) ? -1 : 0;
            }
        }

        /* Tile body: 4 j cells × full 80-trit chunks. (Identical to §20.) */
        for (int j = 0; j < j_tile_end; j += 4) {
            const uint8_t* wj0 = W_packed + (size_t)(j + 0) * Kp;
            const uint8_t* wj1 = W_packed + (size_t)(j + 1) * Kp;
            const uint8_t* wj2 = W_packed + (size_t)(j + 2) * Kp;
            const uint8_t* wj3 = W_packed + (size_t)(j + 3) * Kp;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            for (int k = 0; k < k_tile_end; k += 80) {
                int x_idx = k / 5;
                int8x16_t xv0 = vld1q_s8(X_d[0] + x_idx);
                int8x16_t xv1 = vld1q_s8(X_d[1] + x_idx);
                int8x16_t xv2 = vld1q_s8(X_d[2] + x_idx);
                int8x16_t xv3 = vld1q_s8(X_d[3] + x_idx);
                int8x16_t xv4 = vld1q_s8(X_d[4] + x_idx);

                #define M4T_5IN8_DECODE_AND_SDOT(WJ, ACC) do {                 \
                    uint8x16_t b = vld1q_u8((WJ) + k / 5);                     \
                    uint16x8_t lo16 = vshrq_n_u16(                             \
                        vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 57), 9);         \
                    uint16x8_t hi16 = vshrq_n_u16(                             \
                        vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 57), 9);        \
                    uint8x16_t high = vcombine_u8(                             \
                        vmovn_u16(lo16), vmovn_u16(hi16));                     \
                    uint8x16_t low  = vsubq_u8(b, vmulq_u8(high, nine_v));     \
                    int8x16_t d0 = vqtbl1q_s8(lut_d0, low);                    \
                    int8x16_t d1 = vqtbl1q_s8(lut_d1, low);                    \
                    int8x16_t d2 = vqtbl2q_s8(lut_d2, high);                   \
                    int8x16_t d3 = vqtbl2q_s8(lut_d3, high);                   \
                    int8x16_t d4 = vqtbl2q_s8(lut_d4, high);                   \
                    (ACC) = vdotq_s32((ACC), xv0, d0);                         \
                    (ACC) = vdotq_s32((ACC), xv1, d1);                         \
                    (ACC) = vdotq_s32((ACC), xv2, d2);                         \
                    (ACC) = vdotq_s32((ACC), xv3, d3);                         \
                    (ACC) = vdotq_s32((ACC), xv4, d4);                         \
                } while (0)

                M4T_5IN8_DECODE_AND_SDOT(wj0, acc0);
                M4T_5IN8_DECODE_AND_SDOT(wj1, acc1);
                M4T_5IN8_DECODE_AND_SDOT(wj2, acc2);
                M4T_5IN8_DECODE_AND_SDOT(wj3, acc3);

                #undef M4T_5IN8_DECODE_AND_SDOT
            }

            int32_t s0 = vaddvq_s32(acc0);
            int32_t s1 = vaddvq_s32(acc1);
            int32_t s2 = vaddvq_s32(acc2);
            int32_t s3 = vaddvq_s32(acc3);

            for (int k = k_tile_end; k < K; k++) {
                int byte_idx  = k / 5;
                int digit_pos = k % 5;
                int8_t x_k = xi_scratch[k - k_tile_end];
                #define M4T_5IN8_DECODE_TRIT(WJ) (                             \
                    (uint8_t)((WJ[byte_idx] / POW3[digit_pos]) % 3u) == 1u ?  1 : \
                    (uint8_t)((WJ[byte_idx] / POW3[digit_pos]) % 3u) == 2u ? -1 : \
                                                                              0)
                s0 += (int32_t)x_k * (int32_t)M4T_5IN8_DECODE_TRIT(wj0);
                s1 += (int32_t)x_k * (int32_t)M4T_5IN8_DECODE_TRIT(wj1);
                s2 += (int32_t)x_k * (int32_t)M4T_5IN8_DECODE_TRIT(wj2);
                s3 += (int32_t)x_k * (int32_t)M4T_5IN8_DECODE_TRIT(wj3);
                #undef M4T_5IN8_DECODE_TRIT
            }

            Y[(size_t)i * N + j + 0] = (m4t_mtfp_t)s0;
            Y[(size_t)i * N + j + 1] = (m4t_mtfp_t)s1;
            Y[(size_t)i * N + j + 2] = (m4t_mtfp_t)s2;
            Y[(size_t)i * N + j + 3] = (m4t_mtfp_t)s3;
        }

        for (int j = j_tile_end; j < N; j++) {
            const uint8_t* wj = W_packed + (size_t)j * Kp;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < k_tile_end; k += 80) {
                int x_idx = k / 5;
                int8x16_t xv0 = vld1q_s8(X_d[0] + x_idx);
                int8x16_t xv1 = vld1q_s8(X_d[1] + x_idx);
                int8x16_t xv2 = vld1q_s8(X_d[2] + x_idx);
                int8x16_t xv3 = vld1q_s8(X_d[3] + x_idx);
                int8x16_t xv4 = vld1q_s8(X_d[4] + x_idx);

                uint8x16_t b = vld1q_u8(wj + k / 5);
                uint16x8_t lo16 = vshrq_n_u16(
                    vmulq_n_u16(vmovl_u8(vget_low_u8(b)), 57), 9);
                uint16x8_t hi16 = vshrq_n_u16(
                    vmulq_n_u16(vmovl_u8(vget_high_u8(b)), 57), 9);
                uint8x16_t high = vcombine_u8(
                    vmovn_u16(lo16), vmovn_u16(hi16));
                uint8x16_t low  = vsubq_u8(b, vmulq_u8(high, nine_v));
                int8x16_t d0 = vqtbl1q_s8(lut_d0, low);
                int8x16_t d1 = vqtbl1q_s8(lut_d1, low);
                int8x16_t d2 = vqtbl2q_s8(lut_d2, high);
                int8x16_t d3 = vqtbl2q_s8(lut_d3, high);
                int8x16_t d4 = vqtbl2q_s8(lut_d4, high);
                acc = vdotq_s32(acc, xv0, d0);
                acc = vdotq_s32(acc, xv1, d1);
                acc = vdotq_s32(acc, xv2, d2);
                acc = vdotq_s32(acc, xv3, d3);
                acc = vdotq_s32(acc, xv4, d4);
            }

            int32_t s = vaddvq_s32(acc);
            for (int k = k_tile_end; k < K; k++) {
                int byte_idx  = k / 5;
                int digit_pos = k % 5;
                uint8_t u = (uint8_t)((wj[byte_idx] / POW3[digit_pos]) % 3u);
                int w = (u == 1u) ? 1 : (u == 2u) ? -1 : 0;
                s += (int32_t)xi_scratch[k - k_tile_end] * (int32_t)w;
            }

            Y[(size_t)i * N + j] = (m4t_mtfp_t)s;
        }
    }

    if (X_strided)  free(X_strided);
    if (xi_scratch) free(xi_scratch);
#else
#error "m4t_ternary_5in8_matmul_xpacked_bt requires NEON + ARM_FEATURE_DOTPROD; \
no scalar fallback per project rule. See CONTRIBUTING.md no-scalar audit."
#endif
}

/* Scalar reference for the X-packed variant. Test-only. */
void m4t_ternary_5in8_matmul_xpacked_bt_scalar_ref(
    m4t_mtfp_t* Y,
    const uint8_t* X_packed,
    const uint8_t* W_packed,
    int M, int K, int N)
{
    assert(M >= 0 && K >= 0 && N >= 0);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (M == 0 || N == 0) return;
    assert(Y && (K == 0 || (X_packed && W_packed)));
    assert((const void*)Y != (const void*)X_packed);
    assert((const void*)Y != (const void*)W_packed);

    int Kp = (K + 4) / 5;
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };

    for (int i = 0; i < M; i++) {
        const uint8_t* xi_p = X_packed + (size_t)i * Kp;
        for (int j = 0; j < N; j++) {
            const uint8_t* wj = W_packed + (size_t)j * Kp;
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int b_i = k / 5, d_p = k % 5;
                uint8_t ux = (uint8_t)((xi_p[b_i] / POW3[d_p]) % 3u);
                uint8_t uw = (uint8_t)((wj[b_i]   / POW3[d_p]) % 3u);
                int x = (ux == 1u) ? 1 : (ux == 2u) ? -1 : 0;
                int w = (uw == 1u) ? 1 : (uw == 2u) ? -1 : 0;
                acc += (int32_t)x * (int32_t)w;
            }
            Y[(size_t)i * N + j] = (m4t_mtfp_t)acc;
        }
    }
}
