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
 * implementation. Used by m4t_mtfp_ternary_matmul_bt_scalar_ref (test
 * oracle) AND as the tail / non-NEON fallback inside ternary_dot. */
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

    for (int i = 0; i < M; i++) {
        const m4t_mtfp_t* X_row = X + (size_t)i * K;
        m4t_mtfp_t*       Y_row = Y + (size_t)i * N;

        for (int j = 0; j < N; j++) {
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
