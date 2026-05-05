/*
 * m4t_route.c — ternary routing primitives
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 */

#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_mtfp.h"
#include "m4t_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Threshold extraction ─────────────────────────────────────────────── */

void m4t_route_threshold_extract(
    uint8_t* dst_packed, const int64_t* values, int64_t tau, int n)
{
    assert(tau >= 0);
    assert(n >= 0);

    int n_bytes = M4T_TRIT_PACKED_BYTES(n);
    memset(dst_packed, 0, (size_t)n_bytes);

    for (int i = 0; i < n; i++) {
        int64_t v = values[i];
        /* Strict comparison on the outer bounds so that tau == 0 degenerates
         * to sign-extraction exactly: v == 0 falls through to the zero case. */
        m4t_trit_t t = (v >  tau) ?  1 :
                       (v < -tau) ? -1 :
                                     0;
        uint8_t code = (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
        dst_packed[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

/* ── Dual-threshold extract ────────────────────────────────────────────── */

void m4t_route_threshold_extract_dual(
    uint8_t* dst_trit_packed,
    uint8_t* dst_conf_bits,
    const int64_t* values,
    int64_t tau_weak,
    int64_t tau_strong,
    int n)
{
    assert(dst_trit_packed && dst_conf_bits && values);
    /* H3 fix: writable outputs must not alias each other or the input. */
    assert((const void*)dst_trit_packed != (const void*)dst_conf_bits);
    assert((const void*)dst_trit_packed != (const void*)values);
    assert((const void*)dst_conf_bits   != (const void*)values);
    assert(tau_weak >= 0 && tau_strong >= tau_weak);
    assert(n >= 0);

    int trit_bytes = M4T_TRIT_PACKED_BYTES(n);
    int conf_bytes = (n + 7) / 8;
    memset(dst_trit_packed, 0, (size_t)trit_bytes);
    memset(dst_conf_bits,   0, (size_t)conf_bytes);

    for (int i = 0; i < n; i++) {
        int64_t v = values[i];
        m4t_trit_t t = (v > tau_weak) ?  1 :
                       (v < -tau_weak) ? -1 :
                                          0;
        uint8_t trit_code = (t == 1) ? 0x01u : (t == -1) ? 0x02u : 0x00u;
        dst_trit_packed[i >> 2] |= (uint8_t)(trit_code << ((i & 3) * 2));

        /* Confidence bit: |v| > tau_strong. */
        int64_t abs_v = (v >= 0) ? v : -v;
        if (abs_v > tau_strong) {
            dst_conf_bits[i >> 3] |= (uint8_t)(1u << (i & 7));
        }
    }
}

/* ── Confidence-weighted distance ──────────────────────────────────────── */

int32_t m4t_route_confidence_weighted_dist(
    const uint8_t* query_trit_packed,
    const uint8_t* query_conf_bits,
    const uint8_t* tile_trit_packed,
    const uint8_t* tile_conf_bits,
    const uint8_t* mask,
    int sig_dim)
{
    assert(query_trit_packed && query_conf_bits);
    assert(tile_trit_packed && tile_conf_bits && mask);
    assert(sig_dim >= 0);

    int packed_bytes = M4T_TRIT_PACKED_BYTES(sig_dim);

    /* Standard ternary Hamming as the baseline; weighted variant adds
     * confidence-conditioned extra cost on opposite-sign mismatches. */
    int32_t base_cost = m4t_popcount_dist(query_trit_packed, tile_trit_packed,
                                            mask, packed_bytes);

    /* Confidence weight: per-position scan with early-exit on non-opposite.
     * A branchless variant exists as `m4t_route_confidence_weighted_dist_branchless`
     * for benchmarking; see journal/tier2_perf_redteam.md for history. */
    int32_t bonus = 0;
    for (int i = 0; i < sig_dim; i++) {
        int byte_idx_t = i >> 2;
        int bit_off_t  = (i & 3) * 2;
        uint8_t q_code = (uint8_t)((query_trit_packed[byte_idx_t] >> bit_off_t) & 0x3u);
        uint8_t t_code = (uint8_t)((tile_trit_packed[byte_idx_t]  >> bit_off_t) & 0x3u);

        /* Opposite-sign: q=0b01, t=0b10 OR q=0b10, t=0b01. */
        int opposite = (q_code == 0x01u && t_code == 0x02u) ||
                       (q_code == 0x02u && t_code == 0x01u);
        if (!opposite) continue;

        /* Mask check: same per-byte 0xFF/tail logic as popcount_dist's
         * mask. The trit-field is at bits [bit_off_t, bit_off_t+2).
         * Active iff those bits are set in the mask. */
        uint8_t mask_byte = mask[byte_idx_t];
        uint8_t trit_field_mask = (uint8_t)(0x3u << bit_off_t);
        if ((mask_byte & trit_field_mask) != trit_field_mask) continue;

        /* Confidence bits. */
        int q_conf = (query_conf_bits[i >> 3] >> (i & 7)) & 1u;
        int t_conf = (tile_conf_bits[i >> 3]  >> (i & 7)) & 1u;
        int conf_count = q_conf + t_conf;
        bonus += conf_count;  /* 0, 1, or 2 extra cost beyond the base */
    }

    return base_cost + bonus;
}

/* ── Wildcard distance ─────────────────────────────────────────────────── */

int32_t m4t_route_wildcard_dist(
    const uint8_t* query_packed,
    const uint8_t* tile_packed,
    const uint8_t* mask,
    int sig_dim)
{
    assert(query_packed && tile_packed && mask);
    assert(sig_dim >= 0);

    int packed_bytes = M4T_TRIT_PACKED_BYTES(sig_dim);

    /* Standard ternary Hamming. Reuses the optimized kernel; this
     * accounts for all the cost-table cells except the wildcard
     * exception. */
    int32_t total = m4t_popcount_dist(query_packed, tile_packed,
                                        mask, packed_bytes);

    /* Wildcard correction: subtract 1 per position where
     *   tile-trit-field == 0b00 (zero, wildcard interpretation)
     *   AND query-trit-field is non-zero (0b01 or 0b10)
     *   AND mask field is active (mask byte's 2-bit field is 0b11)
     *
     * Per-byte bit math: each byte holds 4 trit fields at bit positions
     * (0..1), (2..3), (4..5), (6..7). The mask 0x55 = 0b01010101 selects
     * the LOW bit of each 2-bit field. We compute three byte-level masks
     * with one bit per field at the low-bit-of-field position:
     *
     *   tile_field_zero  = (~(b_t | (b_t >> 1))) & 0x55
     *     bit set iff field is 0b00 (both bits zero).
     *
     *   query_field_nonzero = (b_q | (b_q >> 1)) & 0x55
     *     bit set iff field is 0b01 or 0b10 (at least one bit set).
     *
     *   mask_field_active = (mask) & 0x55
     *     bit set iff mask field is 0b11 (caller guarantees mask
     *     bytes are 0xFF for active fields, 0x00 for inactive — see
     *     gesh_forward.c mask construction). The 0x55 isolates the
     *     low bit of each field, which is 1 iff the field is active.
     *
     * Their AND gives a byte where each set bit represents one
     * position to subtract from the standard Hamming total.
     *
     * No NEON path on this corrective pass — the per-byte work is
     * a few ALU ops + popcount; for typical sig_dim ≤ 1024 (Dp ≤ 256)
     * the loop is bounded by a few hundred byte operations. If
     * profiling shows this is a bottleneck (Gate 2: runtime ≤ 1.2×
     * current Hamming), a NEON variant follows (vand/vshr/vbic/vcnt
     * pattern). */
    int32_t correction = 0;
    int i = 0;

    /* 8-byte chunks via __builtin_popcountll for high-throughput
     * scalar. Same shape as popcount_dist's 8-byte path. */
    for (; i + 8 <= packed_bytes; i += 8) {
        uint64_t bt, bq, mb;
        memcpy(&bt, tile_packed  + i, 8);
        memcpy(&bq, query_packed + i, 8);
        memcpy(&mb, mask         + i, 8);
        uint64_t tile_zero       = (~(bt | (bt >> 1))) & 0x5555555555555555ULL;
        uint64_t query_nonzero   = (bq | (bq >> 1))     & 0x5555555555555555ULL;
        uint64_t mask_active     = mb                   & 0x5555555555555555ULL;
        uint64_t correction_mask = tile_zero & query_nonzero & mask_active;
        correction += (int32_t)__builtin_popcountll(correction_mask);
    }
    /* 4-byte chunks for the N_PROJ=16 fast path. */
    for (; i + 4 <= packed_bytes; i += 4) {
        uint32_t bt, bq, mb;
        memcpy(&bt, tile_packed  + i, 4);
        memcpy(&bq, query_packed + i, 4);
        memcpy(&mb, mask         + i, 4);
        uint32_t tile_zero       = (~(bt | (bt >> 1))) & 0x55555555u;
        uint32_t query_nonzero   = (bq | (bq >> 1))     & 0x55555555u;
        uint32_t mask_active     = mb                   & 0x55555555u;
        uint32_t correction_mask = tile_zero & query_nonzero & mask_active;
        correction += (int32_t)__builtin_popcount(correction_mask);
    }
    /* Byte tail. */
    for (; i < packed_bytes; i++) {
        uint8_t bt = tile_packed[i];
        uint8_t bq = query_packed[i];
        uint8_t mb = mask[i];
        uint8_t tile_zero       = (uint8_t)((~(bt | (bt >> 1))) & 0x55u);
        uint8_t query_nonzero   = (uint8_t)((bq | (bq >> 1))     & 0x55u);
        uint8_t mask_active     = (uint8_t)(mb                   & 0x55u);
        uint8_t correction_mask = (uint8_t)(tile_zero & query_nonzero & mask_active);
        correction += (int32_t)__builtin_popcount((unsigned)correction_mask);
    }

    return total - correction;
}

/* ── Wildcard-select mask extraction (compositional routing) ───────────── */

void m4t_route_wildcard_select_mask(
    uint8_t* mask,
    const uint8_t* selector_packed,
    int sig_dim)
{
    assert(mask && selector_packed);
    assert(sig_dim >= 0);

    int packed_bytes = M4T_TRIT_PACKED_BYTES(sig_dim);
    int full_bytes = sig_dim / 4;
    int tail_trits = sig_dim - full_bytes * 4;

    /* For each byte: low_bit_of_field is 1 iff field is 0b00 (wildcard).
     * Expanding low | (low << 1) fills the 2-bit field with 0b11 (active)
     * iff the selector field was a wildcard. Other fields → 0b00 (inactive).
     */
    for (int i = 0; i < full_bytes; i++) {
        uint8_t s = selector_packed[i];
        uint8_t low = (uint8_t)((~(s | (s >> 1))) & 0x55u);
        mask[i] = (uint8_t)(low | (low << 1));
    }

    if (full_bytes < packed_bytes) {
        /* Tail byte: only the first `tail_trits` fields are valid. The
         * remaining fields in the packed byte must be zero (selector
         * itself zeros the tail by construction). Apply the same
         * extraction, then mask off the dead high fields. */
        uint8_t s = selector_packed[full_bytes];
        uint8_t low = (uint8_t)((~(s | (s >> 1))) & 0x55u);
        uint8_t out = (uint8_t)(low | (low << 1));
        uint8_t valid_field_mask = (uint8_t)((1u << (tail_trits * 2)) - 1u);
        mask[full_bytes] = (uint8_t)(out & valid_field_mask);
    }
}

/* ── Pairwise Hamming sum ──────────────────────────────────────────────── */

int32_t m4t_route_pairwise_hamming_sum(
    const uint8_t* tile_sigs_packed,
    const uint8_t* mask,
    int T,
    int sig_dim)
{
    assert(T >= 0 && sig_dim >= 0);
    if (T < 2 || sig_dim == 0) return 0;
    assert(tile_sigs_packed && mask);

    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int32_t sum = 0;
    for (int i = 0; i < T; i++) {
        for (int j = i + 1; j < T; j++) {
            sum += m4t_popcount_dist(
                tile_sigs_packed + (size_t)i * Dp,
                tile_sigs_packed + (size_t)j * Dp,
                mask, Dp);
        }
    }
    return sum;
}

/* ── Distance batch ────────────────────────────────────────────────────── */

void m4t_route_distance_batch(
    int32_t* distances,
    const uint8_t* query_packed,
    const uint8_t* tile_sigs_packed,
    const uint8_t* mask,
    int T, int sig_dim)
{
    assert(distances && query_packed && tile_sigs_packed && mask);
    assert(T >= 0 && sig_dim >= 0);
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);

    for (int t = 0; t < T; t++) {
        distances[t] = m4t_popcount_dist(
            query_packed,
            tile_sigs_packed + (size_t)t * Dp,
            mask, Dp);
    }
}

/* ── Top-k selection ───────────────────────────────────────────────────── */

void m4t_route_topk_abs(
    m4t_route_decision_t* decisions,
    const int32_t* scores,
    int T, int k)
{
    assert(decisions && scores);
    assert(k >= 0 && T >= 0);

    /* Track selected tiles via bitmask. Supports T ≤ M4T_ROUTE_MAX_T
     * explicitly in the type — no stack array, no overflow possible. */
    assert(T <= M4T_ROUTE_MAX_T);
    uint64_t used = 0;

    for (int sel = 0; sel < k; sel++) {
        int best_idx = -1;
        int64_t best_abs = -1;

        for (int t = 0; t < T; t++) {
            if (used & ((uint64_t)1 << t)) continue;
            /* Widen to int64 before negating to avoid UB on INT32_MIN. */
            int64_t a = (int64_t)scores[t];
            int64_t abs_a = (a >= 0) ? a : -a;
            if (abs_a > best_abs) {
                best_abs = abs_a;
                best_idx = t;
            }
        }

        if (best_idx < 0 || best_abs == 0) {
            /* No more nonzero tiles. Fill remaining with sentinel. */
            for (int r = sel; r < k; r++) {
                decisions[r].tile_idx = -1;
                decisions[r].sign = 0;
            }
            break;
        }

        used |= (uint64_t)1 << best_idx;
        decisions[sel].tile_idx = best_idx;
        decisions[sel].sign = (scores[best_idx] > 0) ? 1 : -1;
    }
}

/* ── Signed accumulation ───────────────────────────────────────────────── */

void m4t_route_apply_signed(
    m4t_mtfp_t* result,
    const m4t_mtfp_t* tile_outs,
    const m4t_route_decision_t* decisions,
    int k, int dim)
{
    assert(result && tile_outs && decisions);
    assert(k >= 0 && dim >= 0);

    for (int sel = 0; sel < k; sel++) {
        int idx = decisions[sel].tile_idx;
        m4t_trit_t sign = decisions[sel].sign;
        /* §18 contract — three-state input. tile_idx == -1 is the sentinel
         * (no tile selected); sign ∈ {-1, 0, +1}. The upper-bound on
         * tile_idx is implicit in the caller's tile_outs buffer layout
         * and cannot be checked without an extra parameter. */
        assert(idx >= -1);
        assert(sign == -1 || sign == 0 || sign == 1);
        if (idx < 0) continue;

        const m4t_mtfp_t* tile = tile_outs + (size_t)idx * dim;

        if (sign == 1) {
            m4t_mtfp_vec_add_inplace(result, tile, dim);
        } else if (sign == -1) {
            m4t_mtfp_vec_sub_inplace(result, tile, dim);
        }
    }
}

/* ── Signature update ──────────────────────────────────────────────────── */

void m4t_route_signature_update(
    uint8_t* signatures,
    const uint8_t* weights,
    int64_t* scratch,
    int T, int H, int D)
{
    assert(signatures && weights && scratch);
    assert(T > 0 && H > 0 && D > 0);

    int Dp = M4T_TRIT_PACKED_BYTES(D);

    /* scratch layout: [0 .. T*D-1] = column sums, [T*D .. T*D+D-1] = means */
    int64_t* col_sums = scratch;           /* [T, D] */
    int64_t* means    = scratch + T * D;   /* [D]    */

    /* Phase 1: column sums. For each tile, sum each column of the [H, D]
     * weight matrix. Unpack one row at a time to avoid large temp buffers.
     *
     * This is a compound setup-time function, not a hot-path opcode — the
     * row unpack buffer is heap-allocated so that D is unbounded from the
     * substrate's side. The caller owns whatever upper bound it imposes. */
    memset(col_sums, 0, (size_t)T * D * sizeof(int64_t));

    m4t_trit_t* row_buf = (m4t_trit_t*)malloc((size_t)D * sizeof(m4t_trit_t));
    if (!row_buf) {
        fprintf(stderr, "m4t_route_signature_update: malloc failed (D=%d)\n", D);
        return;
    }

    for (int t = 0; t < T; t++) {
        int64_t* cs = col_sums + (size_t)t * D;
        for (int h = 0; h < H; h++) {
            const uint8_t* row_packed = weights + ((size_t)t * H + h) * Dp;
            m4t_unpack_trits_1d(row_buf, row_packed, D);
            for (int d = 0; d < D; d++) {
                cs[d] += (int64_t)row_buf[d];
            }
        }
    }
    free(row_buf);

    /* Phase 2: mean across tiles per dimension. */
    memset(means, 0, (size_t)D * sizeof(int64_t));
    for (int t = 0; t < T; t++) {
        const int64_t* cs = col_sums + (size_t)t * D;
        for (int d = 0; d < D; d++) {
            means[d] += cs[d];
        }
    }
    for (int d = 0; d < D; d++) {
        means[d] /= T;
    }

    /* Phase 3: sign(col_sum - mean) → packed-trit signature per tile.
     * Reuse the col_sums buffer for the difference values.
     *
     * threshold_extract with tau=0 is the correct shape here: the inputs
     * are integer col_sum − mean differences, where exact zero occurs with
     * non-trivial probability (any tile whose column-sum lands on the
     * cross-tile mean produces a zero trit in that dim). This deployment
     * passes emission coverage — see M4T_SUBSTRATE §18. */
    for (int t = 0; t < T; t++) {
        int64_t* cs = col_sums + (size_t)t * D;
        for (int d = 0; d < D; d++) {
            cs[d] -= means[d];
        }
        m4t_route_threshold_extract(
            signatures + (size_t)t * Dp, cs, 0, D);
    }
}

/* ── select: trit-controlled cell-level mux (elemental floor primitive) ─── */

void m4t_route_select(
    m4t_mtfp_t* out,
    const uint8_t* c_packed,
    const m4t_mtfp_t* a,
    const m4t_mtfp_t* b,
    const m4t_mtfp_t* d,
    int n_cells)
{
    assert(out && c_packed && a && b && d);
    assert(n_cells > 0);

    int i = 0;

#if M4T_HAS_NEON
    /* Process 4 cells (one MTFP19 block, one packed-trit nibble) per iter.
     * For each block:
     *   1. Extract the 4 trit codes from one byte of c_packed.
     *   2. Build mask vectors: a_mask = (code == 0b01) ? all-1s : 0,
     *      b_mask = (code == 0b10) ? all-1s : 0.
     *   3. d_mask = NOT (a_mask | b_mask) — covers both 0b00 and reserved 0b11.
     *   4. out = (a & a_mask) | (b & b_mask) | (d & d_mask).
     *
     * Mask construction per cell uses vceqq_s32 against the constants {1, 2}
     * applied to the per-cell decoded code. ~12 NEON ops per 4 cells. */
    const int32x4_t k_one = vdupq_n_s32(1);
    const int32x4_t k_two = vdupq_n_s32(2);

    for (; i + 4 <= n_cells; i += 4) {
        /* Load packed trit byte covering positions i..i+3. */
        uint8_t byte = c_packed[i >> 2];
        /* Decode the 4 codes into a 4-lane int32 vector.
         * Code at field f = (byte >> (f*2)) & 0x3 */
        int32_t codes[4] = {
            (int32_t)((byte >> 0) & 0x3u),
            (int32_t)((byte >> 2) & 0x3u),
            (int32_t)((byte >> 4) & 0x3u),
            (int32_t)((byte >> 6) & 0x3u)
        };
        int32x4_t code_vec = vld1q_s32(codes);

        /* Mask = all-1s in lanes where code matches. */
        uint32x4_t a_mask = vceqq_s32(code_vec, k_one);   /* +1 trit */
        uint32x4_t b_mask = vceqq_s32(code_vec, k_two);   /* -1 trit */
        /* d_mask = NOT (a_mask OR b_mask) — covers code 0b00 (zero) and
         * reserved 0b11 per the documented "treat as 0" fallback. */
        uint32x4_t ab_mask = vorrq_u32(a_mask, b_mask);
        uint32x4_t d_mask  = vmvnq_u32(ab_mask);

        /* Load the three input cell vectors. */
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        int32x4_t vd = vld1q_s32(d + i);

        /* Bit-select per lane: out = (a & a_mask) | (b & b_mask) | (d & d_mask). */
        int32x4_t result = vorrq_s32(
            vandq_s32(va, vreinterpretq_s32_u32(a_mask)),
            vorrq_s32(
                vandq_s32(vb, vreinterpretq_s32_u32(b_mask)),
                vandq_s32(vd, vreinterpretq_s32_u32(d_mask))));
        vst1q_s32(out + i, result);
    }
#endif

    /* Scalar tail for n_cells not multiple of 4 (geometric, not a
     * fallback — NEON loop above handles whole 4-cell blocks). */
    for (; i < n_cells; i++) {
        uint8_t code = (uint8_t)((c_packed[i >> 2] >> ((i & 3) * 2)) & 0x3u);
        if (code == 0x01u)      out[i] = a[i];   /* trit = +1 */
        else if (code == 0x02u) out[i] = b[i];   /* trit = -1 */
        else                    out[i] = d[i];   /* trit = 0 (or reserved 0b11) */
    }
}

/* Scalar reference for fair-comparison benchmarking. */
void m4t_route_select_scalar_ref(
    m4t_mtfp_t* out,
    const uint8_t* c_packed,
    const m4t_mtfp_t* a,
    const m4t_mtfp_t* b,
    const m4t_mtfp_t* d,
    int n_cells)
{
    assert(out && c_packed && a && b && d);
    assert(n_cells > 0);
    for (int i = 0; i < n_cells; i++) {
        uint8_t code = (uint8_t)((c_packed[i >> 2] >> ((i & 3) * 2)) & 0x3u);
        if (code == 0x01u)      out[i] = a[i];
        else if (code == 0x02u) out[i] = b[i];
        else                    out[i] = d[i];
    }
}

/* Branchless-per-byte reference for fair-comparison benchmarking against
 * the production branchy m4t_route_confidence_weighted_dist. */
int32_t m4t_route_confidence_weighted_dist_branchless(
    const uint8_t* query_trit_packed,
    const uint8_t* query_conf_bits,
    const uint8_t* tile_trit_packed,
    const uint8_t* tile_conf_bits,
    const uint8_t* mask,
    int sig_dim)
{
    assert(query_trit_packed && query_conf_bits);
    assert(tile_trit_packed && tile_conf_bits && mask);
    assert(sig_dim >= 0);

    int packed_bytes = M4T_TRIT_PACKED_BYTES(sig_dim);
    int32_t base_cost = m4t_popcount_dist(query_trit_packed, tile_trit_packed,
                                            mask, packed_bytes);

    int32_t bonus = 0;
    for (int byte_idx = 0; byte_idx < packed_bytes; byte_idx++) {
        uint8_t q_byte = query_trit_packed[byte_idx];
        uint8_t t_byte = tile_trit_packed[byte_idx];
        uint8_t m_byte = mask[byte_idx];

        uint8_t xor_byte     = (uint8_t)(q_byte ^ t_byte);
        uint8_t opposite_low = (uint8_t)((xor_byte & 0x55u) & ((xor_byte & 0xAAu) >> 1));
        uint8_t tracked_low  = (uint8_t)((m_byte & 0x55u) & ((m_byte & 0xAAu) >> 1));
        uint8_t indicator    = (uint8_t)(opposite_low & tracked_low);
        if (indicator == 0) continue;

        uint8_t indicator_nibble = (uint8_t)(
              (indicator       & 0x1u)
            | ((indicator >> 1) & 0x2u)
            | ((indicator >> 2) & 0x4u)
            | ((indicator >> 3) & 0x8u));

        int conf_byte_idx = byte_idx >> 1;
        int nibble_shift  = (byte_idx & 1) * 4;
        uint8_t q_conf_nibble = (uint8_t)((query_conf_bits[conf_byte_idx] >> nibble_shift) & 0x0Fu);
        uint8_t t_conf_nibble = (uint8_t)((tile_conf_bits[conf_byte_idx]  >> nibble_shift) & 0x0Fu);

        bonus += __builtin_popcount((unsigned)(indicator_nibble & q_conf_nibble));
        bonus += __builtin_popcount((unsigned)(indicator_nibble & t_conf_nibble));
    }
    return base_cost + bonus;
}

/* ── Emission-coverage helper ─────────────────────────────────────────── */

void m4t_route_decisions_emit_coverage(
    const m4t_route_decision_t* decisions,
    int k,
    int* out_has_pos,
    int* out_has_neg,
    int* out_has_zero)
{
    assert(k >= 0);
    assert(k == 0 || decisions != NULL);

    int has_pos = 0, has_neg = 0, has_zero = 0;
    for (int i = 0; i < k; i++) {
        m4t_trit_t s = decisions[i].sign;
        if      (s ==  1) has_pos  = 1;
        else if (s == -1) has_neg  = 1;
        else              has_zero = 1;
    }
    if (out_has_pos)  *out_has_pos  = has_pos;
    if (out_has_neg)  *out_has_neg  = has_neg;
    if (out_has_zero) *out_has_zero = has_zero;
}
