/*
 * m4t_ternary_routed16.c — Sparse-routed ternary matmul (encoder + NEON kernel).
 *
 * See m4t_ternary_routed16.h for representation, contract, and design notes.
 */

#include "m4t_ternary_routed16.h"
#include "m4t_mtfp4.h"     /* M4T_SDOT_K_MAX_EXACT */
#include "m4t_internal.h"  /* M4T_HAS_NEON */
#include "m4t_trit_pack.h" /* M4T_TRIT_PACKED5_BYTES */

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if !M4T_HAS_NEON
#error "m4t_ternary_routed16 requires NEON; no scalar fallback per project rule."
#endif

#include <arm_neon.h>

/* ── Packed handle ───────────────────────────────────────────────────── */

struct m4t_routed16_packed {
    int K;
    int N;
    /* Per-column tile counts and pointers into a flat tile arena.
     * tiles[col_offset[j] .. col_offset[j+1]) is column j's tile list. */
    size_t                 total_tiles;
    int*                   col_offset;   /* [N+1] */
    m4t_routed16_tile_t*   tiles;        /* [total_tiles] */
};

int      m4t_routed16_packed_K(const m4t_routed16_packed_t* p) { return p ? p->K : 0; }
int      m4t_routed16_packed_N(const m4t_routed16_packed_t* p) { return p ? p->N : 0; }
size_t   m4t_routed16_packed_total_tiles(const m4t_routed16_packed_t* p) { return p ? p->total_tiles : 0; }
size_t   m4t_routed16_packed_bytes(const m4t_routed16_packed_t* p) {
    if (!p) return 0;
    return sizeof(*p)
         + (size_t)(p->N + 1) * sizeof(int)
         + p->total_tiles * sizeof(m4t_routed16_tile_t);
}

void m4t_ternary_routed16_packed_free(m4t_routed16_packed_t* p) {
    if (!p) return;
    free(p->col_offset);
    free(p->tiles);
    free(p);
}

/* ── Encoder ─────────────────────────────────────────────────────────── */

/* Decode one trit from 5-in-8 column-row stream.
 *   W_5in8[j * Kp + b] where b = k / 5, d = k % 5
 *   u = (byte / 3^d) % 3, then 0→0, 1→+1, 2→-1. */
static inline int8_t routed16_decode_trit(
    const uint8_t* W_5in8, int Kp, int j, int k)
{
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
    int b = k / 5;
    int d = k % 5;
    uint8_t byte = W_5in8[(size_t)j * (size_t)Kp + (size_t)b];
    uint8_t u = (uint8_t)((byte / POW3[d]) % 3u);
    if (u == 1u) return (int8_t) 1;
    if (u == 2u) return (int8_t)-1;
    return 0;
}

/* Pass 1: count tiles per column under the greedy window-32, lanes-16 packing.
 * Pass 2: emit tiles into the arena. Two passes lets us allocate exactly. */
m4t_routed16_packed_t* m4t_ternary_routed16_pack(
    const uint8_t* W_5in8, int K, int N)
{
    assert(K >= 0 && N >= 0);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (K > 0) assert(W_5in8 != NULL);

    m4t_routed16_packed_t* p =
        (m4t_routed16_packed_t*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->K = K;
    p->N = N;
    p->col_offset = (int*)calloc((size_t)N + 1, sizeof(int));
    if (!p->col_offset) { free(p); return NULL; }
    if (N == 0 || K == 0) {
        p->total_tiles = 0;
        p->tiles = NULL;
        return p;
    }

    int Kp = M4T_TRIT_PACKED5_BYTES(K);

    /* ── Pass 1: count tiles per column. ─────────────────────────────── */
    size_t total = 0;
    for (int j = 0; j < N; j++) {
        int n_tiles_col = 0;
        int k = 0;
        while (k < K) {
            /* Advance to next nonzero trit in this column. */
            while (k < K && routed16_decode_trit(W_5in8, Kp, j, k) == 0) k++;
            if (k >= K) break;
            /* This nonzero starts a tile. Take all nonzeros in
             * [k, k + WINDOW) up to LANES. */
            int start = k;
            int taken = 0;
            int kk = k;
            while (kk < K && kk < start + M4T_ROUTED16_WINDOW
                   && taken < M4T_ROUTED16_LANES) {
                if (routed16_decode_trit(W_5in8, Kp, j, kk) != 0) taken++;
                kk++;
            }
            assert(taken >= 1);
            n_tiles_col++;
            k = kk;  /* resume after the last consumed position */
        }
        p->col_offset[j + 1] = p->col_offset[j] + n_tiles_col;
        total += (size_t)n_tiles_col;
    }
    p->total_tiles = total;

    /* ── Pass 2: emit tiles. ──────────────────────────────────────────── */
    if (total > 0) {
        p->tiles = (m4t_routed16_tile_t*)calloc(total, sizeof(m4t_routed16_tile_t));
        if (!p->tiles) {
            free(p->col_offset);
            free(p);
            return NULL;
        }
    }

    for (int j = 0; j < N; j++) {
        int tile_idx = p->col_offset[j];
        int k = 0;
        while (k < K) {
            while (k < K && routed16_decode_trit(W_5in8, Kp, j, k) == 0) k++;
            if (k >= K) break;
            int start = k;
            m4t_routed16_tile_t* t = &p->tiles[tile_idx++];
            t->start_k = start;
            t->n_pos = 0;
            t->n_neg = 0;
            /* Pad sentinel: 0xFF ≥ 32, so vqtbl2q_s8 returns 0. */
            memset(t->idx_pos, 0xFF, sizeof(t->idx_pos));
            memset(t->idx_neg, 0xFF, sizeof(t->idx_neg));

            int kk = k;
            int taken = 0;
            while (kk < K && kk < start + M4T_ROUTED16_WINDOW
                   && taken < M4T_ROUTED16_LANES) {
                int8_t s = routed16_decode_trit(W_5in8, Kp, j, kk);
                if (s != 0) {
                    uint8_t rel = (uint8_t)(kk - start);  /* 0..31 */
                    if (s > 0) {
                        t->idx_pos[t->n_pos++] = rel;
                    } else {
                        t->idx_neg[t->n_neg++] = rel;
                    }
                    taken++;
                }
                kk++;
            }
            k = kk;
        }
        assert(tile_idx == p->col_offset[j + 1]);
    }

    return p;
}

/* ── Production NEON kernel ──────────────────────────────────────────── */

void m4t_ternary_routed16_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_routed16_packed_t* W,
    int M, int K, int N)
{
    assert(M >= 0);
    assert(W != NULL);
    assert(K == W->K);
    assert(N == W->N);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (M == 0 || N == 0) return;
    assert(Y != NULL);
    if (K > 0) assert(X != NULL);

    /* The kernel reads 32 X bytes per tile via two vld1q_s8 at start_k.
     * The encoder guarantees start_k is a position of a nonzero trit
     * (start_k ∈ [0, K)), but start_k + 32 may exceed K-1. We handle
     * tail safely by detecting near-end tiles and using a small
     * stack-buffered load.
     *
     * Loop order: i outer, j inner, t inner-most. Tile metadata is
     * walked once per row; for batched M>1, callers may prefer to
     * pre-pack X or hoist the tile loop — that optimization is future
     * work. The simple ordering keeps the kernel readable and exposes
     * no correctness pitfalls. */

    for (int i = 0; i < M; i++) {
        const m4t_trit_t* xi = X + (size_t)i * K;
        m4t_mtfp_t*       yi = Y + (size_t)i * N;

        for (int j = 0; j < N; j++) {
            int t_lo = W->col_offset[j];
            int t_hi = W->col_offset[j + 1];
            int32_t acc = 0;

            for (int t = t_lo; t < t_hi; t++) {
                const m4t_routed16_tile_t* tile = &W->tiles[t];
                int sk = tile->start_k;
                int avail = K - sk;  /* lanes valid in X starting at sk */

                int8x16_t xa, xb;
                if (avail >= 32) {
                    xa = vld1q_s8((const int8_t*)xi + sk);
                    xb = vld1q_s8((const int8_t*)xi + sk + 16);
                } else {
                    /* Tail: zero-pad into a 32-byte stack buffer. */
                    int8_t buf[32] = {0};
                    if (avail > 0) memcpy(buf, xi + sk, (size_t)avail);
                    xa = vld1q_s8(buf);
                    xb = vld1q_s8(buf + 16);
                }

                uint8x16x2_t xv;
                xv.val[0] = vreinterpretq_u8_s8(xa);
                xv.val[1] = vreinterpretq_u8_s8(xb);

                uint8x16_t idx_pos = vld1q_u8(tile->idx_pos);
                uint8x16_t idx_neg = vld1q_u8(tile->idx_neg);

                /* vqtbl2q returns 0 for out-of-range indices (≥32). */
                int8x16_t pos = vreinterpretq_s8_u8(vqtbl2q_u8(xv, idx_pos));
                int8x16_t neg = vreinterpretq_s8_u8(vqtbl2q_u8(xv, idx_neg));

                /* Reduce 16 int8 → int32. With |X[k]| ≤ 127 and 16 lanes,
                 * sum fits int16 (max 16*127 = 2032). vaddlvq_s8 widens
                 * to int16 and reduces; cast to int32. */
                acc += (int32_t)vaddlvq_s8(pos);
                acc -= (int32_t)vaddlvq_s8(neg);
            }

            yi[j] = (m4t_mtfp_t)acc;
        }
    }
}
