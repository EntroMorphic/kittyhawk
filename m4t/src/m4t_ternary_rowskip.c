/*
 * m4t_ternary_rowskip.c — Row-skip ternary matmul (encoder + kernel).
 *
 * See m4t_ternary_rowskip.h for the strategy and contract.
 */

#include "m4t_ternary_rowskip.h"
#include "m4t_ternary_matmul.h"  /* m4t_ternary_5in8_matmul_bt */
#include "m4t_trit_pack.h"
#include "m4t_internal.h"
#include "m4t_mtfp4.h"           /* M4T_SDOT_K_MAX_EXACT */

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if !M4T_HAS_NEON
#error "m4t_ternary_rowskip requires NEON; no scalar fallback per project rule."
#endif

/* ── Packed handle ───────────────────────────────────────────────────── */

struct m4t_ternary_rowskip_packed {
    int K;              /* original K */
    int K_compressed;   /* count of non-empty K-rows */
    int K_padded;       /* K_compressed rounded up to next multiple of 80
                         * — keeps the dense kernel's NEON tile body
                         * fully populated, avoiding the slow scalar
                         * tail for K%80 != 0. Trits at positions
                         * [K_compressed, K_padded) are zero in W_packed. */
    int N;
    int* nonempty_idx;  /* [K_compressed], values in [0, K) */
    uint8_t* W_packed;  /* [N, M4T_TRIT_PACKED5_BYTES(K_padded)] */
};

#define M4T_ROWSKIP_TILE 80  /* dense kernel's K-tile size */

int m4t_ternary_rowskip_packed_K(const m4t_ternary_rowskip_packed_t* p) {
    return p ? p->K : 0;
}
int m4t_ternary_rowskip_packed_K_compressed(const m4t_ternary_rowskip_packed_t* p) {
    return p ? p->K_compressed : 0;
}
int m4t_ternary_rowskip_packed_N(const m4t_ternary_rowskip_packed_t* p) {
    return p ? p->N : 0;
}
size_t m4t_ternary_rowskip_packed_bytes(const m4t_ternary_rowskip_packed_t* p) {
    if (!p) return 0;
    int Kp_pad = M4T_TRIT_PACKED5_BYTES(p->K_padded);
    return sizeof(*p)
         + (size_t)p->K_compressed * sizeof(int)
         + (size_t)p->N * (size_t)Kp_pad;
}

void m4t_ternary_rowskip_packed_free(m4t_ternary_rowskip_packed_t* p) {
    if (!p) return;
    free(p->nonempty_idx);
    free(p->W_packed);
    free(p);
}

/* ── Encoder ─────────────────────────────────────────────────────────── */

/* Decode one trit from 5-in-8 W^T layout. */
static inline int8_t rowskip_decode_trit(
    const uint8_t* W, int Kp, int j, int k)
{
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
    int b = k / 5;
    int d = k % 5;
    uint8_t byte = W[(size_t)j * (size_t)Kp + (size_t)b];
    uint8_t u = (uint8_t)((byte / POW3[d]) % 3u);
    if (u == 1u) return (int8_t) 1;
    if (u == 2u) return (int8_t)-1;
    return 0;
}

/* Encode trit u (-1, 0, +1) into 5-in-8 packed W byte at digit position d. */
static inline void rowskip_encode_trit(
    uint8_t* byte, int d, int8_t trit)
{
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
    uint8_t u;
    if      (trit ==  1) u = 1u;
    else if (trit == -1) u = 2u;
    else                 u = 0u;
    *byte = (uint8_t)(*byte + (uint8_t)(u * POW3[d]));
}

m4t_ternary_rowskip_packed_t* m4t_ternary_rowskip_pack(
    const uint8_t* W_5in8, int K, int N)
{
    assert(K >= 0 && N >= 0);
    assert(K <= M4T_SDOT_K_MAX_EXACT);
    if (K > 0) assert(W_5in8 != NULL);

    m4t_ternary_rowskip_packed_t* p =
        (m4t_ternary_rowskip_packed_t*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->K = K;
    p->N = N;
    if (K == 0 || N == 0) {
        p->K_compressed = 0;
        p->nonempty_idx = NULL;
        p->W_packed = NULL;
        return p;
    }

    int Kp = M4T_TRIT_PACKED5_BYTES(K);

    /* Pass 1: detect empty K-rows. row k is empty iff for every j in
     * [0, N), W[k, j] == 0. Build an "is_nonempty" bitmap, then list. */
    uint8_t* is_nonempty = (uint8_t*)calloc((size_t)K, 1);
    if (!is_nonempty) { free(p); return NULL; }

    for (int k = 0; k < K; k++) {
        int found = 0;
        for (int j = 0; j < N; j++) {
            if (rowskip_decode_trit(W_5in8, Kp, j, k) != 0) {
                found = 1;
                break;
            }
        }
        is_nonempty[k] = (uint8_t)found;
    }

    int K_c = 0;
    for (int k = 0; k < K; k++) if (is_nonempty[k]) K_c++;
    p->K_compressed = K_c;

    /* Pad K_compressed up to the next multiple of M4T_ROWSKIP_TILE (80),
     * the dense kernel's NEON tile size. This avoids the slow scalar
     * tail in m4t_ternary_5in8_matmul_bt when K%80 != 0. The padding
     * positions hold zero trits in W_packed, contributing 0 to the dot
     * product — bit-exactness preserved. */
    int K_pad = ((K_c + M4T_ROWSKIP_TILE - 1) / M4T_ROWSKIP_TILE) * M4T_ROWSKIP_TILE;
    /* Ensure K_pad fits in the SDOT exact-output bound. */
    assert(K_pad <= M4T_SDOT_K_MAX_EXACT);
    p->K_padded = K_pad;

    if (K_c == 0) {
        free(is_nonempty);
        p->nonempty_idx = NULL;
        p->W_packed = NULL;
        return p;
    }

    p->nonempty_idx = (int*)calloc((size_t)K_c, sizeof(int));
    if (!p->nonempty_idx) {
        free(is_nonempty); free(p); return NULL;
    }
    int idx = 0;
    for (int k = 0; k < K; k++) {
        if (is_nonempty[k]) p->nonempty_idx[idx++] = k;
    }
    assert(idx == K_c);
    free(is_nonempty);

    /* Pass 2: rebuild W_5in8 over the padded K. For each output column j,
     * walk nonempty_idx and copy W[k, j] into the new packed layout.
     * Positions [K_compressed, K_padded) are zero (calloc'd) — they
     * encode trit=0 in 5-in-8, contributing 0 to the dot product. */
    int Kp_pad = M4T_TRIT_PACKED5_BYTES(K_pad);
    p->W_packed = (uint8_t*)calloc((size_t)N * (size_t)Kp_pad, 1);
    if (!p->W_packed) {
        free(p->nonempty_idx); free(p); return NULL;
    }

    for (int j = 0; j < N; j++) {
        uint8_t* row = p->W_packed + (size_t)j * (size_t)Kp_pad;
        for (int i = 0; i < K_c; i++) {
            int k = p->nonempty_idx[i];
            int8_t trit = rowskip_decode_trit(W_5in8, Kp, j, k);
            int b = i / 5;
            int d = i % 5;
            rowskip_encode_trit(&row[b], d, trit);
        }
    }

    return p;
}

/* ── Production kernel ───────────────────────────────────────────────── */

void m4t_ternary_rowskip_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_ternary_rowskip_packed_t* W,
    int M, int K, int N)
{
    assert(M >= 0);
    assert(W != NULL);
    assert(K == W->K);
    assert(N == W->N);
    if (M == 0 || N == 0) return;
    assert(Y != NULL);
    if (K > 0) assert(X != NULL);

    int K_c = W->K_compressed;
    int K_pad = W->K_padded;

    /* If all K-rows are empty, Y is all zeros. */
    if (K_c == 0) {
        memset(Y, 0, (size_t)M * (size_t)N * sizeof(m4t_mtfp_t));
        return;
    }

    /* Allocate X scratch sized to K_padded (≥ K_compressed). The padded
     * positions [K_c, K_pad) are zero — they pair with zero W trits and
     * contribute 0 to the dot product, but keep the dense kernel on its
     * fast NEON tile path (avoiding the K%80 scalar tail). */
    m4t_trit_t* X_compressed = (m4t_trit_t*)calloc((size_t)K_pad, 1);
    if (!X_compressed) {
        memset(Y, 0, (size_t)M * (size_t)N * sizeof(m4t_mtfp_t));
        return;
    }

    for (int i = 0; i < M; i++) {
        const m4t_trit_t* xi = X + (size_t)i * K;
        m4t_mtfp_t* yi = Y + (size_t)i * N;
        for (int c = 0; c < K_c; c++) {
            X_compressed[c] = xi[W->nonempty_idx[c]];
        }
        /* Zero the padding range each iteration in case prior call wrote
         * anything there (defensive — calloc gave us zeros initially,
         * and the gather only writes [0, K_c)). */
        if (i > 0 && K_pad > K_c) {
            memset(X_compressed + K_c, 0, (size_t)(K_pad - K_c));
        }
        m4t_ternary_5in8_matmul_bt(yi, X_compressed, W->W_packed, 1, K_pad, N);
    }

    free(X_compressed);
}
