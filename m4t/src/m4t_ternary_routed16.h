/*
 * m4t_ternary_routed16.h — Sparse-routed ternary matmul (NEON, M=1).
 *
 * The first production routing primitive that exploits weight sparsity
 * for compute savings. Builds on but does not subsume the dense 5-in-8
 * NEON-SDOT path (m4t_ternary_5in8_matmul_bt).
 *
 * ── What this is ─────────────────────────────────────────────────────
 *
 *   Y[i, j] = sum_k X[i, k] * W[k, j]    where W ∈ {-1, 0, +1}^{K,N}
 *
 * Bit-exact to m4t_ternary_5in8_matmul_bt. The math is the same; the
 * data path is different: instead of unpacking every trit in a dense
 * SDOT tile, this kernel walks a per-output-column tile list whose
 * tiles each describe up to 16 nonzero positions within a 32-trit
 * window of K. The kernel never touches X cells where W is zero.
 *
 * ── When this primitive wins ─────────────────────────────────────────
 *
 * Empirical (see test_m4t_ternary_routed16_bench): the crossover with
 * m4t_ternary_5in8_matmul_bt depends on weight sparsity. At BitNet's
 * ~38–50% zero density the per-tile overhead may dominate; at higher
 * sparsity (and especially at structured zero clusters) routing wins.
 *
 * Production callers should select the kernel based on measured
 * sparsity for the target weight tensor. This header exposes both the
 * encoder and the kernel so that callers can pre-pack offline.
 *
 * ── Representation ───────────────────────────────────────────────────
 *
 * For each output column j ∈ [N], a list of fixed-shape tiles. Each
 * tile encodes up to LANES=16 nonzero trits within a contiguous window
 * of WINDOW=32 K-positions. The window enables NEON vqtbl2q_s8 gather
 * over two sequential vld1q_s8 of X — no random gather, no cache miss
 * beyond what the dense path already pays.
 *
 *   typedef struct {
 *       int32_t start_k;        // window start in K-positions; window is [start_k, start_k+32)
 *       uint8_t n_pos;          // active +1 lanes (0..16)
 *       uint8_t n_neg;          // active -1 lanes (0..16)
 *       uint8_t idx_pos[16];    // window-relative positions (0..31); padded with 32+ → vqtbl2 returns 0
 *       uint8_t idx_neg[16];
 *   } m4t_routed16_tile_t;
 *
 * Padding convention: lanes ≥ n_pos in idx_pos are filled with 0xFF
 * (≥32, so vqtbl2q_s8 returns 0 for that lane). Same for idx_neg. This
 * means the inner kernel has no per-tile branch on count — it always
 * gathers 16 lanes and the unused ones contribute 0 to the sum.
 *
 * ── Encoder ──────────────────────────────────────────────────────────
 *
 * Greedy per column: walk the column's nonzero positions; for each
 * tile, take the next nonzero as the window start, then take all
 * subsequent nonzeros that fit within [start, start+32) up to LANES=16.
 * Emit the tile. Repeat until column exhausted.
 *
 * Tiles are emitted in increasing start_k order. Empty columns produce
 * zero tiles (Y[i, j] = 0).
 *
 * ── Storage ──────────────────────────────────────────────────────────
 *
 * 40 bytes per tile (or 32 bytes if we bit-pack signs; not done here
 * for kernel simplicity). Per column with 40% nonzero density:
 *   ~K * 0.6 nonzeros, ~K * 0.6 / 16 = K/27 tiles, ~K * 1.5 bytes.
 * For BitNet q_proj (K=N=2560, ~50% nnz): ~2560 * 1.3 KB * 2560 ≈ 8.5 MB
 * vs 5-in-8 packed at 1.3 MB. ~6.5× expansion. This is the cost of
 * the routing primitive being a different storage layout.
 *
 * ── Constraints ──────────────────────────────────────────────────────
 *
 * Initial primitive supports M=1 only (single-token inference). M>1
 * batched routing is future work — the natural extension is to keep
 * the column-organized tile list and iterate over M inside the j loop.
 *
 * Per project rule (no scalar in production): NEON-only. Compile-time
 * #error if NEON unavailable. Test oracle is m4t_ternary_5in8_matmul_bt_routed_ref.
 */

#ifndef M4T_TERNARY_ROUTED16_H
#define M4T_TERNARY_ROUTED16_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define M4T_ROUTED16_LANES   16
#define M4T_ROUTED16_WINDOW  32

typedef struct {
    int32_t start_k;
    uint8_t n_pos;
    uint8_t n_neg;
    uint8_t _pad[2];
    uint8_t idx_pos[M4T_ROUTED16_LANES];
    uint8_t idx_neg[M4T_ROUTED16_LANES];
} m4t_routed16_tile_t;

/* Opaque packed handle. Owned by the encoder; freed via _free. */
typedef struct m4t_routed16_packed m4t_routed16_packed_t;

/* Encoder. W_5in8 is a 5-trits-in-8-bits packed weight matrix in
 * "W^T" layout (row-major [N, M4T_TRIT_PACKED5_BYTES(K)]) — the same
 * layout consumed by m4t_ternary_5in8_matmul_bt.
 *
 * Returns NULL on allocation failure. Caller must free with
 * m4t_routed16_packed_free.
 *
 * Preconditions:
 *   K >= 0, N >= 0
 *   K <= M4T_SDOT_K_MAX_EXACT
 *   W_5in8 non-NULL when K * N > 0
 */
m4t_routed16_packed_t* m4t_ternary_routed16_pack(
    const uint8_t* W_5in8, int K, int N);

void m4t_ternary_routed16_packed_free(m4t_routed16_packed_t* p);

/* Introspection (for tests + sparsity reporting). */
int      m4t_routed16_packed_K(const m4t_routed16_packed_t* p);
int      m4t_routed16_packed_N(const m4t_routed16_packed_t* p);
size_t   m4t_routed16_packed_total_tiles(const m4t_routed16_packed_t* p);
size_t   m4t_routed16_packed_bytes(const m4t_routed16_packed_t* p);

/* Production NEON kernel.
 *   Y[1, N] = X[1, K] @ W^T[K, N]
 *
 * X: int8 ternary or A8-quantized activations (m4t_trit_t = int8).
 *    The accumulator is int32 with no clamp; |sum| <= K when X is ternary,
 *    or <= 127 * K when X is A8 — caller is responsible for K bound.
 * W: routed16-packed weights produced by m4t_ternary_routed16_pack.
 * Y: int32 outputs (m4t_mtfp_t).
 *
 * Preconditions:
 *   M == 1 (initial primitive constraint; asserted in debug)
 *   K matches W's K (asserted in debug)
 *   N matches W's N (asserted in debug)
 *   K <= M4T_SDOT_K_MAX_EXACT
 *   K + 32 fits in valid X-load range — encoder guarantees window
 *     starts ∈ [0, K), but the load reads start_k+32 bytes; caller
 *     must ensure X allocation has 32 bytes of zero-padded tail or
 *     K is at least M4T_ROUTED16_WINDOW. (See impl.)
 */
void m4t_ternary_routed16_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_routed16_packed_t* W,
    int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TERNARY_ROUTED16_H */
