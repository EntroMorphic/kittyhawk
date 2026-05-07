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
 * Empirical crossover vs m4t_ternary_5in8_matmul_bt (the only valid
 * dense baseline for A8 / int8 activations — xpacked requires ternary
 * X and is NOT a substitute for A8-input BitLinears):
 *
 *   shape               crossover  | win at 99% sparsity
 *   ─────────────────────────────────────────────────────
 *   K=N=2560 (q/o_proj)   96-97%   |   2.2x
 *   K=2560 N=6912 (gate)  ~97%     |   2.1x
 *   K=6912 N=2560 (down)  92-94%   |   3.0x
 *
 * The K=6912 case crosses sooner because each tile's 32-trit window
 * covers a larger fraction of the dense path's per-output work, so the
 * skip benefit accrues faster as sparsity rises.
 *
 * BitNet's measured weight sparsity (38-50%) is well below all three
 * crossovers. routed16 is the right kernel only when sparsity exceeds
 * the relevant crossover for the target shape.
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
 * 40 bytes per tile (4 start_k + 1 n_pos + 1 n_neg + 2 pad + 16 idx_pos
 * + 16 idx_neg). Could be tightened to ~24 bytes by bit-packing signs,
 * not done here for kernel simplicity.
 *
 * Real BitNet layer-0 measurements (this code path):
 *   q_proj   K=N=2560     49.6% sparsity   9.41 MB  vs 1.31 MB 5-in-8 (7.2×)
 *   gate     K=2560 N=6912 39.1% sparsity 27.16 MB vs 3.54 MB 5-in-8 (7.7×)
 *   down     K=6912 N=2560 38.2% sparsity 27.54 MB vs 3.54 MB 5-in-8 (7.8×)
 * Range: 7-8× expansion vs 5-in-8 packed. This is the cost of the
 * routing primitive being a different storage layout.
 *
 * ── Per-tile NEON cost ───────────────────────────────────────────────
 *
 * Per non-tail tile: 2× vld1q_s8 (X load) + 2× vld1q_u8 (idx load) +
 * 2× vqtbl2q_u8 (gather) + 2× vaddlvq_s8 (reduce) + 2 scalar adds on
 * the int32 accumulator ≈ 8 NEON-issue ops + scalar overhead per
 * 16-lane tile. Compare: dense SDOT covers 16 trits per ~3 NEON ops
 * (load + amortized unpack + vdotq). Routed16 wins only when sparsity
 * is high enough that the dense path covers many more trits than
 * nonzeros per tile (empirical crossover ≈ 96-97% on K=N=2560).
 *
 * ── Constraints ──────────────────────────────────────────────────────
 *
 * Per project rule (no scalar in production): NEON-only. Compile-time
 * #error if NEON unavailable. Test oracle is
 * m4t_ternary_5in8_matmul_bt_routed_ref.
 *
 * Supports arbitrary M ≥ 0 (including M=0 which returns immediately).
 * Loop order is i-outer / j-inner / tile-innermost; for very large M
 * a tile-outer / i-inner ordering would amortize tile metadata loads
 * better — that optimization is future work and would require a
 * dedicated batch entry point so M=1 callers are not penalized.
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
 *   Y[M, N] = X[M, K] @ W^T[K, N]
 *
 * X: int8 ternary or A8-quantized activations (m4t_trit_t = int8),
 *    row-major [M, K]. Per-cell accumulator is int32 with no clamp;
 *    |Y[i, j]| ≤ K * max|X| (≤ 127*K for A8, ≤ K for ternary). Caller
 *    is responsible for ensuring K is small enough that this fits.
 * W: routed16-packed weights produced by m4t_ternary_routed16_pack.
 * Y: int32 outputs (m4t_mtfp_t), row-major [M, N].
 *
 * Preconditions:
 *   M >= 0 (M==0 returns immediately)
 *   K matches W's K (asserted in debug)
 *   N matches W's N (asserted in debug)
 *   K <= M4T_SDOT_K_MAX_EXACT
 *   Tail X loads at start_k near K-1 are zero-padded internally via a
 *     stack buffer — caller does NOT need to over-allocate X.
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
