/*
 * m4t_ternary_rowskip.h — Row-skip ternary matmul (NEON, dense kernel
 * over compressed K).
 *
 * Per journal/bitnet_dead_columns.md: BitNet's W1.58 weights have
 * layer-specific empty K-rows (input dims that contribute nothing
 * across all output columns). Layer 0 o_proj has 15.5%, layer 1
 * down_proj has 43.6%, etc.
 *
 * This kernel exploits that structure by:
 *   1. At pack time: build a list of non-empty K indices, and repack
 *      W into 5-in-8 format on only those K positions. Compressed
 *      K = K - count(empty rows).
 *   2. At call time: gather X into X_compressed using the index list,
 *      then call the existing dense kernel m4t_ternary_5in8_matmul_bt
 *      on the compressed K.
 *
 * The win: kernel work is linear in K, so reducing K from 6912 to
 * 3898 (43.6% empty) gives ~43.6% per-call speedup on that BitLinear,
 * minus the gather overhead.
 *
 * ── Selection policy (per measured benches) ─────────────────────────
 *
 * After the K%80 fix to m4t_ternary_5in8_matmul_bt (journal/k80_fix_lmm.md),
 * rowskip's net benefit is roughly proportional to the empty-row
 * fraction, minus a small gather/pad overhead. Smart dispatch:
 *
 *   skip% ≥ 5%   → use rowskip
 *   skip% <  5%  → use dense (m4t_ternary_5in8_matmul_bt)
 *
 * BitNet layer-0 BitLinears that benefit from rowskip:
 *   layer 1 down_proj  —  43.6% empty (largest single win, ~50% reduction)
 *   layer 2 down_proj  —  27.7%
 *   layer 29 o_proj    —  24.0%
 *   layer 0 o_proj     —  15.5%
 *   layer 3 down_proj  —  14.9%
 *   most other layers  —  <5%   (use dense, rowskip pays only gather overhead)
 *
 * Aggregate across BitNet's 210 calls/token at smart dispatch:
 * ~+0.9% over dense alone. Most of the per-token compute savings are
 * concentrated in 4-5 specific (layer, BitLinear) pairs.
 *
 * ── Bit-exactness ───────────────────────────────────────────────────
 *
 * Bit-exact to m4t_ternary_5in8_matmul_bt because:
 *   - Empty K positions contribute 0 to every output (W[k, :] = 0
 *     for all j means dot product term is 0 regardless of X[k]).
 *   - Removing those terms from the sum doesn't change the result.
 *
 * Verified against the dense kernel oracle in
 * test_m4t_ternary_rowskip.c.
 *
 * ── Constraints ──────────────────────────────────────────────────────
 *
 * NEON-only (delegates to dense kernel which is NEON-only). Gather
 * uses a scalar loop (~3 ns per index on Apple Silicon); for K up
 * to ~16K positions this is negligible relative to the matmul.
 *
 * Supports arbitrary M, including M=0.
 */

#ifndef M4T_TERNARY_ROWSKIP_H
#define M4T_TERNARY_ROWSKIP_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque packed handle. */
typedef struct m4t_ternary_rowskip_packed m4t_ternary_rowskip_packed_t;

/* Encoder: convert 5-in-8 packed weights to rowskip-packed format.
 *
 * W_5in8: source weight matrix in W^T layout (row-major [N, Kp]
 *   where Kp = (K+4)/5), as consumed by m4t_ternary_5in8_matmul_bt.
 * K, N: original dimensions.
 *
 * Returns NULL on allocation failure. Caller must free with
 * m4t_ternary_rowskip_packed_free.
 *
 * Preconditions:
 *   K >= 0, N >= 0
 *   W_5in8 non-NULL when K * N > 0
 */
m4t_ternary_rowskip_packed_t* m4t_ternary_rowskip_pack(
    const uint8_t* W_5in8, int K, int N);

void m4t_ternary_rowskip_packed_free(m4t_ternary_rowskip_packed_t* p);

/* Introspection. */
int m4t_ternary_rowskip_packed_K(const m4t_ternary_rowskip_packed_t* p);
int m4t_ternary_rowskip_packed_K_compressed(const m4t_ternary_rowskip_packed_t* p);
int m4t_ternary_rowskip_packed_N(const m4t_ternary_rowskip_packed_t* p);
size_t m4t_ternary_rowskip_packed_bytes(const m4t_ternary_rowskip_packed_t* p);

/* Production NEON kernel.
 *   Y[M, N] = X[M, K] @ W^T[K, N]
 *
 * X: int8 ternary or A8-quantized activations (m4t_trit_t = int8),
 *    row-major [M, K_original]. The kernel internally gathers
 *    X[i, nonempty_indices] into a scratch buffer of size K_compressed
 *    per row, then calls the dense kernel.
 * W: rowskip-packed weights produced by m4t_ternary_rowskip_pack.
 * Y: int32 outputs (m4t_mtfp_t), row-major [M, N].
 *
 * Preconditions:
 *   M >= 0 (M==0 returns immediately)
 *   K matches W's K_original (asserted in debug)
 *   N matches W's N (asserted in debug)
 */
void m4t_ternary_rowskip_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_ternary_rowskip_packed_t* W,
    int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TERNARY_ROWSKIP_H */
