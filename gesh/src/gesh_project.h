/*
 * gesh_project.h — substrate-routed ternary projection.
 *
 * Phase B substrate-discipline remediation. Replaces the open-coded
 * `acc = sum(R[j]*x[j]); s = sign(acc)` patterns scattered across
 * gesh/src/{gesh_forward,gesh_train,gesh_bank}.c and the bench code
 * with a single wrapper that routes through `libm4t` kernels:
 *
 *   1. m4t_pack_trits_1d        (R packed once, amortized)
 *   2. widen ternary x → MTFP19 (sanctioned: one-shot per-call)
 *   3. m4t_mtfp_ternary_matmul_bt   (the real work; NEON-vectorized)
 *   4. m4t_route_threshold_extract  (sign quantize at tau = 0)
 *   5. m4t_unpack_trits_1d      (caller-format conversion)
 *
 * The inner-loop arithmetic lives entirely inside libm4t. This header
 * exposes the only API gesh consumers and benches should use to project
 * ternary signatures through ternary R.
 *
 * Substrate-claim integrity:
 *   - All multiply-accumulate runs in m4t_mtfp_ternary_matmul_bt.
 *   - All ternarize-to-trit runs in m4t_route_threshold_extract.
 *   - tau = 0 is sanctioned per m4t_route.h's second input class
 *     (integer-arithmetic inputs with non-trivial probability of
 *     exact zero — the case we always have for ternary × ternary
 *     integer matmul on integer-valued inputs).
 *
 * Bit-equivalence with the prior open-coded path: the kernel and the
 * open-coded loop produce identical ternary outputs for any pair of
 * ternary inputs (both are deterministic integer operations with the
 * same arithmetic semantics). Verified by `tests/test_gesh_project.c`.
 */

#ifndef GESH_PROJECT_H
#define GESH_PROJECT_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Project a batch of n ternary signatures through a ternary projection
 * R, producing n ternary output signatures, all UNPACKED (m4t_trit_t).
 *
 *   x_batch:    [n × input_dim]      input ternary trits
 *   R:          [sig_dim × input_dim] projection ternary trits
 *   out_batch:  [n × sig_dim]        output ternary trits, caller-allocated
 *
 * Operation: out_batch[i, oi] = sign( Σ_j R[oi, j] · x_batch[i, j] ).
 * Tie at zero → 0.
 *
 * Routes through libm4t kernels end-to-end (no open-coded
 * multiply-accumulate or sign-threshold). Allocates O(n·sig_dim +
 * n·input_dim + sig_dim·Dp_in + sig_dim·8) scratch internally.
 *
 * Preconditions:
 *   n, sig_dim, input_dim > 0
 *   out_batch must not alias x_batch or R
 */
void gesh_project_batch_unpacked(
    m4t_trit_t*       out_batch,
    const m4t_trit_t* x_batch,
    int               n,
    const m4t_trit_t* R,
    int               sig_dim,
    int               input_dim);

/* Project a single ternary signature through R into one packed-trit
 * output signature. Convenience wrapper around the batch path with
 * n = 1 plus a final pack step (so callers that already maintain
 * packed signatures save the unpack→pack round-trip).
 *
 *   x:          [input_dim]                     unpacked ternary
 *   R:          [sig_dim × input_dim]           unpacked ternary
 *   out_packed: [M4T_TRIT_PACKED_BYTES(sig_dim)] packed ternary,
 *                                                caller-allocated
 *
 * Preconditions: as for the batch path; n == 1. */
void gesh_project_one_packed(
    uint8_t*          out_packed,
    const m4t_trit_t* x,
    const m4t_trit_t* R,
    int               sig_dim,
    int               input_dim);

/* ── Scratch-aware variant for hot loops ─────────────────────────────────
 *
 * The wrappers above allocate per call. For hot-loop use (gesh_train's
 * coordinate-descent inner loop), the per-call malloc/free overhead
 * dominates. The scratch struct below holds all kernel buffers in
 * caller-managed memory; one alloc at startup, reused for every call.
 */

typedef struct {
    uint8_t*    R_packed;     /* [sig_dim × M4T_TRIT_PACKED_BYTES(input_dim)] */
    m4t_mtfp_t* X_mtfp;       /* [n_max × input_dim] */
    m4t_mtfp_t* Y_mtfp;       /* [n_max × sig_dim] */
    int64_t*    row64;        /* [sig_dim] */
    uint8_t*    row_packed;   /* [M4T_TRIT_PACKED_BYTES(sig_dim)] */
    int sig_dim;
    int input_dim;
    int n_max;
} gesh_project_scratch_t;

/* Allocate scratch sized for projecting up to n_max queries through R
 * of shape [sig_dim × input_dim]. Zero-init's nothing except struct
 * dimensions; buffers may contain garbage on entry. */
void gesh_project_scratch_init(
    gesh_project_scratch_t* sc,
    int sig_dim, int input_dim, int n_max);

void gesh_project_scratch_free(gesh_project_scratch_t* sc);

/* Same operation as gesh_project_batch_unpacked, but uses caller-supplied
 * scratch — no per-call malloc. R is re-packed into sc->R_packed every
 * call (callers in hot loops where R changes need this; a future
 * scratch_pack_R_row helper could amortize when only a single trit
 * changed). x widening into sc->X_mtfp is done per call.
 *
 * Preconditions: sc was initialized with sig_dim ≥ this call's sig_dim,
 * input_dim ≥ this call's input_dim, n_max ≥ n. */
void gesh_project_batch_unpacked_scratch(
    m4t_trit_t* out_batch,
    const m4t_trit_t* x_batch,
    int n,
    const m4t_trit_t* R,
    int sig_dim, int input_dim,
    gesh_project_scratch_t* sc);

/* Threshold an int32 mantissa array to unpacked trits at tau = 0.
 * Wrapper around m4t_route_threshold_extract that handles the int32
 * → int64 widen + post-extract unpack so callers with an int32 acc
 * don't open-code the sign step.
 *
 *   values: [n] int32 mantissas
 *   out:    [n] unpacked ternary trits, caller-allocated
 *
 * Substrate-discipline: lifts every "v > 0 ? 1 : v < 0 ? -1 : 0"
 * scatter-site through the kernel.
 *
 * Preconditions: n > 0; out must not alias values. */
void gesh_threshold_int32_to_trit(
    m4t_trit_t* out,
    const int32_t* values,
    int n);

#ifdef __cplusplus
}
#endif

#endif /* GESH_PROJECT_H */
