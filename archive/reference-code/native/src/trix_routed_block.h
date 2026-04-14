/*
 * trix_routed_block.h — Fully Routed Block
 *
 * One block = TrixRoutedMixer + TrixTernaryRoutedFFN with residual.
 * The mixer provides cross-position interaction (ternary basis decomposition).
 * The FFN provides per-token computation (ternary routed tiles).
 * Stack L blocks for depth.
 *
 * No dense matmul. No attention. All ternary-routed.
 */

#ifndef TRIX_ROUTED_BLOCK_H
#define TRIX_ROUTED_BLOCK_H

#include "trix_routed_mixer.h"
#include "trix_ternary_route.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int d_model;
    int num_tiles;
    int tile_hidden;
    int active_k;
    float mixer_scale_init;
    float ffn_scale_init;
    float ln_eps;
} TrixRoutedBlockConfig;

typedef struct TrixRoutedBlock {
    TrixRoutedBlockConfig cfg;
    TrixRoutedMixer*      mixer;
    TrixTernaryRoutedFFN* ffn;
} TrixRoutedBlock;

/* ── Lifecycle ── */

TrixRoutedBlock* trix_routed_block_create(TrixRoutedBlockConfig cfg, uint64_t seed);
void trix_routed_block_destroy(TrixRoutedBlock* blk);

/* ── Forward: x[seq, D] → out[seq, D] ── */
void trix_routed_block_forward(TrixRoutedBlock* blk,
                                const float* x, float* out, int seq_len);

/* ── Causal Forward: position i sees only 0..i-1 through the mixer ── */
void trix_routed_block_forward_causal(TrixRoutedBlock* blk,
                                       const float* x, float* out, int seq_len);

/* ── MTFP Causal Forward: entire path in integer arithmetic ── */
void trix_routed_block_forward_causal_mtfp(TrixRoutedBlock* blk,
                                            const mtfp_t* x, mtfp_t* out, int seq_len);

/* ── Backward ── */
void trix_routed_block_backward(TrixRoutedBlock* blk,
                                 const float* x, const float* dy, float* dx,
                                 int seq_len);

void trix_routed_block_zero_grad(TrixRoutedBlock* blk);

/* ── Optimizer ── */
void trix_routed_block_adamw_step(TrixRoutedBlock* blk,
                                   float lr, float b1, float b2,
                                   float eps, float wd);

float trix_routed_block_clip_grad_norm(TrixRoutedBlock* blk, float max_norm);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_ROUTED_BLOCK_H */
