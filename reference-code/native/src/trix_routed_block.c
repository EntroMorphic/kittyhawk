/*
 * trix_routed_block.c — Fully Routed Block: Mixer + FFN
 */

#include "trix_routed_block.h"
#include "trix_routed_mixer_causal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

TrixRoutedBlock* trix_routed_block_create(TrixRoutedBlockConfig cfg, uint64_t seed) {
    TrixRoutedBlock* blk = calloc(1, sizeof(TrixRoutedBlock));
    if (!blk) return NULL;
    blk->cfg = cfg;

    TrixRoutedMixerConfig mc = {
        .d_model = cfg.d_model,
        .num_tiles = cfg.num_tiles,
        .tile_hidden = cfg.tile_hidden,
        .active_k = cfg.active_k,
        .output_scale_init = cfg.mixer_scale_init,
        .ln_eps = cfg.ln_eps
    };
    blk->mixer = trix_routed_mixer_create(mc, seed);

    TrixTernaryRouteConfig fc = {
        .d_model = cfg.d_model,
        .num_tiles = cfg.num_tiles,
        .tile_hidden = cfg.tile_hidden,
        .active_k = cfg.active_k,
        .output_scale_init = cfg.ffn_scale_init,
        .ln_eps = cfg.ln_eps
    };
    blk->ffn = trix_ternary_route_create(fc, seed + 10000);

    return blk;
}

void trix_routed_block_destroy(TrixRoutedBlock* blk) {
    if (!blk) return;
    trix_routed_mixer_destroy(blk->mixer);
    trix_ternary_route_destroy(blk->ffn);
    free(blk);
}

void trix_routed_block_forward(TrixRoutedBlock* blk,
                                const float* x, float* out, int seq_len)
{
    /* Mixer: cross-position interaction */
    trix_routed_mixer_forward(blk->mixer, x, out, seq_len);

    /* FFN: per-token computation (operates on mixer output, includes residual) */
    /* FFN reads from out and writes back to out.
     * It applies: out = mid + scale * ffn(LN(mid))
     * We need a temp buffer because FFN reads x and writes out. */
    int D = blk->cfg.d_model;
    float* mid = malloc(seq_len * D * sizeof(float));
    memcpy(mid, out, seq_len * D * sizeof(float));
    trix_ternary_route_forward(blk->ffn, mid, out, seq_len);
    free(mid);
}

void trix_routed_block_forward_causal(TrixRoutedBlock* blk,
                                       const float* x, float* out, int seq_len)
{
    /* Causal mixer: position i sees only 0..i-1 */
    trix_routed_mixer_forward_causal(blk->mixer, x, out, seq_len);

    /* FFN: per-token (no causality issue — already per-position) */
    int D = blk->cfg.d_model;
    float* mid = malloc(seq_len * D * sizeof(float));
    memcpy(mid, out, seq_len * D * sizeof(float));
    trix_ternary_route_forward(blk->ffn, mid, out, seq_len);
    free(mid);
}

void trix_routed_block_forward_causal_mtfp(TrixRoutedBlock* blk,
                                            const mtfp_t* x, mtfp_t* out, int seq_len)
{
    /* Causal mixer: MTFP end to end */
    trix_routed_mixer_forward_causal_mtfp(blk->mixer, x, out, seq_len);

    /* FFN: MTFP end to end */
    int D = blk->cfg.d_model;
    mtfp_t* mid = malloc(seq_len * D * sizeof(mtfp_t));
    memcpy(mid, out, seq_len * D * sizeof(mtfp_t));
    trix_ternary_route_forward_mtfp(blk->ffn, mid, out, seq_len);
    free(mid);
}

void trix_routed_block_backward(TrixRoutedBlock* blk,
                                 const float* x, const float* dy, float* dx,
                                 int seq_len)
{
    int D = blk->cfg.d_model;

    /* We need the mixer output (mid) for FFN backward.
     * Recompute mixer forward to get mid. */
    float* mid = malloc(seq_len * D * sizeof(float));
    trix_routed_mixer_forward(blk->mixer, x, mid, seq_len);

    /* FFN backward: dy → d_mid
     * NOTE: do NOT zero_grad here — gradients accumulate across
     * samples in a batch. Caller is responsible for zero_grad
     * before the batch starts. */
    float* d_mid = malloc(seq_len * D * sizeof(float));
    trix_ternary_route_backward(blk->ffn, mid, dy, d_mid, seq_len);

    /* Mixer backward: d_mid → dx */
    trix_routed_mixer_backward(blk->mixer, x, d_mid, dx, seq_len);

    free(mid);
    free(d_mid);
}

void trix_routed_block_zero_grad(TrixRoutedBlock* blk) {
    trix_routed_mixer_zero_grad(blk->mixer);
    trix_ternary_route_zero_grad(blk->ffn);
}

void trix_routed_block_adamw_step(TrixRoutedBlock* blk,
                                   float lr, float b1, float b2,
                                   float eps, float wd)
{
    trix_routed_mixer_adamw_step(blk->mixer, lr, b1, b2, eps, wd);
    trix_ternary_route_adamw_step(blk->ffn, lr, b1, b2, eps, wd);
}

float trix_routed_block_clip_grad_norm(TrixRoutedBlock* blk, float max_norm) {
    /* Global clipping across mixer + FFN */
    int D = blk->cfg.d_model, T = blk->cfg.num_tiles, H = blk->cfg.tile_hidden;
    TrixRoutedMixer* rm = blk->mixer;
    TrixTernaryRoutedFFN* tr = blk->ffn;

    float sq = 0.0f;

    /* Mixer grads */
    sq += trix_sum_sq(rm->dln_weight, D);
    sq += trix_sum_sq(rm->dln_bias, D);
    sq += trix_sum_sq(rm->dW1, T * H * D);
    sq += trix_sum_sq(rm->db1, T * H);
    sq += trix_sum_sq(rm->dW2, T * D * H);
    sq += trix_sum_sq(rm->db2, T * D);
    sq += rm->doutput_scale * rm->doutput_scale;

    /* FFN grads */
    sq += trix_sum_sq(tr->dln_weight, D);
    sq += trix_sum_sq(tr->dln_bias, D);
    sq += trix_sum_sq(tr->dW1, T * H * D);
    sq += trix_sum_sq(tr->db1, T * H);
    sq += trix_sum_sq(tr->dW2, T * D * H);
    sq += trix_sum_sq(tr->db2, T * D);
    sq += tr->doutput_scale * tr->doutput_scale;

    float norm = sqrtf(sq);
    if (norm > max_norm && norm > 0.0f) {
        float s = max_norm / norm;
        /* Scale mixer grads */
        trix_vec_scale(rm->dln_weight, rm->dln_weight, s, D);
        trix_vec_scale(rm->dln_bias, rm->dln_bias, s, D);
        trix_vec_scale(rm->dW1, rm->dW1, s, T * H * D);
        trix_vec_scale(rm->db1, rm->db1, s, T * H);
        trix_vec_scale(rm->dW2, rm->dW2, s, T * D * H);
        trix_vec_scale(rm->db2, rm->db2, s, T * D);
        rm->doutput_scale *= s;
        /* Scale FFN grads */
        trix_vec_scale(tr->dln_weight, tr->dln_weight, s, D);
        trix_vec_scale(tr->dln_bias, tr->dln_bias, s, D);
        trix_vec_scale(tr->dW1, tr->dW1, s, T * H * D);
        trix_vec_scale(tr->db1, tr->db1, s, T * H);
        trix_vec_scale(tr->dW2, tr->dW2, s, T * D * H);
        trix_vec_scale(tr->db2, tr->db2, s, T * D);
        tr->doutput_scale *= s;
    }
    return norm;
}
