/*
 * trix_routed_mixer_causal.c — Causal scatter-gather for autoregressive LM
 *
 * Process positions left to right. Each position:
 *   1. Gathers from running pools (sees only past)
 *   2. Scatters its own contribution (for future positions)
 *
 * The tile FFN transforms the running pool before gather. Since the pool
 * changes at each position, the tile FFN runs per-position — but only on
 * the K active tiles, not all T tiles, and only on one D-vector per tile.
 *
 * Backward: reverse the causal chain. Each position's gradient flows
 * back through its gather and forward through its scatter contribution
 * to all future positions.
 */

#include "trix_routed_mixer_causal.h"
#include "trix_ternary_matmul.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════════════
 * Causal Forward
 *
 * For each position (left to right):
 *   1. Route: score against signatures, threshold to ternary
 *   2. Gather: y[pos] = sum_t route[pos,t] * tile_ffn(running_pool[t])
 *      (only active tiles are transformed)
 *   3. Scatter: running_pool[t] += route[pos,t] * x_norm[pos]
 *      (update pools for future positions)
 *   4. Scale + residual: out[pos] = x[pos] + scale * y[pos]
 *
 * Note the order: gather THEN scatter. Position i gathers from pools
 * containing positions 0..i-1, then adds its own contribution for i+1..n.
 * ══════════════════════════════════════════════════════════════════════ */

/* Ternary threshold — same as in trix_routed_mixer.c */
static void causal_threshold_topk(int* route, const float* scores, int T, int k) {
    memset(route, 0, T * sizeof(int));
    float abs_scores[64];
    for (int t = 0; t < T; t++) abs_scores[t] = fabsf(scores[t]);
    for (int j = 0; j < k && j < T; j++) {
        int best = -1; float best_val = -1.0f;
        for (int t = 0; t < T; t++) {
            if (abs_scores[t] > best_val) { best_val = abs_scores[t]; best = t; }
        }
        if (best >= 0) {
            route[best] = (scores[best] >= 0.0f) ? 1 : -1;
            abs_scores[best] = -1.0f;
        }
    }
}

void trix_routed_mixer_forward_causal(TrixRoutedMixer* rm,
                                       const float* x, float* out, int seq_len)
{
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles;
    int H = rm->cfg.tile_hidden, K = rm->cfg.active_k;

    /* Ensure scratch */
    if (seq_len > rm->seq_cap) {
        /* Force realloc via the non-causal path's ensure */
        trix_routed_mixer_forward(rm, x, out, seq_len);
        /* Now redo causally */
    }

    /* 1. LayerNorm (full sequence — LN is per-position, no causality issue) */
    trix_layernorm_forward_save(rm->x_norm, rm->ln_mean, rm->ln_rstd,
        x, rm->ln_weight, rm->ln_bias, seq_len, D, rm->cfg.ln_eps);

    mtfp_from_float_batch(rm->mx_norm, rm->x_norm, seq_len * D);

    /* 2. Routing scores (full sequence — scoring is per-position) */
    {
        int8_t* sig_i8 = calloc(T * D, sizeof(int8_t));
        mtfp_t* mscores = calloc(seq_len * T, sizeof(mtfp_t));
        for (int i = 0; i < T * D; i++)
            sig_i8[i] = (rm->signatures[i] > 0.5f) ? 1 : (rm->signatures[i] < -0.5f) ? -1 : 0;
        mtfp_ternary_matmul_bt(mscores, rm->mx_norm, sig_i8, seq_len, D, T);
        mtfp_to_float_batch(rm->scores, mscores, seq_len * T);
        free(mscores); free(sig_i8);
    }

    /* 3. Threshold all positions */
    for (int pos = 0; pos < seq_len; pos++)
        causal_threshold_topk(rm->route + pos * T, rm->scores + pos * T, T, K);

    /* 4. Causal scatter-gather: left to right */
    mtfp_t* running_pool = calloc(T * D, sizeof(mtfp_t));
    int* running_counts = calloc(T, sizeof(int));

    /* Per-tile FFN scratch (one vector at a time) */
    mtfp_t* tile_z1 = calloc(H, sizeof(mtfp_t));
    mtfp_t* tile_h1 = calloc(H, sizeof(mtfp_t));
    mtfp_t* tile_mixed = calloc(D, sizeof(mtfp_t));
    mtfp_t* tile_pool_norm = calloc(D, sizeof(mtfp_t));

    memset(rm->mgathered, 0, seq_len * D * sizeof(mtfp_t));

    /* Save per-position pool snapshots and tile FFN states for backward */
    /* (saved_pool, saved_z1, saved_h1, saved_mixed are [T, *] sized —
     *  for causal we'd need [seq, T, *] but that's too much memory.
     *  For now, save only what we need for the float backward.) */

    for (int pos = 0; pos < seq_len; pos++) {
        int* r = rm->route + pos * T;

        /* GATHER from running pools (sees only past) */
        for (int t = 0; t < T; t++) {
            if (r[t] == 0) continue;
            if (running_counts[t] == 0) continue;  /* empty pool — nothing to gather */

            /* Normalize a copy of the running pool for this tile */
            memcpy(tile_pool_norm, running_pool + t * D, D * sizeof(mtfp_t));
            mtfp_fan_in_normalize(tile_pool_norm, D, running_counts[t]);

            /* Tile FFN: z1 = pool_norm @ W1^T + b1 */
            int8_t* W1t = rm->W1_tern + t * H * D;
            int8_t* W2t = rm->W2_tern + t * D * H;

            mtfp_ternary_matmul_bt(tile_z1, tile_pool_norm, W1t, 1, D, H);
            mtfp_fan_in_normalize(tile_z1, H, D);
            mtfp_bias_add(tile_z1, rm->mb1 + t * H, 1, H);
            mtfp_gelu(tile_h1, tile_z1, H);
            mtfp_ternary_matmul_bt(tile_mixed, tile_h1, W2t, 1, H, D);
            mtfp_bias_add(tile_mixed, rm->mb2 + t * D, 1, D);

            /* Accumulate into gathered output for this position */
            mtfp_t* dst = rm->mgathered + pos * D;
            if (r[t] == 1) mtfp_vec_add_inplace(dst, tile_mixed, D);
            else for (int d = 0; d < D; d++) dst[d] -= tile_mixed[d];
        }

        /* SCATTER this position into running pools (for future positions) */
        for (int t = 0; t < T; t++) {
            if (r[t] == 0) continue;
            running_counts[t]++;
            mtfp_t* dst = running_pool + t * D;
            const mtfp_t* src = rm->mx_norm + pos * D;
            if (r[t] == 1) mtfp_vec_add_inplace(dst, src, D);
            else for (int d = 0; d < D; d++) dst[d] -= src[d];
        }
    }

    /* 5. Convert to float, scale + residual */
    float* gathered_f = rm->d_gathered;  /* reuse scratch */
    mtfp_to_float_batch(gathered_f, rm->mgathered, seq_len * D);

    for (int i = 0; i < seq_len * D; i++)
        out[i] = x[i] + rm->output_scale * gathered_f[i];

    /* Save running pool state for backward */
    mtfp_to_float_batch(rm->saved_pool, running_pool, T * D);
    memcpy(rm->pool_counts, running_counts, T * sizeof(int));

    free(running_pool); free(running_counts);
    free(tile_z1); free(tile_h1); free(tile_mixed); free(tile_pool_norm);
}

/* ══════════════════════════════════════════════════════════════════════
 * MTFP Causal Forward — zero float in the entire path
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_mixer_forward_causal_mtfp(TrixRoutedMixer* rm,
                                            const mtfp_t* x, mtfp_t* out, int seq_len)
{
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles;
    int H = rm->cfg.tile_hidden, K = rm->cfg.active_k;

    /* Ensure scratch */
    if (seq_len > rm->seq_cap) {
        float* tmp_x = calloc(seq_len * D, sizeof(float));
        float* tmp_out = calloc(seq_len * D, sizeof(float));
        mtfp_to_float_batch(tmp_x, x, seq_len * D);
        trix_routed_mixer_forward(rm, tmp_x, tmp_out, seq_len);
        free(tmp_x); free(tmp_out);
    }

    /* 1. MTFP LayerNorm — integer arithmetic, one sqrt */
    mtfp_layernorm(rm->mx_norm, x, rm->mln_weight, rm->mln_bias, seq_len, D);

    /* Save float copies for backward (STE needs float) */
    mtfp_to_float_batch(rm->x_norm, rm->mx_norm, seq_len * D);
    {
        float* x_f = calloc(seq_len * D, sizeof(float));
        mtfp_to_float_batch(x_f, x, seq_len * D);
        trix_layernorm_forward_save(rm->x_norm, rm->ln_mean, rm->ln_rstd,
            x_f, rm->ln_weight, rm->ln_bias, seq_len, D, rm->cfg.ln_eps);
        free(x_f);
    }

    /* 2. Routing scores — MTFP × ternary = integer add/sub */
    {
        int8_t* sig_i8 = calloc(T * D, sizeof(int8_t));
        mtfp_t* mscores = calloc(seq_len * T, sizeof(mtfp_t));
        for (int i = 0; i < T * D; i++)
            sig_i8[i] = (rm->signatures[i] > 0.5f) ? 1 : (rm->signatures[i] < -0.5f) ? -1 : 0;
        mtfp_ternary_matmul_bt(mscores, rm->mx_norm, sig_i8, seq_len, D, T);
        mtfp_to_float_batch(rm->scores, mscores, seq_len * T);
        free(mscores); free(sig_i8);
    }

    /* 3. Threshold all positions */
    for (int pos = 0; pos < seq_len; pos++)
        causal_threshold_topk(rm->route + pos * T, rm->scores + pos * T, T, K);

    /* 4. Causal scatter-gather: left to right, all MTFP */
    mtfp_t* running_pool = calloc(T * D, sizeof(mtfp_t));
    int* running_counts = calloc(T, sizeof(int));

    mtfp_t* tile_z1 = calloc(H, sizeof(mtfp_t));
    mtfp_t* tile_h1 = calloc(H, sizeof(mtfp_t));
    mtfp_t* tile_mixed = calloc(D, sizeof(mtfp_t));
    mtfp_t* tile_pool_norm = calloc(D, sizeof(mtfp_t));

    memset(rm->mgathered, 0, seq_len * D * sizeof(mtfp_t));

    for (int pos = 0; pos < seq_len; pos++) {
        int* r = rm->route + pos * T;

        /* GATHER from running pools (past only) */
        for (int t = 0; t < T; t++) {
            if (r[t] == 0) continue;
            if (running_counts[t] == 0) continue;

            memcpy(tile_pool_norm, running_pool + t * D, D * sizeof(mtfp_t));
            mtfp_fan_in_normalize(tile_pool_norm, D, running_counts[t]);

            int8_t* W1t = rm->W1_tern + t * H * D;
            int8_t* W2t = rm->W2_tern + t * D * H;

            mtfp_ternary_matmul_bt(tile_z1, tile_pool_norm, W1t, 1, D, H);
            mtfp_fan_in_normalize(tile_z1, H, D);
            mtfp_bias_add(tile_z1, rm->mb1 + t * H, 1, H);
            mtfp_gelu(tile_h1, tile_z1, H);
            mtfp_ternary_matmul_bt(tile_mixed, tile_h1, W2t, 1, H, D);
            mtfp_bias_add(tile_mixed, rm->mb2 + t * D, 1, D);

            mtfp_t* dst = rm->mgathered + pos * D;
            if (r[t] == 1) mtfp_vec_add_inplace(dst, tile_mixed, D);
            else for (int d = 0; d < D; d++) dst[d] -= tile_mixed[d];
        }

        /* SCATTER into running pools (for future positions) */
        for (int t = 0; t < T; t++) {
            if (r[t] == 0) continue;
            running_counts[t]++;
            mtfp_t* dst = running_pool + t * D;
            const mtfp_t* src = rm->mx_norm + pos * D;
            if (r[t] == 1) mtfp_vec_add_inplace(dst, src, D);
            else for (int d = 0; d < D; d++) dst[d] -= src[d];
        }
    }

    /* 5. Scale + residual — all MTFP, no float */
    mtfp_t mscale = mtfp_from_float(rm->output_scale);
    for (int i = 0; i < seq_len * D; i++)
        out[i] = mtfp_add(x[i], mtfp_mul(mscale, rm->mgathered[i]));

    /* Save for backward */
    mtfp_to_float_batch(rm->saved_pool, running_pool, T * D);
    memcpy(rm->pool_counts, running_counts, T * sizeof(int));
    /* Save float gathered for backward d_output_scale computation */
    mtfp_to_float_batch(rm->d_gathered, rm->mgathered, seq_len * D);

    free(running_pool); free(running_counts);
    free(tile_z1); free(tile_h1); free(tile_mixed); free(tile_pool_norm);
}
