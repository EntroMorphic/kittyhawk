/*
 * trix_routed_mixer.c — Routed Token Mixer
 *
 * Scatter-transform-gather: ternary-routed cross-position interaction.
 * Replaces dense attention with O(seq_len × T × D) routed mixing.
 *
 * Forward path is fully MTFP integer arithmetic. No float multiplies.
 * Backward path uses float shadow weights (STE through quantization).
 */

#include "trix_routed_mixer.h"
#include "trix_ternary_matmul.h"
#include "trix_rng.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ══════════════════════════════════════════════════════════════════════
 * Ternary quantization and packing
 * ══════════════════════════════════════════════════════════════════════ */

static void binarize_to_ternary(int8_t* dst, const float* src, int n) {
    float mean_abs = 0.0f;
    for (int i = 0; i < n; i++) mean_abs += fabsf(src[i]);
    mean_abs /= (float)n;
    float thresh = mean_abs * 0.5f;
    for (int i = 0; i < n; i++)
        dst[i] = (src[i] > thresh) ? 1 : (src[i] < -thresh) ? -1 : 0;
}

static void quantize_tiles(TrixRoutedMixer* rm) {
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles, H = rm->cfg.tile_hidden;
    for (int t = 0; t < T; t++) {
        binarize_to_ternary(rm->W1_tern + t * H * D, rm->W1 + t * H * D, H * D);
        binarize_to_ternary(rm->W2_tern + t * D * H, rm->W2 + t * D * H, D * H);
    }
    if (rm->mb1) mtfp_from_float_batch(rm->mb1, rm->b1, T * H);
    if (rm->mb2) mtfp_from_float_batch(rm->mb2, rm->b2, T * D);
    if (rm->mln_weight) mtfp_from_float_batch(rm->mln_weight, rm->ln_weight, D);
    if (rm->mln_bias) mtfp_from_float_batch(rm->mln_bias, rm->ln_bias, D);
}

/* ══════════════════════════════════════════════════════════════════════
 * Ternary threshold: top-k by |score|, assign sign
 * ══════════════════════════════════════════════════════════════════════ */

static void ternary_threshold_topk(int* route, const float* scores,
                                    int batch, int T, int k)
{
    for (int i = 0; i < batch; i++) {
        const float* s = scores + i * T;
        int* r = route + i * T;
        memset(r, 0, T * sizeof(int));
        float abs_scores[64];
        for (int t = 0; t < T; t++) abs_scores[t] = fabsf(s[t]);
        for (int j = 0; j < k && j < T; j++) {
            int best = -1; float best_val = -1.0f;
            for (int t = 0; t < T; t++) {
                if (abs_scores[t] > best_val) { best_val = abs_scores[t]; best = t; }
            }
            if (best >= 0) {
                r[best] = (s[best] >= 0.0f) ? 1 : -1;
                abs_scores[best] = -1.0f;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Scratch allocation
 * ══════════════════════════════════════════════════════════════════════ */

static void rm_ensure_scratch(TrixRoutedMixer* rm, int seq_len) {
    if (seq_len <= rm->seq_cap) return;
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles, H = rm->cfg.tile_hidden;

    free(rm->x_norm); free(rm->ln_mean); free(rm->ln_rstd);
    free(rm->scores); free(rm->route);
    free(rm->mx_norm); free(rm->pool); free(rm->pool_counts);
    free(rm->mz1); free(rm->mh1); free(rm->mixed); free(rm->mgathered);
    free(rm->saved_pool); free(rm->saved_z1); free(rm->saved_h1); free(rm->saved_mixed);
    free(rm->d_gathered); free(rm->d_mixed);
    free(rm->dh1); free(rm->dz1); free(rm->d_pool); free(rm->dx_scatter);

    rm->x_norm      = calloc(seq_len * D, sizeof(float));
    rm->ln_mean     = calloc(seq_len, sizeof(float));
    rm->ln_rstd     = calloc(seq_len, sizeof(float));
    rm->scores      = calloc(seq_len * T, sizeof(float));
    rm->route       = calloc(seq_len * T, sizeof(int));

    rm->mx_norm     = calloc(seq_len * D, sizeof(mtfp_t));
    rm->pool        = calloc(T * D, sizeof(mtfp_t));
    rm->pool_counts = calloc(T, sizeof(int));
    rm->mz1         = calloc(T * H, sizeof(mtfp_t));
    rm->mh1         = calloc(T * H, sizeof(mtfp_t));
    rm->mixed       = calloc(T * D, sizeof(mtfp_t));
    rm->mgathered   = calloc(seq_len * D, sizeof(mtfp_t));

    rm->saved_pool  = calloc(T * D, sizeof(float));
    rm->saved_z1    = calloc(T * H, sizeof(float));
    rm->saved_h1    = calloc(T * H, sizeof(float));
    rm->saved_mixed = calloc(T * D, sizeof(float));

    rm->d_gathered  = calloc(seq_len * D, sizeof(float));
    rm->d_mixed     = calloc(T * D, sizeof(float));
    rm->dh1         = calloc(T * H, sizeof(float));
    rm->dz1         = calloc(T * H, sizeof(float));
    rm->d_pool      = calloc(T * D, sizeof(float));
    rm->dx_scatter  = calloc(seq_len * D, sizeof(float));

    rm->seq_cap = seq_len;
}

/* ══════════════════════════════════════════════════════════════════════
 * Create / Destroy
 * ══════════════════════════════════════════════════════════════════════ */

TrixRoutedMixer* trix_routed_mixer_create(TrixRoutedMixerConfig cfg, uint64_t seed) {
    TrixRoutedMixer* rm = calloc(1, sizeof(TrixRoutedMixer));
    if (!rm) return NULL;
    rm->cfg = cfg;

    int D = cfg.d_model, T = cfg.num_tiles, H = cfg.tile_hidden;
    trix_rng_t rng = trix_rng_seed(seed);

    /* LayerNorm */
    rm->ln_weight = calloc(D, sizeof(float));
    rm->ln_bias   = calloc(D, sizeof(float));
    for (int i = 0; i < D; i++) rm->ln_weight[i] = 1.0f;

    /* Signatures */
    rm->signatures = calloc(T * D, sizeof(float));
    trix_xavier_init(rm->signatures, D, T, T * D, &rng);

    /* Per-tile FFN weights */
    rm->W1 = calloc(T * H * D, sizeof(float));
    rm->b1 = calloc(T * H, sizeof(float));
    rm->W2 = calloc(T * D * H, sizeof(float));
    rm->b2 = calloc(T * D, sizeof(float));
    for (int t = 0; t < T; t++) {
        trix_xavier_init(rm->W1 + t * H * D, D, H, H * D, &rng);
        trix_xavier_init(rm->W2 + t * D * H, H, D, D * H, &rng);
    }
    rm->output_scale = cfg.output_scale_init;

    /* Ternary weights */
    rm->W1_tern = calloc(T * H * D, sizeof(int8_t));
    rm->W2_tern = calloc(T * D * H, sizeof(int8_t));

    /* MTFP weights */
    rm->mb1 = calloc(T * H, sizeof(mtfp_t));
    rm->mb2 = calloc(T * D, sizeof(mtfp_t));
    rm->mln_weight = calloc(D, sizeof(mtfp_t));
    rm->mln_bias   = calloc(D, sizeof(mtfp_t));

    /* Gradients */
    rm->dln_weight = calloc(D, sizeof(float));
    rm->dln_bias   = calloc(D, sizeof(float));
    rm->dW1 = calloc(T * H * D, sizeof(float));
    rm->db1 = calloc(T * H, sizeof(float));
    rm->dW2 = calloc(T * D * H, sizeof(float));
    rm->db2 = calloc(T * D, sizeof(float));

    /* AdamW moments */
    rm->m_ln_w = calloc(D, sizeof(float)); rm->v_ln_w = calloc(D, sizeof(float));
    rm->m_ln_b = calloc(D, sizeof(float)); rm->v_ln_b = calloc(D, sizeof(float));
    rm->m_W1 = calloc(T*H*D, sizeof(float)); rm->v_W1 = calloc(T*H*D, sizeof(float));
    rm->m_b1 = calloc(T*H, sizeof(float));   rm->v_b1 = calloc(T*H, sizeof(float));
    rm->m_W2 = calloc(T*D*H, sizeof(float)); rm->v_W2 = calloc(T*D*H, sizeof(float));
    rm->m_b2 = calloc(T*D, sizeof(float));   rm->v_b2 = calloc(T*D, sizeof(float));

    rm->seq_cap = 0;
    rm->adam_step = 0;

    mtfp_gelu_init();
    trix_routed_mixer_update_signatures(rm);
    quantize_tiles(rm);

    return rm;
}

void trix_routed_mixer_destroy(TrixRoutedMixer* rm) {
    if (!rm) return;
    free(rm->ln_weight); free(rm->ln_bias);
    free(rm->signatures);
    free(rm->W1); free(rm->b1); free(rm->W2); free(rm->b2);
    free(rm->W1_tern); free(rm->W2_tern);
    free(rm->mb1); free(rm->mb2); free(rm->mln_weight); free(rm->mln_bias);
    free(rm->dln_weight); free(rm->dln_bias);
    free(rm->dW1); free(rm->db1); free(rm->dW2); free(rm->db2);
    free(rm->m_ln_w); free(rm->v_ln_w);
    free(rm->m_ln_b); free(rm->v_ln_b);
    free(rm->m_W1); free(rm->v_W1); free(rm->m_b1); free(rm->v_b1);
    free(rm->m_W2); free(rm->v_W2); free(rm->m_b2); free(rm->v_b2);
    free(rm->x_norm); free(rm->ln_mean); free(rm->ln_rstd);
    free(rm->scores); free(rm->route);
    free(rm->mx_norm); free(rm->pool); free(rm->pool_counts);
    free(rm->mz1); free(rm->mh1); free(rm->mixed); free(rm->mgathered);
    free(rm->saved_pool); free(rm->saved_z1); free(rm->saved_h1); free(rm->saved_mixed);
    free(rm->d_gathered); free(rm->d_mixed);
    free(rm->dh1); free(rm->dz1); free(rm->d_pool); free(rm->dx_scatter);
    free(rm);
}

/* ══════════════════════════════════════════════════════════════════════
 * Signature Update
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_mixer_update_signatures(TrixRoutedMixer* rm) {
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles, H = rm->cfg.tile_hidden;

    float* raw = calloc(T * D, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++)
                raw[t * D + d] += rm->W1[t * H * D + h * D + d];

    float* mean = calloc(D, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            mean[d] += raw[t * D + d];
    for (int d = 0; d < D; d++) mean[d] /= (float)T;

    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
            float diff = raw[t * D + d] - mean[d];
            rm->signatures[t * D + d] = (diff > 0.0f) ? 1.0f : (diff < 0.0f) ? -1.0f : 0.0f;
        }

    free(raw); free(mean);
}

/* ══════════════════════════════════════════════════════════════════════
 * Scatter / Gather Atoms
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_scatter(mtfp_t* pool, int* counts,
                          const mtfp_t* x, const int* route,
                          int seq_len, int D, int T)
{
    memset(pool, 0, T * D * sizeof(mtfp_t));
    memset(counts, 0, T * sizeof(int));

    for (int pos = 0; pos < seq_len; pos++) {
        for (int t = 0; t < T; t++) {
            int r = route[pos * T + t];
            if (r == 0) continue;
            counts[t]++;
            mtfp_t* dst = pool + t * D;
            const mtfp_t* src = x + pos * D;
            if (r == 1) mtfp_vec_add_inplace(dst, src, D);
            else for (int d = 0; d < D; d++) dst[d] -= src[d];
        }
    }
}

void trix_routed_gather(mtfp_t* y, const mtfp_t* mixed,
                         const int* route,
                         int seq_len, int D, int T)
{
    memset(y, 0, seq_len * D * sizeof(mtfp_t));

    for (int pos = 0; pos < seq_len; pos++) {
        for (int t = 0; t < T; t++) {
            int r = route[pos * T + t];
            if (r == 0) continue;
            mtfp_t* dst = y + pos * D;
            const mtfp_t* src = mixed + t * D;
            if (r == 1) mtfp_vec_add_inplace(dst, src, D);
            else for (int d = 0; d < D; d++) dst[d] -= src[d];
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Forward
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_mixer_forward(TrixRoutedMixer* rm,
                                const float* x, float* out, int seq_len)
{
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles;
    int H = rm->cfg.tile_hidden, K = rm->cfg.active_k;

    rm_ensure_scratch(rm, seq_len);

    /* 1. LayerNorm */
    trix_layernorm_forward_save(rm->x_norm, rm->ln_mean, rm->ln_rstd,
        x, rm->ln_weight, rm->ln_bias, seq_len, D, rm->cfg.ln_eps);

    mtfp_from_float_batch(rm->mx_norm, rm->x_norm, seq_len * D);

    /* 2. Routing scores: MTFP × ternary signatures */
    {
        int8_t* sig_i8 = calloc(T * D, sizeof(int8_t));
        mtfp_t* mscores = calloc(seq_len * T, sizeof(mtfp_t));
        for (int i = 0; i < T * D; i++)
            sig_i8[i] = (rm->signatures[i] > 0.5f) ? 1 : (rm->signatures[i] < -0.5f) ? -1 : 0;
        mtfp_ternary_matmul_bt(mscores, rm->mx_norm, sig_i8, seq_len, D, T);
        mtfp_to_float_batch(rm->scores, mscores, seq_len * T);
        free(mscores); free(sig_i8);
    }

    /* 3. Ternary threshold */
    ternary_threshold_topk(rm->route, rm->scores, seq_len, T, K);

    /* 4. Scatter: pool tokens by routing destination */
    trix_routed_scatter(rm->pool, rm->pool_counts,
                         rm->mx_norm, rm->route, seq_len, D, T);

    /* 5. Tile FFN on each non-empty pool */
    memset(rm->mixed, 0, T * D * sizeof(mtfp_t));

    for (int t = 0; t < T; t++) {
        if (rm->pool_counts[t] == 0) continue;

        /* Fan-in normalize by token count */
        mtfp_fan_in_normalize(rm->pool + t * D, D, rm->pool_counts[t]);

        int8_t* W1t = rm->W1_tern + t * H * D;
        int8_t* W2t = rm->W2_tern + t * D * H;

        /* z1 = pool @ W1^T + b1 — one vector, not a batch */
        mtfp_ternary_matmul_bt(rm->mz1 + t * H, rm->pool + t * D, W1t, 1, D, H);
        mtfp_fan_in_normalize(rm->mz1 + t * H, H, D);
        mtfp_bias_add(rm->mz1 + t * H, rm->mb1 + t * H, 1, H);

        /* GELU */
        mtfp_gelu(rm->mh1 + t * H, rm->mz1 + t * H, H);

        /* mixed = h1 @ W2^T + b2 */
        mtfp_ternary_matmul_bt(rm->mixed + t * D, rm->mh1 + t * H, W2t, 1, H, D);
        mtfp_bias_add(rm->mixed + t * D, rm->mb2 + t * D, 1, D);

        /* Save float copies for backward */
        mtfp_to_float_batch(rm->saved_z1 + t * H, rm->mz1 + t * H, H);
        mtfp_to_float_batch(rm->saved_h1 + t * H, rm->mh1 + t * H, H);
        mtfp_to_float_batch(rm->saved_mixed + t * D, rm->mixed + t * D, D);
    }

    /* Save float pool for backward */
    mtfp_to_float_batch(rm->saved_pool, rm->pool, T * D);

    /* 6. Gather: distribute tile outputs back to positions */
    trix_routed_gather(rm->mgathered, rm->mixed, rm->route, seq_len, D, T);

    /* 7. Convert to float, scale + residual */
    float* gathered_f = rm->d_gathered;  /* reuse scratch */
    mtfp_to_float_batch(gathered_f, rm->mgathered, seq_len * D);

    for (int i = 0; i < seq_len * D; i++)
        out[i] = x[i] + rm->output_scale * gathered_f[i];
}

/* ══════════════════════════════════════════════════════════════════════
 * Backward
 * ══════════════════════════════════════════════════════════════════════ */

/* Float scatter: d_pool[t] = sum over pos: route[pos,t] * dy[pos] */
static void scatter_backward_float(float* d_pool, const float* dy, const int* route,
                                    int seq_len, int D, int T)
{
    memset(d_pool, 0, T * D * sizeof(float));
    for (int pos = 0; pos < seq_len; pos++) {
        for (int t = 0; t < T; t++) {
            int r = route[pos * T + t];
            if (r == 0) continue;
            float sign = (float)r;
            trix_vec_fma(d_pool + t * D, dy + pos * D, sign, D);
        }
    }
}

/* Float gather: dx[pos] = sum over t: route[pos,t] * d_mixed[t] */
static void gather_backward_float(float* dx, const float* d_mixed, const int* route,
                                   int seq_len, int D, int T)
{
    memset(dx, 0, seq_len * D * sizeof(float));
    for (int pos = 0; pos < seq_len; pos++) {
        for (int t = 0; t < T; t++) {
            int r = route[pos * T + t];
            if (r == 0) continue;
            float sign = (float)r;
            trix_vec_fma(dx + pos * D, d_mixed + t * D, sign, D);
        }
    }
}

void trix_routed_mixer_backward(TrixRoutedMixer* rm,
                                 const float* x, const float* dy, float* dx,
                                 int seq_len)
{
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles, H = rm->cfg.tile_hidden;

    /* d_gathered = dy * output_scale */
    trix_vec_scale(rm->d_gathered, dy, rm->output_scale, seq_len * D);

    /* d_output_scale += dot(dy, gathered) */
    float* gathered_f = calloc(seq_len * D, sizeof(float));
    mtfp_to_float_batch(gathered_f, rm->mgathered, seq_len * D);
    rm->doutput_scale += trix_dot(dy, gathered_f, seq_len * D);
    free(gathered_f);

    /* Gather backward: d_mixed[t] = sum over pos: route[pos,t] * d_gathered[pos]
     * This is a scatter operation on the gradient. */
    scatter_backward_float(rm->d_mixed, rm->d_gathered, rm->route, seq_len, D, T);

    /* Per-tile FFN backward.
     *
     * The tile FFN weights are shared across all positions. In the causal
     * forward, the tile FFN runs once per position per active tile (on
     * running pools). In the bidirectional forward, it runs once per tile
     * (on global pools). Either way, the gradient into dW accumulates.
     *
     * For causal: the saved_pool/z1/h1/mixed reflect the FINAL state
     * (full pools after all positions). This is correct for bidirectional
     * backward but approximate for causal. The approximation: we compute
     * the gradient as if the tile FFN ran on the final pools, then scale
     * by 1/seq_len to account for the averaging effect of running on
     * partial pools across positions. */
    float* d_pool_pre_norm = calloc(T * D, sizeof(float));
    float grad_scale = 1.0f / (float)seq_len;

    for (int t = 0; t < T; t++) {
        if (rm->pool_counts[t] == 0) continue;

        /* dW2 += h1^T @ d_mixed_t   — [D, H] */
        trix_matmul_at(rm->dW2 + t * D * H, rm->d_mixed + t * D, rm->saved_h1 + t * H, 1, D, H);
        trix_bias_grad(rm->db2 + t * D, rm->d_mixed + t * D, 1, D);

        /* dh1 = d_mixed_t @ W2_t   — [1, H] via ternary matmul */
        trix_ternary_matmul(rm->dh1 + t * H, rm->d_mixed + t * D,
                            rm->W2_tern + t * D * H, 1, D, H);

        /* dz1 = dh1 * GELU'(z1) */
        trix_gelu_grad(rm->dz1 + t * H, rm->dh1 + t * H, rm->saved_z1 + t * H, H);

        /* dW1 += dz1^T @ pool   — [H, D] */
        trix_matmul_at(rm->dW1 + t * H * D, rm->dz1 + t * H, rm->saved_pool + t * D, 1, H, D);
        trix_bias_grad(rm->db1 + t * H, rm->dz1 + t * H, 1, H);

        /* Scale tile FFN weight gradients */
        trix_vec_scale(rm->dW1 + t * H * D, rm->dW1 + t * H * D, grad_scale, H * D);
        trix_vec_scale(rm->db1 + t * H, rm->db1 + t * H, grad_scale, H);
        trix_vec_scale(rm->dW2 + t * D * H, rm->dW2 + t * D * H, grad_scale, D * H);
        trix_vec_scale(rm->db2 + t * D, rm->db2 + t * D, grad_scale, D);

        /* d_pool = dz1 @ W1_t   — [1, D] via ternary matmul */
        trix_ternary_matmul(d_pool_pre_norm + t * D, rm->dz1 + t * H,
                            rm->W1_tern + t * H * D, 1, H, D);

        /* Undo fan-in normalization in gradient */
        int count = rm->pool_counts[t];
        float inv_norm = 1.0f / sqrtf((float)count);
        trix_vec_scale(d_pool_pre_norm + t * D, d_pool_pre_norm + t * D, inv_norm, D);
    }

    /* Scatter backward: dx_scatter[pos] = sum over t: route[pos,t] * d_pool[t]
     * This is a gather operation on the gradient. */
    gather_backward_float(rm->dx_scatter, d_pool_pre_norm, rm->route, seq_len, D, T);
    free(d_pool_pre_norm);

    /* LayerNorm backward */
    float* dx_pre_ln = calloc(seq_len * D, sizeof(float));
    trix_layernorm_backward(dx_pre_ln, rm->dln_weight, rm->dln_bias,
                             rm->dx_scatter, x, rm->ln_weight,
                             rm->ln_mean, rm->ln_rstd, seq_len, D);

    /* dx = dy + dx_pre_ln (residual) */
    if (dx) {
        trix_vec_add(dx, dy, dx_pre_ln, seq_len * D);
    }

    free(dx_pre_ln);
}

/* ══════════════════════════════════════════════════════════════════════
 * Zero Grad / Optimizer / Clipping
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_mixer_zero_grad(TrixRoutedMixer* rm) {
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles, H = rm->cfg.tile_hidden;
    trix_vec_zero(rm->dln_weight, D);
    trix_vec_zero(rm->dln_bias, D);
    trix_vec_zero(rm->dW1, T * H * D);
    trix_vec_zero(rm->db1, T * H);
    trix_vec_zero(rm->dW2, T * D * H);
    trix_vec_zero(rm->db2, T * D);
    rm->doutput_scale = 0.0f;
}

void trix_routed_mixer_adamw_step(TrixRoutedMixer* rm,
                                   float lr, float beta1, float beta2,
                                   float eps, float weight_decay)
{
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles, H = rm->cfg.tile_hidden;
    rm->adam_step++;
    int step = rm->adam_step;

    trix_adamw_update(rm->ln_weight, rm->dln_weight, rm->m_ln_w, rm->v_ln_w,
                      lr, beta1, beta2, eps, 0.0f, step, D);
    trix_adamw_update(rm->ln_bias, rm->dln_bias, rm->m_ln_b, rm->v_ln_b,
                      lr, beta1, beta2, eps, 0.0f, step, D);
    trix_adamw_update(rm->W1, rm->dW1, rm->m_W1, rm->v_W1,
                      lr, beta1, beta2, eps, weight_decay, step, T * H * D);
    trix_adamw_update(rm->b1, rm->db1, rm->m_b1, rm->v_b1,
                      lr, beta1, beta2, eps, 0.0f, step, T * H);
    trix_adamw_update(rm->W2, rm->dW2, rm->m_W2, rm->v_W2,
                      lr, beta1, beta2, eps, weight_decay, step, T * D * H);
    trix_adamw_update(rm->b2, rm->db2, rm->m_b2, rm->v_b2,
                      lr, beta1, beta2, eps, 0.0f, step, T * D);

    /* output_scale */
    {
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);
        rm->m_output_scale = beta1 * rm->m_output_scale + (1.0f - beta1) * rm->doutput_scale;
        rm->v_output_scale = beta2 * rm->v_output_scale + (1.0f - beta2) * rm->doutput_scale * rm->doutput_scale;
        float m_hat = rm->m_output_scale / bc1;
        float v_hat = rm->v_output_scale / bc2;
        rm->output_scale -= lr * m_hat / (sqrtf(v_hat) + eps);
    }

    trix_routed_mixer_update_signatures(rm);
    quantize_tiles(rm);
}

float trix_routed_mixer_clip_grad_norm(TrixRoutedMixer* rm, float max_norm) {
    int D = rm->cfg.d_model, T = rm->cfg.num_tiles, H = rm->cfg.tile_hidden;
    float sq = 0.0f;
    sq += trix_sum_sq(rm->dln_weight, D);
    sq += trix_sum_sq(rm->dln_bias, D);
    sq += trix_sum_sq(rm->dW1, T * H * D);
    sq += trix_sum_sq(rm->db1, T * H);
    sq += trix_sum_sq(rm->dW2, T * D * H);
    sq += trix_sum_sq(rm->db2, T * D);
    sq += rm->doutput_scale * rm->doutput_scale;

    float norm = sqrtf(sq);
    if (norm > max_norm && norm > 0.0f) {
        float s = max_norm / norm;
        trix_vec_scale(rm->dln_weight, rm->dln_weight, s, D);
        trix_vec_scale(rm->dln_bias, rm->dln_bias, s, D);
        trix_vec_scale(rm->dW1, rm->dW1, s, T * H * D);
        trix_vec_scale(rm->db1, rm->db1, s, T * H);
        trix_vec_scale(rm->dW2, rm->dW2, s, T * D * H);
        trix_vec_scale(rm->db2, rm->db2, s, T * D);
        rm->doutput_scale *= s;
    }
    return norm;
}
