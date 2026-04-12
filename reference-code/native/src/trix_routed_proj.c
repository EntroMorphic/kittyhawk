/*
 * trix_routed_proj.c — Ternary-routed linear projection
 *
 * Same routing mechanism as trix_ternary_route.c (k-of-T, weight-derived
 * signatures, signed composition) applied to a linear projection.
 *
 * Used for: routed QKV, routed W_O, routed LM head.
 */

#include "trix_routed_proj.h"
#include "trix_ternary_matmul.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Binarize float weights to ternary {-1,0,+1} — same as trix_ternary_route.c */
static void rp_binarize(int8_t* dst, const float* src, int n) {
    float mean_abs = 0.0f;
    for (int i = 0; i < n; i++) mean_abs += fabsf(src[i]);
    mean_abs /= (float)n;
    float thresh = mean_abs * 0.5f;
    for (int i = 0; i < n; i++)
        dst[i] = (src[i] > thresh) ? 1 : (src[i] < -thresh) ? -1 : 0;
}

/* Quantize weights to ternary and convert biases to MTFP */
static void rp_quantize_weights(TrixRoutedProj* rp) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    if (rp->W_tern) {
        for (int t = 0; t < T; t++)
            rp_binarize(rp->W_tern + t * O * I, rp->W + t * O * I, O * I);
    }
    if (rp->mb)
        mtfp_from_float_batch(rp->mb, rp->b, T * O);
    if (rp->mln_w && rp->ln_weight)
        mtfp_from_float_batch(rp->mln_w, rp->ln_weight, rp->cfg.in_dim);
    if (rp->mln_b && rp->ln_bias)
        mtfp_from_float_batch(rp->mln_b, rp->ln_bias, rp->cfg.in_dim);
}

/* ── PRNG (same xoshiro128+ as everywhere else) ── */
static inline uint32_t rp_rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
typedef struct { uint32_t s[4]; } rp_rng_t;
static uint32_t rp_rng_next(rp_rng_t* r) {
    uint32_t result = r->s[0] + r->s[3], t = r->s[1] << 9;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1]; r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = rp_rotl(r->s[3], 11); return result;
}
static float rp_rng_uniform(rp_rng_t* r) { return (float)(rp_rng_next(r) >> 8) / 16777216.0f; }
static rp_rng_t rp_rng_seed(uint64_t seed) {
    rp_rng_t r; r.s[0]=(uint32_t)seed; r.s[1]=(uint32_t)(seed>>32);
    r.s[2]=(uint32_t)(seed*2654435761ULL); r.s[3]=(uint32_t)((seed*2654435761ULL)>>32);
    for(int i=0;i<16;i++) rp_rng_next(&r); return r;
}
static void rp_xavier_init(float* w, int fan_in, int fan_out, int n, rp_rng_t* rng) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < n; i++) w[i] = (2.0f * rp_rng_uniform(rng) - 1.0f) * limit;
}

/* ── Ternary threshold (same as trix_ternary_route.c) ── */
static void ternary_threshold_topk(int* route, const float* scores, int batch, int T, int k) {
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

/* ── Scratch ── */
static void rp_ensure_scratch(TrixRoutedProj* rp, int batch) {
    if (batch <= rp->batch_cap) return;
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;

    free(rp->x_norm); free(rp->ln_mean); free(rp->ln_rstd);
    free(rp->scores); free(rp->route);
    free(rp->tile_out); free(rp->combined);
    free(rp->d_combined); free(rp->d_tile_out); free(rp->dx_accum);
    free(rp->dW_tmp); free(rp->dx_tmp); free(rp->dx_pre_ln);
    free(rp->mx_norm); free(rp->mtile_out); free(rp->mcombined);

    rp->x_norm    = calloc(batch * I, sizeof(float));
    rp->ln_mean   = calloc(batch, sizeof(float));
    rp->ln_rstd   = calloc(batch, sizeof(float));
    rp->scores    = calloc(batch * T, sizeof(float));
    rp->route     = calloc(batch * T, sizeof(int));
    rp->tile_out  = calloc(batch * O, sizeof(float));
    rp->combined  = calloc(batch * O, sizeof(float));
    rp->d_combined = calloc(batch * O, sizeof(float));
    rp->d_tile_out = calloc(batch * O, sizeof(float));
    rp->dx_accum  = calloc(batch * I, sizeof(float));
    rp->dW_tmp    = calloc(O * I, sizeof(float));
    rp->dx_tmp    = calloc(batch * I, sizeof(float));
    rp->dx_pre_ln = calloc(batch * I, sizeof(float));

    /* MTFP scratch */
    rp->mx_norm   = calloc(batch * I, sizeof(mtfp_t));
    rp->mtile_out = calloc(batch * O, sizeof(mtfp_t));
    rp->mcombined = calloc(batch * O, sizeof(mtfp_t));

    rp->batch_cap = batch;
}

/* ══════════════════════════════════════════════════════════════════════
 * Create / Destroy
 * ══════════════════════════════════════════════════════════════════════ */

TrixRoutedProj* trix_routed_proj_create(TrixRoutedProjConfig cfg, uint64_t seed) {
    TrixRoutedProj* rp = calloc(1, sizeof(TrixRoutedProj));
    if (!rp) return NULL;
    rp->cfg = cfg;
    int I = cfg.in_dim, O = cfg.out_dim, T = cfg.num_tiles;
    rp_rng_t rng = rp_rng_seed(seed);

    /* LayerNorm */
    if (cfg.use_layernorm) {
        rp->ln_weight = calloc(I, sizeof(float));
        rp->ln_bias   = calloc(I, sizeof(float));
        for (int i = 0; i < I; i++) rp->ln_weight[i] = 1.0f;
        rp->dln_weight = calloc(I, sizeof(float));
        rp->dln_bias   = calloc(I, sizeof(float));
        rp->m_ln_w = calloc(I, sizeof(float)); rp->v_ln_w = calloc(I, sizeof(float));
        rp->m_ln_b = calloc(I, sizeof(float)); rp->v_ln_b = calloc(I, sizeof(float));
    }

    /* Per-tile weights: Xavier init */
    rp->W = calloc(T * O * I, sizeof(float));
    rp->b = calloc(T * O, sizeof(float));
    for (int t = 0; t < T; t++)
        rp_xavier_init(rp->W + t * O * I, I, O, O * I, &rng);

    /* Signatures */
    rp->signatures = calloc(T * I, sizeof(float));

    /* Ternary weights + MTFP biases + LN weights */
    rp->W_tern = calloc(T * O * I, sizeof(int8_t));
    rp->mb = calloc(T * O, sizeof(mtfp_t));
    if (cfg.use_layernorm) {
        rp->mln_w = calloc(I, sizeof(mtfp_t));
        rp->mln_b = calloc(I, sizeof(mtfp_t));
    }

    /* Output scale */
    rp->output_scale = cfg.output_scale_init;

    /* Gradients */
    rp->dW = calloc(T * O * I, sizeof(float));
    rp->db = calloc(T * O, sizeof(float));

    /* AdamW moments */
    rp->m_W = calloc(T * O * I, sizeof(float)); rp->v_W = calloc(T * O * I, sizeof(float));
    rp->m_b = calloc(T * O, sizeof(float));     rp->v_b = calloc(T * O, sizeof(float));

    rp->batch_cap = 0;
    rp->adam_step = 0;

    trix_routed_proj_update_signatures(rp);
    rp_quantize_weights(rp);
    return rp;
}

void trix_routed_proj_destroy(TrixRoutedProj* rp) {
    if (!rp) return;
    free(rp->ln_weight); free(rp->ln_bias);
    free(rp->dln_weight); free(rp->dln_bias);
    free(rp->m_ln_w); free(rp->v_ln_w);
    free(rp->m_ln_b); free(rp->v_ln_b);
    free(rp->W); free(rp->b); free(rp->signatures);
    free(rp->W_tern); free(rp->mb); free(rp->mln_w); free(rp->mln_b);
    free(rp->dW); free(rp->db);
    free(rp->m_W); free(rp->v_W); free(rp->m_b); free(rp->v_b);
    free(rp->x_norm); free(rp->ln_mean); free(rp->ln_rstd);
    free(rp->scores); free(rp->route);
    free(rp->tile_out); free(rp->combined);
    free(rp->d_combined); free(rp->d_tile_out); free(rp->dx_accum);
    free(rp->dW_tmp); free(rp->dx_tmp); free(rp->dx_pre_ln);
    free(rp->mx_norm); free(rp->mtile_out); free(rp->mcombined);
    free(rp);
}

/* ══════════════════════════════════════════════════════════════════════
 * Signature Update
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_proj_update_signatures(TrixRoutedProj* rp) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    int sig_cols = rp->cfg.sig_cols > 0 ? rp->cfg.sig_cols : O;
    if (sig_cols > O) sig_cols = O;

    /* raw_t[d] = sum over first sig_cols rows of W_t[:, d]
     * W_t is [O, I], row-major. Element (o, d) at W_t[o * I + d]. */
    float* raw = calloc(T * I, sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < sig_cols; o++) {
            for (int d = 0; d < I; d++) {
                raw[t * I + d] += rp->W[t * O * I + o * I + d];
            }
        }
    }

    /* mean[d] = mean_over_t(raw_t[d]) */
    float* mean = calloc(I, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int d = 0; d < I; d++)
            mean[d] += raw[t * I + d];
    for (int d = 0; d < I; d++) mean[d] /= (float)T;

    /* sig_t[d] = sign(raw_t[d] - mean[d]) */
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < I; d++) {
            float diff = raw[t * I + d] - mean[d];
            rp->signatures[t * I + d] = (diff > 0.0f) ? 1.0f : (diff < 0.0f) ? -1.0f : 0.0f;
        }
    }

    free(raw); free(mean);
}

/* ══════════════════════════════════════════════════════════════════════
 * Forward
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_proj_forward(TrixRoutedProj* rp, const float* x, float* out, int batch) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    int K = rp->cfg.active_k;
    rp_ensure_scratch(rp, batch);

    /* 1. LayerNorm (optional) */
    const float* input;
    if (rp->ln_weight) {
        trix_layernorm_forward_save(rp->x_norm, rp->ln_mean, rp->ln_rstd,
                                     x, rp->ln_weight, rp->ln_bias,
                                     batch, I, rp->cfg.ln_eps);
        input = rp->x_norm;
    } else {
        memcpy(rp->x_norm, x, batch * I * sizeof(float)); /* save for backward */
        input = x;
    }

    /* 2. Routing scores: scores[batch, T] = input[batch, I] @ sigs[T, I]^T */
    trix_matmul_bt(rp->scores, input, rp->signatures, batch, I, T);

    /* 3. Ternary threshold */
    ternary_threshold_topk(rp->route, rp->scores, batch, T, K);

    /* 4. Signed tile sum */
    trix_vec_zero(rp->combined, batch * O);
    for (int t = 0; t < T; t++) {
        float* W_t = rp->W + t * O * I;

        /* tile_out = input @ W_t^T + b_t   [batch, O] */
        trix_matmul_bt(rp->tile_out, input, W_t, batch, I, O);
        trix_bias_add(rp->tile_out, rp->b + t * O, batch, O);

        /* combined += route_t * tile_out */
        for (int i = 0; i < batch; i++) {
            int r = rp->route[i * T + t];
            if (r == 0) continue;
            trix_vec_fma(rp->combined + i * O, rp->tile_out + i * O, (float)r, O);
        }
    }

    /* 5. Scale (no residual — caller handles that) */
    for (int i = 0; i < batch * O; i++)
        out[i] = rp->output_scale * rp->combined[i];
}

/* ══════════════════════════════════════════════════════════════════════
 * Forward (MTFP native — zero float in tile computation)
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_proj_forward_mtfp(TrixRoutedProj* rp, const mtfp_t* x, mtfp_t* out, int batch) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    int K = rp->cfg.active_k;
    rp_ensure_scratch(rp, batch);

    /* 1. LayerNorm (optional) — MTFP with integer sqrt */
    if (rp->mln_w) {
        mtfp_layernorm(rp->mx_norm, x, rp->mln_w, rp->mln_b, batch, I);
    } else {
        memcpy(rp->mx_norm, x, batch * I * sizeof(mtfp_t));
    }

    /* Save float for backward */
    mtfp_to_float_batch(rp->x_norm, rp->mx_norm, batch * I);
    if (rp->ln_weight) {
        float* x_f = calloc(batch * I, sizeof(float));
        mtfp_to_float_batch(x_f, x, batch * I);
        trix_layernorm_forward_save(rp->x_norm, rp->ln_mean, rp->ln_rstd,
                                     x_f, rp->ln_weight, rp->ln_bias,
                                     batch, I, rp->cfg.ln_eps);
        free(x_f);
    }

    /* 2. Routing scores — MTFP × ternary = integer add/sub */
    {
        int8_t* sig_i8 = calloc(T * I, sizeof(int8_t));
        for (int i = 0; i < T * I; i++)
            sig_i8[i] = (rp->signatures[i] > 0.5f) ? 1 : (rp->signatures[i] < -0.5f) ? -1 : 0;
        mtfp_t* mscores = calloc(batch * T, sizeof(mtfp_t));
        mtfp_ternary_matmul_bt(mscores, rp->mx_norm, sig_i8, batch, I, T);
        mtfp_to_float_batch(rp->scores, mscores, batch * T);
        free(mscores); free(sig_i8);
    }

    /* 3. Ternary threshold */
    ternary_threshold_topk(rp->route, rp->scores, batch, T, K);

    /* 4. MTFP tile computation — zero float multiplies */
    memset(rp->mcombined, 0, batch * O * sizeof(mtfp_t));
    for (int t = 0; t < T; t++) {
        int8_t* W_t = rp->W_tern + t * O * I;

        mtfp_ternary_matmul_bt(rp->mtile_out, rp->mx_norm, W_t, batch, I, O);
        mtfp_fan_in_normalize(rp->mtile_out, batch * O, I);
        mtfp_bias_add(rp->mtile_out, rp->mb + t * O, batch, O);

        for (int i = 0; i < batch; i++) {
            int r = rp->route[i * T + t];
            if (r == 0) continue;
            mtfp_t* dst = rp->mcombined + i * O;
            mtfp_t* src = rp->mtile_out + i * O;
            if (r == 1) mtfp_vec_add_inplace(dst, src, O);
            else for (int j = 0; j < O; j++) dst[j] -= src[j];
        }
    }

    /* 5. Scale — MTFP multiply */
    mtfp_t mscale = mtfp_from_float(rp->output_scale);
    for (int i = 0; i < batch * O; i++)
        out[i] = mtfp_mul(mscale, rp->mcombined[i]);

    /* Save float combined for backward */
    mtfp_to_float_batch(rp->combined, rp->mcombined, batch * O);
}

/* ══════════════════════════════════════════════════════════════════════
 * Backward
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_proj_backward(TrixRoutedProj* rp, const float* x, const float* dy, float* dx, int batch) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;

    /* d_combined = dy * output_scale */
    trix_vec_scale(rp->d_combined, dy, rp->output_scale, batch * O);

    /* d_output_scale += dot(dy, combined) */
    rp->doutput_scale += trix_dot(dy, rp->combined, batch * O);

    /* Per-tile backward */
    trix_vec_zero(rp->dx_accum, batch * I);

    for (int t = 0; t < T; t++) {
        float* W_t = rp->W + t * O * I;

        /* d_tile_out[i] = route[i,t] * d_combined[i] */
        for (int i = 0; i < batch; i++) {
            int r = rp->route[i * T + t];
            float sign = (float)r;
            for (int j = 0; j < O; j++)
                rp->d_tile_out[i * O + j] = sign * rp->d_combined[i * O + j];
        }

        /* dW_t += d_tile_out^T @ x_norm   [O, I] */
        trix_matmul_at(rp->dW_tmp, rp->d_tile_out, rp->x_norm, batch, O, I);
        trix_vec_add_inplace(rp->dW + t * O * I, rp->dW_tmp, O * I);

        /* db_t += sum_batch(d_tile_out) */
        trix_bias_grad(rp->db + t * O, rp->d_tile_out, batch, O);

        /* dx_accum += d_tile_out @ W_t   [batch, I] */
        trix_matmul(rp->dx_tmp, rp->d_tile_out, W_t, batch, O, I);
        trix_vec_add_inplace(rp->dx_accum, rp->dx_tmp, batch * I);
    }

    /* LayerNorm backward (if enabled) */
    if (rp->ln_weight && dx) {
        trix_layernorm_backward(rp->dx_pre_ln, rp->dln_weight, rp->dln_bias,
                                 rp->dx_accum, x, rp->ln_weight,
                                 rp->ln_mean, rp->ln_rstd, batch, I);
        memcpy(dx, rp->dx_pre_ln, batch * I * sizeof(float));
    } else if (dx) {
        memcpy(dx, rp->dx_accum, batch * I * sizeof(float));
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Optimizer
 * ══════════════════════════════════════════════════════════════════════ */

void trix_routed_proj_zero_grad(TrixRoutedProj* rp) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    memset(rp->dW, 0, T * O * I * sizeof(float));
    memset(rp->db, 0, T * O * sizeof(float));
    rp->doutput_scale = 0.0f;
    if (rp->dln_weight) {
        memset(rp->dln_weight, 0, I * sizeof(float));
        memset(rp->dln_bias, 0, I * sizeof(float));
    }
}

void trix_routed_proj_adamw_step(TrixRoutedProj* rp, float lr, float b1, float b2, float eps, float wd) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    rp->adam_step++;
    int step = rp->adam_step;

    if (rp->ln_weight) {
        trix_adamw_update(rp->ln_weight, rp->dln_weight, rp->m_ln_w, rp->v_ln_w,
                           lr, b1, b2, eps, 0.0f, step, I);
        trix_adamw_update(rp->ln_bias, rp->dln_bias, rp->m_ln_b, rp->v_ln_b,
                           lr, b1, b2, eps, 0.0f, step, I);
    }
    trix_adamw_update(rp->W, rp->dW, rp->m_W, rp->v_W,
                       lr, b1, b2, eps, wd, step, T * O * I);
    trix_adamw_update(rp->b, rp->db, rp->m_b, rp->v_b,
                       lr, b1, b2, eps, 0.0f, step, T * O);

    /* output_scale */
    {
        float bc1 = 1.0f - powf(b1, (float)step);
        float bc2 = 1.0f - powf(b2, (float)step);
        rp->m_os = b1 * rp->m_os + (1.0f - b1) * rp->doutput_scale;
        rp->v_os = b2 * rp->v_os + (1.0f - b2) * rp->doutput_scale * rp->doutput_scale;
        float mh = rp->m_os / bc1, vh = rp->v_os / bc2;
        rp->output_scale -= lr * mh / (sqrtf(vh) + eps);
    }

    /* Update signatures and re-quantize ternary weights */
    trix_routed_proj_update_signatures(rp);
    rp_quantize_weights(rp);
}

float trix_routed_proj_grad_sq(const TrixRoutedProj* rp) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    float sq = 0.0f;
    sq += trix_sum_sq(rp->dW, T * O * I);
    sq += trix_sum_sq(rp->db, T * O);
    sq += rp->doutput_scale * rp->doutput_scale;
    if (rp->dln_weight) {
        sq += trix_sum_sq(rp->dln_weight, I);
        sq += trix_sum_sq(rp->dln_bias, I);
    }
    return sq;
}

void trix_routed_proj_scale_grad(TrixRoutedProj* rp, float scale) {
    int I = rp->cfg.in_dim, O = rp->cfg.out_dim, T = rp->cfg.num_tiles;
    trix_vec_scale(rp->dW, rp->dW, scale, T * O * I);
    trix_vec_scale(rp->db, rp->db, scale, T * O);
    rp->doutput_scale *= scale;
    if (rp->dln_weight) {
        trix_vec_scale(rp->dln_weight, rp->dln_weight, scale, I);
        trix_vec_scale(rp->dln_bias, rp->dln_bias, scale, I);
    }
}
