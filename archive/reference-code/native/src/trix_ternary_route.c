/*
 * trix_ternary_route.c — Ternary-routed FFN implementation
 *
 * All tile computation uses ternary weights via SDOT kernel — zero float
 * multiplies in the forward path. Float32 shadow weights are maintained
 * for gradient updates (STE through the quantization).
 *
 * Forward: quantize(W) → packed ternary → SDOT matvec (multiply-free)
 * Backward dx: same ternary matvec (multiply-free)
 * Backward dW: float32 matmul (gradient into shadow weights, via Accelerate/AMX)
 */

#include "trix_ternary_route.h"
#include "trix_ternary_matvec.h"
#include "trix_ternary_matmul.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

/* Binarize float32 weights to ternary int8 {-1, 0, +1}.
 * Threshold: values within ±0.5*mean_abs are zero. */
static void binarize_to_ternary(int8_t* dst, const float* src, int n) {
    float mean_abs = 0.0f;
    for (int i = 0; i < n; i++) mean_abs += fabsf(src[i]);
    mean_abs /= (float)n;
    float thresh = mean_abs * 0.5f;
    for (int i = 0; i < n; i++)
        dst[i] = (src[i] > thresh) ? 1 : (src[i] < -thresh) ? -1 : 0;
}

/* Quantize all tile weights to ternary, pack for SDOT, and convert biases to MTFP */
static void quantize_and_pack_tiles(TrixTernaryRoutedFFN* tr) {
    int D = tr->cfg.d_model, T = tr->cfg.num_tiles, H = tr->cfg.tile_hidden;
    int w1_elems = H * D, w2_elems = D * H;
    int w1_packed = H * ((D + 3) / 4);
    int w2_packed = D * ((H + 3) / 4);
    for (int t = 0; t < T; t++) {
        binarize_to_ternary(tr->W1_tern + t * w1_elems, tr->W1 + t * w1_elems, w1_elems);
        trix_ternary_pack_weights_i8(tr->W1_packed + t * w1_packed, tr->W1_tern + t * w1_elems, H, D);
        binarize_to_ternary(tr->W2_tern + t * w2_elems, tr->W2 + t * w2_elems, w2_elems);
        trix_ternary_pack_weights_i8(tr->W2_packed + t * w2_packed, tr->W2_tern + t * w2_elems, D, H);
    }
    /* Convert biases and LN weights to MTFP */
    if (tr->mb1) mtfp_from_float_batch(tr->mb1, tr->b1, T * H);
    if (tr->mb2) mtfp_from_float_batch(tr->mb2, tr->b2, T * D);
    if (tr->mln_weight) mtfp_from_float_batch(tr->mln_weight, tr->ln_weight, D);
    if (tr->mln_bias) mtfp_from_float_batch(tr->mln_bias, tr->ln_bias, D);
}

/* Quantize float activations to int8 for SDOT */
static void quantize_to_i8(int8_t* dst, const float* src, int n, float* inv_scale_out) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(src[i]); if (a > max_abs) max_abs = a; }
    if (max_abs < 1e-8f) { memset(dst, 0, (size_t)n); *inv_scale_out = 0.0f; return; }
    float scale = 127.0f / max_abs;
    *inv_scale_out = max_abs / 127.0f;
    for (int i = 0; i < n; i++) {
        float v = src[i] * scale;
        if (v > 127.0f) v = 127.0f; if (v < -127.0f) v = -127.0f;
        dst[i] = (int8_t)lrintf(v);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * PRNG (same as trix_multitrit.c)
 * ══════════════════════════════════════════════════════════════════════ */

static inline uint32_t tr_rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
typedef struct { uint32_t s[4]; } tr_rng_t;
static uint32_t tr_rng_next(tr_rng_t* r) {
    uint32_t result = r->s[0] + r->s[3];
    uint32_t t = r->s[1] << 9;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = tr_rotl(r->s[3], 11);
    return result;
}
static float tr_rng_uniform(tr_rng_t* r) { return (float)(tr_rng_next(r) >> 8) / 16777216.0f; }
static tr_rng_t tr_rng_seed(uint64_t seed) {
    tr_rng_t r;
    r.s[0] = (uint32_t)seed; r.s[1] = (uint32_t)(seed >> 32);
    r.s[2] = (uint32_t)(seed * 2654435761ULL); r.s[3] = (uint32_t)((seed * 2654435761ULL) >> 32);
    for (int i = 0; i < 16; i++) tr_rng_next(&r);
    return r;
}
static void tr_xavier_init(float* w, int fan_in, int fan_out, int n, tr_rng_t* rng) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < n; i++) w[i] = (2.0f * tr_rng_uniform(rng) - 1.0f) * limit;
}

/* ══════════════════════════════════════════════════════════════════════
 * Ternary Routing: threshold scores to {-1, 0, +1}
 *
 * For each token, compute |score| for all tiles, pick top-k,
 * assign route = sign(score) for those k tiles, 0 for the rest.
 * ══════════════════════════════════════════════════════════════════════ */

static void ternary_threshold_topk(
    int* route,             /* [batch, T] output: -1, 0, +1 */
    const float* scores,    /* [batch, T] input: raw dot products */
    int batch, int T, int k)
{
    for (int i = 0; i < batch; i++) {
        const float* s = scores + i * T;
        int* r = route + i * T;

        /* Zero all routes */
        memset(r, 0, T * sizeof(int));

        /* Find top-k by |score| using simple selection */
        /* For small T (4-8), brute force is fine */
        float abs_scores[64]; /* T <= 64 */
        for (int t = 0; t < T; t++) abs_scores[t] = fabsf(s[t]);

        for (int j = 0; j < k && j < T; j++) {
            int best = -1;
            float best_val = -1.0f;
            for (int t = 0; t < T; t++) {
                if (abs_scores[t] > best_val) {
                    best_val = abs_scores[t];
                    best = t;
                }
            }
            if (best >= 0) {
                r[best] = (s[best] >= 0.0f) ? 1 : -1;
                abs_scores[best] = -1.0f; /* mark as used */
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Scratch allocation
 * ══════════════════════════════════════════════════════════════════════ */

static void tr_ensure_scratch(TrixTernaryRoutedFFN* tr, int batch) {
    if (batch <= tr->batch_cap) return;
    int D = tr->cfg.d_model;
    int T = tr->cfg.num_tiles;
    int H = tr->cfg.tile_hidden;

    free(tr->x_norm); free(tr->ln_mean); free(tr->ln_rstd);
    free(tr->scores); free(tr->route);
    free(tr->z1); free(tr->h1);
    free(tr->tile_out); free(tr->combined);
    free(tr->d_combined); free(tr->d_tile_out);
    free(tr->dh1); free(tr->dz1); free(tr->dx_tile);
    free(tr->mx_norm); free(tr->mz1); free(tr->mh1);
    free(tr->mtile_out); free(tr->mcombined);

    if (tr->saved_z1) {
        for (int t = 0; t < T; t++) { free(tr->saved_z1[t]); free(tr->saved_h1[t]); }
    }
    free(tr->saved_z1); free(tr->saved_h1);

    tr->x_norm     = calloc(batch * D, sizeof(float));
    tr->ln_mean    = calloc(batch, sizeof(float));
    tr->ln_rstd    = calloc(batch, sizeof(float));
    tr->scores     = calloc(batch * T, sizeof(float));
    tr->route      = calloc(batch * T, sizeof(int));
    tr->z1         = calloc(batch * H, sizeof(float));
    tr->h1         = calloc(batch * H, sizeof(float));
    tr->tile_out   = calloc(batch * D, sizeof(float));
    tr->combined   = calloc(batch * D, sizeof(float));
    tr->d_combined = calloc(batch * D, sizeof(float));
    tr->d_tile_out = calloc(batch * D, sizeof(float));
    tr->dh1        = calloc(batch * H, sizeof(float));
    tr->dz1        = calloc(batch * H, sizeof(float));
    tr->dx_tile    = calloc(batch * D, sizeof(float));

    /* MTFP scratch */
    tr->mx_norm    = calloc(batch * D, sizeof(mtfp_t));
    tr->mz1        = calloc(batch * H, sizeof(mtfp_t));
    tr->mh1        = calloc(batch * H, sizeof(mtfp_t));
    tr->mtile_out  = calloc(batch * D, sizeof(mtfp_t));
    tr->mcombined  = calloc(batch * D, sizeof(mtfp_t));

    tr->saved_z1 = calloc(T, sizeof(float*));
    tr->saved_h1 = calloc(T, sizeof(float*));
    for (int t = 0; t < T; t++) {
        tr->saved_z1[t] = calloc(batch * H, sizeof(float));
        tr->saved_h1[t] = calloc(batch * H, sizeof(float));
    }

    tr->batch_cap = batch;
}

/* ══════════════════════════════════════════════════════════════════════
 * Create / Destroy
 * ══════════════════════════════════════════════════════════════════════ */

TrixTernaryRoutedFFN* trix_ternary_route_create(
    TrixTernaryRouteConfig cfg, uint64_t seed)
{
    TrixTernaryRoutedFFN* tr = calloc(1, sizeof(TrixTernaryRoutedFFN));
    if (!tr) return NULL;
    tr->cfg = cfg;

    int D = cfg.d_model, T = cfg.num_tiles, H = cfg.tile_hidden;
    tr_rng_t rng = tr_rng_seed(seed);

    /* LayerNorm: weight=1, bias=0 */
    tr->ln_weight = calloc(D, sizeof(float));
    tr->ln_bias   = calloc(D, sizeof(float));
    for (int i = 0; i < D; i++) tr->ln_weight[i] = 1.0f;

    /* Signatures: random init, will be updated from weights */
    tr->signatures = calloc(T * D, sizeof(float));
    tr_xavier_init(tr->signatures, D, T, T * D, &rng);

    /* Per-tile weights */
    tr->W1 = calloc(T * H * D, sizeof(float));
    tr->b1 = calloc(T * H, sizeof(float));
    tr->W2 = calloc(T * D * H, sizeof(float));
    tr->b2 = calloc(T * D, sizeof(float));
    for (int t = 0; t < T; t++) {
        tr_xavier_init(tr->W1 + t * H * D, D, H, H * D, &rng);
        tr_xavier_init(tr->W2 + t * D * H, H, D, D * H, &rng);
    }
    tr->output_scale = cfg.output_scale_init;

    /* MTFP LN weights + biases — converted from float, used in MTFP forward path */
    tr->mln_weight = calloc(D, sizeof(mtfp_t));
    tr->mln_bias = calloc(D, sizeof(mtfp_t));
    tr->mb1 = calloc(T * H, sizeof(mtfp_t));
    tr->mb2 = calloc(T * D, sizeof(mtfp_t));

    /* Ternary packed weights for SDOT kernel */
    tr->W1_tern   = calloc(T * H * D, sizeof(int8_t));
    tr->W2_tern   = calloc(T * D * H, sizeof(int8_t));
    tr->W1_packed = calloc((size_t)T * H * ((D + 3) / 4), sizeof(uint8_t));
    tr->W2_packed = calloc((size_t)T * D * ((H + 3) / 4), sizeof(uint8_t));

    /* Gradients */
    tr->dln_weight = calloc(D, sizeof(float));
    tr->dln_bias   = calloc(D, sizeof(float));
    tr->dW1 = calloc(T * H * D, sizeof(float));
    tr->db1 = calloc(T * H, sizeof(float));
    tr->dW2 = calloc(T * D * H, sizeof(float));
    tr->db2 = calloc(T * D, sizeof(float));

    /* AdamW moments */
    tr->m_ln_w = calloc(D, sizeof(float)); tr->v_ln_w = calloc(D, sizeof(float));
    tr->m_ln_b = calloc(D, sizeof(float)); tr->v_ln_b = calloc(D, sizeof(float));
    tr->m_W1 = calloc(T*H*D, sizeof(float)); tr->v_W1 = calloc(T*H*D, sizeof(float));
    tr->m_b1 = calloc(T*H, sizeof(float));   tr->v_b1 = calloc(T*H, sizeof(float));
    tr->m_W2 = calloc(T*D*H, sizeof(float)); tr->v_W2 = calloc(T*D*H, sizeof(float));
    tr->m_b2 = calloc(T*D, sizeof(float));   tr->v_b2 = calloc(T*D, sizeof(float));

    tr->batch_cap = 0;
    tr->adam_step = 0;

    /* Initialize GELU lookup table (once, shared across all instances) */
    mtfp_gelu_init();

    /* Initialize signatures from weights and pack ternary */
    trix_ternary_route_update_signatures(tr);
    quantize_and_pack_tiles(tr);

    return tr;
}

void trix_ternary_route_destroy(TrixTernaryRoutedFFN* tr) {
    if (!tr) return;
    int T = tr->cfg.num_tiles;

    free(tr->ln_weight); free(tr->ln_bias);
    free(tr->signatures);
    free(tr->W1); free(tr->b1); free(tr->W2); free(tr->b2);
    free(tr->W1_tern); free(tr->W2_tern);
    free(tr->W1_packed); free(tr->W2_packed);
    free(tr->mln_weight); free(tr->mln_bias);
    free(tr->mb1); free(tr->mb2);
    free(tr->mx_norm); free(tr->mz1); free(tr->mh1);
    free(tr->mtile_out); free(tr->mcombined);
    free(tr->dln_weight); free(tr->dln_bias);
    free(tr->dW1); free(tr->db1); free(tr->dW2); free(tr->db2);
    free(tr->m_ln_w); free(tr->v_ln_w);
    free(tr->m_ln_b); free(tr->v_ln_b);
    free(tr->m_W1); free(tr->v_W1);
    free(tr->m_b1); free(tr->v_b1);
    free(tr->m_W2); free(tr->v_W2);
    free(tr->m_b2); free(tr->v_b2);

    free(tr->x_norm); free(tr->ln_mean); free(tr->ln_rstd);
    free(tr->scores); free(tr->route);
    free(tr->z1); free(tr->h1);
    free(tr->tile_out); free(tr->combined);
    free(tr->d_combined); free(tr->d_tile_out);
    free(tr->dh1); free(tr->dz1); free(tr->dx_tile);

    if (tr->saved_z1) {
        for (int t = 0; t < T; t++) { free(tr->saved_z1[t]); free(tr->saved_h1[t]); }
        free(tr->saved_z1); free(tr->saved_h1);
    }
    free(tr);
}

/* ══════════════════════════════════════════════════════════════════════
 * Signature Update (weight-derived, mean-subtracted, ternarized)
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_route_update_signatures(TrixTernaryRoutedFFN* tr) {
    int D = tr->cfg.d_model, T = tr->cfg.num_tiles, H = tr->cfg.tile_hidden;

    /* raw_t[d] = sum_h W1_t[h, d] */
    float* raw = calloc(T * D, sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < D; d++) {
                raw[t * D + d] += tr->W1[t * H * D + h * D + d];
            }
        }
    }

    /* mean[d] = mean_over_t(raw_t[d]) */
    float* mean = calloc(D, sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            mean[d] += raw[t * D + d];
        }
    }
    for (int d = 0; d < D; d++) mean[d] /= (float)T;

    /* sig_t[d] = sign(raw_t[d] - mean[d]) */
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            float diff = raw[t * D + d] - mean[d];
            tr->signatures[t * D + d] = (diff > 0.0f) ? 1.0f : (diff < 0.0f) ? -1.0f : 0.0f;
        }
    }

    free(raw);
    free(mean);
}

/* ══════════════════════════════════════════════════════════════════════
 * Forward
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_route_forward(
    TrixTernaryRoutedFFN* tr,
    const float* x, float* out, int batch)
{
    int D = tr->cfg.d_model;
    int T = tr->cfg.num_tiles;
    int H = tr->cfg.tile_hidden;
    int K = tr->cfg.active_k;

    tr_ensure_scratch(tr, batch);

    /* 1. Float LayerNorm — saves stats for backward (STE needs float) */
    trix_layernorm_forward_save(tr->x_norm, tr->ln_mean, tr->ln_rstd,
        x, tr->ln_weight, tr->ln_bias, batch, D, tr->cfg.ln_eps);

    /* Convert the LN output to MTFP for the tile computation */
    mtfp_from_float_batch(tr->mx_norm, tr->x_norm, batch * D);

    /* 2. Routing scores: signatures are ternary {-1,0,+1}.
     * Score = dot(mx_norm, sig) — MTFP × ternary = integer add/sub.
     * Convert scores to float for the threshold (tiny operation). */
    {
        mtfp_t* mscores = calloc(batch * T, sizeof(mtfp_t));
        /* signatures stored as float ternary, but they ARE {-1,0,+1}.
         * Cast to int8 for ternary matmul. */
        int8_t* sig_i8 = calloc(T * D, sizeof(int8_t));
        for (int i = 0; i < T * D; i++)
            sig_i8[i] = (tr->signatures[i] > 0.5f) ? 1 : (tr->signatures[i] < -0.5f) ? -1 : 0;
        mtfp_ternary_matmul_bt(mscores, tr->mx_norm, sig_i8, batch, D, T);
        mtfp_to_float_batch(tr->scores, mscores, batch * T);
        free(mscores); free(sig_i8);
    }

    /* 3. Ternary threshold: top-k by |score|, assign sign */
    ternary_threshold_topk(tr->route, tr->scores, batch, T, K);
    memset(tr->mcombined, 0, batch * D * sizeof(mtfp_t));

    for (int t = 0; t < T; t++) {
        int8_t* W1t = tr->W1_tern + t * H * D;
        int8_t* W2t = tr->W2_tern + t * D * H;

        /* mz1 = mx_norm @ W1_t^T + mb1 — MTFP × ternary, ZERO multiplies */
        mtfp_ternary_matmul_bt(tr->mz1, tr->mx_norm, W1t, batch, D, H);
        mtfp_fan_in_normalize(tr->mz1, batch * H, D);  /* before GELU */
        mtfp_bias_add(tr->mz1, tr->mb1 + t * H, batch, H);

        mtfp_gelu(tr->mh1, tr->mz1, batch * H);

        mtfp_to_float_batch(tr->saved_z1[t], tr->mz1, batch * H);
        mtfp_to_float_batch(tr->saved_h1[t], tr->mh1, batch * H);

        /* mtile_out — NO normalization (output_scale handles magnitude) */
        mtfp_ternary_matmul_bt(tr->mtile_out, tr->mh1, W2t, batch, H, D);
        mtfp_bias_add(tr->mtile_out, tr->mb2 + t * D, batch, D);

        /* Accumulate: mcombined[i] += route[i,t] * mtile_out[i]
         * Route is {-1, 0, +1} — integer add/subtract/skip */
        for (int i = 0; i < batch; i++) {
            int r = tr->route[i * T + t];
            if (r == 0) continue;
            mtfp_t* dst = tr->mcombined + i * D;
            mtfp_t* src = tr->mtile_out + i * D;
            if (r == 1) mtfp_vec_add_inplace(dst, src, D);
            else { /* r == -1 */
                for (int d = 0; d < D; d++) dst[d] -= src[d];
            }
        }
    }

    /* 5. Convert MTFP combined back to float, then scale + residual in float.
     * The output_scale multiply and residual add stay float for now to avoid
     * mtfp_mul quantization noise on the residual stream. The tile computation
     * (matmul, GELU, bias, accumulation) is 100% MTFP integer. */
    mtfp_to_float_batch(tr->combined, tr->mcombined, batch * D);
    for (int i = 0; i < batch * D; i++)
        out[i] = x[i] + tr->output_scale * tr->combined[i];
}

/* ══════════════════════════════════════════════════════════════════════
 * Forward (MTFP native — zero conversion inside)
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_route_forward_mtfp(
    TrixTernaryRoutedFFN* tr,
    const mtfp_t* x, mtfp_t* out, int batch)
{
    int D = tr->cfg.d_model;
    int T = tr->cfg.num_tiles;
    int H = tr->cfg.tile_hidden;
    int K = tr->cfg.active_k;

    tr_ensure_scratch(tr, batch);

    /* 1. MTFP LayerNorm — one sqrt, everything else integer */
    mtfp_layernorm(tr->mx_norm, x, tr->mln_weight, tr->mln_bias, batch, D);

    /* Save float versions for backward (STE) */
    mtfp_to_float_batch(tr->x_norm, tr->mx_norm, batch * D);
    /* Recompute float LN stats for backward */
    {
        float* x_f = calloc(batch * D, sizeof(float));
        mtfp_to_float_batch(x_f, x, batch * D);
        trix_layernorm_forward_save(tr->x_norm, tr->ln_mean, tr->ln_rstd,
            x_f, tr->ln_weight, tr->ln_bias, batch, D, tr->cfg.ln_eps);
        free(x_f);
    }

    /* 2. Routing scores — MTFP × ternary = integer add/sub */
    {
        int8_t* sig_i8 = calloc(T * D, sizeof(int8_t));
        for (int i = 0; i < T * D; i++)
            sig_i8[i] = (tr->signatures[i] > 0.5f) ? 1 : (tr->signatures[i] < -0.5f) ? -1 : 0;
        mtfp_t* mscores = calloc(batch * T, sizeof(mtfp_t));
        mtfp_ternary_matmul_bt(mscores, tr->mx_norm, sig_i8, batch, D, T);
        mtfp_to_float_batch(tr->scores, mscores, batch * T);
        free(mscores); free(sig_i8);
    }

    /* 3. Ternary threshold */
    ternary_threshold_topk(tr->route, tr->scores, batch, T, K);

    /* 4. MTFP tile computation — 100% integer, zero float */
    memset(tr->mcombined, 0, batch * D * sizeof(mtfp_t));

    for (int t = 0; t < T; t++) {
        int8_t* W1t = tr->W1_tern + t * H * D;
        int8_t* W2t = tr->W2_tern + t * D * H;

        mtfp_ternary_matmul_bt(tr->mz1, tr->mx_norm, W1t, batch, D, H);
        mtfp_fan_in_normalize(tr->mz1, batch * H, D);  /* before GELU */
        mtfp_bias_add(tr->mz1, tr->mb1 + t * H, batch, H);
        mtfp_gelu(tr->mh1, tr->mz1, batch * H);

        mtfp_to_float_batch(tr->saved_z1[t], tr->mz1, batch * H);
        mtfp_to_float_batch(tr->saved_h1[t], tr->mh1, batch * H);

        /* W2 — NO normalization */
        mtfp_ternary_matmul_bt(tr->mtile_out, tr->mh1, W2t, batch, H, D);
        mtfp_bias_add(tr->mtile_out, tr->mb2 + t * D, batch, D);

        for (int i = 0; i < batch; i++) {
            int r = tr->route[i * T + t];
            if (r == 0) continue;
            mtfp_t* dst = tr->mcombined + i * D;
            mtfp_t* src = tr->mtile_out + i * D;
            if (r == 1) mtfp_vec_add_inplace(dst, src, D);
            else for (int d = 0; d < D; d++) dst[d] -= src[d];
        }
    }

    /* 5. Scale + residual — all MTFP, no float conversion */
    mtfp_t mscale = mtfp_from_float(tr->output_scale);
    for (int i = 0; i < batch * D; i++)
        out[i] = mtfp_add(x[i], mtfp_mul(mscale, tr->mcombined[i]));

    /* Save float combined for backward d_output_scale */
    mtfp_to_float_batch(tr->combined, tr->mcombined, batch * D);
}

/* ══════════════════════════════════════════════════════════════════════
 * Backward
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_route_backward(
    TrixTernaryRoutedFFN* tr,
    const float* x, const float* dy, float* dx, int batch)
{
    int D = tr->cfg.d_model;
    int T = tr->cfg.num_tiles;
    int H = tr->cfg.tile_hidden;

    /* d_combined = dy * output_scale */
    trix_vec_scale(tr->d_combined, dy, tr->output_scale, batch * D);

    /* d_output_scale += dot(dy, combined) */
    tr->doutput_scale += trix_dot(dy, tr->combined, batch * D);

    /* Per-tile backward: route modulates gradient direction */
    float* dx_accum = calloc(batch * D, sizeof(float));

    for (int t = 0; t < T; t++) {
        float* W1_t = tr->W1 + t * H * D;
        float* W2_t = tr->W2 + t * D * H;

        /* d_tile_out[i] = route[i,t] * d_combined[i]
         * Sign of route modulates the gradient direction.
         * +1 tiles: gradient flows as-is (expert contributes)
         * -1 tiles: gradient is negated (anti-expert subtracts)
         *  0 tiles: no gradient (skipped) */
        trix_vec_zero(tr->d_tile_out, batch * D);
        for (int i = 0; i < batch; i++) {
            int r = tr->route[i * T + t];
            if (r == 0) continue;
            float sign = (float)r;
            trix_vec_scale(tr->d_tile_out + i * D, tr->d_combined + i * D, sign, D);
        }

        /* dW2 += h1^T @ d_tile_out   [D, H] — float32 via Accelerate/AMX (shadow weight grad) */
        trix_matmul_at(tr->dW2 + t * D * H, tr->d_tile_out, tr->saved_h1[t], batch, D, H);
        trix_bias_grad(tr->db2 + t * D, tr->d_tile_out, batch, D);

        /* dh1 = d_tile_out @ W2_t   [batch, H] — float × ternary, ZERO multiplies
         * d_tile_out[batch, D] @ W2_tern[D, H] */
        trix_ternary_matmul(tr->dh1, tr->d_tile_out, tr->W2_tern + t * D * H, batch, D, H);

        /* dz1 = dh1 * GELU'(z1) */
        trix_gelu_grad(tr->dz1, tr->dh1, tr->saved_z1[t], batch * H);

        /* dW1 += dz1^T @ x_norm   [H, D] — float32 via Accelerate/AMX (shadow weight grad) */
        trix_matmul_at(tr->dW1 + t * H * D, tr->dz1, tr->x_norm, batch, H, D);
        trix_bias_grad(tr->db1 + t * H, tr->dz1, batch, H);

        /* dx_norm += dz1 @ W1_t   [batch, D] — float × ternary, ZERO multiplies */
        trix_ternary_matmul(tr->dx_tile, tr->dz1, tr->W1_tern + t * H * D, batch, H, D);
        trix_vec_add(dx_accum, dx_accum, tr->dx_tile, batch * D);
    }

    /* LayerNorm backward */
    float* dx_pre_ln = calloc(batch * D, sizeof(float));
    trix_layernorm_backward(
        dx_pre_ln, tr->dln_weight, tr->dln_bias,
        dx_accum, x, tr->ln_weight,
        tr->ln_mean, tr->ln_rstd,
        batch, D
    );

    /* dx = dy + dx_pre_ln (residual) */
    if (dx) {
        trix_vec_add(dx, dy, dx_pre_ln, batch * D);
    }

    free(dx_accum);
    free(dx_pre_ln);
}

/* ══════════════════════════════════════════════════════════════════════
 * Zero Grad / Optimizer / Clipping
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_route_zero_grad(TrixTernaryRoutedFFN* tr) {
    int D = tr->cfg.d_model, T = tr->cfg.num_tiles, H = tr->cfg.tile_hidden;
    trix_vec_zero(tr->dln_weight, D);
    trix_vec_zero(tr->dln_bias, D);
    trix_vec_zero(tr->dW1, T * H * D);
    trix_vec_zero(tr->db1, T * H);
    trix_vec_zero(tr->dW2, T * D * H);
    trix_vec_zero(tr->db2, T * D);
    tr->doutput_scale = 0.0f;
}

void trix_ternary_route_adamw_step(
    TrixTernaryRoutedFFN* tr,
    float lr, float beta1, float beta2, float eps, float wd)
{
    int D = tr->cfg.d_model, T = tr->cfg.num_tiles, H = tr->cfg.tile_hidden;
    tr->adam_step++;
    int step = tr->adam_step;

    trix_adamw_update(tr->ln_weight, tr->dln_weight, tr->m_ln_w, tr->v_ln_w,
                      lr, beta1, beta2, eps, 0.0f, step, D);
    trix_adamw_update(tr->ln_bias, tr->dln_bias, tr->m_ln_b, tr->v_ln_b,
                      lr, beta1, beta2, eps, 0.0f, step, D);
    trix_adamw_update(tr->W1, tr->dW1, tr->m_W1, tr->v_W1,
                      lr, beta1, beta2, eps, wd, step, T * H * D);
    trix_adamw_update(tr->b1, tr->db1, tr->m_b1, tr->v_b1,
                      lr, beta1, beta2, eps, 0.0f, step, T * H);
    trix_adamw_update(tr->W2, tr->dW2, tr->m_W2, tr->v_W2,
                      lr, beta1, beta2, eps, wd, step, T * D * H);
    trix_adamw_update(tr->b2, tr->db2, tr->m_b2, tr->v_b2,
                      lr, beta1, beta2, eps, 0.0f, step, T * D);

    /* output_scale */
    {
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);
        tr->m_output_scale = beta1 * tr->m_output_scale + (1.0f - beta1) * tr->doutput_scale;
        tr->v_output_scale = beta2 * tr->v_output_scale + (1.0f - beta2) * tr->doutput_scale * tr->doutput_scale;
        float m_hat = tr->m_output_scale / bc1;
        float v_hat = tr->v_output_scale / bc2;
        tr->output_scale -= lr * m_hat / (sqrtf(v_hat) + eps);
    }

    /* Update signatures and re-pack ternary weights from new shadow weights */
    trix_ternary_route_update_signatures(tr);
    quantize_and_pack_tiles(tr);
}

float trix_ternary_route_clip_grad_norm(TrixTernaryRoutedFFN* tr, float max_norm) {
    int D = tr->cfg.d_model, T = tr->cfg.num_tiles, H = tr->cfg.tile_hidden;
    float sq = 0.0f;
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

/* ══════════════════════════════════════════════════════════════════════
 * Diagnostics
 * ══════════════════════════════════════════════════════════════════════ */

void trix_ternary_route_get_routing(
    const TrixTernaryRoutedFFN* tr, int* route_out, int batch)
{
    memcpy(route_out, tr->route, batch * tr->cfg.num_tiles * sizeof(int));
}

void trix_ternary_route_get_tile_activity(
    const TrixTernaryRoutedFFN* tr,
    int* pos_count, int* neg_count, int batch)
{
    int T = tr->cfg.num_tiles;
    memset(pos_count, 0, T * sizeof(int));
    memset(neg_count, 0, T * sizeof(int));
    for (int i = 0; i < batch; i++) {
        for (int t = 0; t < T; t++) {
            int r = tr->route[i * T + t];
            if (r > 0) pos_count[t]++;
            else if (r < 0) neg_count[t]++;
        }
    }
}
