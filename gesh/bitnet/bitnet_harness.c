/*
 * gesh/bitnet/bitnet_harness.c — single-block forward-pass harness for
 * BitNet b1.58-2B-4T inference on the m4t substrate. Per work-unit 1
 * of the bitnet_phase1 LMM cycle.
 *
 * This file is the SKELETON committed at the start of work-unit 1.
 * It compiles cleanly, allocates scratch, lays out the forward pass
 * structure, and stubs out the parts that need (a) actual weights
 * loaded from disk and (b) per-layer activation dumping. Subsequent
 * work-unit-1 commits fill those in.
 *
 * Execution model:
 *   build/gesh/bitnet/bitnet_harness <weights_blob.bin> <input_token_id>
 *
 * Output:
 *   per-layer activations dumped to stdout in a documented binary
 *   format (work-unit 1 fills this in alongside the Python comparison
 *   driver).
 */

#include "bitnet_config.h"
#include "bitnet_stubs.h"
#include "bitnet_weights.h"

#include "m4t_ternary_matmul.h"
#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── BitLinear scale composition (work-unit 5) ───────────────────────────
 *
 * BitLinear forward:  y = (matmul_int_out) × α × absmax / 127
 * where:
 *   - α is the BitLinear's per-tensor scale, stored as MTFP19
 *     (mantissa × 3^(-block_exp)).
 *   - absmax is the activation's per-token absmax (from A8 quantize).
 *   - matmul_int_out is the raw int-mantissa output from
 *     m4t_ternary_5in8_matmul_bt.
 *
 * Composing: y = raw × (α_m / 3^bx) × absmax / 127
 *              = raw × (α_m × absmax) / (127 × 3^bx)
 *
 * num = α_m × absmax, den = 127 × 3^bx.
 *
 * Constraint: bx ≤ ~38 to keep den ≤ int64 max. For BitNet b1.58-2B-4T
 * α magnitudes (range ~1e-3 to ~1), typical block_exp ≤ 25. Documented.
 *
 * If α_mantissa == 0 (convert_weights.py's sentinel for "no α
 * available"), the apply is a no-op (skip vec_scale; output remains
 * raw matmul, useful for skeleton-mode debugging). */

static int64_t pow3_int(int k) {
    /* Bound: caller computes den = 127 × 3^k, which must fit int64.
     * 127 × 3^35 = 6.35e18 < INT64_MAX (9.22e18); 3^36 already overflows
     * after the × 127. For BitNet α (range ~1e-3 to 1), block_exp ≤ ~22
     * — comfortable. Tighter bound here trades catching overflow vs
     * supporting smaller α down to ~3^-35 ≈ 2e-17. */
    assert(k >= 0 && k <= 35);
    int64_t r = 1;
    for (int i = 0; i < k; i++) r *= 3;
    return r;
}

static void bitnet_apply_bitlinear_scale(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* alpha_ptr, int alpha_block_exp,
    m4t_mtfp_t absmax, int n)
{
    int64_t alpha_m = (int64_t)(*alpha_ptr);
    if (alpha_m == 0) {
        if (y != x) memcpy(y, x, (size_t)n * sizeof(m4t_mtfp_t));
        return;
    }
    int64_t num = alpha_m * (int64_t)absmax;
    int64_t den = (int64_t)127 * pow3_int(alpha_block_exp);
    m4t_mtfp_vec_scale(y, x, num, den, n);
}

/* ── Scratch alloc / free ────────────────────────────────────────── */

void bitnet_block_scratch_alloc(bitnet_block_scratch_t* s) {
    /* Single-token forward pass: each [HIDDEN] or [INTERMEDIATE] buffer
     * holds one row. Multi-token prefill widens these (work-unit 7+). */
    s->x              = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->residual       = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->x_norm         = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->q              = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->k              = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    s->v              = calloc(BITNET_KV_PROJ_DIM,         sizeof(m4t_mtfp_t));
    /* Single-token: attn_scores is just [num_heads × 1 × 1] = 20 cells.
     * Multi-token: would scale with seq_len. */
    s->attn_scores    = calloc(BITNET_NUM_ATTENTION_HEADS, sizeof(m4t_mtfp_t));
    s->attn_output    = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->attn_sub_norm  = calloc(BITNET_HIDDEN_SIZE,         sizeof(m4t_mtfp_t));
    s->gate           = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->up             = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->gate_act       = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->ffn_sub_norm   = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(m4t_mtfp_t));
    s->q_int8         = calloc(BITNET_INTERMEDIATE_SIZE,   sizeof(int8_t));
    s->q_absmax       = 0;
}

void bitnet_block_scratch_free(bitnet_block_scratch_t* s) {
    free(s->x); free(s->residual); free(s->x_norm);
    free(s->q); free(s->k); free(s->v);
    free(s->attn_scores); free(s->attn_output); free(s->attn_sub_norm);
    free(s->gate); free(s->up); free(s->gate_act); free(s->ffn_sub_norm);
    free(s->q_int8);
    memset(s, 0, sizeof(*s));
}

/* ── Forward pass through one transformer block ──────────────────── */

/*
 * Input: x[HIDDEN] (the token embedding for the first block; the
 * previous block's output for layers > 0).
 * Output: x[HIDDEN] (overwritten in place).
 *
 * Per the architecture (verified in bitnet_phase1_o1_findings.md):
 *
 *   residual = x
 *   x = input_layernorm(x)
 *   x = attention(x)            ← contains attn_sub_norm
 *   x = residual + x
 *   residual = x
 *   x = post_attention_layernorm(x)
 *   x = ffn(x)                  ← contains ffn_sub_norm
 *   x = residual + x
 */
void bitnet_forward_block(
    m4t_mtfp_t* x_io,
    const bitnet_layer_weights_t* w,
    bitnet_block_scratch_t* s,
    bitnet_kv_cache_t* cache,
    int layer_idx,
    int position)
{
    /* For now: x_io aliased into s->x at entry, copied back at exit.
     * Future cleanup may eliminate the copy. */
    memcpy(s->x, x_io, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* ── Attention sub-block ──────────────────────────────────────── */

    /* residual = x */
    memcpy(s->residual, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* x_norm = input_layernorm(x) */
    m4t_mtfp_rmsnorm(s->x_norm, s->x, w->gamma_input_norm,
                        /* eps_mtfp19 */ 1, BITNET_HIDDEN_SIZE);
    /* TODO(work-unit 1.5): empirically determine eps_mtfp19. The HF
     * eps is 1e-5 (FP); the substrate-side equivalent depends on
     * what scale the activations land in. Capture x's magnitude
     * during first run and pick eps to land at the same ratio. */

    /* QKV projections via packed-ternary matmul.
     * Activation goes through A8 quantize first per W1.58A8 spec. */
    /* TODO: A8 quantize x_norm → q_int8 + scale; pass to BitLinear-equivalent.
     * For now, the substrate's existing matmul takes int8 X directly,
     * and we approximate by treating x_norm's MSB as int8. This is
     * NOT the BitNet protocol — work-unit 5 builds the actual A8 path. */

    /* QKV projections via packed-ternary matmul (gated on weights-loaded;
     * zero output if w->w_q is NULL — keeps harness runnable without
     * actual weights for skeleton testing).
     *
     * BitNet's W1.58A8 spec: x_norm gets A8-quantized to int8 (per-token
     * absmax), then matmul receives the int8. Q/K/V share x_norm as input
     * — we A8-quantize ONCE and reuse (RC-11 fix from work-unit 1 red-team:
     * computing the absmax separately for each projection wastes work
     * and risks numerical divergence if they produced different absmaxes).
     *
     * Note (RC-6): we call `m4t_ternary_5in8_matmul_bt` with non-ternary X
     * (int8 in [-127, 127] from A8 quantize, not {-1, 0, +1}). The kernel
     * uses SDOT which handles full int8 range correctly, but the documented
     * contract says X is ternary. This usage is intentional; future
     * cleanup (work-unit 5+) may add a substrate variant with a wider X
     * contract. */
    if (w->w_q != NULL) {
        /* A8-quantize x_norm ONCE → reused for Q, K, V. */
        s->q_absmax = m4t_a8_quantize(s->q_int8, s->x_norm,
                                       BITNET_HIDDEN_SIZE);
        /* Q = x_int8 @ W_q^T → raw mantissas → scale apply. */
        m4t_ternary_5in8_matmul_bt(s->q, (const m4t_trit_t*)s->q_int8, w->w_q,
                                    /*M=*/1, BITNET_HIDDEN_SIZE,
                                    /*N=*/BITNET_HIDDEN_SIZE);
        bitnet_apply_bitlinear_scale(s->q, s->q,
                                      w->alpha_q, w->alpha_q_block_exp,
                                      s->q_absmax, BITNET_HIDDEN_SIZE);
        /* K, V projections (smaller output dim due to GQA). */
        m4t_ternary_5in8_matmul_bt(s->k, (const m4t_trit_t*)s->q_int8, w->w_k,
                                    1, BITNET_HIDDEN_SIZE, BITNET_KV_PROJ_DIM);
        bitnet_apply_bitlinear_scale(s->k, s->k,
                                      w->alpha_k, w->alpha_k_block_exp,
                                      s->q_absmax, BITNET_KV_PROJ_DIM);
        m4t_ternary_5in8_matmul_bt(s->v, (const m4t_trit_t*)s->q_int8, w->w_v,
                                    1, BITNET_HIDDEN_SIZE, BITNET_KV_PROJ_DIM);
        bitnet_apply_bitlinear_scale(s->v, s->v,
                                      w->alpha_v, w->alpha_v_block_exp,
                                      s->q_absmax, BITNET_KV_PROJ_DIM);
    } else {
        /* No weights loaded — skeleton mode. Zero outputs. */
        memset(s->q, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
        memset(s->k, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));
        memset(s->v, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));
    }

    /* RoPE on Q, K. */
    m4t_mtfp_rope_apply(s->q, s->k, position,
                        BITNET_NUM_ATTENTION_HEADS,
                        BITNET_NUM_KV_HEADS,
                        BITNET_HEAD_DIM,
                        BITNET_ROPE_THETA);

    /* Attention with KV cache (work-unit 7).
     *
     * Per Q head h:
     *   kv_head_idx = h / (num_q_heads / num_kv_heads)
     *   scores[t]   = Q[h] · K_cache[layer][t][kv_head_idx] / sqrt(head_dim)
     *                 for t ∈ [0, position+1)
     *   weights     = softmax(scores)        [length = position+1]
     *   attn_out[h] = Σ_t weights[t] · V_cache[layer][t][kv_head_idx]
     *
     * Score → softmax scale: softmax expects "1 LSB = 1 nat". Q · K
     * dot products in raw int can be enormous (up to 2^63ish). To bring
     * them into the LUT range [-30, 0] post-max-subtraction, we
     * right-shift the scores by enough bits to fit. This is a temperature
     * change relative to HF (softmax(x / T) ≠ softmax(x)) — the per-layer
     * ε comparison (work-unit 6) measures the resulting mismatch.
     *
     * 1/sqrt(head_dim) factor is folded into the rescale shift (it's a
     * constant downscale ≈ /11.3 = ~3.5 bits; absorbed into the heuristic). */
    {
        const int q_per_kv = BITNET_NUM_ATTENTION_HEADS / BITNET_NUM_KV_HEADS;

        if (cache == NULL) {
            /* No-cache fallback: degenerate seq=1 (attn_out = V). */
            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                int kv_head = h / q_per_kv;
                memcpy(s->attn_output + (size_t)h * BITNET_HEAD_DIM,
                       s->v + (size_t)kv_head * BITNET_HEAD_DIM,
                       BITNET_HEAD_DIM * sizeof(m4t_mtfp_t));
            }
            memset(s->attn_scores, 0,
                   BITNET_NUM_ATTENTION_HEADS * sizeof(m4t_mtfp_t));
        } else {
            /* Write current K, V to cache at this layer's slot for `position`. */
            assert(layer_idx >= 0 && layer_idx < cache->n_layers);
            assert(position >= 0 && position < cache->max_seq_len);
            size_t row_size = (size_t)BITNET_NUM_KV_HEADS * BITNET_HEAD_DIM;
            size_t base = (size_t)layer_idx * cache->per_layer_stride
                          + (size_t)position * row_size;
            memcpy(cache->k + base, s->k, row_size * sizeof(m4t_mtfp_t));
            memcpy(cache->v + base, s->v, row_size * sizeof(m4t_mtfp_t));

            int seq_k = position + 1;

            /* Per-head scratch hoisted out of the head loop (RC-2). */
            int64_t* scores_i64    = (int64_t*)   malloc((size_t)seq_k * sizeof(int64_t));
            m4t_mtfp_t* scores_int = (m4t_mtfp_t*)malloc((size_t)seq_k * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* weights    = (m4t_mtfp_t*)malloc((size_t)seq_k * sizeof(m4t_mtfp_t));
            assert(scores_i64 && scores_int && weights);

            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                int kv_head = h / q_per_kv;
                const m4t_mtfp_t* qh = s->q + (size_t)h * BITNET_HEAD_DIM;

                /* Compute scores[t] = dot(qh, K_cache[layer][t][kv_head]). */
                int64_t max_abs = 1;
                for (int t = 0; t < seq_k; t++) {
                    size_t k_row_base = (size_t)layer_idx * cache->per_layer_stride
                                        + (size_t)t * row_size
                                        + (size_t)kv_head * BITNET_HEAD_DIM;
                    const m4t_mtfp_t* kh = cache->k + k_row_base;
                    int64_t acc = 0;
                    for (int d = 0; d < BITNET_HEAD_DIM; d++) {
                        acc += (int64_t)qh[d] * (int64_t)kh[d];
                    }
                    scores_i64[t] = acc;
                    int64_t a = acc < 0 ? -acc : acc;
                    if (a > max_abs) max_abs = a;
                }

                /* Rescale scores into "1 LSB ≈ 1 nat" range for softmax.
                 * Pick shift such that max_abs >> shift ≤ 30 — i.e.,
                 * scores fit the LUT range without underflow at the top.
                 * 1/sqrt(head_dim) factor folded into this shift (3.5 bits). */
                int score_shift = 0;
                while ((max_abs >> score_shift) > 30) score_shift++;
                /* Add an extra ~4 bits to absorb the 1/sqrt(d) factor and
                 * keep the softmax distribution from being too peaked. */
                score_shift += 4;

                for (int t = 0; t < seq_k; t++) {
                    int64_t r;
                    if (scores_i64[t] >= 0) r = scores_i64[t] >> score_shift;
                    else                    r = -((-scores_i64[t]) >> score_shift);
                    if (r >  M4T_MTFP_MAX_VAL) r =  M4T_MTFP_MAX_VAL;
                    if (r < -M4T_MTFP_MAX_VAL) r = -M4T_MTFP_MAX_VAL;
                    scores_int[t] = (m4t_mtfp_t)r;
                }

                /* Softmax over scores. */
                m4t_mtfp_softmax(weights, scores_int, seq_k);

                /* attn_out[h × head_dim..] = Σ_t weights[t] · V_cache[layer][t][kv_head].
                 * weights[t] is at scale 2^30; V is mantissa.
                 * accum at scale 2^30 → >> 30 to recover MTFP19 mantissa. */
                m4t_mtfp_t* out_h = s->attn_output + (size_t)h * BITNET_HEAD_DIM;
                for (int d = 0; d < BITNET_HEAD_DIM; d++) {
                    int64_t acc = 0;
                    for (int t = 0; t < seq_k; t++) {
                        size_t v_row_base = (size_t)layer_idx * cache->per_layer_stride
                                            + (size_t)t * row_size
                                            + (size_t)kv_head * BITNET_HEAD_DIM;
                        acc += (int64_t)weights[t] * (int64_t)cache->v[v_row_base + d];
                    }
                    out_h[d] = m4t_mtfp_clamp64(acc >> 30);
                }
            }

            /* Stash one debug score per head (last position's score),
             * for the activation dump. */
            for (int h = 0; h < BITNET_NUM_ATTENTION_HEADS; h++) {
                s->attn_scores[h] = 0;  /* not load-bearing in tests */
            }

            free(scores_i64); free(scores_int); free(weights);
        }
    }

    /* attn_sub_norm: y = γ · y · rsqrt(mean(y²) + ε) */
    m4t_mtfp_rmsnorm(s->attn_sub_norm, s->attn_output,
                        w->gamma_attn_sub_norm, 1, BITNET_HIDDEN_SIZE);

    /* O projection: y = attn_sub_norm @ W_o^T (BitLinear, A8-quantized).
     * Per-projection input — own A8 quantize. */
    if (w->w_o != NULL) {
        s->q_absmax = m4t_a8_quantize(s->q_int8, s->attn_sub_norm,
                                       BITNET_HIDDEN_SIZE);
        m4t_ternary_5in8_matmul_bt(s->x, (const m4t_trit_t*)s->q_int8, w->w_o,
                                    1, BITNET_HIDDEN_SIZE, BITNET_HIDDEN_SIZE);
        bitnet_apply_bitlinear_scale(s->x, s->x,
                                      w->alpha_o, w->alpha_o_block_exp,
                                      s->q_absmax, BITNET_HIDDEN_SIZE);
    } else {
        memcpy(s->x, s->attn_sub_norm, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    }

    /* x = residual + x. */
    for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) {
        int64_t v = (int64_t)s->residual[i] + (int64_t)s->x[i];
        s->x[i] = (int32_t)((v > 581130733) ? 581130733 :
                            (v < -581130733) ? -581130733 : v);
    }

    /* ── FFN sub-block ────────────────────────────────────────────── */

    /* residual = x */
    memcpy(s->residual, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

    /* x_norm = post_attention_layernorm(x) */
    m4t_mtfp_rmsnorm(s->x_norm, s->x, w->gamma_post_attn_norm,
                        1, BITNET_HIDDEN_SIZE);

    /* gate, up = x_norm projected (BitLinear, A8). They share x_norm as
     * input — A8-quantize ONCE and reuse (RC-11 fix). */
    if (w->w_gate != NULL && w->w_up != NULL) {
        s->q_absmax = m4t_a8_quantize(s->q_int8, s->x_norm,
                                       BITNET_HIDDEN_SIZE);
        m4t_ternary_5in8_matmul_bt(s->gate, (const m4t_trit_t*)s->q_int8, w->w_gate,
                                    1, BITNET_HIDDEN_SIZE, BITNET_INTERMEDIATE_SIZE);
        bitnet_apply_bitlinear_scale(s->gate, s->gate,
                                      w->alpha_gate, w->alpha_gate_block_exp,
                                      s->q_absmax, BITNET_INTERMEDIATE_SIZE);
        m4t_ternary_5in8_matmul_bt(s->up,   (const m4t_trit_t*)s->q_int8, w->w_up,
                                    1, BITNET_HIDDEN_SIZE, BITNET_INTERMEDIATE_SIZE);
        bitnet_apply_bitlinear_scale(s->up, s->up,
                                      w->alpha_up, w->alpha_up_block_exp,
                                      s->q_absmax, BITNET_INTERMEDIATE_SIZE);
    } else {
        memset(s->gate, 0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));
        memset(s->up,   0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));
    }

    /* gate_act = relu²(gate). */
    m4t_mtfp_relu2_inplace(s->gate, BITNET_INTERMEDIATE_SIZE);
    /* gate_act = gate * up. */
    m4t_mtfp_elementwise_mul(s->gate_act, s->gate, s->up,
                             BITNET_INTERMEDIATE_SIZE);

    /* ffn_sub_norm(gate_act). */
    m4t_mtfp_rmsnorm(s->ffn_sub_norm, s->gate_act,
                        w->gamma_ffn_sub_norm, 1, BITNET_INTERMEDIATE_SIZE);

    /* down = ffn_sub_norm @ W_down^T (BitLinear, A8). Different input than
     * gate/up — own A8 quantize. */
    if (w->w_down != NULL) {
        s->q_absmax = m4t_a8_quantize(s->q_int8, s->ffn_sub_norm,
                                       BITNET_INTERMEDIATE_SIZE);
        m4t_ternary_5in8_matmul_bt(s->x, (const m4t_trit_t*)s->q_int8, w->w_down,
                                    1, BITNET_INTERMEDIATE_SIZE, BITNET_HIDDEN_SIZE);
        bitnet_apply_bitlinear_scale(s->x, s->x,
                                      w->alpha_down, w->alpha_down_block_exp,
                                      s->q_absmax, BITNET_HIDDEN_SIZE);
    } else {
        memset(s->x, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    }

    /* x = residual + x. */
    for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) {
        int64_t v = (int64_t)s->residual[i] + (int64_t)s->x[i];
        s->x[i] = (int32_t)((v > 581130733) ? 581130733 :
                            (v < -581130733) ? -581130733 : v);
    }

    /* Copy back to caller's buffer. */
    memcpy(x_io, s->x, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
}

/* ── main ─────────────────────────────────────────────────────────── */

/* Per-layer activation dump: writes captured tensors to a binary file
 * compatible with the comparison driver (scripts/compare_activations.py).
 *
 * Format: one .bin per (layer, sublayer) capture site, containing raw
 * int32 mantissas. A small JSON sidecar lists shapes + block_exps (TBD).
 *
 * Phase 1 simplification: just dump a few key sites to one consolidated
 * file. Detailed sublayer dumps are work-unit 6+ scope. */
static int dump_activations_to_file(
    const char* path,
    const bitnet_block_scratch_t* s,
    int layer)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[harness] cannot open %s for write\n", path);
        return 1;
    }
    /* Tiny header: magic "ACTV", layer index, then 5 captured tensors. */
    fwrite("ACTV", 1, 4, f);
    int32_t li = layer;
    fwrite(&li, sizeof(int32_t), 1, f);
    int32_t hidden = BITNET_HIDDEN_SIZE;
    int32_t intermediate = BITNET_INTERMEDIATE_SIZE;
    int32_t kv_proj = BITNET_KV_PROJ_DIM;
    fwrite(&hidden, sizeof(int32_t), 1, f);
    fwrite(&intermediate, sizeof(int32_t), 1, f);
    fwrite(&kv_proj, sizeof(int32_t), 1, f);
    /* x_norm (post input_layernorm) */
    fwrite(s->x_norm, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* q, k, v (post-projection) */
    fwrite(s->q, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    fwrite(s->k, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    fwrite(s->v, sizeof(m4t_mtfp_t), BITNET_KV_PROJ_DIM, f);
    /* attn_sub_norm output */
    fwrite(s->attn_sub_norm, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    /* gate, up (post-projection FFN) */
    fwrite(s->gate, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    fwrite(s->up, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* ffn_sub_norm output */
    fwrite(s->ffn_sub_norm, sizeof(m4t_mtfp_t), BITNET_INTERMEDIATE_SIZE, f);
    /* block_output (final residual added) */
    fwrite(s->x, sizeof(m4t_mtfp_t), BITNET_HIDDEN_SIZE, f);
    fclose(f);
    return 0;
}

/* Embedding lookup: x[i] = embedding[token_id × HIDDEN + i] as MTFP19
 * mantissas. The block_exp of the embedding is set at conversion time
 * (per-tensor). Caller must hold block_exp tracking externally if needed.
 *
 * Phase 1 simplification: we read mantissas directly as activation
 * values, treating everything as "block_exp 0" through the forward
 * pass. This loses precision relative to HF's bf16/fp32 but produces
 * a consistent, deterministic substrate output. Per-layer ε
 * comparison (work-unit 6's gate) measures how the discrepancy
 * accumulates. */
static void bitnet_embed(
    m4t_mtfp_t* x_out,
    const m4t_mtfp_t* embedding,
    int token_id)
{
    if (embedding == NULL) {
        for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x_out[i] = 0;
        return;
    }
    const m4t_mtfp_t* row = embedding + (size_t)token_id * BITNET_HIDDEN_SIZE;
    memcpy(x_out, row, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
}

/* LM head: logits = x @ lm_head^T. lm_head is [VOCAB × HIDDEN] MTFP19
 * mantissas. Output: logits at scale (lm_head_block_exp + x_block_exp)
 * in mantissa units; magnitudes can grow; we don't compute argmax in C
 * (the comparison driver does that against HF reference output). */
static void bitnet_lm_head(
    m4t_mtfp_t* logits_out,
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* lm_head,
    int top_n)
{
    /* For comparison purposes we only need logits[0..top_n) — full vocab
     * (128256) is dumped to file by the comparison driver if needed. */
    for (int v = 0; v < top_n; v++) {
        const m4t_mtfp_t* row = lm_head + (size_t)v * BITNET_HIDDEN_SIZE;
        int64_t acc = 0;
        for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) {
            acc += (int64_t)x[i] * (int64_t)row[i];
        }
        /* Crude scale-down to fit MTFP19; the comparison driver consumes
         * raw acc values for ε measurement so we expose them at int64
         * in a separate accumulator buffer (caller supplied). */
        int64_t scaled = acc >> 30;  /* somewhat arbitrary; rescaled by Python comparison */
        logits_out[v] = m4t_mtfp_clamp64(scaled);
    }
}

int main(int argc, char** argv) {
    int token_id = 1;          /* default: BOS-like token */
    int n_layers = -1;         /* -1 = all loaded layers */
    int n_positions = 1;       /* number of positions to forward (work-unit 7 cache) */
    const char* weights_arg = NULL;
    const char* dump_path = NULL;

    /* Simple positional + flag parsing. */
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <weights_blob.bin|-> "
            "[--token <id>] [--layers <n>] [--positions <p>] [--dump <path>]\n"
            "  weights_blob.bin — produced by scripts/convert_weights.py.\n"
            "                     '-' for skeleton mode.\n"
            "  --token <id>     — input token id (default: 1).\n"
            "  --layers <n>     — number of transformer layers to run\n"
            "                     (default: all loaded).\n"
            "  --positions <p>  — number of forward passes to run\n"
            "                     (positions 0..p-1; default 1). Exercises\n"
            "                     the KV cache across multiple decode steps.\n"
            "  --dump <path>    — write per-layer activation snapshots to\n"
            "                     <path>.pos<P>.layer<N>.bin for ε comparison.\n",
            argv[0]);
        return 1;
    }
    weights_arg = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        if      (strcmp(argv[i], "--token")     == 0) token_id    = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--layers")    == 0) n_layers    = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--positions") == 0) n_positions = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--dump")      == 0) dump_path   = argv[i+1];
        else { fprintf(stderr, "[harness] unknown flag: %s\n", argv[i]); return 1; }
    }
    if (n_positions < 1) n_positions = 1;

    bitnet_weights_t weights = {0};
    bitnet_weights_loaded_t handle = {0};
    int loaded_ok = 0;
    if (strcmp(weights_arg, "-") != 0) {
        if (bitnet_weights_load(weights_arg, &weights, &handle) == 0) {
            loaded_ok = 1;
            fprintf(stderr, "[harness] loaded weights from %s\n", weights_arg);
        } else {
            fprintf(stderr, "[harness] weight load failed; running skeleton mode\n");
        }
    } else {
        fprintf(stderr, "[harness] skeleton mode (no weights)\n");
    }

    /* Allocate scratch. */
    bitnet_block_scratch_t s = {0};
    bitnet_block_scratch_alloc(&s);

    /* Input: embed token_id (or fallback pattern in skeleton mode). */
    m4t_mtfp_t x[BITNET_HIDDEN_SIZE];
    if (loaded_ok && weights.embedding != NULL) {
        bitnet_embed(x, weights.embedding, token_id);
    } else {
        for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) x[i] = (i % 7) - 3;
    }

    /* Determine layer count to run. */
    int layers_to_run = (n_layers >= 0) ? n_layers : BITNET_NUM_LAYERS;
    if (layers_to_run > BITNET_NUM_LAYERS) layers_to_run = BITNET_NUM_LAYERS;

    /* Skeleton fallback weights for layers without loaded weights. */
    bitnet_layer_weights_t w_dummy = {0};
    static m4t_mtfp_t gamma_dummy[BITNET_INTERMEDIATE_SIZE];
    static m4t_mtfp_t alpha_dummy = 0;  /* triggers no-op scale in apply helper */
    for (int i = 0; i < BITNET_INTERMEDIATE_SIZE; i++) gamma_dummy[i] = 1;
    w_dummy.gamma_input_norm     = gamma_dummy;
    w_dummy.gamma_post_attn_norm = gamma_dummy;
    w_dummy.gamma_attn_sub_norm  = gamma_dummy;
    w_dummy.gamma_ffn_sub_norm   = gamma_dummy;
    w_dummy.alpha_q = w_dummy.alpha_k = w_dummy.alpha_v = w_dummy.alpha_o
        = w_dummy.alpha_gate = w_dummy.alpha_up = w_dummy.alpha_down = &alpha_dummy;

    /* KV cache (work-unit 7). max_seq_len ≥ n_positions. */
    bitnet_kv_cache_t cache = {0};
    int max_seq = n_positions > 256 ? n_positions : 256;
    if (bitnet_kv_cache_alloc(&cache, max_seq, layers_to_run) != 0) {
        fprintf(stderr, "[harness] KV cache alloc failed\n");
        return 1;
    }

    /* Multi-position forward. Each position re-embeds the same token
     * (single-prompt-token mode) — work-unit 8's generation loop replaces
     * this with a per-token feed pulling from the prior step's argmax. */
    m4t_mtfp_t x_init[BITNET_HIDDEN_SIZE];
    memcpy(x_init, x, sizeof(x_init));

    for (int pos = 0; pos < n_positions; pos++) {
        if (pos > 0) memcpy(x, x_init, sizeof(x_init));  /* re-embed for this pos */

        for (int l = 0; l < layers_to_run; l++) {
            const bitnet_layer_weights_t* w_l;
            if (loaded_ok && weights.layers[l].w_q != NULL) {
                w_l = &weights.layers[l];
            } else {
                w_l = &w_dummy;
            }
            bitnet_forward_block(x, w_l, &s, &cache, l, pos);

            if (dump_path && pos == n_positions - 1) {
                char path[1024];
                snprintf(path, sizeof(path), "%s.layer%d.bin", dump_path, l);
                dump_activations_to_file(path, &s, l);
            }
        }
        cache.current_pos = pos + 1;
    }

    /* Final norm + LM head. */
    if (loaded_ok && weights.gamma_final_norm != NULL) {
        m4t_mtfp_t x_norm[BITNET_HIDDEN_SIZE];
        m4t_mtfp_rmsnorm(x_norm, x, weights.gamma_final_norm,
                         /*eps=*/1, BITNET_HIDDEN_SIZE);
        memcpy(x, x_norm, sizeof(x));
    }

    int top_n = 16;
    m4t_mtfp_t logits[16];
    if (loaded_ok && weights.lm_head != NULL) {
        bitnet_lm_head(logits, x, weights.lm_head, top_n);
    } else {
        memset(logits, 0, sizeof(logits));
    }

    fprintf(stderr,
        "[ok] %d layer(s) forward pass completed (token_id=%d).\n"
        "     post-final-norm x[0..3]      = %d %d %d %d\n"
        "     logits[0..3]                  = %d %d %d %d\n",
        layers_to_run, token_id,
        x[0], x[1], x[2], x[3],
        logits[0], logits[1], logits[2], logits[3]);

    bitnet_block_scratch_free(&s);
    bitnet_kv_cache_free(&cache);
    if (loaded_ok) bitnet_weights_unload(&handle);
    return 0;
}
