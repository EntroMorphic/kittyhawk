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
static void bitnet_forward_block(
    m4t_mtfp_t* x_io,
    const bitnet_layer_weights_t* w,
    bitnet_block_scratch_t* s,
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

    /* Attention scores = Q @ K^T * (1/sqrt(head_dim)).
     * Single-token decode: K from cache is [n_kv_heads × seq_k × head_dim];
     * for work-unit 1 we don't have a cache yet, so seq_k = 1 (just the
     * current token attending to itself). Sanity-check shape only. */
    /* TODO: Q @ K^T scaled, then softmax, then × V.
     * Stubbed for now. */
    memset(s->attn_scores, 0, BITNET_NUM_ATTENTION_HEADS * sizeof(m4t_mtfp_t));
    memset(s->attn_output, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

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
    bitnet_stub_relu2_inplace(s->gate, BITNET_INTERMEDIATE_SIZE);
    /* gate_act = gate * up. */
    bitnet_stub_elementwise_mul(s->gate_act, s->gate, s->up,
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

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <weights_blob.bin> [dump_path]\n"
            "  weights_blob.bin — produced by scripts/convert_weights.py.\n"
            "                     If file doesn't exist or arg is '-', runs\n"
            "                     in skeleton mode (zero output, no weights).\n"
            "  dump_path        — optional. If set, dumps layer 0 captured\n"
            "                     activations to <dump_path> for comparison\n"
            "                     against scripts/dump_reference.py output.\n",
            argv[0]);
        return 1;
    }

    bitnet_weights_t weights = {0};
    bitnet_weights_loaded_t handle = {0};
    int loaded_ok = 0;

    if (strcmp(argv[1], "-") != 0) {
        if (bitnet_weights_load(argv[1], &weights, &handle) == 0) {
            loaded_ok = 1;
            fprintf(stderr, "[harness] loaded weights from %s\n", argv[1]);
        } else {
            fprintf(stderr, "[harness] weight load failed; running skeleton mode\n");
        }
    } else {
        fprintf(stderr, "[harness] skeleton mode (no weights)\n");
    }

    /* Allocate scratch. */
    bitnet_block_scratch_t s = {0};
    bitnet_block_scratch_alloc(&s);

    /* Input vector. For real inference, this would be the embedding lookup
     * for a token id. For skeleton, use a small non-zero pattern so the
     * forward pass exercises non-trivial values. */
    m4t_mtfp_t x[BITNET_HIDDEN_SIZE];
    for (int i = 0; i < BITNET_HIDDEN_SIZE; i++) {
        x[i] = (i % 7) - 3;  /* small values in [-3, +3] */
    }

    /* Choose layer 0's weights — real if loaded, dummy otherwise. */
    bitnet_layer_weights_t* w_layer0;
    bitnet_layer_weights_t w_dummy = {0};
    static m4t_mtfp_t gamma_dummy[BITNET_INTERMEDIATE_SIZE];
    if (loaded_ok && weights.layers[0].w_q != NULL) {
        w_layer0 = &weights.layers[0];
    } else {
        for (int i = 0; i < BITNET_INTERMEDIATE_SIZE; i++) gamma_dummy[i] = 1;
        w_dummy.gamma_input_norm     = gamma_dummy;
        w_dummy.gamma_post_attn_norm = gamma_dummy;
        w_dummy.gamma_attn_sub_norm  = gamma_dummy;
        w_dummy.gamma_ffn_sub_norm   = gamma_dummy;
        w_layer0 = &w_dummy;
    }

    bitnet_forward_block(x, w_layer0, &s, /*position=*/0);

    fprintf(stderr,
        "[ok] layer 0 forward pass completed.\n"
        "     output[0..3] = %d %d %d %d\n",
        x[0], x[1], x[2], x[3]);

    /* Optional activation dump. */
    if (argc >= 3) {
        const char* dump_path = argv[2];
        if (dump_activations_to_file(dump_path, &s, /*layer=*/0) == 0) {
            fprintf(stderr, "[ok] dumped activations to %s\n", dump_path);
        }
    }

    bitnet_block_scratch_free(&s);
    if (loaded_ok) bitnet_weights_unload(&handle);
    return 0;
}
