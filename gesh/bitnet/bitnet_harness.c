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

#include "m4t_ternary_matmul.h"
#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
    bitnet_stub_rmsnorm(s->x_norm, s->x, w->gamma_input_norm,
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

    /* Q = x_norm @ W_q^T   (packed-ternary matmul, output MTFP19) */
    /* m4t_ternary_5in8_matmul_xpacked_bt(s->q, x_packed, w->w_q,
     *                                     1, BITNET_HIDDEN_SIZE,
     *                                     BITNET_HIDDEN_SIZE);
     * STUB: zero out for now. */
    memset(s->q, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));
    /* K = x_norm @ W_k^T */
    memset(s->k, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));
    /* V = x_norm @ W_v^T */
    memset(s->v, 0, BITNET_KV_PROJ_DIM * sizeof(m4t_mtfp_t));

    /* RoPE on Q, K. */
    bitnet_stub_rope_apply(s->q, s->k, position,
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
    bitnet_stub_rmsnorm(s->attn_sub_norm, s->attn_output,
                        w->gamma_attn_sub_norm, 1, BITNET_HIDDEN_SIZE);

    /* O projection: y = attn_sub_norm @ W_o^T. STUB. */
    /* Result goes back into s->x. */
    memcpy(s->x, s->attn_sub_norm, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

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
    bitnet_stub_rmsnorm(s->x_norm, s->x, w->gamma_post_attn_norm,
                        1, BITNET_HIDDEN_SIZE);

    /* gate = x_norm @ W_gate^T. STUB. */
    memset(s->gate, 0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));
    /* up = x_norm @ W_up^T. STUB. */
    memset(s->up, 0, BITNET_INTERMEDIATE_SIZE * sizeof(m4t_mtfp_t));

    /* gate_act = relu²(gate). */
    bitnet_stub_relu2_inplace(s->gate, BITNET_INTERMEDIATE_SIZE);
    /* gate_act = gate * up. */
    bitnet_stub_elementwise_mul(s->gate_act, s->gate, s->up,
                                BITNET_INTERMEDIATE_SIZE);

    /* ffn_sub_norm(gate_act). */
    bitnet_stub_rmsnorm(s->ffn_sub_norm, s->gate_act,
                        w->gamma_ffn_sub_norm, 1, BITNET_INTERMEDIATE_SIZE);

    /* down = ffn_sub_norm @ W_down^T. STUB. */
    memset(s->x, 0, BITNET_HIDDEN_SIZE * sizeof(m4t_mtfp_t));

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

int main(int argc, char** argv) {
    fprintf(stderr,
        "bitnet_harness — Phase 1 work-unit 1 skeleton.\n"
        "Status: SCAFFOLDING. Forward pass shape laid out; weight loader\n"
        "and per-layer matmul calls stubbed. Subsequent work-unit-1 commits\n"
        "will fill these in alongside the Python conversion + reference\n"
        "comparison drivers.\n\n"
    );

    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <weights_blob.bin>\n"
            "  weights_blob.bin — produced by scripts/convert_weights.py\n"
            "                     from the released BitNet b1.58-2B-4T checkpoint.\n",
            argv[0]);
        return 1;
    }

    /* TODO: mmap weights blob, populate bitnet_weights_t. */
    fprintf(stderr, "[stub] would load weights from: %s\n", argv[1]);

    /* Allocate scratch + a dummy input vector to verify the forward pass
     * shape compiles and runs without crashing. */
    bitnet_block_scratch_t s = {0};
    bitnet_block_scratch_alloc(&s);

    m4t_mtfp_t x[BITNET_HIDDEN_SIZE];
    memset(x, 0, sizeof(x));
    /* Dummy weights — all-zero pointers will be replaced by real load. */
    bitnet_layer_weights_t w_layer0 = {0};

    /* Currently the forward pass uses only γ pointers (for stubbed
     * RMSNorm calls); BitLinear weights are stubbed to zero output.
     * γ = NULL would crash in the stub — fake it with an all-ones γ.
     * Sized for the largest norm (FFN sub-norm, INTERMEDIATE_SIZE);
     * the smaller HIDDEN-sized norms read only the first HIDDEN cells. */
    m4t_mtfp_t gamma_dummy[BITNET_INTERMEDIATE_SIZE];
    for (int i = 0; i < BITNET_INTERMEDIATE_SIZE; i++) gamma_dummy[i] = 1;
    w_layer0.gamma_input_norm     = gamma_dummy;
    w_layer0.gamma_post_attn_norm = gamma_dummy;
    w_layer0.gamma_attn_sub_norm  = gamma_dummy;
    w_layer0.gamma_ffn_sub_norm   = gamma_dummy;

    bitnet_forward_block(x, &w_layer0, &s, /*position=*/0);

    fprintf(stderr,
        "[ok] layer 0 forward pass completed without crashing.\n"
        "     output[0..3] = %d %d %d %d\n",
        x[0], x[1], x[2], x[3]);

    bitnet_block_scratch_free(&s);
    return 0;
}
