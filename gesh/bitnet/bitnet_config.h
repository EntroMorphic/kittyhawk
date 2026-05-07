/*
 * gesh/bitnet/bitnet_config.h — BitNet b1.58-2B-4T model configuration
 * and per-block scratch layout. Per the LMM cycle in journal/bitnet_phase1_*.
 *
 * All dimensions are compile-time constants matching the released
 * checkpoint. A future-fork (b2.0, different parameter count) would
 * recompile this header against the new config.json.
 */

#ifndef GESH_BITNET_CONFIG_H
#define GESH_BITNET_CONFIG_H

#include "m4t_types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Model dimensions (locked to BitNet b1.58-2B-4T) ────────────────── */

#define BITNET_HIDDEN_SIZE          2560
#define BITNET_INTERMEDIATE_SIZE    6912
#define BITNET_NUM_LAYERS             30
#define BITNET_NUM_ATTENTION_HEADS    20
#define BITNET_NUM_KV_HEADS            5
#define BITNET_HEAD_DIM              128
#define BITNET_NUM_KV_GROUPS           (BITNET_NUM_ATTENTION_HEADS / BITNET_NUM_KV_HEADS)  /* GQA 4:1 */
#define BITNET_MAX_POSITION         4096
#define BITNET_VOCAB_SIZE         128256

/* RMSNorm epsilon, matching config.json. */
#define BITNET_RMS_NORM_EPS         (1e-5)

/* RoPE base frequency, matching config.json. */
#define BITNET_ROPE_THETA      500000.0

/* Static checks: head_dim derives from hidden_size and num_attention_heads. */
_Static_assert(
    BITNET_HEAD_DIM * BITNET_NUM_ATTENTION_HEADS == BITNET_HIDDEN_SIZE,
    "BitNet config: head_dim × num_attention_heads must equal hidden_size"
);
_Static_assert(
    BITNET_NUM_ATTENTION_HEADS % BITNET_NUM_KV_HEADS == 0,
    "BitNet config: num_attention_heads must be divisible by num_kv_heads"
);

/* ── Per-layer weight shapes ────────────────────────────────────────── */
/*
 * All BitLinear layers store ternary weights in 5-in-8 packed format
 * (substrate-side packing per work-unit 1's R3 decision in synthesize).
 * Shapes assume row-major W^T layout (each row j holds K trits of
 * column j of W) — matches m4t_ternary_5in8_matmul_*'s W_packed convention.
 *
 * Linear shapes (output_dim × input_dim, packed bytes per row):
 *   W_q:    HIDDEN × HIDDEN                  — Q projection
 *   W_k:    (NUM_KV_HEADS × HEAD_DIM) × HIDDEN — K projection (GQA: smaller than Q)
 *   W_v:    (NUM_KV_HEADS × HEAD_DIM) × HIDDEN — V projection (GQA: smaller than Q)
 *   W_o:    HIDDEN × HIDDEN                  — output projection
 *   W_gate: INTERMEDIATE × HIDDEN
 *   W_up:   INTERMEDIATE × HIDDEN
 *   W_down: HIDDEN × INTERMEDIATE
 *
 * RMSNorm γ vectors (4 per layer):
 *   input_layernorm.γ:           [HIDDEN]
 *   post_attention_layernorm.γ:  [HIDDEN]
 *   attn_sub_norm.γ:             [HIDDEN]    (sub-LN inside attention)
 *   ffn_sub_norm.γ:              [INTERMEDIATE] (sub-LN inside FFN, between gate*up and down)
 */

#define BITNET_KV_PROJ_DIM   (BITNET_NUM_KV_HEADS * BITNET_HEAD_DIM)  /* = 640 */

/* ── Storage helpers ────────────────────────────────────────────────── */

/* 5-in-8 packed bytes for a row of K trits. */
#define BITNET_PACKED5(K)    (((K) + 4) / 5)

/* Per-layer packed weight sizes (for sizing the binary blob). */
#define BITNET_W_Q_PACKED_BYTES    (BITNET_HIDDEN_SIZE  * BITNET_PACKED5(BITNET_HIDDEN_SIZE))
#define BITNET_W_K_PACKED_BYTES    (BITNET_KV_PROJ_DIM   * BITNET_PACKED5(BITNET_HIDDEN_SIZE))
#define BITNET_W_V_PACKED_BYTES    (BITNET_KV_PROJ_DIM   * BITNET_PACKED5(BITNET_HIDDEN_SIZE))
#define BITNET_W_O_PACKED_BYTES    (BITNET_HIDDEN_SIZE  * BITNET_PACKED5(BITNET_HIDDEN_SIZE))
#define BITNET_W_GATE_PACKED_BYTES (BITNET_INTERMEDIATE_SIZE * BITNET_PACKED5(BITNET_HIDDEN_SIZE))
#define BITNET_W_UP_PACKED_BYTES   (BITNET_INTERMEDIATE_SIZE * BITNET_PACKED5(BITNET_HIDDEN_SIZE))
#define BITNET_W_DOWN_PACKED_BYTES (BITNET_HIDDEN_SIZE  * BITNET_PACKED5(BITNET_INTERMEDIATE_SIZE))

/* ── Per-layer weight set ───────────────────────────────────────────── */

/* Pointers into the mmap'd weights blob. The owner sets these once during
 * load and they're read-only thereafter. */
typedef struct {
    /* BitLinear weights (5-in-8 packed ternary, W^T layout). */
    const uint8_t* w_q;        /* [HIDDEN × ⌈HIDDEN/5⌉] */
    const uint8_t* w_k;        /* [KV_PROJ × ⌈HIDDEN/5⌉] */
    const uint8_t* w_v;        /* [KV_PROJ × ⌈HIDDEN/5⌉] */
    const uint8_t* w_o;        /* [HIDDEN × ⌈HIDDEN/5⌉] */
    const uint8_t* w_gate;     /* [INTERMEDIATE × ⌈HIDDEN/5⌉] */
    const uint8_t* w_up;       /* [INTERMEDIATE × ⌈HIDDEN/5⌉] */
    const uint8_t* w_down;     /* [HIDDEN × ⌈INTERMEDIATE/5⌉] */

    /* BitLinear α scales (work-unit 5). One scalar per projection.
     * α stored as MTFP19 (mantissa + per-tensor block_exp).
     * Real α = mantissa[0] × 3^block_exp. Used at the BitLinear scale
     * apply step: y = matmul_out × α × absmax / 127. */
    const m4t_mtfp_t* alpha_q;     int alpha_q_block_exp;
    const m4t_mtfp_t* alpha_k;     int alpha_k_block_exp;
    const m4t_mtfp_t* alpha_v;     int alpha_v_block_exp;
    const m4t_mtfp_t* alpha_o;     int alpha_o_block_exp;
    const m4t_mtfp_t* alpha_gate;  int alpha_gate_block_exp;
    const m4t_mtfp_t* alpha_up;    int alpha_up_block_exp;
    const m4t_mtfp_t* alpha_down;  int alpha_down_block_exp;

    /* RMSNorm γ scales (MTFP19 mantissas). RC-4: use m4t_mtfp_t to make
     * the substrate-native semantic explicit (typedef'd to int32_t but
     * the type carries the intent). */
    const m4t_mtfp_t* gamma_input_norm;       /* [HIDDEN] */
    const m4t_mtfp_t* gamma_post_attn_norm;   /* [HIDDEN] */
    const m4t_mtfp_t* gamma_attn_sub_norm;    /* [HIDDEN] */
    const m4t_mtfp_t* gamma_ffn_sub_norm;     /* [INTERMEDIATE] */
} bitnet_layer_weights_t;

/* ── Whole-model weight set ─────────────────────────────────────────── */

typedef struct {
    /* Embedding + LM head (likely tied; same buffer, different access). */
    const m4t_mtfp_t* embedding;        /* [VOCAB × HIDDEN] MTFP19 mantissas */
    const m4t_mtfp_t* lm_head;          /* [VOCAB × HIDDEN] MTFP19 mantissas (may alias embedding) */
    /* Final pre-LM-head norm. */
    const m4t_mtfp_t* gamma_final_norm; /* [HIDDEN] */
    /* Per-layer weight sets. */
    bitnet_layer_weights_t layers[BITNET_NUM_LAYERS];
} bitnet_weights_t;

/* ── Per-call activation buffers ────────────────────────────────────── */

/* Activation buffers for ONE token's forward pass through ONE block.
 * Allocated once and reused across blocks (overwritten per-block). KV
 * cache buffers are separate (work-unit 7).
 *
 * RC-4: all activation buffers use m4t_mtfp_t (== int32_t) to make the
 * substrate-native semantic explicit at every call site.
 *
 * Naming: shape suffixes denote `[seq_len, hidden]` or similar. M=1 for
 * single-token prefill; multi-token prefill is a Phase-1-batch concern
 * (work-unit 8 if we get there).
 */
typedef struct {
    /* Block input/output (also residual buffer). MTFP19. */
    m4t_mtfp_t* x;                /* [HIDDEN] */
    m4t_mtfp_t* residual;         /* [HIDDEN] — pre-norm copy of x */
    /* Post-norm intermediate. MTFP19. */
    m4t_mtfp_t* x_norm;           /* [HIDDEN] */
    /* QKV projections (post-BitLinear, MTFP19). */
    m4t_mtfp_t* q;                /* [NUM_ATTENTION_HEADS × HEAD_DIM] = [HIDDEN] */
    m4t_mtfp_t* k;                /* [NUM_KV_HEADS × HEAD_DIM] = [KV_PROJ_DIM] */
    m4t_mtfp_t* v;                /* [NUM_KV_HEADS × HEAD_DIM] = [KV_PROJ_DIM] */
    /* Attention scratch. */
    m4t_mtfp_t* attn_scores;      /* [NUM_ATTENTION_HEADS × seq_q × seq_k] — small for single-token */
    m4t_mtfp_t* attn_output;      /* [HIDDEN] post-attention, pre-O */
    m4t_mtfp_t* attn_sub_norm;    /* [HIDDEN] post-sub-LN, pre-O */
    /* FFN scratch. */
    m4t_mtfp_t* gate;             /* [INTERMEDIATE] */
    m4t_mtfp_t* up;               /* [INTERMEDIATE] */
    m4t_mtfp_t* gate_act;         /* [INTERMEDIATE] = relu²(gate) * up */
    m4t_mtfp_t* ffn_sub_norm;     /* [INTERMEDIATE] post-sub-LN, pre-down */
    /* A8 quantization scratch (per-token: one int8 buffer + absmax per quantize call). */
    int8_t*    q_int8;            /* [INTERMEDIATE or HIDDEN] worst case */
    m4t_mtfp_t q_absmax;          /* per-token absmax (RC-3: rename from q_scale) */
} bitnet_block_scratch_t;

/* ── Allocation / free ──────────────────────────────────────────────── */

/* Allocate scratch buffers sized for one transformer block. Caller must
 * call bitnet_block_scratch_free before exit. */
void bitnet_block_scratch_alloc(bitnet_block_scratch_t* s);
void bitnet_block_scratch_free (bitnet_block_scratch_t* s);

#ifdef __cplusplus
}
#endif

#endif /* GESH_BITNET_CONFIG_H */
