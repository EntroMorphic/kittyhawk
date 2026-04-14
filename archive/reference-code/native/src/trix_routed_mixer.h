/*
 * trix_routed_mixer.h — Routed Token Mixer
 *
 * Replaces attention with ternary-routed cross-position interaction.
 * Tokens that share a routing destination pool together. Tiles transform
 * the pooled representation. The routing pattern distributes tile outputs
 * back to positions.
 *
 * Forward:
 *   LayerNorm → Score → Threshold → Scatter → Tile FFN → Gather → Scale → Residual
 *
 * Scatter: pool[t] = signed sum of tokens routed to tile t
 * Gather:  y[pos] = signed sum of tile outputs for pos's routing pattern
 *
 * No dense matmul. No softmax. No Q@K^T.
 * All ternary-routed, all MTFP integer arithmetic in the forward path.
 *
 * Complexity: O(seq_len × T × D) — linear in sequence length.
 */

#ifndef TRIX_ROUTED_MIXER_H
#define TRIX_ROUTED_MIXER_H

#include "trix_types.h"
#include "trix_atoms.h"
#include "trix_mtfp.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int d_model;
    int num_tiles;
    int tile_hidden;
    int active_k;
    float output_scale_init;
    float ln_eps;
} TrixRoutedMixerConfig;

struct TrixRoutedMixer {
    TrixRoutedMixerConfig cfg;

    /* LayerNorm */
    float* ln_weight;       /* [D] */
    float* ln_bias;         /* [D] */

    /* Routing signatures: ternary {-1, 0, +1}, weight-derived */
    float* signatures;      /* [T, D] */

    /* Per-tile FFN: float32 shadow weights */
    float* W1;              /* [T, H, D] */
    float* b1;              /* [T, H] */
    float* W2;              /* [T, D, H] */
    float* b2;              /* [T, D] */

    /* Per-tile ternary weights: quantized from shadow weights */
    int8_t* W1_tern;        /* [T, H, D] */
    int8_t* W2_tern;        /* [T, D, H] */

    float output_scale;

    /* Gradients */
    float* dln_weight;
    float* dln_bias;
    float* dW1;
    float* db1;
    float* dW2;
    float* db2;
    float  doutput_scale;

    /* AdamW moments */
    float* m_ln_w; float* v_ln_w;
    float* m_ln_b; float* v_ln_b;
    float* m_W1; float* v_W1;
    float* m_b1; float* v_b1;
    float* m_W2; float* v_W2;
    float* m_b2; float* v_b2;
    float  m_output_scale;
    float  v_output_scale;

    /* MTFP tile FFN weights */
    mtfp_t* mb1;            /* [T * H] */
    mtfp_t* mb2;            /* [T * D] */
    mtfp_t* mln_weight;     /* [D] */
    mtfp_t* mln_bias;       /* [D] */

    /* Scratch — forward */
    float* x_norm;          /* [seq_cap, D] */
    float* ln_mean;         /* [seq_cap] */
    float* ln_rstd;         /* [seq_cap] */
    float* scores;          /* [seq_cap, T] */
    int*   route;           /* [seq_cap, T] */

    /* Scatter/gather scratch */
    mtfp_t* mx_norm;        /* [seq_cap, D] MTFP */
    mtfp_t* pool;           /* [T, D] MTFP — scattered token pools */
    int*    pool_counts;    /* [T] — tokens routed to each tile */
    mtfp_t* mz1;            /* [T, H] MTFP — tile FFN pre-activation */
    mtfp_t* mh1;            /* [T, H] MTFP — tile FFN post-GELU */
    mtfp_t* mixed;          /* [T, D] MTFP — tile FFN output */
    mtfp_t* mgathered;      /* [seq_cap, D] MTFP — gathered output */

    /* Saved for backward (float) */
    float* saved_pool;      /* [T, D] */
    float* saved_z1;        /* [T, H] */
    float* saved_h1;        /* [T, H] */
    float* saved_mixed;     /* [T, D] */

    /* Backward scratch */
    float* d_gathered;      /* [seq_cap, D] */
    float* d_mixed;         /* [T, D] */
    float* dh1;             /* [T, H] */
    float* dz1;             /* [T, H] */
    float* d_pool;          /* [T, D] */
    float* dx_scatter;      /* [seq_cap, D] */

    int seq_cap;
    int adam_step;
};

typedef struct TrixRoutedMixer TrixRoutedMixer;

/* ── Lifecycle ── */

TrixRoutedMixer* trix_routed_mixer_create(TrixRoutedMixerConfig cfg, uint64_t seed);
void trix_routed_mixer_destroy(TrixRoutedMixer* rm);

/* ── Forward ──
 *
 * x:   [seq_len, D] input token representations
 * out: [seq_len, D] output: x + output_scale * gather(tile_ffn(scatter(LN(x))))
 */
void trix_routed_mixer_forward(TrixRoutedMixer* rm,
                                const float* x, float* out, int seq_len);

/* ── Backward ── */

void trix_routed_mixer_backward(TrixRoutedMixer* rm,
                                 const float* x, const float* dy, float* dx,
                                 int seq_len);

void trix_routed_mixer_zero_grad(TrixRoutedMixer* rm);

/* ── Optimizer ── */

void trix_routed_mixer_adamw_step(TrixRoutedMixer* rm,
                                   float lr, float beta1, float beta2,
                                   float eps, float weight_decay);

float trix_routed_mixer_clip_grad_norm(TrixRoutedMixer* rm, float max_norm);

/* ── Signatures ── */

void trix_routed_mixer_update_signatures(TrixRoutedMixer* rm);

/* ── Scatter/Gather Atoms ── */

/* Scatter: pool[t] = signed sum of x[pos] for positions routed to tile t.
 * pool:    [T, D] output, zeroed then accumulated.
 * counts:  [T] output, number of contributing tokens per tile.
 * x:       [seq_len, D] input.
 * route:   [seq_len, T] ternary routing pattern. */
void trix_routed_scatter(mtfp_t* pool, int* counts,
                          const mtfp_t* x, const int* route,
                          int seq_len, int D, int T);

/* Gather: y[pos] = signed sum of mixed[t] for tiles routed to position pos.
 * y:       [seq_len, D] output, zeroed then accumulated.
 * mixed:   [T, D] input (tile FFN outputs).
 * route:   [seq_len, T] ternary routing pattern. */
void trix_routed_gather(mtfp_t* y, const mtfp_t* mixed,
                         const int* route,
                         int seq_len, int D, int T);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_ROUTED_MIXER_H */
