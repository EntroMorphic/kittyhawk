/*
 * trix_ternary_route.h — Ternary-routed FFN
 *
 * Instead of argmax (one tile per token), each tile gets a ternary
 * relationship to each token: {+1, -1, 0}.
 *
 *   +1: tile contributes positively (expert)
 *   -1: tile contributes negatively (anti-expert)
 *    0: tile is irrelevant (skip)
 *
 * Output: sum_t( route[t] * tile_t(x) )  for route[t] ∈ {-1, 0, +1}
 *
 * Routing decision: compute score = dot(x, sig_t) for each tile,
 * then threshold to ternary. Top-k by |score| determines which tiles
 * are active; sign(score) determines +1 or -1.
 *
 * With T=8 tiles and top-1 routing: 8 patterns.
 * With T=8 tiles and ternary routing (k=3): C(8,3) * 2^3 = 448 patterns.
 * With T=8 tiles and full ternary: 3^8 = 6,561 patterns.
 *
 * Built on trix_atoms. All computation in NEON-optimized C.
 */

#ifndef TRIX_TERNARY_ROUTE_H
#define TRIX_TERNARY_ROUTE_H

#include "trix_types.h"  /* forward declarations */
#include "trix_atoms.h"  /* LayerNorm, AdamW, vector ops */
#include "trix_mtfp.h"   /* Multi-Trit Fixed Point arithmetic */
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int d_model;        /* input/output dimension */
    int num_tiles;      /* number of specialist tiles */
    int tile_hidden;    /* hidden dimension per tile */
    int active_k;       /* number of active tiles per token (top-k by |score|) */
    float output_scale_init;
    float ln_eps;
} TrixTernaryRouteConfig;

struct TrixTernaryRoutedFFN {
    TrixTernaryRouteConfig cfg;

    /* LayerNorm */
    float* ln_weight;       /* [d_model] */
    float* ln_bias;         /* [d_model] */

    /* Routing signatures: ternary {-1, 0, +1} derived from tile weights */
    float* signatures;      /* [num_tiles, d_model] */

    /* Per-tile FFN weights: float32 shadow weights for gradient updates */
    float* W1;              /* [num_tiles, tile_hidden, d_model] */
    float* b1;              /* [num_tiles, tile_hidden] */
    float* W2;              /* [num_tiles, d_model, tile_hidden] */
    float* b2;              /* [num_tiles, d_model] */

    /* Per-tile ternary packed weights: quantized from shadow weights each step.
     * Forward uses these via SDOT kernel — zero float multiplies. */
    int8_t* W1_tern;        /* [num_tiles, tile_hidden, d_model] ternary {-1,0,+1} */
    int8_t* W2_tern;        /* [num_tiles, d_model, tile_hidden] ternary {-1,0,+1} */
    uint8_t* W1_packed;     /* [num_tiles, tile_hidden, d_model/4] 2-bit packed */
    uint8_t* W2_packed;     /* [num_tiles, d_model, tile_hidden/4] 2-bit packed */

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

    /* Scratch */
    float* x_norm;          /* [batch_cap, d_model] */
    float* ln_mean;         /* [batch_cap] */
    float* ln_rstd;         /* [batch_cap] */
    float* scores;          /* [batch_cap, num_tiles] — dot product scores */
    int*   route;           /* [batch_cap, num_tiles] — ternary route {-1,0,+1} */
    float* z1;              /* [batch_cap, tile_hidden] */
    float* h1;              /* [batch_cap, tile_hidden] */
    float* tile_out;        /* [batch_cap, d_model] — one tile's output */
    float* combined;        /* [batch_cap, d_model] — sum of signed tile outputs */

    /* Saved for backward (per tile) */
    float** saved_z1;       /* [num_tiles] -> [batch_cap, tile_hidden] */
    float** saved_h1;       /* [num_tiles] -> [batch_cap, tile_hidden] */

    /* MTFP LayerNorm weights */
    mtfp_t* mln_weight;     /* [d_model] */
    mtfp_t* mln_bias;       /* [d_model] */

    /* MTFP scratch — tile computation runs in integer arithmetic */
    mtfp_t* mx_norm;        /* [batch_cap, d_model] MTFP activations */
    mtfp_t* mz1;            /* [batch_cap, tile_hidden] */
    mtfp_t* mh1;            /* [batch_cap, tile_hidden] */
    mtfp_t* mtile_out;      /* [batch_cap, d_model] */
    mtfp_t* mcombined;      /* [batch_cap, d_model] */
    mtfp_t* mb1;            /* [num_tiles * tile_hidden] biases in MTFP */
    mtfp_t* mb2;            /* [num_tiles * d_model] biases in MTFP */

    /* Backward scratch */
    float* d_combined;      /* [batch_cap, d_model] */
    float* d_tile_out;      /* [batch_cap, d_model] */
    float* dh1;             /* [batch_cap, tile_hidden] */
    float* dz1;             /* [batch_cap, tile_hidden] */
    float* dx_tile;         /* [batch_cap, d_model] pre-allocated for backward */

    int batch_cap;
    int adam_step;
};

/* ── Forward (MTFP native — zero conversion inside) ── */

/* x and out are mtfp_t*. No float↔MTFP conversion.
 * The entire computation stays in balanced ternary fixed-point. */
void trix_ternary_route_forward_mtfp(
    TrixTernaryRoutedFFN* tr,
    const mtfp_t* x, mtfp_t* out, int batch
);

/* ── Lifecycle ── */

TrixTernaryRoutedFFN* trix_ternary_route_create(
    TrixTernaryRouteConfig cfg, uint64_t seed
);

void trix_ternary_route_destroy(TrixTernaryRoutedFFN* tr);

/* ── Forward ── */

/*
 * Forward: LN → score → ternary threshold → signed tile sum → scale → residual
 *
 * x:   [batch, d_model] input
 * out: [batch, d_model] output: x + output_scale * sum_t(route_t * tile_t(LN(x)))
 */
void trix_ternary_route_forward(
    TrixTernaryRoutedFFN* tr,
    const float* x, float* out, int batch
);

/* ── Backward ── */

void trix_ternary_route_backward(
    TrixTernaryRoutedFFN* tr,
    const float* x, const float* dy, float* dx, int batch
);

void trix_ternary_route_zero_grad(TrixTernaryRoutedFFN* tr);

/* ── Optimizer ── */

void trix_ternary_route_adamw_step(
    TrixTernaryRoutedFFN* tr,
    float lr, float beta1, float beta2,
    float eps, float weight_decay
);

float trix_ternary_route_clip_grad_norm(TrixTernaryRoutedFFN* tr, float max_norm);

/* ── Signatures ── */

/* Update signatures from tile weights: sig_t = sign(sum_h(W1_t) - mean) */
void trix_ternary_route_update_signatures(TrixTernaryRoutedFFN* tr);

/* ── Diagnostics ── */

/* Get the ternary routing pattern from the last forward pass.
 * route_out: [batch, num_tiles] — each entry is -1, 0, or +1 */
void trix_ternary_route_get_routing(
    const TrixTernaryRoutedFFN* tr,
    int* route_out, int batch
);

/* Get per-tile activation counts: how many tokens had route[t] != 0 */
void trix_ternary_route_get_tile_activity(
    const TrixTernaryRoutedFFN* tr,
    int* pos_count, int* neg_count, int batch
);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_TERNARY_ROUTE_H */
