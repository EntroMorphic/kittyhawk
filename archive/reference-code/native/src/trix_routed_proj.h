/*
 * trix_routed_proj.h — Ternary-routed linear projection
 *
 * Replaces a dense linear projection [in_dim → out_dim] with T ternary-routed
 * tile projections. Each tile has its own weight matrix; the output is a signed
 * sum of active tile outputs: sum_t(route_t * (x @ W_t^T + b_t)).
 *
 * route_t in {-1, 0, +1}: top-k by |score|, sign preserved.
 * Signatures derived from Q-portion column sums of tile weights.
 *
 * This is the building block for routed QKV, routed W_O, and routed LM head.
 * Same mechanism as the ternary routed FFN, but for linear projections.
 *
 * With T=4 and K=4, the effective projection is one of 3^4 = 81 signed
 * combinations of 4 basis matrices. This is a structured adaptive projection —
 * mathematically equivalent to selecting from a discrete set of weight blends.
 */

#ifndef TRIX_ROUTED_PROJ_H
#define TRIX_ROUTED_PROJ_H

#include "trix_types.h"
#include "trix_atoms.h"
#include "trix_mtfp.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int in_dim;
    int out_dim;
    int num_tiles;
    int active_k;
    int sig_cols;       /* columns of W to use for signature derivation (0 = all) */
    float output_scale_init;
    float ln_eps;
    bool use_layernorm; /* true for QKV (pre-norm), false for W_O */
} TrixRoutedProjConfig;

typedef struct TrixRoutedProj TrixRoutedProj;
struct TrixRoutedProj {
    TrixRoutedProjConfig cfg;

    /* LayerNorm (optional) */
    float* ln_weight;       /* [in_dim] or NULL */
    float* ln_bias;         /* [in_dim] or NULL */

    /* Per-tile weights: each tile is a full [out_dim, in_dim] projection */
    float* W;               /* [T * out_dim * in_dim] */
    float* b;               /* [T * out_dim] */

    /* Signatures (derived from tile W columns) */
    float* signatures;      /* [T * in_dim] */

    /* Ternary weights + MTFP biases (quantized from shadow weights each step) */
    int8_t* W_tern;         /* [T * out_dim * in_dim] ternary {-1,0,+1} */
    mtfp_t* mb;             /* [T * out_dim] biases in MTFP */
    mtfp_t* mln_w;          /* [in_dim] LN weight in MTFP (pre-computed) */
    mtfp_t* mln_b;          /* [in_dim] LN bias in MTFP (pre-computed) */

    /* Output scale */
    float output_scale;

    /* Gradients */
    float* dln_weight;
    float* dln_bias;
    float* dW;              /* [T * out_dim * in_dim] */
    float* db;              /* [T * out_dim] */
    float doutput_scale;

    /* AdamW moments */
    float* m_ln_w; float* v_ln_w;
    float* m_ln_b; float* v_ln_b;
    float* m_W; float* v_W;
    float* m_b; float* v_b;
    float m_os; float v_os;

    /* Scratch */
    float* x_norm;          /* [batch, in_dim] */
    float* ln_mean;         /* [batch] */
    float* ln_rstd;         /* [batch] */
    float* scores;          /* [batch, T] */
    int*   route;           /* [batch, T] */
    float* tile_out;        /* [batch, out_dim] */
    float* combined;        /* [batch, out_dim] */

    /* MTFP scratch */
    mtfp_t* mx_norm;        /* [batch, in_dim] */
    mtfp_t* mtile_out;      /* [batch, out_dim] */
    mtfp_t* mcombined;      /* [batch, out_dim] */

    /* Backward scratch */
    float* d_combined;      /* [batch, out_dim] */
    float* d_tile_out;      /* [batch, out_dim] */
    float* dx_accum;        /* [batch, in_dim] */
    float* dW_tmp;          /* [out_dim, in_dim] temp for matmul_at */
    float* dx_tmp;          /* [batch, in_dim] temp for matmul dx */
    float* dx_pre_ln;       /* [batch, in_dim] temp for LN backward */

    int batch_cap;
    int adam_step;
};

/* Lifecycle */
TrixRoutedProj* trix_routed_proj_create(TrixRoutedProjConfig cfg, uint64_t seed);
void trix_routed_proj_destroy(TrixRoutedProj* rp);

/* Forward (MTFP native): x and out are mtfp_t*. Zero float in tile computation. */
void trix_routed_proj_forward_mtfp(TrixRoutedProj* rp, const mtfp_t* x, mtfp_t* out, int batch);

/* Forward (float): x[batch, in_dim] → out[batch, out_dim]
 * out = output_scale * sum_t(route_t * (x_norm @ W_t^T + b_t))
 * NOTE: no residual. Caller adds residual if needed. */
void trix_routed_proj_forward(TrixRoutedProj* rp, const float* x, float* out, int batch);

/* Backward: dy[batch, out_dim] → dx[batch, in_dim] (may be NULL)
 * Accumulates into dW, db, dln, doutput_scale. */
void trix_routed_proj_backward(TrixRoutedProj* rp, const float* x, const float* dy, float* dx, int batch);

/* Optimizer */
void trix_routed_proj_zero_grad(TrixRoutedProj* rp);
void trix_routed_proj_adamw_step(TrixRoutedProj* rp, float lr, float b1, float b2, float eps, float wd);

/* Signatures: derive from tile weight columns */
void trix_routed_proj_update_signatures(TrixRoutedProj* rp);

/* Grad norm (returns sum of squares for global clipping) */
float trix_routed_proj_grad_sq(const TrixRoutedProj* rp);

/* Scale all gradients (for global clipping) */
void trix_routed_proj_scale_grad(TrixRoutedProj* rp, float scale);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_ROUTED_PROJ_H */
