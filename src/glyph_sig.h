/*
 * glyph_sig.h — ternary signature builders.
 *
 * Two signature paths:
 *
 *   DIRECT QUANTIZATION (preferred for image classification):
 *     glyph_sig_quantize / glyph_sig_quantize_batch — quantize each
 *     input dimension to a trit via per-value thresholding. Each trit
 *     represents a SPECIFIC input value (pixel, gradient), not a
 *     random mixture. Preserves spatial identity. Use this for image
 *     data after normalization.
 *
 *   RANDOM PROJECTION (legacy, retained for non-image domains):
 *     glyph_sig_builder_init — generates a random ternary projection
 *     matrix and calibrates τ. Each trit is a random linear combination
 *     of ~D/3 input dimensions. DESTROYS spatial structure. Do NOT use
 *     for image classification — direct quantization is strictly
 *     superior on every measured dataset (MNIST, Fashion-MNIST, CIFAR-10).
 *
 * Under the hood both paths produce packed-trit signatures compatible
 * with glyph_bucket, glyph_multiprobe, and all resolver variants.
 *
 * Typical use (init → encode → free):
 *
 *     glyph_sig_builder_t sb;
 *     if (glyph_sig_builder_init(
 *             &sb,
 *             16,                         // N_PROJ (signature dimension)
 *             ds.input_dim,               // e.g. 784 for MNIST
 *             0.33,                       // balanced base-3 density
 *             42, 123, 456, 789,          // RNG seed quadruple
 *             ds.x_train, 1000) != 0) {   // calibration subset
 *         // handle OOM
 *     }
 *
 *     uint8_t* train_sigs = calloc((size_t)ds.n_train * sb.sig_bytes, 1);
 *     glyph_sig_encode_batch(&sb, ds.x_train, ds.n_train, train_sigs);
 *
 *     uint8_t* test_sigs = calloc((size_t)ds.n_test * sb.sig_bytes, 1);
 *     glyph_sig_encode_batch(&sb, ds.x_test, ds.n_test, test_sigs);
 *
 *     // ... feed train_sigs into glyph_bucket_build, query via test_sigs ...
 *
 *     free(train_sigs);
 *     free(test_sigs);
 *     glyph_sig_builder_free(&sb);
 */

#ifndef GLYPH_SIG_H
#define GLYPH_SIG_H

#include <stdint.h>
#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int       n_proj;       /* signature dimension in trits                */
    int       input_dim;    /* vector dimension                            */
    int       sig_bytes;    /* M4T_TRIT_PACKED_BYTES(n_proj)               */
    uint8_t*  proj_packed;  /* [n_proj × ceil(input_dim/4)] packed trits   */
    int64_t   tau_q;        /* calibrated threshold                        */
    double    density;      /* target density used for calibration         */
    uint32_t  seed[4];      /* seed used at init, for reproducibility      */
} glyph_sig_builder_t;

/* Initialize a signature builder. Generates a random ternary projection
 * matrix from the given seed, then calibrates tau on the provided
 * calibration set (typically a subset of the training data).
 *
 * calibration_set points at n_calib × input_dim MTFP values. The
 * calibration uses the |W @ x| distribution percentile at `density`
 * to set tau. density=0.33 gives the balanced base-3 deployment.
 *
 * Returns 0 on success, non-zero on allocation failure. */
int glyph_sig_builder_init(
    glyph_sig_builder_t* sb,
    int n_proj,
    int input_dim,
    double density,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
    const m4t_mtfp_t* calibration_set,
    int n_calib);

/* Encode a single vector into a packed trit signature (sig_bytes bytes). */
void glyph_sig_encode(const glyph_sig_builder_t* sb,
                      const m4t_mtfp_t* x,
                      uint8_t* out_sig);

/* Encode n vectors stored contiguously at x_batch; writes n × sig_bytes
 * bytes to out_sigs. */
void glyph_sig_encode_batch(const glyph_sig_builder_t* sb,
                            const m4t_mtfp_t* x_batch,
                            int n,
                            uint8_t* out_sigs);

void glyph_sig_builder_free(glyph_sig_builder_t* sb);

/* ================================================================
 * Direct ternary quantization (preferred for image classification).
 *
 * Quantize each input dimension to a trit:
 *   value >  +tau → +1
 *   value <  -tau → -1
 *   |value| <= tau →  0  (structural zero: "this dimension is uninformative")
 *
 * The structural zero filters noise. On normalized images, small
 * values near the mean map to zero (genuinely uninformative). Strong
 * deviations map to ±1 (class-discriminative signal).
 *
 * The output is a packed-trit signature of n_dims trits, compatible
 * with glyph_bucket, glyph_multiprobe, and all resolver variants.
 *
 * DO NOT use glyph_sig_builder_init for image classification.
 * Direct quantization preserves spatial identity (each trit = one
 * pixel or gradient). Random projection destroys it (each trit =
 * random mixture of ~D/3 pixels).
 * ================================================================ */

/* Quantize a single MTFP vector of length n_dims into a packed-trit
 * signature. out_sig must be at least M4T_TRIT_PACKED_BYTES(n_dims)
 * bytes, zeroed by caller. */
void glyph_sig_quantize(const m4t_mtfp_t* x, int n_dims,
                        int64_t tau, uint8_t* out_sig);

/* Quantize n vectors stored contiguously; writes
 * n × M4T_TRIT_PACKED_BYTES(n_dims) bytes to out_sigs. */
void glyph_sig_quantize_batch(const m4t_mtfp_t* x_batch, int n,
                              int n_dims, int64_t tau, uint8_t* out_sigs);

/* Compute tau for direct quantization from a calibration sample.
 * Returns the density-th percentile of |x| across all dimensions
 * in the sample. At density=0.60, ~60% of values map to zero. */
int64_t glyph_sig_quantize_tau(const m4t_mtfp_t* x_sample,
                               int n_sample, int n_dims, double density);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_SIG_H */
