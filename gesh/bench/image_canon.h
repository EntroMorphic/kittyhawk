/*
 * image_canon.h — canonical image-data pipeline for Gesh consumer probes.
 *
 * The substrate-legal path from raw MNIST IDX bytes to unpacked ternary
 * trits suitable for Gesh's forward + lattice-update pipeline:
 *
 *   IDX bytes
 *     → MTFP-encoded pixels (one-shot ingestion; per `M4T_SUBSTRATE.md` §12)
 *     → per-image normalize (zero-mean, unit-variance, integer arithmetic)
 *     → direct ternary quantization (per-pixel sign-of-deviation)
 *     → unpacked m4t_trit_t (Gesh forward input)
 *
 * Discipline:
 *   - NO RANDOM PROJECTIONS. Each output trit corresponds to one specific
 *     input pixel. Spatial structure preserved end-to-end.
 *   - NO DENSE RESOLVERS. Output is purely ternary; downstream uses
 *     Hamming-distance routing.
 *   - Float arithmetic appears ONLY in the IDX → MTFP one-shot conversion
 *     (sanctioned non-runtime float site). All per-image work is integer.
 *
 * Lifted from `01MAY26_archived/src/glyph_dataset.c` and `glyph_sig.c`,
 * scoped to MNIST-only (CIFAR-10 and Fashion-MNIST loaders deferred until
 * a consumer asks). No deskew or gradients in this pass; if Gate 1 needs
 * them, they re-lift.
 */

#ifndef GESH_IMAGE_CANON_H
#define GESH_IMAGE_CANON_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MTFP-encoded image dataset. Per-pixel value range: [-3·SCALE, +3·SCALE]
 * after normalization (clipped at ±3σ). Pre-normalize: [0, SCALE]. */
typedef struct {
    m4t_mtfp_t* x_train;    /* [n_train × input_dim]   MTFP-encoded pixels */
    int*        y_train;    /* [n_train]               integer labels      */
    m4t_mtfp_t* x_test;     /* [n_test  × input_dim]                       */
    int*        y_test;     /* [n_test]                                    */
    int         n_train;
    int         n_test;
    int         input_dim;  /* img_w × img_h (784 for MNIST)               */
    int         img_w;
    int         img_h;
} image_canon_dataset_t;

/* Load MNIST IDX files from <dir>:
 *   <dir>/train-images-idx3-ubyte
 *   <dir>/train-labels-idx1-ubyte
 *   <dir>/t10k-images-idx3-ubyte
 *   <dir>/t10k-labels-idx1-ubyte
 *
 * Returns 0 on success. On failure leaves whatever allocations succeeded;
 * caller still calls image_canon_free to clean up. */
int image_canon_load_mnist(image_canon_dataset_t* ds, const char* dir);

/* Per-image zero-mean / unit-variance normalization, integer arithmetic
 * via Newton's-method int64 isqrt. Maps ±1σ to ±M4T_MTFP_SCALE; clips at
 * ±3σ to keep mantissas in int32 range. Idempotent only modulo the
 * stddev computation; calling twice will degenerate. Call once. */
void image_canon_normalize(image_canon_dataset_t* ds);

/* Compute tau for direct ternary quantization. Uses the `density`-th
 * percentile of |x| over all dims of the supplied sample. At density=0.6,
 * approximately 60% of pixels map to the structural zero. */
int64_t image_canon_quantize_tau(const m4t_mtfp_t* x_sample,
                                 int n_sample, int n_dims,
                                 double density);

/* Direct ternary quantization writing UNPACKED m4t_trit_t (Gesh forward's
 * native input format). One output trit per input pixel; no projection.
 *   x[i] >  +tau →  +1
 *   x[i] < -tau  → -1
 *   else         →  0 (structural zero — "this dim is uninformative")
 *
 * Output buffer must hold at least n × n_dims trits. */
void image_canon_quantize_unpacked_batch(const m4t_mtfp_t* x_batch,
                                          int n, int n_dims,
                                          int64_t tau,
                                          m4t_trit_t* out_trits);

void image_canon_free(image_canon_dataset_t* ds);

#ifdef __cplusplus
}
#endif

#endif /* GESH_IMAGE_CANON_H */
