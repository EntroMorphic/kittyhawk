/*
 * glyph_dataset.h — MNIST IDX loader and integer-moment deskew.
 *
 * The dataset struct owns MTFP-encoded pixel data for train and test
 * splits, plus integer labels. glyph_dataset_deskew applies per-image
 * integer shear correction based on image moments (no float) that
 * improves downstream k-NN accuracy on MNIST by ~0.5-1 point.
 *
 * IDX file format: http://yann.lecun.com/exdb/mnist/
 */

#ifndef GLYPH_DATASET_H
#define GLYPH_DATASET_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    m4t_mtfp_t* x_train;    /* [n_train × input_dim] MTFP-encoded pixels   */
    int*        y_train;    /* [n_train] integer labels                    */
    m4t_mtfp_t* x_test;     /* [n_test  × input_dim]                       */
    int*        y_test;     /* [n_test]                                    */
    int         n_train;
    int         n_test;
    int         input_dim;  /* rows × cols (784 for MNIST)                 */
    int         img_w;
    int         img_h;
} glyph_dataset_t;

/* Load MNIST IDX files from:
 *   <dir>/train-images-idx3-ubyte
 *   <dir>/train-labels-idx1-ubyte
 *   <dir>/t10k-images-idx3-ubyte
 *   <dir>/t10k-labels-idx1-ubyte
 *
 * Populates every field of ds. Returns 0 on success, non-zero on any
 * fopen/read failure (with diagnostic to stderr). On failure, the ds
 * is left with whatever allocations succeeded; call glyph_dataset_free
 * to clean up.
 */
int glyph_dataset_load_mnist(glyph_dataset_t* ds, const char* dir);

/* Apply integer-moment deskew to every image in x_train and x_test.
 * Computes per-image shear from the image's own second moments (int64)
 * and applies it via pixel shifts per row. Zero float. Idempotent — a
 * second call is a no-op. */
void glyph_dataset_deskew(glyph_dataset_t* ds);

/* Free all heap allocations. Safe to call multiple times. */
void glyph_dataset_free(glyph_dataset_t* ds);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_DATASET_H */
