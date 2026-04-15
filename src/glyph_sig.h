/*
 * glyph_sig.h — ternary signature builder.
 *
 * A signature builder bundles the random ternary projection matrix,
 * the density-calibrated threshold, and the packed-trit encode path
 * that turns an MTFP vector into a routing signature.
 *
 * Under the hood it uses m4t_route_threshold_extract + m4t_ternary_matmul_bt.
 * The abstraction exists so consumers don't re-do the projection +
 * calibration dance in every tool.
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

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_SIG_H */
