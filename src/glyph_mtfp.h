/*
 * glyph_mtfp.h — glyph MTFP aliases over M4T
 *
 * Includes the full M4T MTFP surface (MTFP19, MTFP39, MTFP4) and
 * re-exports under glyph_* names where needed by application code.
 */

#ifndef GLYPH_MTFP_H
#define GLYPH_MTFP_H

#include "m4t_mtfp.h"
#include "m4t_mtfp_w.h"
#include "m4t_mtfp4.h"

#define glyph_mtfp_add           m4t_mtfp_add
#define glyph_mtfp_sub           m4t_mtfp_sub
#define glyph_mtfp_neg           m4t_mtfp_neg
#define glyph_mtfp_mul           m4t_mtfp_mul
#define glyph_mtfp_mul_trit      m4t_mtfp_mul_trit
#define glyph_mtfp_clamp64       m4t_mtfp_clamp64
#define glyph_mtfp_vec_zero      m4t_mtfp_vec_zero
#define glyph_mtfp_vec_add       m4t_mtfp_vec_add
#define glyph_mtfp_vec_add_inplace  m4t_mtfp_vec_add_inplace
#define glyph_mtfp_vec_sub_inplace  m4t_mtfp_vec_sub_inplace
#define glyph_mtfp_vec_scale     m4t_mtfp_vec_scale
#define glyph_mtfp_matmul        m4t_mtfp_matmul
#define glyph_mtfp_matmul_bt     m4t_mtfp_matmul_bt
#define glyph_mtfp_bias_add      m4t_mtfp_bias_add
#define glyph_mtfp_fan_in_normalize  m4t_mtfp_fan_in_normalize
#define glyph_mtfp_layernorm     m4t_mtfp_layernorm
#define glyph_isqrt64            m4t_isqrt64
#define glyph_mtfp_isqrt_inv     m4t_mtfp_isqrt_inv

#endif /* GLYPH_MTFP_H */
