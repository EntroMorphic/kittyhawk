/*
 * glyph_types.h — glyph type aliases over M4T
 *
 * Glyph is a consumer of the M4T ternary substrate. This header aliases
 * M4T types into the glyph namespace for application-layer code.
 */

#ifndef GLYPH_TYPES_H
#define GLYPH_TYPES_H

#include "m4t_types.h"

typedef m4t_mtfp4_t   glyph_mtfp4_t;
typedef m4t_mtfp9_t   glyph_mtfp9_t;
typedef m4t_mtfp_t    glyph_mtfp_t;
typedef m4t_trit_t    glyph_trit_t;

#define GLYPH_MTFP_SCALE      M4T_MTFP_SCALE
#define GLYPH_MTFP_MAX_VAL    M4T_MTFP_MAX_VAL
#define GLYPH_TRIT_PACKED_BYTES(n) M4T_TRIT_PACKED_BYTES(n)

#endif /* GLYPH_TYPES_H */
