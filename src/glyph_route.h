/*
 * glyph_route.h — glyph routing aliases over M4T
 */

#ifndef GLYPH_ROUTE_H
#define GLYPH_ROUTE_H

#include "m4t_route.h"
#include "m4t_ops.h"

typedef m4t_route_decision_t glyph_route_decision_t;

#define glyph_route_sign_extract      m4t_route_sign_extract
#define glyph_route_distance_batch    m4t_route_distance_batch
#define glyph_route_topk_abs          m4t_route_topk_abs
#define glyph_route_apply_signed      m4t_route_apply_signed
#define glyph_route_signature_update  m4t_route_signature_update

#endif /* GLYPH_ROUTE_H */
