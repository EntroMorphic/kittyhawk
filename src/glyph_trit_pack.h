/*
 * glyph_trit_pack.h — glyph trit-packing aliases over M4T
 */

#ifndef GLYPH_TRIT_PACK_H
#define GLYPH_TRIT_PACK_H

#include "m4t_trit_pack.h"
#include "m4t_trit_ops.h"
#include "m4t_trit_reducers.h"

#define glyph_pack_trits_1d       m4t_pack_trits_1d
#define glyph_unpack_trits_1d     m4t_unpack_trits_1d
#define glyph_pack_trits_rowmajor m4t_pack_trits_rowmajor
#define glyph_unpack_trits_rowmajor m4t_unpack_trits_rowmajor
#define glyph_popcount_dist       m4t_popcount_dist

#endif /* GLYPH_TRIT_PACK_H */
