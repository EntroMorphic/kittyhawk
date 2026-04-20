/*
 * glyph_probe.h — shared multi-probe LSH candidate collection.
 *
 * Provides the probe_state / probe_cb / probe_table pattern used by
 * every routed consumer. Probe a bucket table at ternary Hamming
 * radii 0..max_radius, collecting unique prototype indices into a
 * candidate union with vote counts.
 */

#ifndef GLYPH_PROBE_H
#define GLYPH_PROBE_H

#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t* votes;           /* [n_train] vote count per prototype       */
    uint8_t*  min_radius;      /* [n_train] smallest radius at which each
                                  proto was found (NULL to skip tracking)  */
    int32_t*  hit_list;        /* [max_union] proto indices in union       */
    int       n_hit;           /* unique protos in current union           */
    int       max_union;       /* capacity of hit_list                     */
    int       n_probes;        /* total probe_cb invocations               */
    int       per_table_cands; /* candidates from current table's probing  */
} glyph_probe_state_t;

/* Reset the probe state between queries. Lazy-zeros only the slots
 * touched by the previous query (O(n_hit), not O(n_train)). */
void glyph_probe_reset(glyph_probe_state_t* st);

/* Probe a single bucket table at radii 0..max_radius, collecting
 * candidates into st. Stops expanding radius once per_table_cands
 * reaches min_cands. scratch must have at least sig_bytes bytes. */
void glyph_probe_table(const glyph_bucket_table_t* bt,
                       const uint8_t* query_sig,
                       int n_proj, int sig_bytes,
                       int max_radius, int min_cands,
                       glyph_probe_state_t* st,
                       uint8_t* scratch);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_PROBE_H */
