/*
 * bitnet_kv_cache.c — KV cache lifecycle (work-unit 7).
 *
 * Per-layer K and V buffers, sized [n_layers × max_seq_len × NUM_KV_HEADS
 * × HEAD_DIM]. Caller writes the current token's K, V at the end of each
 * layer's attention; subsequent tokens read past positions for the
 * attention QK^T computation.
 */

#include "bitnet_config.h"

#include <stdlib.h>
#include <string.h>

int bitnet_kv_cache_alloc(bitnet_kv_cache_t* cache, int max_seq_len, int n_layers) {
    if (!cache || max_seq_len <= 0 || n_layers <= 0) return 1;
    size_t per_layer = (size_t)max_seq_len * (size_t)BITNET_NUM_KV_HEADS * (size_t)BITNET_HEAD_DIM;
    size_t total_cells = per_layer * (size_t)n_layers;
    cache->max_seq_len = max_seq_len;
    cache->n_layers = n_layers;
    cache->current_pos = 0;
    cache->per_layer_stride = per_layer;
    cache->k = (m4t_mtfp_t*)calloc(total_cells, sizeof(m4t_mtfp_t));
    cache->v = (m4t_mtfp_t*)calloc(total_cells, sizeof(m4t_mtfp_t));
    if (!cache->k || !cache->v) {
        bitnet_kv_cache_free(cache);
        return 2;
    }
    return 0;
}

void bitnet_kv_cache_free(bitnet_kv_cache_t* cache) {
    if (!cache) return;
    free(cache->k);
    free(cache->v);
    memset(cache, 0, sizeof(*cache));
}

void bitnet_kv_cache_reset(bitnet_kv_cache_t* cache) {
    if (!cache) return;
    cache->current_pos = 0;
    /* No need to zero buffers — they're overwritten on next use. */
}
