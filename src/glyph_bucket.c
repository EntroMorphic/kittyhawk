/*
 * glyph_bucket.c — sorted bucket table implementation.
 */

#include "glyph_bucket.h"

#include <stdlib.h>
#include <string.h>

uint32_t glyph_sig_to_key_u32(const uint8_t* sig) {
    return (uint32_t)sig[0]
         | ((uint32_t)sig[1] << 8)
         | ((uint32_t)sig[2] << 16)
         | ((uint32_t)sig[3] << 24);
}

static int cmp_entry(const void* a, const void* b) {
    uint32_t x = ((const glyph_bucket_entry_t*)a)->key;
    uint32_t y = ((const glyph_bucket_entry_t*)b)->key;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

int glyph_bucket_build(glyph_bucket_table_t* bt,
                       const uint8_t* sigs,
                       int n_entries,
                       int sig_bytes)
{
    memset(bt, 0, sizeof(*bt));
    if (sig_bytes < 4) return 1;

    bt->entries = malloc((size_t)n_entries * sizeof(glyph_bucket_entry_t));
    if (!bt->entries) return 1;
    bt->n_entries = n_entries;
    bt->sig_bytes = sig_bytes;

    for (int i = 0; i < n_entries; i++) {
        bt->entries[i].key       = glyph_sig_to_key_u32(sigs + (size_t)i * sig_bytes);
        bt->entries[i].proto_idx = i;
    }
    qsort(bt->entries, (size_t)n_entries, sizeof(glyph_bucket_entry_t), cmp_entry);
    return 0;
}

int glyph_bucket_lower_bound(const glyph_bucket_table_t* bt, uint32_t target) {
    int lo = 0, hi = bt->n_entries;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (bt->entries[mid].key < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int glyph_bucket_count_distinct(const glyph_bucket_table_t* bt) {
    if (bt->n_entries == 0) return 0;
    int count = 0;
    int i = 0;
    while (i < bt->n_entries) {
        int j = i + 1;
        while (j < bt->n_entries && bt->entries[j].key == bt->entries[i].key) j++;
        count++;
        i = j;
    }
    return count;
}

void glyph_bucket_table_free(glyph_bucket_table_t* bt) {
    if (!bt) return;
    free(bt->entries);
    memset(bt, 0, sizeof(*bt));
}
