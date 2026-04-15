/*
 * glyph_bucket.h — sorted bucket table keyed on packed trit signatures.
 *
 * The bucket table is Glyph's production index structure for routed
 * k-NN. Given N prototypes and their packed-trit signatures, it sorts
 * (sig_key, proto_idx) pairs by sig_key so that lookup is a binary
 * search in O(log N) and exact-match buckets are contiguous runs.
 *
 * Current constraint: 4-byte signatures (N_PROJ=16) fit in uint32 keys.
 * Longer signatures would need a different key type (uint64 for 8 bytes)
 * or a different structure (hash table). Kept simple until N_PROJ>16
 * bucket variants become interesting. Generalizing to uint64 keys is
 * tracked as the next libglyph API extension (enables the fused-filter
 * variant with concatenated H1+H2 signatures).
 *
 * Typical lookup pattern (matching-run scan):
 *
 *     uint32_t key = glyph_sig_to_key_u32(query_sig);
 *     int lb = glyph_bucket_lower_bound(&bt, key);
 *     if (lb < bt.n_entries && bt.entries[lb].key == key) {
 *         // Same-key entries form a contiguous run because the table
 *         // is sorted. Walk forward while the key still matches.
 *         for (int i = lb;
 *              i < bt.n_entries && bt.entries[i].key == key;
 *              i++) {
 *             int proto_idx = bt.entries[i].proto_idx;
 *             // ... score / vote / accumulate candidate ...
 *         }
 *     }
 *
 * This pattern is what the multi-probe callback in the bucket consumer
 * tools uses to collect candidates from the query's signature
 * neighborhood. For a complete working example see
 * tools/mnist_routed_bucket.c and tools/mnist_routed_bucket_multi.c.
 */

#ifndef GLYPH_BUCKET_H
#define GLYPH_BUCKET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t key;
    int32_t  proto_idx;
} glyph_bucket_entry_t;

typedef struct {
    glyph_bucket_entry_t* entries;
    int                   n_entries;
    int                   sig_bytes;   /* must be 4 in this version */
} glyph_bucket_table_t;

/* Pack a 4-byte signature as a little-endian uint32 for use as a
 * bucket key. Only defined for sig_bytes == 4. */
uint32_t glyph_sig_to_key_u32(const uint8_t* sig);

/* Build a sorted bucket table over n_entries packed signatures. Each
 * signature is sig_bytes bytes, stored contiguously at sigs. Returns
 * 0 on success. Allocates bt->entries via malloc; caller must call
 * glyph_bucket_table_free. */
int glyph_bucket_build(glyph_bucket_table_t* bt,
                       const uint8_t* sigs,
                       int n_entries,
                       int sig_bytes);

/* Binary-search for the first entry with key >= target. Returns
 * n_entries if no entry has key >= target. */
int glyph_bucket_lower_bound(const glyph_bucket_table_t* bt, uint32_t target);

/* Count distinct keys in the sorted table (diagnostic). */
int glyph_bucket_count_distinct(const glyph_bucket_table_t* bt);

void glyph_bucket_table_free(glyph_bucket_table_t* bt);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_BUCKET_H */
