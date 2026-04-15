/*
 * glyph_multiprobe.h — ternary Hamming multi-probe enumeration.
 *
 * The Trit Lattice signature is packed as 2-bit trit codes (see
 * m4t_trit_pack.h): +1=0b01, 0=0b00, -1=0b10. Multi-probe LSH walks
 * signatures at ternary-Hamming cost radii 0, 1, 2 by enumerating
 * neighbor codes that differ from the query signature by the
 * corresponding cost.
 *
 * Cost function:
 *   same trit (any state)   -> cost 0
 *   ±1 vs 0                 -> cost 1
 *   +1 vs -1                -> cost 2
 *
 * Radius-1 neighborhood has ~16×1.67 ≈ 27 probes at density 0.33.
 * Radius-2 neighborhood has ~340 probes (one sign-flip OR two cost-1
 * moves at different positions).
 */

#ifndef GLYPH_MULTIPROBE_H
#define GLYPH_MULTIPROBE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Read trit j from a packed signature. Returns -1, 0, or +1. */
int8_t glyph_read_trit(const uint8_t* sig, int j);

/* Write trit j into a packed signature (in-place). t must be in {-1,0,+1}. */
void   glyph_write_trit(uint8_t* sig, int j, int8_t t);

/* Callback invoked once per enumerated probe signature. Return non-zero
 * to stop enumeration early (e.g., when the caller's candidate budget
 * is filled).
 *
 * Typical use — look up each probe in a bucket index, accumulate
 * candidates into a hit_list, stop when the budget fills:
 *
 *     typedef struct {
 *         const glyph_bucket_table_t* bt;
 *         int32_t* hit_list;
 *         int      n_hit;
 *         int      max_hit;
 *     } my_ctx_t;
 *
 *     static int my_cb(const uint8_t* probe_sig, void* vctx) {
 *         my_ctx_t* c = (my_ctx_t*)vctx;
 *         uint32_t key = glyph_sig_to_key_u32(probe_sig);
 *         int lb = glyph_bucket_lower_bound(c->bt, key);
 *         for (int i = lb;
 *              i < c->bt->n_entries && c->bt->entries[i].key == key;
 *              i++) {
 *             if (c->n_hit >= c->max_hit) return 1;  // stop enumeration
 *             c->hit_list[c->n_hit++] = c->bt->entries[i].proto_idx;
 *         }
 *         return 0;  // continue enumeration
 *     }
 *
 *     uint8_t scratch[4];
 *     my_ctx_t ctx = { &bt, hit_list, 0, MAX_UNION };
 *     for (int r = 0; r <= 2; r++) {
 *         if (ctx.n_hit >= MIN_CANDS) break;
 *         glyph_multiprobe_enumerate(q_sig, 16, 4, r, scratch, my_cb, &ctx);
 *     }
 */
typedef int (*glyph_probe_cb)(const uint8_t* probe_sig, void* ctx);

/* Enumerate every signature at EXACTLY ternary Hamming cost equal to
 * `radius` around `query_sig`. Supports radius in {0, 1, 2}. scratch
 * must have at least sig_bytes capacity; it is used as the edit
 * buffer for each probe.
 *
 * Returns non-zero if the callback requested early stop. */
int glyph_multiprobe_enumerate(
    const uint8_t* query_sig,
    int n_proj,
    int sig_bytes,
    int radius,
    uint8_t* scratch,
    glyph_probe_cb cb,
    void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_MULTIPROBE_H */
