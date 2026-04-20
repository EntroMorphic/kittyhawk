/*
 * glyph_probe.c — shared multi-probe LSH candidate collection.
 */

#include "glyph_probe.h"

typedef struct {
    const glyph_bucket_table_t* table;
    glyph_probe_state_t*        state;
    int                         current_radius;
} probe_ctx_t;

static int probe_cb(const uint8_t* probe_sig, void* vctx) {
    probe_ctx_t* pc = (probe_ctx_t*)vctx;
    glyph_probe_state_t* st = pc->state;
    const glyph_bucket_table_t* bt = pc->table;
    uint8_t cur_r = (uint8_t)pc->current_radius;

    st->n_probes++;
    uint32_t key = glyph_sig_to_key_u32(probe_sig);
    int lb = glyph_bucket_lower_bound(bt, key);
    if (lb >= bt->n_entries || bt->entries[lb].key != key) return 0;

    for (int i = lb; i < bt->n_entries && bt->entries[i].key == key; i++) {
        int idx = bt->entries[i].proto_idx;
        if (st->votes[idx] == 0) {
            if (st->n_hit >= st->max_union) return 1;
            st->hit_list[st->n_hit++] = idx;
            if (st->min_radius) st->min_radius[idx] = cur_r;
        } else if (st->min_radius && cur_r < st->min_radius[idx]) {
            st->min_radius[idx] = cur_r;
        }
        st->votes[idx]++;
        st->per_table_cands++;
        if (st->n_hit >= st->max_union) return 1;
    }
    return 0;
}

void glyph_probe_reset(glyph_probe_state_t* st) {
    for (int j = 0; j < st->n_hit; j++) {
        int idx = st->hit_list[j];
        st->votes[idx] = 0;
        if (st->min_radius) st->min_radius[idx] = 0;
    }
    st->n_hit = 0;
    st->n_probes = 0;
}

void glyph_probe_table(const glyph_bucket_table_t* bt,
                       const uint8_t* query_sig,
                       int n_proj, int sig_bytes,
                       int max_radius, int min_cands,
                       glyph_probe_state_t* st,
                       uint8_t* scratch)
{
    probe_ctx_t pc = { bt, st, 0 };
    st->per_table_cands = 0;
    for (int r = 0; r <= max_radius; r++) {
        if (st->per_table_cands >= min_cands && r > 0) break;
        pc.current_radius = r;
        glyph_multiprobe_enumerate(query_sig, n_proj, sig_bytes, r,
                                   scratch, probe_cb, &pc);
        if (st->n_hit >= st->max_union) break;
    }
}
