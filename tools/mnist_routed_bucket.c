/*
 * mnist_routed_bucket.c — single-table routed bucket consumer (Axis 5).
 *
 * Thin CLI wrapper on libglyph. Builds one bucket-indexed signature
 * table from H1 and runs a (MAX_RADIUS × MIN_CANDIDATES) tuning sweep
 * with a routed secondary-hash resolver (H2+H3+H4 summed popcount_dist).
 *
 * This is the "first production-shape consumer" — the architecture
 * that the Axis 5 reframe produced, before the Axis 6 multi-table
 * extension. Reaches 82.58% at ~9.9 μs/query on deskewed MNIST at
 * N_PROJ=16 in its best sweep cell (MAX_R=2, MIN_C=100).
 *
 * Usage:
 *   mnist_routed_bucket [options]
 *
 * See `--help`. Default reproduces the Axis 5 measurement
 * (journal/routed_bucket_consumer.md).
 *
 * For the multi-table Axis 6 architecture that exceeds 97%, see
 * tools/mnist_routed_bucket_multi.c.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_sig.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"

#include "m4t_trit_pack.h"
#include "m4t_route.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Per-query probe state: sorted-bucket hit list with dense vote array.
 * Lazy-zero pattern keeps reset O(|union|) rather than O(n_train). */
typedef struct {
    uint16_t* votes;
    int32_t*  hit_list;
    int       n_hit;
    int       max_union;
    int       n_probes;
    int       n_candidates;   /* total additions, including duplicates */
} probe_state_t;

typedef struct {
    const glyph_bucket_table_t* table;
    probe_state_t* state;
} probe_ctx_t;

static void probe_state_reset(probe_state_t* st) {
    for (int j = 0; j < st->n_hit; j++) st->votes[st->hit_list[j]] = 0;
    st->n_hit = 0;
    st->n_probes = 0;
    st->n_candidates = 0;
}

static int probe_cb(const uint8_t* probe_sig, void* vctx) {
    probe_ctx_t* pc = (probe_ctx_t*)vctx;
    probe_state_t* st = pc->state;
    const glyph_bucket_table_t* bt = pc->table;
    st->n_probes++;
    uint32_t key = glyph_sig_to_key_u32(probe_sig);
    int lb = glyph_bucket_lower_bound(bt, key);
    if (lb >= bt->n_entries || bt->entries[lb].key != key) return 0;
    for (int i = lb; i < bt->n_entries && bt->entries[i].key == key; i++) {
        int idx = bt->entries[i].proto_idx;
        if (st->votes[idx] == 0) {
            if (st->n_hit >= st->max_union) return 1;
            st->hit_list[st->n_hit++] = idx;
        }
        st->votes[idx]++;
        st->n_candidates++;
    }
    return 0;
}

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    /* The single-table consumer only runs oracle+resolver at a single M=1.
     * We ignore --m_max and --single_m. If the user passes --mode full
     * we run the resolver; otherwise we skip it. */
    int full_mode = (strcmp(cfg.mode, "full") == 0);

    glyph_dataset_t ds;
    if (glyph_dataset_load_mnist(&ds, cfg.data_dir) != 0) {
        fprintf(stderr, "failed to load MNIST from %s\n", cfg.data_dir);
        return 1;
    }
    glyph_dataset_deskew(&ds);

    printf("mnist_routed_bucket (libglyph single-table consumer, Axis 5)\n");
    printf("  data_dir=%s\n", cfg.data_dir);
    printf("  n_proj=%d  density=%.2f  max_radius=%d  min_cands=%d  max_union=%d\n",
           cfg.n_proj, cfg.density, cfg.max_radius, cfg.min_cands, cfg.max_union);
    printf("  base_seed=%u,%u,%u,%u\n",
           cfg.base_seed[0], cfg.base_seed[1], cfg.base_seed[2], cfg.base_seed[3]);
    printf("  n_train=%d  n_test=%d  input_dim=%d\n\n",
           ds.n_train, ds.n_test, ds.input_dim);

    int sig_bytes = M4T_TRIT_PACKED_BYTES(cfg.n_proj);
    if (sig_bytes != 4) {
        fprintf(stderr, "this tool supports N_PROJ=16 (4-byte sigs) only\n");
        glyph_dataset_free(&ds);
        return 1;
    }

    /* Four signature builders: H1 (filter), H2/H3/H4 (resolver).
     * Seeds mirror the original mnist_routed_bucket.c for exact
     * reproduction. */
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    glyph_sig_builder_t sb_H1, sb_H2, sb_H3, sb_H4;
    struct { glyph_sig_builder_t* sb; uint32_t s[4]; } hashes[4] = {
        {&sb_H1, {cfg.base_seed[0], cfg.base_seed[1], cfg.base_seed[2], cfg.base_seed[3]}},
        {&sb_H2, {1337u, 2718u, 3141u, 5923u}},
        {&sb_H3, {1009u, 2017u, 3041u, 5059u}},
        {&sb_H4, {9001u, 9002u, 9003u, 9004u}},
    };
    for (int h = 0; h < 4; h++) {
        if (glyph_sig_builder_init(
                hashes[h].sb, cfg.n_proj, ds.input_dim, cfg.density,
                hashes[h].s[0], hashes[h].s[1], hashes[h].s[2], hashes[h].s[3],
                ds.x_train, n_calib) != 0) {
            fprintf(stderr, "sig builder init failed for hash %d\n", h);
            return 1;
        }
    }

    /* Encode all train/test sigs for each hash. */
    uint8_t* train_H1 = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* train_H2 = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* train_H3 = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* train_H4 = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* test_H1  = calloc((size_t)ds.n_test  * sig_bytes, 1);
    uint8_t* test_H2  = calloc((size_t)ds.n_test  * sig_bytes, 1);
    uint8_t* test_H3  = calloc((size_t)ds.n_test  * sig_bytes, 1);
    uint8_t* test_H4  = calloc((size_t)ds.n_test  * sig_bytes, 1);

    glyph_sig_encode_batch(&sb_H1, ds.x_train, ds.n_train, train_H1);
    glyph_sig_encode_batch(&sb_H2, ds.x_train, ds.n_train, train_H2);
    glyph_sig_encode_batch(&sb_H3, ds.x_train, ds.n_train, train_H3);
    glyph_sig_encode_batch(&sb_H4, ds.x_train, ds.n_train, train_H4);
    glyph_sig_encode_batch(&sb_H1, ds.x_test,  ds.n_test,  test_H1);
    glyph_sig_encode_batch(&sb_H2, ds.x_test,  ds.n_test,  test_H2);
    glyph_sig_encode_batch(&sb_H3, ds.x_test,  ds.n_test,  test_H3);
    glyph_sig_encode_batch(&sb_H4, ds.x_test,  ds.n_test,  test_H4);

    /* Build bucket index on H1. */
    clock_t t_build_start = clock();
    glyph_bucket_table_t bt = {0};
    if (glyph_bucket_build(&bt, train_H1, ds.n_train, sig_bytes) != 0) {
        fprintf(stderr, "bucket build failed\n");
        return 1;
    }
    double t_build = (double)(clock() - t_build_start) / CLOCKS_PER_SEC;

    printf("H1 bucket index:\n");
    printf("  %d prototypes -> %d distinct buckets (%.2fx compression)\n",
           ds.n_train, glyph_bucket_count_distinct(&bt),
           (double)ds.n_train / (double)glyph_bucket_count_distinct(&bt));
    printf("  build time: %.3fs\n\n", t_build);

    uint8_t* mask = malloc(sig_bytes);
    memset(mask, 0xFF, sig_bytes);

    /* Per-query state (reused). */
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    st.n_hit = 0;

    /* Sweep over (MAX_RADIUS, MIN_CANDIDATES). If --single_m is not
     * supplied via the shared config (it's an M-sweep knob, irrelevant
     * here) we always run the full 3×4 grid. */
    const int radii[] = {0, 1, 2};
    const int mins[]  = {1, 20, 100, 400};
    const int n_radii = (int)(sizeof(radii)/sizeof(radii[0]));
    const int n_mins  = (int)(sizeof(mins)/sizeof(mins[0]));

    printf("Sweep (rows=MAX_RADIUS, cols=MIN_CANDIDATES):\n");
    printf("  MAX_R  MIN_C   accuracy   avg_cands   avg_probes   empty   us/qry\n");

    uint8_t scratch[4];

    for (int ri = 0; ri < n_radii; ri++) {
        for (int mi = 0; mi < n_mins; mi++) {
            int MAX_R = radii[ri];
            int MIN_C = mins[mi];

            int correct = 0;
            long total_probes = 0;
            long total_cands  = 0;
            int  empty_queries = 0;

            clock_t t_start = clock();
            for (int s = 0; s < ds.n_test; s++) {
                const uint8_t* q_H1 = test_H1 + (size_t)s * sig_bytes;
                const uint8_t* q_H2 = test_H2 + (size_t)s * sig_bytes;
                const uint8_t* q_H3 = test_H3 + (size_t)s * sig_bytes;
                const uint8_t* q_H4 = test_H4 + (size_t)s * sig_bytes;
                int y = ds.y_test[s];

                probe_state_reset(&st);
                probe_ctx_t pc = {&bt, &st};

                for (int r = 0; r <= MAX_R; r++) {
                    if (st.n_candidates >= MIN_C) break;
                    glyph_multiprobe_enumerate(
                        q_H1, cfg.n_proj, sig_bytes, r, scratch, probe_cb, &pc);
                    if (st.n_hit >= st.max_union) break;
                }

                if (st.n_hit == 0) { empty_queries++; continue; }
                total_probes += st.n_probes;
                total_cands  += st.n_hit;

                /* Resolver: summed H2+H3+H4 popcount_dist 1-NN over union. */
                int32_t best_score = INT32_MAX;
                int     best_label = -1;
                for (int j = 0; j < st.n_hit; j++) {
                    int idx = st.hit_list[j];
                    int32_t score =
                        m4t_popcount_dist(q_H2, train_H2 + (size_t)idx * sig_bytes, mask, sig_bytes) +
                        m4t_popcount_dist(q_H3, train_H3 + (size_t)idx * sig_bytes, mask, sig_bytes) +
                        m4t_popcount_dist(q_H4, train_H4 + (size_t)idx * sig_bytes, mask, sig_bytes);
                    if (score < best_score) {
                        best_score = score;
                        best_label = ds.y_train[idx];
                    }
                }
                if (best_label == y) correct++;
            }
            double t_sec = (double)(clock() - t_start) / CLOCKS_PER_SEC;

            int divisor = ds.n_test - empty_queries;
            if (divisor <= 0) divisor = 1;
            printf("    %d      %4d   %6.2f%%   %8.1f  %10.1f   %4d   %6.1f\n",
                   MAX_R, MIN_C,
                   100.0 * correct / ds.n_test,
                   (double)total_cands / divisor,
                   (double)total_probes / divisor,
                   empty_queries,
                   1e6 * t_sec / ds.n_test);
            fflush(stdout);
        }
    }
    printf("\n");

    (void)full_mode;

    /* Cleanup. */
    free(st.votes); free(st.hit_list);
    free(mask);
    glyph_bucket_table_free(&bt);
    free(train_H1); free(train_H2); free(train_H3); free(train_H4);
    free(test_H1);  free(test_H2);  free(test_H3);  free(test_H4);
    for (int h = 0; h < 4; h++) glyph_sig_builder_free(hashes[h].sb);
    glyph_dataset_free(&ds);
    return 0;
}
