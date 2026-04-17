/*
 * layered_lsh.c — LSH + GSH in concert.
 *
 * LSH: standard multi-table bucket-indexed LSH on pixel signatures.
 *      Finds prototypes the query LOOKS LIKE (geometric distance).
 *
 * GSH: hashes the LSH routing pattern (per-table 1-NN vote labels)
 *      directly as a multi-trit signature — no random projection.
 *      Finds training images the query ROUTES LIKE (topological
 *      distance = number of tables that disagree).
 *
 * Both fully routed: bucket index, multi-probe, k-NN resolve.
 * The two instruments play together — agreement confirms,
 * disagreement reveals boundary cases.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_rng.h"
#include "glyph_sig.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include "glyph_resolver.h"
#include "m4t_trit_pack.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10
#define TRITS_PER_VOTE 4

/* 10 classes → 4 trits each, unique codewords. */
static const int8_t vote_trits[10][TRITS_PER_VOTE] = {
    {-1,-1,-1,-1},  /* 0 */
    {-1,-1,-1, 0},  /* 1 */
    {-1,-1,-1,+1},  /* 2 */
    {-1,-1, 0,-1},  /* 3 */
    {-1,-1, 0, 0},  /* 4 */
    {-1,-1, 0,+1},  /* 5 */
    {-1,-1,+1,-1},  /* 6 */
    {-1,-1,+1, 0},  /* 7 */
    {-1,-1,+1,+1},  /* 8 */
    {-1, 0,-1,-1},  /* 9 */
};

typedef struct {
    uint16_t* votes;
    int32_t*  hit_list;
    int       n_hit;
    int       max_union;
    int       n_probes;
    int       per_table_cands;
} probe_state_t;

typedef struct {
    const glyph_bucket_table_t* table;
    probe_state_t* state;
} probe_ctx_t;

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
        st->per_table_cands++;
        if (st->n_hit >= st->max_union) return 1;
    }
    return 0;
}

static void probe_state_reset(probe_state_t* st) {
    for (int j = 0; j < st->n_hit; j++) st->votes[st->hit_list[j]] = 0;
    st->n_hit = 0; st->n_probes = 0;
}

static void probe_table(const glyph_bucket_table_t* bt, const uint8_t* q_sig,
                        int n_proj, int sig_bytes, int max_radius, int min_cands,
                        probe_state_t* st, uint8_t* scratch) {
    probe_ctx_t pc = { bt, st };
    st->per_table_cands = 0;
    for (int r = 0; r <= max_radius; r++) {
        if (st->per_table_cands >= min_cands && r > 0) break;
        glyph_multiprobe_enumerate(q_sig, n_proj, sig_bytes, r, scratch, probe_cb, &pc);
        if (st->n_hit >= st->max_union) break;
    }
}

static void derive_seed(uint32_t m, const uint32_t base[4], uint32_t out[4]) {
    if (m == 0) { out[0]=base[0]; out[1]=base[1]; out[2]=base[2]; out[3]=base[3]; return; }
    out[0] = 2654435761u * m + 1013904223u;
    out[1] = 1597334677u * m + 2246822519u;
    out[2] = 3266489917u * m +  668265263u;
    out[3] =  374761393u * m + 3266489917u;
}

/* Extract per-table 1-NN labels from the LSH union. */
static void union_per_table_labels(
    const probe_state_t* st, int n_tables, int sig_bytes,
    uint8_t** train_sigs, const uint8_t** q_sigs,
    const uint8_t* mask, const int* y_train,
    int exclude_idx, int* out_labels)
{
    for (int m = 0; m < n_tables; m++) {
        int32_t best_d = INT32_MAX;
        int best_label = -1;
        for (int j = 0; j < st->n_hit; j++) {
            int idx = st->hit_list[j];
            if (idx == exclude_idx) continue;
            int32_t d = m4t_popcount_dist(
                q_sigs[m], train_sigs[m] + (size_t)idx * sig_bytes,
                mask, sig_bytes);
            if (d < best_d) { best_d = d; best_label = y_train[idx]; }
        }
        out_labels[m] = best_label;
    }
}

/* Encode M vote labels into a packed-trit GSH signature.
 * Each label → 4 trits. Total: M*4 trits, packed at 4 trits/byte. */
static void encode_gsh_sig(const int* labels, int M, uint8_t* out_sig, int gsh_sig_bytes) {
    memset(out_sig, 0, gsh_sig_bytes);
    for (int m = 0; m < M; m++) {
        int lbl = labels[m];
        if (lbl < 0 || lbl >= N_CLASSES) lbl = 0;
        for (int t = 0; t < TRITS_PER_VOTE; t++) {
            int trit_pos = m * TRITS_PER_VOTE + t;
            glyph_write_trit(out_sig, trit_pos, vote_trits[lbl][t]);
        }
    }
}

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) {
        fprintf(stderr, "failed to load dataset from %s\n", cfg.data_dir);
        return 1;
    }
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    const int L1_M = cfg.m_max;
    const int L1_NP = 16;
    const int L1_SB = M4T_TRIT_PACKED_BYTES(L1_NP);
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    const int KNN_K = 5;

    /* GSH: M votes × 4 trits/vote. */
    const int GSH_NTRITS = L1_M * TRITS_PER_VOTE;
    const int GSH_SB = M4T_TRIT_PACKED_BYTES(GSH_NTRITS);
    /* Bucket key uses first 16 trits (4 bytes). */
    const int GSH_KEY_NP = 16;

    printf("layered_lsh: LSH + GSH in concert\n");
    printf("  data_dir=%s  deskew=%s  density=%.2f\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on", cfg.density);
    printf("  LSH: M=%d  N_PROJ=%d  (pixel geometry)\n", L1_M, L1_NP);
    printf("  GSH: %d trits  %d bytes  (routing topology, no random projection)\n",
           GSH_NTRITS, GSH_SB);
    printf("  n_train=%d  n_test=%d  input_dim=%d  knn_k=%d\n\n",
           ds.n_train, ds.n_test, ds.input_dim, KNN_K);

    /* ============================================================
     * Build LSH tables.
     * ============================================================ */
    clock_t t0 = clock();
    printf("Building LSH tables...\n");

    glyph_sig_builder_t* l1_builders = calloc((size_t)L1_M, sizeof(glyph_sig_builder_t));
    uint8_t** l1_train = calloc((size_t)L1_M, sizeof(uint8_t*));
    uint8_t** l1_test  = calloc((size_t)L1_M, sizeof(uint8_t*));
    glyph_bucket_table_t* l1_tables = calloc((size_t)L1_M, sizeof(glyph_bucket_table_t));

    for (int m = 0; m < L1_M; m++) {
        uint32_t seeds[4]; derive_seed((uint32_t)m, cfg.base_seed, seeds);
        glyph_sig_builder_init(&l1_builders[m], L1_NP, ds.input_dim, cfg.density,
                                seeds[0], seeds[1], seeds[2], seeds[3],
                                ds.x_train, n_calib);
        l1_train[m] = calloc((size_t)ds.n_train * L1_SB, 1);
        l1_test[m]  = calloc((size_t)ds.n_test  * L1_SB, 1);
        glyph_sig_encode_batch(&l1_builders[m], ds.x_train, ds.n_train, l1_train[m]);
        glyph_sig_encode_batch(&l1_builders[m], ds.x_test,  ds.n_test,  l1_test[m]);
        glyph_bucket_build(&l1_tables[m], l1_train[m], ds.n_train, L1_SB);
    }
    printf("  LSH: %d tables built.\n", L1_M);

    /* ============================================================
     * Compute training GSH signatures via LSH probing.
     * ============================================================ */
    printf("Computing training routing patterns via LSH...\n");
    clock_t t_rsig = clock();

    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    st.n_hit = 0;
    uint8_t l1_scratch[4];
    uint8_t l1_mask[4]; memset(l1_mask, 0xFF, L1_SB);
    const uint8_t** q_ptrs = calloc((size_t)L1_M, sizeof(uint8_t*));

    int* vote_labels = malloc((size_t)L1_M * sizeof(int));
    uint8_t* gsh_train_sigs = calloc((size_t)ds.n_train * GSH_SB, 1);

    for (int i = 0; i < ds.n_train; i++) {
        for (int m = 0; m < L1_M; m++)
            q_ptrs[m] = l1_train[m] + (size_t)i * L1_SB;
        probe_state_reset(&st);
        for (int m = 0; m < L1_M; m++)
            probe_table(&l1_tables[m], q_ptrs[m], L1_NP, L1_SB,
                        cfg.max_radius, cfg.min_cands, &st, l1_scratch);
        union_per_table_labels(&st, L1_M, L1_SB, l1_train, q_ptrs,
                               l1_mask, ds.y_train, i, vote_labels);
        encode_gsh_sig(vote_labels, L1_M,
                       gsh_train_sigs + (size_t)i * GSH_SB, GSH_SB);

        if ((i + 1) % 10000 == 0) {
            printf("  %d/%d (%.1fs)\n", i + 1, ds.n_train,
                   (double)(clock() - t_rsig) / CLOCKS_PER_SEC);
            fflush(stdout);
        }
    }
    printf("  Training GSH sigs: %.1fs\n",
           (double)(clock() - t_rsig) / CLOCKS_PER_SEC);

    /* ============================================================
     * Build GSH bucket index (no random projection — direct hash).
     * Key on first 4 bytes (16 trits = first 4 table votes).
     * ============================================================ */
    printf("Building GSH bucket index...\n");
    glyph_bucket_table_t gsh_table;
    glyph_bucket_build(&gsh_table, gsh_train_sigs, ds.n_train, GSH_SB);
    printf("  GSH: %d distinct buckets.\n",
           glyph_bucket_count_distinct(&gsh_table));

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Total build: %.1fs\n\n", build_sec);

    /* ============================================================
     * Classify: LSH + GSH in concert.
     * ============================================================ */
    probe_state_t gst;
    gst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gst.max_union = cfg.max_union;
    gst.n_hit = 0;
    uint8_t* gsh_scratch = calloc(GSH_SB, 1);
    uint8_t* gsh_mask = malloc(GSH_SB);
    memset(gsh_mask, 0xFF, GSH_SB);
    uint8_t* q_gsh_sig = calloc(GSH_SB, 1);

    glyph_union_t u = {0};
    u.y_train = ds.y_train;
    u.n_classes = N_CLASSES;

    int lsh_correct = 0, gsh_correct = 0;
    int agree_correct = 0, agree_count = 0;
    int disagree_lsh_correct = 0, disagree_gsh_correct = 0, disagree_count = 0;

    printf("Classifying %d test queries...\n", ds.n_test);
    clock_t t_sweep = clock();

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];

        /* --- LSH pass --- */
        for (int m = 0; m < L1_M; m++)
            q_ptrs[m] = l1_test[m] + (size_t)qi * L1_SB;
        probe_state_reset(&st);
        for (int m = 0; m < L1_M; m++)
            probe_table(&l1_tables[m], q_ptrs[m], L1_NP, L1_SB,
                        cfg.max_radius, cfg.min_cands, &st, l1_scratch);

        u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;
        int lsh_pred = glyph_resolver_sum_knn(&u, L1_M, L1_SB,
                                               l1_train, q_ptrs, l1_mask, KNN_K);
        if (lsh_pred == y) lsh_correct++;

        /* Extract routing pattern from LSH union. */
        union_per_table_labels(&st, L1_M, L1_SB, l1_train, q_ptrs,
                               l1_mask, ds.y_train, -1, vote_labels);
        encode_gsh_sig(vote_labels, L1_M, q_gsh_sig, GSH_SB);

        /* --- GSH pass --- */
        probe_state_reset(&gst);
        /* Probe with first 16 trits as the key. Multi-probe on the
         * GSH signature operates on the first GSH_KEY_NP trits. */
        probe_table(&gsh_table, q_gsh_sig, GSH_KEY_NP, L1_SB,
                    cfg.max_radius, cfg.min_cands, &gst, l1_scratch);

        glyph_union_t gu = {0};
        gu.hit_list = gst.hit_list; gu.n_hit = gst.n_hit;
        gu.votes = gst.votes;
        gu.y_train = ds.y_train; gu.n_classes = N_CLASSES;

        /* Score GSH union by Hamming distance on FULL GSH sigs. */
        int gsh_pred = -1;
        {
            typedef struct { int32_t s; int l; } tk_t;
            tk_t topk[64]; int ntk = 0;
            for (int j = 0; j < gst.n_hit; j++) {
                int idx = gst.hit_list[j];
                int32_t d = m4t_popcount_dist(
                    q_gsh_sig, gsh_train_sigs + (size_t)idx * GSH_SB,
                    gsh_mask, GSH_SB);
                int lbl = ds.y_train[idx];
                if (ntk < KNN_K) {
                    int pos = ntk;
                    while (pos > 0 && topk[pos-1].s > d) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = d; topk[pos].l = lbl; ntk++;
                } else if (d < topk[KNN_K-1].s) {
                    int pos = KNN_K-1;
                    while (pos > 0 && topk[pos-1].s > d) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = d; topk[pos].l = lbl;
                }
            }
            int cv[N_CLASSES] = {0};
            for (int i = 0; i < ntk; i++) cv[topk[i].l] += (KNN_K - i);
            gsh_pred = 0;
            for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[gsh_pred]) gsh_pred = c;
        }
        if (gsh_pred == y) gsh_correct++;

        /* --- Agreement analysis --- */
        if (lsh_pred == gsh_pred) {
            agree_count++;
            if (lsh_pred == y) agree_correct++;
        } else {
            disagree_count++;
            if (lsh_pred == y) disagree_lsh_correct++;
            if (gsh_pred == y) disagree_gsh_correct++;
        }

        if ((qi + 1) % 1000 == 0 || qi == ds.n_test - 1) {
            printf("  %d/%d  LSH=%.2f%%  GSH=%.2f%%  agree=%.1f%%\n",
                   qi + 1, ds.n_test,
                   100.0 * lsh_correct / (qi + 1),
                   100.0 * gsh_correct / (qi + 1),
                   100.0 * agree_count / (qi + 1));
            fflush(stdout);
        }
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("\n=== Results ===\n");
    printf("  LSH k=%d-NN:                 %6.2f%%\n", KNN_K, 100.0 * lsh_correct / ds.n_test);
    printf("  GSH k=%d-NN:                 %6.2f%%\n", KNN_K, 100.0 * gsh_correct / ds.n_test);
    printf("\n  Agreement rate:              %6.2f%%  (%d / %d)\n",
           100.0 * agree_count / ds.n_test, agree_count, ds.n_test);
    printf("  P(correct | agree):          %6.2f%%  (%d / %d)\n",
           agree_count ? 100.0 * agree_correct / agree_count : 0.0,
           agree_correct, agree_count);
    printf("  P(LSH correct | disagree):   %6.2f%%  (%d / %d)\n",
           disagree_count ? 100.0 * disagree_lsh_correct / disagree_count : 0.0,
           disagree_lsh_correct, disagree_count);
    printf("  P(GSH correct | disagree):   %6.2f%%  (%d / %d)\n",
           disagree_count ? 100.0 * disagree_gsh_correct / disagree_count : 0.0,
           disagree_gsh_correct, disagree_count);
    printf("\n  Trust-LSH accuracy:          %6.2f%%  (always take LSH)\n",
           100.0 * lsh_correct / ds.n_test);
    printf("  Trust-agree accuracy:        %6.2f%%  (agree→accept, disagree→LSH)\n",
           100.0 * (agree_correct + disagree_lsh_correct) / ds.n_test);
    printf("  Trust-GSH-override:          %6.2f%%  (agree→accept, disagree→GSH)\n",
           100.0 * (agree_correct + disagree_gsh_correct) / ds.n_test);
    printf("  Sweep time: %.1fs\n\n", sweep_sec);

    /* Cleanup. */
    free(vote_labels); free(q_gsh_sig); free(gsh_scratch); free(gsh_mask);
    free(gsh_train_sigs);
    glyph_bucket_table_free(&gsh_table);
    free(gst.votes); free(gst.hit_list);
    free(st.votes); free(st.hit_list); free(q_ptrs);
    for (int m = 0; m < L1_M; m++) {
        glyph_sig_builder_free(&l1_builders[m]);
        glyph_bucket_table_free(&l1_tables[m]);
        free(l1_train[m]); free(l1_test[m]);
    }
    free(l1_builders); free(l1_tables); free(l1_train); free(l1_test);
    glyph_dataset_free(&ds);
    return 0;
}
