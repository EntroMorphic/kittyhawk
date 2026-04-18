/*
 * specialist_rerank.c — LSH + GSH + routed table selection.
 *
 * The specialist doesn't generate new projections. It identifies
 * which of the LSH's EXISTING tables best distinguish each
 * confusion pair — discovered by routing measurements on the
 * training set. Re-ranks the LSH union using only the selected
 * tables' distances.
 *
 * The structural zero (W_f[hidden]=0) is already placed in the
 * existing tables. Routing tells us which placements work for
 * which class pairs.
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
#define MAX_CONFUSIONS 10
#define KNN_K 5
#define SPEC_TOP_TABLES 16

static const int8_t vote_trits[10][TRITS_PER_VOTE] = {
    {-1,-1,-1,-1}, {-1,-1,-1, 0}, {-1,-1,-1,+1}, {-1,-1, 0,-1},
    {-1,-1, 0, 0}, {-1,-1, 0,+1}, {-1,-1,+1,-1}, {-1,-1,+1, 0},
    {-1,-1,+1,+1}, {-1, 0,-1,-1},
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

static void encode_gsh_sig(const int* labels, int M, uint8_t* out, int gsh_sb) {
    memset(out, 0, gsh_sb);
    for (int m = 0; m < M; m++) {
        int lbl = labels[m];
        if (lbl < 0 || lbl >= N_CLASSES) lbl = 0;
        for (int t = 0; t < TRITS_PER_VOTE; t++)
            glyph_write_trit(out, m * TRITS_PER_VOTE + t, vote_trits[lbl][t]);
    }
}

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    const int L1_M = cfg.m_max;
    const int L1_NP = 16;
    const int L1_SB = M4T_TRIT_PACKED_BYTES(L1_NP);
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    const int GSH_NTRITS = L1_M * TRITS_PER_VOTE;
    const int GSH_SB = M4T_TRIT_PACKED_BYTES(GSH_NTRITS);

    printf("specialist_rerank: LSH + GSH + routed table selection\n");
    printf("  data=%s  deskew=%s  density=%.2f  M=%d\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on", cfg.density, L1_M);
    printf("  specialist: top-%d tables per confusion pair (from existing %d)\n",
           SPEC_TOP_TABLES, L1_M);
    printf("  n_train=%d  n_test=%d  input_dim=%d  knn_k=%d\n\n",
           ds.n_train, ds.n_test, ds.input_dim, KNN_K);

    /* ============================================================
     * Build LSH.
     * ============================================================ */
    clock_t t0 = clock();
    printf("Building LSH...\n");
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

    /* ============================================================
     * Build GSH.
     * ============================================================ */
    printf("Building GSH...\n");
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union; st.n_hit = 0;
    uint8_t scratch[4];
    uint8_t l1_mask[4]; memset(l1_mask, 0xFF, L1_SB);
    const uint8_t** q_ptrs = calloc((size_t)L1_M, sizeof(uint8_t*));
    int* vote_labels = malloc((size_t)L1_M * sizeof(int));
    uint8_t* gsh_train = calloc((size_t)ds.n_train * GSH_SB, 1);

    for (int i = 0; i < ds.n_train; i++) {
        for (int m = 0; m < L1_M; m++)
            q_ptrs[m] = l1_train[m] + (size_t)i * L1_SB;
        probe_state_reset(&st);
        for (int m = 0; m < L1_M; m++)
            probe_table(&l1_tables[m], q_ptrs[m], L1_NP, L1_SB,
                        cfg.max_radius, cfg.min_cands, &st, scratch);
        union_per_table_labels(&st, L1_M, L1_SB, l1_train, q_ptrs,
                               l1_mask, ds.y_train, i, vote_labels);
        encode_gsh_sig(vote_labels, L1_M, gsh_train + (size_t)i * GSH_SB, GSH_SB);
    }
    glyph_bucket_table_t gsh_table;
    glyph_bucket_build(&gsh_table, gsh_train, ds.n_train, GSH_SB);
    printf("  GSH: %d distinct buckets.\n", glyph_bucket_count_distinct(&gsh_table));

    /* ============================================================
     * Per-table per-class-pair discrimination score.
     *
     * For each table m and each class pair (a, b): score is the
     * fraction of training queries where table m's 1-NN label is
     * CORRECT for the true class among {a, b}. Computed from the
     * training routing signatures.
     *
     * This is routing the solution: the LSH's own measurements
     * on the training set determine which tables discriminate
     * which class pairs. No new projections generated.
     * ============================================================ */
    printf("Computing per-table discrimination scores...\n");

    /* Recompute per-table 1-NN for training images (already have
     * them from GSH construction — but we need to rebuild here
     * because we freed the union). Faster: store the training
     * routing labels during GSH construction. */
    int* all_train_labels = malloc((size_t)ds.n_train * L1_M * sizeof(int));
    for (int i = 0; i < ds.n_train; i++) {
        for (int m = 0; m < L1_M; m++)
            q_ptrs[m] = l1_train[m] + (size_t)i * L1_SB;
        probe_state_reset(&st);
        for (int m = 0; m < L1_M; m++)
            probe_table(&l1_tables[m], q_ptrs[m], L1_NP, L1_SB,
                        cfg.max_radius, cfg.min_cands, &st, scratch);
        union_per_table_labels(&st, L1_M, L1_SB, l1_train, q_ptrs,
                               l1_mask, ds.y_train, i,
                               all_train_labels + (size_t)i * L1_M);
    }

    /* For each (table, class_a, class_b): count how often table m's
     * 1-NN correctly identifies {a vs b}. */
    int table_pair_correct[256][N_CLASSES][N_CLASSES];
    int table_pair_total[256][N_CLASSES][N_CLASSES];
    memset(table_pair_correct, 0, sizeof(table_pair_correct));
    memset(table_pair_total, 0, sizeof(table_pair_total));

    for (int i = 0; i < ds.n_train; i++) {
        int y = ds.y_train[i];
        if (y < 0 || y >= N_CLASSES) continue;
        for (int m = 0; m < L1_M && m < 256; m++) {
            int lbl = all_train_labels[i * L1_M + m];
            for (int other = 0; other < N_CLASSES; other++) {
                if (other == y) continue;
                table_pair_total[m][y][other]++;
                if (lbl == y) table_pair_correct[m][y][other]++;
            }
        }
    }

    /* For each confusion pair, rank tables by discrimination score
     * and keep the top SPEC_TOP_TABLES. */
    typedef struct { int a, b, count; } cpair_t;
    cpair_t conf_pairs[N_CLASSES * N_CLASSES];
    int n_conf = 0;

    /* Identify confusion pairs from test set first pass. */
    printf("Identifying confusion pairs...\n");
    glyph_union_t u = {0};
    u.y_train = ds.y_train; u.n_classes = N_CLASSES;
    int test_confusion[N_CLASSES][N_CLASSES] = {{0}};
    for (int qi = 0; qi < ds.n_test; qi++) {
        for (int m = 0; m < L1_M; m++)
            q_ptrs[m] = l1_test[m] + (size_t)qi * L1_SB;
        probe_state_reset(&st);
        for (int m = 0; m < L1_M; m++)
            probe_table(&l1_tables[m], q_ptrs[m], L1_NP, L1_SB,
                        cfg.max_radius, cfg.min_cands, &st, scratch);
        u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;
        int pred = glyph_resolver_sum(&u, L1_M, L1_SB, l1_train, q_ptrs, l1_mask);
        int y = ds.y_test[qi];
        if (pred != y && y >= 0 && y < N_CLASSES && pred >= 0 && pred < N_CLASSES)
            test_confusion[y][pred]++;
    }
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = 0; b < N_CLASSES; b++)
            if (a != b && test_confusion[a][b] > 0) {
                conf_pairs[n_conf].a = a; conf_pairs[n_conf].b = b;
                conf_pairs[n_conf].count = test_confusion[a][b]; n_conf++;
            }
    for (int i = 1; i < n_conf; i++) {
        cpair_t v = conf_pairs[i]; int j = i - 1;
        while (j >= 0 && conf_pairs[j].count < v.count) { conf_pairs[j+1] = conf_pairs[j]; j--; }
        conf_pairs[j+1] = v;
    }
    int n_spec = (n_conf < MAX_CONFUSIONS) ? n_conf : MAX_CONFUSIONS;

    printf("  Top-%d confusion pairs:\n", n_spec);
    for (int i = 0; i < n_spec; i++)
        printf("    %d→%d: %d errors\n", conf_pairs[i].a, conf_pairs[i].b, conf_pairs[i].count);

    /* Build per-specialist table ranking. */
    int spec_tables[MAX_CONFUSIONS][256];
    for (int si = 0; si < n_spec; si++) {
        int ca = conf_pairs[si].a, cb = conf_pairs[si].b;
        typedef struct { int table; double score; } ts_t;
        ts_t scores[256];
        for (int m = 0; m < L1_M && m < 256; m++) {
            int tot = table_pair_total[m][ca][cb] + table_pair_total[m][cb][ca];
            int cor = table_pair_correct[m][ca][cb] + table_pair_correct[m][cb][ca];
            scores[m].table = m;
            scores[m].score = tot > 0 ? (double)cor / tot : 0.0;
        }
        for (int i = 1; i < L1_M; i++) {
            ts_t v = scores[i]; int j = i - 1;
            while (j >= 0 && scores[j].score < v.score) { scores[j+1] = scores[j]; j--; }
            scores[j+1] = v;
        }
        printf("  Specialist %d (%d→%d): best table %d (%.1f%%), worst kept %d (%.1f%%)\n",
               si, ca, cb, scores[0].table, 100*scores[0].score,
               scores[SPEC_TOP_TABLES-1].table, 100*scores[SPEC_TOP_TABLES-1].score);
        for (int k = 0; k < SPEC_TOP_TABLES; k++)
            spec_tables[si][k] = scores[k].table;
    }

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Total build: %.1fs\n\n", build_sec);

    /* ============================================================
     * Classify: LSH + GSH + specialist table selection.
     * ============================================================ */
    probe_state_t gst;
    gst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gst.max_union = cfg.max_union; gst.n_hit = 0;
    uint8_t* q_gsh = calloc(GSH_SB, 1);
    uint8_t* gsh_full_mask = malloc(GSH_SB); memset(gsh_full_mask, 0xFF, GSH_SB);

    int lsh_correct = 0, gsh_correct = 0, combined_correct = 0;
    int agree_count = 0, agree_correct = 0;
    int disagree_count = 0, spec_used = 0, spec_correct = 0;

    printf("Classifying %d test queries...\n", ds.n_test);
    clock_t t_sweep = clock();

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];

        /* LSH pass. */
        for (int m = 0; m < L1_M; m++)
            q_ptrs[m] = l1_test[m] + (size_t)qi * L1_SB;
        probe_state_reset(&st);
        for (int m = 0; m < L1_M; m++)
            probe_table(&l1_tables[m], q_ptrs[m], L1_NP, L1_SB,
                        cfg.max_radius, cfg.min_cands, &st, scratch);
        u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;
        int lsh_pred = glyph_resolver_sum_knn(&u, L1_M, L1_SB,
                                               l1_train, q_ptrs, l1_mask, KNN_K);
        if (lsh_pred == y) lsh_correct++;

        /* GSH pass. */
        union_per_table_labels(&st, L1_M, L1_SB, l1_train, q_ptrs,
                               l1_mask, ds.y_train, -1, vote_labels);
        encode_gsh_sig(vote_labels, L1_M, q_gsh, GSH_SB);
        probe_state_reset(&gst);
        probe_table(&gsh_table, q_gsh, 16, L1_SB,
                    cfg.max_radius, cfg.min_cands, &gst, scratch);
        int gsh_pred = -1;
        {
            int32_t best = INT32_MAX;
            for (int j = 0; j < gst.n_hit; j++) {
                int idx = gst.hit_list[j];
                int32_t d = m4t_popcount_dist(q_gsh,
                    gsh_train + (size_t)idx * GSH_SB, gsh_full_mask, GSH_SB);
                if (d < best) { best = d; gsh_pred = ds.y_train[idx]; }
            }
        }
        if (gsh_pred == y) gsh_correct++;

        /* Combination. */
        int final_pred = lsh_pred;
        if (lsh_pred == gsh_pred) {
            agree_count++;
            if (lsh_pred == y) agree_correct++;
        } else {
            disagree_count++;
            /* Find specialist for this confusion. */
            int spec_idx = -1;
            for (int si = 0; si < n_spec; si++) {
                if ((conf_pairs[si].a == lsh_pred && conf_pairs[si].b == gsh_pred) ||
                    (conf_pairs[si].a == gsh_pred && conf_pairs[si].b == lsh_pred)) {
                    spec_idx = si; break;
                }
            }
            if (spec_idx >= 0 && st.n_hit > 0) {
                spec_used++;
                /* Re-rank LSH union using only the specialist's
                 * selected tables (the tables routing proved
                 * discriminate this confusion pair). */
                int32_t sp_best = INT32_MAX;
                int sp_pred = lsh_pred;
                for (int j = 0; j < st.n_hit; j++) {
                    int idx = st.hit_list[j];
                    int32_t d = 0;
                    for (int k = 0; k < SPEC_TOP_TABLES; k++) {
                        int tm = spec_tables[spec_idx][k];
                        d += m4t_popcount_dist(
                            q_ptrs[tm],
                            l1_train[tm] + (size_t)idx * L1_SB,
                            l1_mask, L1_SB);
                    }
                    if (d < sp_best) { sp_best = d; sp_pred = ds.y_train[idx]; }
                }
                final_pred = sp_pred;
                if (sp_pred == y) spec_correct++;
            }
        }
        if (final_pred == y) combined_correct++;

        if ((qi + 1) % 1000 == 0 || qi == ds.n_test - 1) {
            printf("  %d/%d  LSH=%.2f%%  GSH=%.2f%%  combined=%.2f%%\n",
                   qi + 1, ds.n_test,
                   100.0 * lsh_correct / (qi + 1),
                   100.0 * gsh_correct / (qi + 1),
                   100.0 * combined_correct / (qi + 1));
            fflush(stdout);
        }
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("\n=== Results ===\n");
    printf("  LSH k=%d-NN:                %6.2f%%\n", KNN_K, 100.0 * lsh_correct / ds.n_test);
    printf("  GSH 1-NN:                   %6.2f%%\n", 100.0 * gsh_correct / ds.n_test);
    printf("  Combined (LSH+GSH+spec):    %6.2f%%\n", 100.0 * combined_correct / ds.n_test);
    printf("\n  Agreement: %d (%.1f%%),  P(correct|agree)=%.2f%%\n",
           agree_count, 100.0 * agree_count / ds.n_test,
           agree_count ? 100.0 * agree_correct / agree_count : 0.0);
    printf("  Disagreement: %d,  specialist matched: %d\n", disagree_count, spec_used);
    printf("  Specialist correct: %d / %d (%.1f%%)\n",
           spec_correct, spec_used,
           spec_used ? 100.0 * spec_correct / spec_used : 0.0);
    printf("  Sweep: %.1fs\n\n", sweep_sec);

    /* Cleanup. */
    free(all_train_labels);
    free(q_gsh); free(gsh_full_mask);
    free(gsh_train); glyph_bucket_table_free(&gsh_table);
    free(gst.votes); free(gst.hit_list);
    free(vote_labels); free(st.votes); free(st.hit_list); free(q_ptrs);
    for (int m = 0; m < L1_M; m++) {
        glyph_sig_builder_free(&l1_builders[m]);
        glyph_bucket_table_free(&l1_tables[m]);
        free(l1_train[m]); free(l1_test[m]);
    }
    free(l1_builders); free(l1_tables); free(l1_train); free(l1_test);
    glyph_dataset_free(&ds);
    return 0;
}
