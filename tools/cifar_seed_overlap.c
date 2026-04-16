/*
 * cifar_seed_overlap.c — measures error correlation across seeds.
 *
 * Runs 3 seeds at M=64 on the same dataset, tracks per-query
 * correctness, and reports how many queries are correct in all 3,
 * exactly 2, exactly 1, and 0 runs. High overlap → failures are
 * input-driven (genuinely hard images). Low overlap → failures
 * are projection-driven (different projections see different
 * structure).
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_rng.h"
#include "glyph_sig.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include "glyph_resolver.h"
#include "m4t_trit_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_SEEDS 3
#define N_CLASSES 10

static void derive_seed(uint32_t m, const uint32_t base[4], uint32_t out[4]) {
    if (m == 0) { out[0]=base[0]; out[1]=base[1]; out[2]=base[2]; out[3]=base[3]; return; }
    out[0] = 2654435761u * m + 1013904223u;
    out[1] = 1597334677u * m + 2246822519u;
    out[2] = 3266489917u * m +  668265263u;
    out[3] =  374761393u * m + 3266489917u;
}

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

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    int sig_bytes = M4T_TRIT_PACKED_BYTES(cfg.n_proj);
    if (sig_bytes != 4) { fprintf(stderr, "n_proj must be 16\n"); return 1; }
    const int M = cfg.m_max;
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    uint32_t all_seeds[N_SEEDS][4] = {
        {42, 123, 456, 789},
        {7, 31, 127, 511},
        {101, 307, 601, 907}
    };

    /* Per-query correctness: correct[seed][query] = 0 or 1. */
    uint8_t* correct[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++)
        correct[s] = calloc((size_t)ds.n_test, 1);

    /* Seed-0 confidence metrics, filled during the first seed run. */
    int32_t* s0_gap     = calloc((size_t)ds.n_test, sizeof(int32_t));
    int*     s0_votes_w = calloc((size_t)ds.n_test, sizeof(int));
    int*     s0_votes_t = calloc((size_t)ds.n_test, sizeof(int));

    printf("cifar_seed_overlap: %d seeds, M=%d, n_proj=%d, density=%.2f\n",
           N_SEEDS, M, cfg.n_proj, cfg.density);
    printf("  n_train=%d  n_test=%d  input_dim=%d\n\n", ds.n_train, ds.n_test, ds.input_dim);

    for (int si = 0; si < N_SEEDS; si++) {
        printf("Seed %d: {%u,%u,%u,%u} ... ",
               si, all_seeds[si][0], all_seeds[si][1], all_seeds[si][2], all_seeds[si][3]);
        fflush(stdout);
        clock_t t0 = clock();

        glyph_sig_builder_t* builders = calloc((size_t)M, sizeof(glyph_sig_builder_t));
        uint8_t** train_sigs = calloc((size_t)M, sizeof(uint8_t*));
        uint8_t** test_sigs  = calloc((size_t)M, sizeof(uint8_t*));
        glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));

        for (int m = 0; m < M; m++) {
            uint32_t seeds[4];
            derive_seed((uint32_t)m, all_seeds[si], seeds);
            glyph_sig_builder_init(&builders[m], cfg.n_proj, ds.input_dim, cfg.density,
                                    seeds[0], seeds[1], seeds[2], seeds[3],
                                    ds.x_train, n_calib);
            train_sigs[m] = calloc((size_t)ds.n_train * sig_bytes, 1);
            test_sigs[m]  = calloc((size_t)ds.n_test  * sig_bytes, 1);
            glyph_sig_encode_batch(&builders[m], ds.x_train, ds.n_train, train_sigs[m]);
            glyph_sig_encode_batch(&builders[m], ds.x_test,  ds.n_test,  test_sigs[m]);
            glyph_bucket_build(&tables[m], train_sigs[m], ds.n_train, sig_bytes);
        }

        probe_state_t st;
        st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
        st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
        st.max_union = cfg.max_union;
        st.n_hit = 0;
        uint8_t scratch[4];
        uint8_t mask[4]; memset(mask, 0xFF, 4);
        const uint8_t** q_sigs_p = calloc((size_t)M, sizeof(uint8_t*));

        int n_correct = 0;
        int32_t* cand_sum = NULL;
        if (si == 0) cand_sum = malloc((size_t)cfg.max_union * sizeof(int32_t));

        for (int q = 0; q < ds.n_test; q++) {
            for (int m = 0; m < M; m++)
                q_sigs_p[m] = test_sigs[m] + (size_t)q * sig_bytes;
            probe_state_reset(&st);
            for (int m = 0; m < M; m++)
                probe_table(&tables[m], q_sigs_p[m], cfg.n_proj, sig_bytes,
                            cfg.max_radius, cfg.min_cands, &st, scratch);

            glyph_union_t u = {0};
            u.hit_list = st.hit_list; u.n_hit = st.n_hit;
            u.votes = st.votes; u.y_train = ds.y_train; u.n_classes = N_CLASSES;
            int pred = glyph_resolver_sum(&u, M, sig_bytes, train_sigs, q_sigs_p, mask);
            if (pred == ds.y_test[q]) { correct[si][q] = 1; n_correct++; }

            /* On seed 0: compute confidence metrics. */
            if (si == 0) {
                int y_true = ds.y_test[q];
                /* Compute sum_dist for all candidates. */
                int32_t best_sum = INT32_MAX, runner_sum = INT32_MAX;
                int32_t best_true_sum = INT32_MAX;
                for (int j = 0; j < st.n_hit; j++) {
                    int idx = st.hit_list[j];
                    int32_t d = 0;
                    for (int m = 0; m < M; m++)
                        d += m4t_popcount_dist(q_sigs_p[m],
                                               train_sigs[m] + (size_t)idx * sig_bytes,
                                               mask, sig_bytes);
                    if (d < best_sum) {
                        runner_sum = best_sum;
                        best_sum = d;
                    } else if (d < runner_sum) {
                        runner_sum = d;
                    }
                    if (ds.y_train[idx] == y_true && d < best_true_sum)
                        best_true_sum = d;
                }
                s0_gap[q] = (pred == y_true)
                    ? (runner_sum - best_sum)   /* positive = winner margin */
                    : (best_true_sum - best_sum);  /* positive = how far behind true is */

                /* Per-table 1-NN vote for winner class. */
                int vw = 0, vt = 0;
                for (int m = 0; m < M; m++) {
                    int32_t bd = INT32_MAX; int bl = -1;
                    for (int j = 0; j < st.n_hit; j++) {
                        int idx = st.hit_list[j];
                        int32_t d = m4t_popcount_dist(q_sigs_p[m],
                                       train_sigs[m] + (size_t)idx * sig_bytes,
                                       mask, sig_bytes);
                        if (d < bd) { bd = d; bl = ds.y_train[idx]; }
                    }
                    if (bl == pred) vw++;
                    if (bl == y_true) vt++;
                }
                s0_votes_w[q] = vw;
                s0_votes_t[q] = vt;
            }
        }
        free(cand_sum);

        double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("%d/%d correct (%.2f%%) in %.1fs\n", n_correct, ds.n_test,
               100.0 * n_correct / ds.n_test, sec);

        free(st.votes); free(st.hit_list); free(q_sigs_p);
        for (int m = 0; m < M; m++) {
            glyph_sig_builder_free(&builders[m]);
            glyph_bucket_table_free(&tables[m]);
            free(train_sigs[m]); free(test_sigs[m]);
        }
        free(builders); free(train_sigs); free(test_sigs); free(tables);
    }

    /* Overlap analysis. */
    int hist[N_SEEDS + 1]; memset(hist, 0, sizeof(hist));
    for (int q = 0; q < ds.n_test; q++) {
        int sum = 0;
        for (int s = 0; s < N_SEEDS; s++) sum += correct[s][q];
        hist[sum]++;
    }

    printf("\n=== Cross-seed overlap (N_SEEDS=%d) ===\n", N_SEEDS);
    printf("  correct in 0/%d (always wrong): %5d  %5.1f%%\n",
           N_SEEDS, hist[0], 100.0 * hist[0] / ds.n_test);
    for (int k = 1; k < N_SEEDS; k++)
        printf("  correct in %d/%d              : %5d  %5.1f%%\n",
               k, N_SEEDS, hist[k], 100.0 * hist[k] / ds.n_test);
    printf("  correct in %d/%d (always right): %5d  %5.1f%%\n",
           N_SEEDS, N_SEEDS, hist[N_SEEDS], 100.0 * hist[N_SEEDS] / ds.n_test);

    /* Per-class breakdown of always-wrong and always-right. */
    int always_wrong_class[N_CLASSES] = {0};
    int always_right_class[N_CLASSES] = {0};
    int class_total[N_CLASSES] = {0};
    for (int q = 0; q < ds.n_test; q++) {
        int y = ds.y_test[q];
        if (y >= 0 && y < N_CLASSES) {
            class_total[y]++;
            int sum = 0;
            for (int s = 0; s < N_SEEDS; s++) sum += correct[s][q];
            if (sum == 0) always_wrong_class[y]++;
            if (sum == N_SEEDS) always_right_class[y]++;
        }
    }
    printf("\nPer-class always-wrong / always-right:\n");
    printf("  class  total  always_wrong  always_right  aw%%    ar%%\n");
    for (int c = 0; c < N_CLASSES; c++) {
        if (class_total[c] == 0) continue;
        printf("   %2d   %5d     %5d         %5d     %5.1f  %5.1f\n",
               c, class_total[c], always_wrong_class[c], always_right_class[c],
               100.0 * always_wrong_class[c] / class_total[c],
               100.0 * always_right_class[c] / class_total[c]);
    }

    /* Confidence metrics split by always-right / always-wrong / swing. */
    printf("\n=== Seed-0 confidence: always-right vs always-wrong ===\n");
    {
        long ar_gap_sum = 0, aw_gap_sum = 0;
        long ar_vw_sum = 0, aw_vw_sum = 0;
        long ar_vt_sum = 0, aw_vt_sum = 0;
        int ar_n = 0, aw_n = 0;
        /* Gap histogram for always-right. */
        int ar_gap_hist[6] = {0};  /* 0-1, 2-3, 4-7, 8-15, 16-31, 32+ */
        int aw_gap_hist[6] = {0};
        const int gap_edges[6] = {2, 4, 8, 16, 32, INT32_MAX};
        const char* gap_labels[6] = {"0-1","2-3","4-7","8-15","16-31","32+"};

        for (int q = 0; q < ds.n_test; q++) {
            int sum = 0;
            for (int s = 0; s < N_SEEDS; s++) sum += correct[s][q];

            int bin = 0;
            int32_t g = s0_gap[q] < 0 ? -s0_gap[q] : s0_gap[q];
            for (int b = 0; b < 6; b++) { if (g < gap_edges[b]) { bin = b; break; } }

            if (sum == N_SEEDS) {
                ar_gap_sum += s0_gap[q];
                ar_vw_sum += s0_votes_w[q];
                ar_vt_sum += s0_votes_t[q];
                ar_gap_hist[bin]++;
                ar_n++;
            } else if (sum == 0) {
                aw_gap_sum += s0_gap[q];
                aw_vw_sum += s0_votes_w[q];
                aw_vt_sum += s0_votes_t[q];
                aw_gap_hist[bin]++;
                aw_n++;
            }
        }

        printf("  always-right (n=%d):\n", ar_n);
        printf("    mean sum_dist margin (winner - runner-up): %.2f\n",
               ar_n ? (double)ar_gap_sum / ar_n : 0.0);
        printf("    mean per-table votes for winner: %.2f / %d (%.1f%%)\n",
               ar_n ? (double)ar_vw_sum / ar_n : 0.0, M,
               ar_n ? 100.0 * ar_vw_sum / ((double)ar_n * M) : 0.0);
        printf("    mean per-table votes for true:   %.2f / %d (%.1f%%)\n",
               ar_n ? (double)ar_vt_sum / ar_n : 0.0, M,
               ar_n ? 100.0 * ar_vt_sum / ((double)ar_n * M) : 0.0);
        printf("    margin distribution:\n");
        for (int b = 0; b < 6; b++)
            printf("      %-6s %5d  %5.1f%%\n", gap_labels[b], ar_gap_hist[b],
                   ar_n ? 100.0 * ar_gap_hist[b] / ar_n : 0.0);

        printf("  always-wrong (n=%d):\n", aw_n);
        printf("    mean sum_dist gap (true_best - winner): %.2f\n",
               aw_n ? (double)aw_gap_sum / aw_n : 0.0);
        printf("    mean per-table votes for winner: %.2f / %d (%.1f%%)\n",
               aw_n ? (double)aw_vw_sum / aw_n : 0.0, M,
               aw_n ? 100.0 * aw_vw_sum / ((double)aw_n * M) : 0.0);
        printf("    mean per-table votes for true:   %.2f / %d (%.1f%%)\n",
               aw_n ? (double)aw_vt_sum / aw_n : 0.0, M,
               aw_n ? 100.0 * aw_vt_sum / ((double)aw_n * M) : 0.0);
        printf("    gap distribution:\n");
        for (int b = 0; b < 6; b++)
            printf("      %-6s %5d  %5.1f%%\n", gap_labels[b], aw_gap_hist[b],
                   aw_n ? 100.0 * aw_gap_hist[b] / aw_n : 0.0);
    }

    free(s0_gap); free(s0_votes_w); free(s0_votes_t);
    for (int s = 0; s < N_SEEDS; s++) free(correct[s]);
    glyph_dataset_free(&ds);
    return 0;
}
