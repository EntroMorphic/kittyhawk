/*
 * fashion_atomics.c — diagnostic instrumentation for the Fashion-MNIST
 * resolver gap.
 *
 * Runs a single M=cfg.m_max pass over the test set with a scalar SUM
 * resolver and, for every failing query, measures three atomic signals
 * about *where inside the routing lattice* the failure lives:
 *
 *   Atom 1 — rank & gap
 *     Find the true-class prototype with the smallest sum_dist inside
 *     the union. Rank = number of candidates with strictly smaller
 *     sum_dist. Gap = sum_dist(best_true) − sum_dist(winner). Tells us
 *     whether the correct answer is rank-2 by 1 Hamming unit (rescorable)
 *     or rank-50 by 30 units (representation-bound).
 *
 *   Atom 2 — per-table 1-NN agreement
 *     For each of M tables, run a 1-NN restricted to the union under
 *     that table's projection alone, read the predicted label. Count
 *     per failing query how many tables individually vote true vs
 *     winner vs other. If the median failing query has >M/2 tables
 *     already voting wrong, the projection itself is the bottleneck
 *     and fusion cannot rescue it.
 *
 *   Atom 3 — per-table signature-distance gap
 *     For each of M tables, compute the min popcount_dist to any
 *     candidate labeled y_true and to any candidate labeled y_winner.
 *     Accumulate (d_winner − d_true) across all (failing query, table)
 *     pairs. Positive mean ⇒ on average the lattice does place the
 *     true class closer per-table, but the signal gets lost in the
 *     summed resolver. Negative mean ⇒ the lattice genuinely places
 *     the wrong garment closer and B.2 density-mixing is needed.
 *
 * All three measurements are routing-native — only popcount_dist on
 * packed trit signatures. No pixel math, no dense fallback.
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
    st->n_hit = 0;
    st->n_probes = 0;
}

static void probe_table(const glyph_bucket_table_t* bt,
                        const uint8_t* q_sig,
                        int n_proj,
                        int sig_bytes,
                        int max_radius,
                        int min_cands,
                        probe_state_t* st,
                        uint8_t* scratch)
{
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

/* Histogram bins. */
#define RANK_BINS 8
static const int rank_edges[RANK_BINS] = {1, 2, 4, 8, 16, 64, 256, INT_MAX};
static const char* rank_labels[RANK_BINS] = {
    "rank 0", "rank 1", "rank 2-3", "rank 4-7", "rank 8-15",
    "rank 16-63", "rank 64-255", "rank >=256"
};

#define GAP_BINS 7
static const int gap_edges[GAP_BINS] = {1, 2, 4, 8, 16, 32, INT_MAX};
static const char* gap_labels[GAP_BINS] = {
    "gap 0",   "gap 1",   "gap 2-3", "gap 4-7",
    "gap 8-15","gap 16-31","gap >=32"
};

static int bin_of(int v, const int* edges, int n_bins) {
    for (int i = 0; i < n_bins; i++) if (v < edges[i]) return i;
    return n_bins - 1;
}

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;
    cfg.mode = "full";  /* atomics tool always runs full pipeline */

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) {
        fprintf(stderr, "failed to load dataset from %s\n", cfg.data_dir);
        return 1;
    }
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    const int mixed_mode = (strcmp(cfg.density_schedule, "mixed") == 0);

    printf("fashion_atomics (libglyph diagnostic)\n");
    printf("  data_dir=%s  deskew=%s\n", cfg.data_dir, cfg.no_deskew ? "off" : "on");
    printf("  n_proj=%d  density=%.2f  M=%d  max_radius=%d  min_cands=%d  max_union=%d\n",
           cfg.n_proj, cfg.density, cfg.m_max, cfg.max_radius, cfg.min_cands, cfg.max_union);
    if (mixed_mode) {
        printf("  density_schedule=mixed  density_triple=%.2f,%.2f,%.2f\n",
               cfg.density_triple[0], cfg.density_triple[1], cfg.density_triple[2]);
    } else {
        printf("  density_schedule=fixed\n");
    }
    printf("  n_train=%d  n_test=%d\n\n", ds.n_train, ds.n_test);

    int sig_bytes = M4T_TRIT_PACKED_BYTES(cfg.n_proj);
    if (sig_bytes != 4) { fprintf(stderr, "n_proj must be 16\n"); return 1; }
    const int M = cfg.m_max;
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    glyph_sig_builder_t* builders = calloc((size_t)M, sizeof(glyph_sig_builder_t));
    uint8_t** train_sigs = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** test_sigs  = calloc((size_t)M, sizeof(uint8_t*));
    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));

    for (int m = 0; m < M; m++) {
        uint32_t seeds[4];
        derive_seed((uint32_t)m, cfg.base_seed, seeds);
        double table_density = mixed_mode ? cfg.density_triple[m % 3] : cfg.density;
        if (glyph_sig_builder_init(&builders[m], cfg.n_proj, ds.input_dim, table_density,
                                    seeds[0], seeds[1], seeds[2], seeds[3],
                                    ds.x_train, n_calib) != 0) return 1;
        train_sigs[m] = calloc((size_t)ds.n_train * sig_bytes, 1);
        test_sigs[m]  = calloc((size_t)ds.n_test  * sig_bytes, 1);
        glyph_sig_encode_batch(&builders[m], ds.x_train, ds.n_train, train_sigs[m]);
        glyph_sig_encode_batch(&builders[m], ds.x_test,  ds.n_test,  test_sigs[m]);
        if (glyph_bucket_build(&tables[m], train_sigs[m], ds.n_train, sig_bytes) != 0) return 1;
    }
    printf("Built %d tables.\n\n", M);

    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    st.n_hit = 0;

    uint8_t scratch[4];
    uint8_t mask[4]; memset(mask, 0xFF, 4);
    const uint8_t** q_sigs_p = calloc((size_t)M, sizeof(uint8_t*));

    /* Per-candidate sum_dist scratch (length max_union). */
    int32_t* cand_sum = malloc((size_t)cfg.max_union * sizeof(int32_t));

    /* Magnet audit: count how many failing queries each training
     * prototype won as the SUM winner. Length n_train. */
    int* proto_fail_wins = calloc((size_t)ds.n_train, sizeof(int));

    /* Atomics accumulators. */
    int rank_hist[RANK_BINS] = {0};
    int gap_hist[GAP_BINS]   = {0};
    int64_t gap_sum_total = 0;
    int     gap_count = 0;
    int     n_fail = 0, n_correct = 0, n_fail_no_true_in_union = 0;
    long    votes_true_sum = 0, votes_winner_sum = 0, votes_other_sum = 0;
    int64_t table_gap_sum = 0;    /* sum of (d_winner - d_true) across all (fail, table) pairs where both labels appear */
    int64_t table_gap_count = 0;
    int     table_gap_positive = 0; /* # pairs where d_true < d_winner */
    int     table_gap_zero     = 0;
    int     table_gap_negative = 0;

    /* Per-(true, winner) class pair failure counts, plus per-pair
     * accumulators so we can report Atoms 1-3 broken down by the
     * dominant confusion pairs. */
    int  pair_fail[N_CLASSES][N_CLASSES] = {{0}};
    long pair_votes_true[N_CLASSES][N_CLASSES]   = {{0}};
    long pair_votes_winner[N_CLASSES][N_CLASSES] = {{0}};
    int64_t pair_gap[N_CLASSES][N_CLASSES] = {{0}};  /* sum of sum-dist gap */
    int64_t pair_table_gap[N_CLASSES][N_CLASSES] = {{0}}; /* sum of per-table d_winner-d_true */
    int64_t pair_table_gap_ct[N_CLASSES][N_CLASSES] = {{0}};

    clock_t t0 = clock();
    for (int s = 0; s < ds.n_test; s++) {
        int y_true = ds.y_test[s];
        for (int m = 0; m < M; m++) q_sigs_p[m] = test_sigs[m] + (size_t)s * sig_bytes;

        probe_state_reset(&st);
        for (int m = 0; m < M; m++) {
            probe_table(&tables[m], q_sigs_p[m], cfg.n_proj, sig_bytes,
                        cfg.max_radius, cfg.min_cands, &st, scratch);
        }

        /* Precompute sum_dist for every candidate in the union. */
        int32_t winner_sum = INT32_MAX;
        int winner_label = -1;
        int winner_proto = -1;
        for (int j = 0; j < st.n_hit; j++) {
            int idx = st.hit_list[j];
            int32_t d = 0;
            for (int m = 0; m < M; m++) {
                d += m4t_popcount_dist(q_sigs_p[m],
                                       train_sigs[m] + (size_t)idx * sig_bytes,
                                       mask, sig_bytes);
            }
            cand_sum[j] = d;
            if (d < winner_sum) {
                winner_sum = d;
                winner_label = ds.y_train[idx];
                winner_proto = idx;
            }
        }

        if (winner_label == y_true) { n_correct++; continue; }
        n_fail++;
        if (winner_proto >= 0) proto_fail_wins[winner_proto]++;

        /* Atom 1: find best-true sum_dist and its rank. */
        int32_t best_true_sum = INT32_MAX;
        int     best_true_hit = -1;
        for (int j = 0; j < st.n_hit; j++) {
            int idx = st.hit_list[j];
            if (ds.y_train[idx] == y_true && cand_sum[j] < best_true_sum) {
                best_true_sum = cand_sum[j];
                best_true_hit = j;
            }
        }
        if (best_true_hit < 0) { n_fail_no_true_in_union++; continue; }
        int rank = 0;
        for (int j = 0; j < st.n_hit; j++) if (cand_sum[j] < best_true_sum) rank++;
        rank_hist[bin_of(rank, rank_edges, RANK_BINS)]++;

        int32_t gap = best_true_sum - winner_sum;  /* ≥0 since winner is argmin */
        gap_hist[bin_of((int)gap, gap_edges, GAP_BINS)]++;
        gap_sum_total += gap;
        gap_count++;

        if (y_true < N_CLASSES && winner_label >= 0 && winner_label < N_CLASSES) {
            pair_fail[y_true][winner_label]++;
            pair_gap [y_true][winner_label] += gap;
        }

        /* Atom 2 + 3: per-table scan. Single pass finds (per-table
         * 1-NN label, min d_true, min d_winner). */
        int votes_true = 0, votes_winner = 0, votes_other = 0;
        for (int m = 0; m < M; m++) {
            const uint8_t* qm = q_sigs_p[m];
            const uint8_t* tm = train_sigs[m];
            int32_t best_d      = INT32_MAX; int     best_lbl = -1;
            int32_t best_d_true = INT32_MAX;
            int32_t best_d_win  = INT32_MAX;
            for (int j = 0; j < st.n_hit; j++) {
                int idx = st.hit_list[j];
                int32_t d = m4t_popcount_dist(qm, tm + (size_t)idx * sig_bytes, mask, sig_bytes);
                int lbl = ds.y_train[idx];
                if (d < best_d) { best_d = d; best_lbl = lbl; }
                if (lbl == y_true        && d < best_d_true) best_d_true = d;
                if (lbl == winner_label  && d < best_d_win)  best_d_win  = d;
            }
            if (best_lbl == y_true)              votes_true++;
            else if (best_lbl == winner_label)   votes_winner++;
            else                                 votes_other++;

            if (best_d_true != INT32_MAX && best_d_win != INT32_MAX) {
                int32_t d_gap = best_d_win - best_d_true;
                table_gap_sum += d_gap;
                table_gap_count++;
                if (d_gap > 0)      table_gap_positive++;
                else if (d_gap < 0) table_gap_negative++;
                else                table_gap_zero++;
                if (y_true < N_CLASSES && winner_label < N_CLASSES) {
                    pair_table_gap   [y_true][winner_label] += d_gap;
                    pair_table_gap_ct[y_true][winner_label]++;
                }
            }
        }
        votes_true_sum  += votes_true;
        votes_winner_sum += votes_winner;
        votes_other_sum  += votes_other;
        if (y_true < N_CLASSES && winner_label < N_CLASSES) {
            pair_votes_true  [y_true][winner_label] += votes_true;
            pair_votes_winner[y_true][winner_label] += votes_winner;
        }
    }
    double t_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Scan complete: %.2fs, %d correct, %d fail (%d with true class absent from union).\n\n",
           t_sec, n_correct, n_fail, n_fail_no_true_in_union);

    if (n_fail == 0 || gap_count == 0) { printf("no failures; nothing to report\n"); return 0; }

    printf("=== Atom 1: rank of true-class best prototype (by sum_dist) ===\n");
    printf("  failing queries with true class present in union: %d\n", gap_count);
    for (int b = 0; b < RANK_BINS; b++) {
        printf("  %-14s %6d  %5.1f%%\n", rank_labels[b], rank_hist[b],
               100.0 * rank_hist[b] / gap_count);
    }
    printf("\n");

    printf("=== Atom 1b: sum_dist gap (best_true - winner) ===\n");
    printf("  mean gap: %.2f Hamming units\n", (double)gap_sum_total / gap_count);
    for (int b = 0; b < GAP_BINS; b++) {
        printf("  %-14s %6d  %5.1f%%\n", gap_labels[b], gap_hist[b],
               100.0 * gap_hist[b] / gap_count);
    }
    printf("\n");

    printf("=== Atom 2: per-table 1-NN vote (failing queries only) ===\n");
    printf("  mean per-query votes out of M=%d:\n", M);
    printf("    true class    : %6.2f  (%5.1f%%)\n",
           (double)votes_true_sum / n_fail,
           100.0 * votes_true_sum / ((double)n_fail * M));
    printf("    winner (wrong): %6.2f  (%5.1f%%)\n",
           (double)votes_winner_sum / n_fail,
           100.0 * votes_winner_sum / ((double)n_fail * M));
    printf("    other         : %6.2f  (%5.1f%%)\n",
           (double)votes_other_sum / n_fail,
           100.0 * votes_other_sum / ((double)n_fail * M));
    printf("\n");

    printf("=== Atom 3: per-table sig-distance (d_winner - d_true) ===\n");
    printf("  samples: %lld (failing query × table pairs where both labels present)\n",
           (long long)table_gap_count);
    if (table_gap_count > 0) {
        printf("  mean per-table gap : %+.3f Hamming bits\n",
               (double)table_gap_sum / table_gap_count);
        printf("    true closer  (gap>0): %7d  %5.1f%%\n", table_gap_positive,
               100.0 * table_gap_positive / table_gap_count);
        printf("    tied         (gap=0): %7d  %5.1f%%\n", table_gap_zero,
               100.0 * table_gap_zero     / table_gap_count);
        printf("    winner closer(gap<0): %7d  %5.1f%%\n", table_gap_negative,
               100.0 * table_gap_negative / table_gap_count);
    }
    printf("\n");

    /* Top confusion pairs with per-pair atomics breakdown. */
    typedef struct { int t, p, n; } cf_t;
    cf_t pairs[N_CLASSES * N_CLASSES];
    int n_pairs = 0;
    for (int t = 0; t < N_CLASSES; t++)
        for (int p = 0; p < N_CLASSES; p++)
            if (t != p && pair_fail[t][p] > 0) {
                pairs[n_pairs].t = t; pairs[n_pairs].p = p; pairs[n_pairs].n = pair_fail[t][p];
                n_pairs++;
            }
    for (int i = 1; i < n_pairs; i++) {
        cf_t v = pairs[i]; int j = i - 1;
        while (j >= 0 && pairs[j].n < v.n) { pairs[j+1] = pairs[j]; j--; }
        pairs[j+1] = v;
    }
    int shown = (n_pairs < 8) ? n_pairs : 8;
    printf("=== Top-%d confusion pairs: per-pair atomics ===\n", shown);
    printf(" true->pred  count  mean_sum_gap  vote_true  vote_winner  mean_tbl_gap\n");
    for (int i = 0; i < shown; i++) {
        int t = pairs[i].t, p = pairs[i].p, n = pairs[i].n;
        double mean_sum_gap = (double)pair_gap[t][p] / n;
        double mean_vt = (double)pair_votes_true  [t][p] / n;
        double mean_vw = (double)pair_votes_winner[t][p] / n;
        double mean_tg = pair_table_gap_ct[t][p]
                       ? (double)pair_table_gap[t][p] / pair_table_gap_ct[t][p]
                       : 0.0;
        printf("   %2d->%-2d   %5d   %10.2f   %6.2f    %7.2f      %+7.3f\n",
               t, p, n, mean_sum_gap, mean_vt, mean_vw, mean_tg);
    }
    printf("\n");

    /* ============================================================
     * Magnet audit: training prototypes that win the most failing
     * queries. If the distribution is concentrated on a few indices
     * they are geometric magnets in signature space; if it's flat
     * it's a structural cluster-center problem, not a pathological
     * prototype problem.
     * ============================================================ */
    {
        int top_k = 20;
        typedef struct { int idx; int wins; } mag_t;
        mag_t top[32];
        for (int i = 0; i < top_k; i++) { top[i].idx = -1; top[i].wins = 0; }
        int total_wins = 0;
        int n_magnets = 0;  /* prototypes that won ≥1 failing query */
        for (int i = 0; i < ds.n_train; i++) {
            int w = proto_fail_wins[i];
            if (w == 0) continue;
            n_magnets++;
            total_wins += w;
            if (w > top[top_k - 1].wins) {
                int k = top_k - 1;
                top[k].idx = i; top[k].wins = w;
                while (k > 0 && top[k-1].wins < top[k].wins) {
                    mag_t tmp = top[k-1]; top[k-1] = top[k]; top[k] = tmp; k--;
                }
            }
        }
        printf("=== Magnet audit: top-%d SUM-winning prototypes on failures ===\n", top_k);
        printf("  %d distinct prototypes won at least one failing query\n", n_magnets);
        printf("  total failing wins: %d (= n_fail check: %d)\n", total_wins, n_fail);
        double top_share = 0.0;
        for (int i = 0; i < top_k && top[i].idx >= 0; i++) top_share += top[i].wins;
        printf("  top-%d share: %.1f%% of all failing wins\n\n",
               top_k, 100.0 * top_share / (total_wins ? total_wins : 1));
        printf("  rank  proto_idx  label  wins  %%_of_fails\n");
        for (int i = 0; i < top_k && top[i].idx >= 0; i++) {
            int idx = top[i].idx;
            printf("   %2d   %7d     %2d   %4d     %5.2f%%\n",
                   i, idx, ds.y_train[idx], top[i].wins,
                   100.0 * top[i].wins / (n_fail ? n_fail : 1));
        }
        printf("\n");
    }

    /* Cleanup. */
    free(proto_fail_wins);
    free(cand_sum);
    free(st.votes); free(st.hit_list);
    free(q_sigs_p);
    for (int m = 0; m < M; m++) {
        glyph_sig_builder_free(&builders[m]);
        glyph_bucket_table_free(&tables[m]);
        free(train_sigs[m]); free(test_sigs[m]);
    }
    free(builders); free(train_sigs); free(test_sigs); free(tables);
    glyph_dataset_free(&ds);
    return 0;
}
