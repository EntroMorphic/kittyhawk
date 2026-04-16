/*
 * dynamic_nproj.c — resolution-adaptive routing with recycled failures.
 *
 * Filters at N_PROJ=16, then re-ranks uncertain queries through
 * progressively wider projections (32, 64, 128, 256, 512, 1024)
 * until confident. Every stage is routing-native: the wider
 * signatures re-score the Stage-1 union, not a new index.
 *
 * Reports per-stage standalone accuracy and cascade accuracy at
 * several confidence thresholds so the optimal operating point
 * can be identified without committing to a threshold a priori.
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
#define N_STAGES  7

static const int stage_nproj[N_STAGES] = {16, 32, 64, 128, 256, 512, 1024};

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

/* Resolve and return both prediction and margin (winner - runner_up). */
static int resolve_with_margin(
    const glyph_union_t* u, int m_active, int sig_bytes,
    uint8_t* const* train_sigs, const uint8_t* const* q_sigs,
    const uint8_t* mask, int32_t* out_margin)
{
    int32_t best = INT32_MAX, runner = INT32_MAX;
    int     best_label = -1;
    for (int j = 0; j < u->n_hit; j++) {
        int idx = u->hit_list[j];
        int32_t d = 0;
        for (int m = 0; m < m_active; m++)
            d += m4t_popcount_dist(q_sigs[m],
                                   train_sigs[m] + (size_t)idx * sig_bytes,
                                   mask, sig_bytes);
        if (d < best) {
            runner = best; best = d; best_label = u->y_train[idx];
        } else if (d < runner) {
            runner = d;
        }
    }
    *out_margin = (runner == INT32_MAX) ? 0 : (runner - best);
    return best_label;
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

    const int M_filter = 16;
    const int M_rr = 32;
    const int filter_nproj = 16;
    const int filter_sig_bytes = M4T_TRIT_PACKED_BYTES(filter_nproj);
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    printf("dynamic_nproj: resolution-adaptive routing cascade\n");
    printf("  data_dir=%s  deskew=%s  density=%.2f\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on", cfg.density);
    printf("  filter: N_PROJ=%d  M=%d\n", filter_nproj, M_filter);
    printf("  re-rank stages: M_rr=%d  N_PROJ={", M_rr);
    for (int s = 1; s < N_STAGES; s++) printf("%s%d", s>1?",":"", stage_nproj[s]);
    printf("}\n");
    printf("  n_train=%d  n_test=%d  input_dim=%d\n\n", ds.n_train, ds.n_test, ds.input_dim);

    /* Stage 0: filter tables at N_PROJ=16. */
    clock_t t_build = clock();
    glyph_sig_builder_t* filter_builders = calloc((size_t)M_filter, sizeof(glyph_sig_builder_t));
    uint8_t** filter_train = calloc((size_t)M_filter, sizeof(uint8_t*));
    uint8_t** filter_test  = calloc((size_t)M_filter, sizeof(uint8_t*));
    glyph_bucket_table_t* filter_tables = calloc((size_t)M_filter, sizeof(glyph_bucket_table_t));

    for (int m = 0; m < M_filter; m++) {
        uint32_t seeds[4]; derive_seed((uint32_t)m, cfg.base_seed, seeds);
        glyph_sig_builder_init(&filter_builders[m], filter_nproj, ds.input_dim, cfg.density,
                                seeds[0], seeds[1], seeds[2], seeds[3], ds.x_train, n_calib);
        filter_train[m] = calloc((size_t)ds.n_train * filter_sig_bytes, 1);
        filter_test[m]  = calloc((size_t)ds.n_test  * filter_sig_bytes, 1);
        glyph_sig_encode_batch(&filter_builders[m], ds.x_train, ds.n_train, filter_train[m]);
        glyph_sig_encode_batch(&filter_builders[m], ds.x_test,  ds.n_test,  filter_test[m]);
        glyph_bucket_build(&filter_tables[m], filter_train[m], ds.n_train, filter_sig_bytes);
    }
    printf("  filter tables built.\n");

    /* Stages 1..6: re-rank encoders at N_PROJ=32..1024.
     * Each stage uses M_rr tables with independent seeds (offset by
     * stage index × 1000 to avoid seed overlap with other stages). */
    glyph_sig_builder_t* rr_builders[N_STAGES];
    uint8_t**            rr_train[N_STAGES];
    int                  rr_sig_bytes[N_STAGES];

    for (int s = 0; s < N_STAGES; s++) {
        rr_sig_bytes[s] = M4T_TRIT_PACKED_BYTES(stage_nproj[s]);
        int M_s = (s == 0) ? M_filter : M_rr;
        rr_builders[s] = calloc((size_t)M_s, sizeof(glyph_sig_builder_t));
        rr_train[s] = calloc((size_t)M_s, sizeof(uint8_t*));
        for (int m = 0; m < M_s; m++) {
            uint32_t seeds[4];
            derive_seed((uint32_t)(s * 1000 + m), cfg.base_seed, seeds);
            glyph_sig_builder_init(&rr_builders[s][m], stage_nproj[s], ds.input_dim,
                                    cfg.density, seeds[0], seeds[1], seeds[2], seeds[3],
                                    ds.x_train, n_calib);
            rr_train[s][m] = calloc((size_t)ds.n_train * rr_sig_bytes[s], 1);
            glyph_sig_encode_batch(&rr_builders[s][m], ds.x_train, ds.n_train, rr_train[s][m]);
        }
        printf("  stage %d (N_PROJ=%d, M=%d, sig_bytes=%d) built.\n",
               s, stage_nproj[s], M_s, rr_sig_bytes[s]);
    }
    double t_build_sec = (double)(clock() - t_build) / CLOCKS_PER_SEC;
    printf("Total build: %.1fs\n\n", t_build_sec);

    /* Query-time state. */
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    st.n_hit = 0;
    uint8_t scratch[4];
    uint8_t filter_mask[4]; memset(filter_mask, 0xFF, 4);

    const uint8_t** fq_ptrs = calloc((size_t)M_filter, sizeof(uint8_t*));

    /* Per-query, per-stage scratch for re-rank query sigs. */
    uint8_t** rr_q_bufs[N_STAGES];
    const uint8_t** rr_q_ptrs[N_STAGES];
    uint8_t* rr_masks[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        int M_s = (s == 0) ? M_filter : M_rr;
        rr_q_bufs[s] = calloc((size_t)M_s, sizeof(uint8_t*));
        rr_q_ptrs[s] = calloc((size_t)M_s, sizeof(const uint8_t*));
        for (int m = 0; m < M_s; m++) {
            rr_q_bufs[s][m] = malloc(rr_sig_bytes[s]);
            rr_q_ptrs[s][m] = rr_q_bufs[s][m];
        }
        rr_masks[s] = malloc(rr_sig_bytes[s]);
        memset(rr_masks[s], 0xFF, rr_sig_bytes[s]);
    }

    /* Per-stage counters. */
    int stage_correct[N_STAGES] = {0};
    int32_t stage_margin_sum[N_STAGES] = {0};

    /* Multi-resolution combined scoring. */
    int combined_correct_1nn = 0;
    int combined_correct_knn = 0;
    const int KNN_K = 5;

    /* Per-query stage predictions and margins (for cascade analysis). */
    int8_t*  q_pred   = malloc((size_t)ds.n_test * N_STAGES);
    int8_t*  q_correct= malloc((size_t)ds.n_test * N_STAGES);
    int32_t* q_margin = malloc((size_t)ds.n_test * N_STAGES * sizeof(int32_t));

    glyph_union_t u = {0};
    u.y_train = ds.y_train;
    u.n_classes = N_CLASSES;

    clock_t t_sweep = clock();
    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        const m4t_mtfp_t* qvec = ds.x_test + (size_t)qi * ds.input_dim;

        /* Filter pass: build union at N_PROJ=16. */
        for (int m = 0; m < M_filter; m++)
            fq_ptrs[m] = filter_test[m] + (size_t)qi * filter_sig_bytes;
        probe_state_reset(&st);
        for (int m = 0; m < M_filter; m++)
            probe_table(&filter_tables[m], fq_ptrs[m], filter_nproj, filter_sig_bytes,
                        cfg.max_radius, cfg.min_cands, &st, scratch);

        u.hit_list = st.hit_list;
        u.n_hit    = st.n_hit;
        u.votes    = st.votes;

        /* Score at each stage. */
        for (int s = 0; s < N_STAGES; s++) {
            int M_s = (s == 0) ? M_filter : M_rr;

            /* Encode query sigs for this stage. */
            for (int m = 0; m < M_s; m++)
                glyph_sig_encode(&rr_builders[s][m], qvec, rr_q_bufs[s][m]);

            int32_t margin = 0;
            int pred = resolve_with_margin(
                &u, M_s, rr_sig_bytes[s],
                rr_train[s], rr_q_ptrs[s], rr_masks[s], &margin);

            int c = (pred == y) ? 1 : 0;
            stage_correct[s] += c;
            stage_margin_sum[s] += margin;

            q_pred  [qi * N_STAGES + s] = (int8_t)pred;
            q_correct[qi * N_STAGES + s] = (int8_t)c;
            q_margin [qi * N_STAGES + s] = margin;
        }

        /* Multi-resolution combined scoring: for each candidate,
         * sum normalized distances across ALL stages.
         * Normalization: divide by (2 × n_proj × M_s) so each stage
         * contributes equally regardless of N_PROJ and table count. */
        {
            typedef struct { int64_t score; int label; } topk_t;
            topk_t topk[64];
            int n_topk = 0;
            int64_t best_combined = INT64_MAX;
            int     best_combined_label = -1;

            for (int j = 0; j < u.n_hit; j++) {
                int idx = u.hit_list[j];
                int64_t combined = 0;
                for (int s = 0; s < N_STAGES; s++) {
                    int M_s = (s == 0) ? M_filter : M_rr;
                    int32_t d = 0;
                    for (int m = 0; m < M_s; m++) {
                        d += m4t_popcount_dist(
                            rr_q_ptrs[s][m],
                            rr_train[s][m] + (size_t)idx * rr_sig_bytes[s],
                            rr_masks[s], rr_sig_bytes[s]);
                    }
                    /* Scale by 1024 / (2 * n_proj * M_s) to normalize
                     * across stages with integer arithmetic. */
                    combined += ((int64_t)d * 1024) /
                                (2 * stage_nproj[s] * M_s);
                }

                /* 1-NN tracking. */
                if (combined < best_combined) {
                    best_combined = combined;
                    best_combined_label = ds.y_train[idx];
                }

                /* Top-K tracking for k-NN. */
                int lbl = ds.y_train[idx];
                if (n_topk < KNN_K) {
                    int pos = n_topk;
                    while (pos > 0 && topk[pos-1].score > combined) {
                        topk[pos] = topk[pos-1]; pos--;
                    }
                    topk[pos].score = combined;
                    topk[pos].label = lbl;
                    n_topk++;
                } else if (combined < topk[KNN_K-1].score) {
                    int pos = KNN_K - 1;
                    while (pos > 0 && topk[pos-1].score > combined) {
                        topk[pos] = topk[pos-1]; pos--;
                    }
                    topk[pos].score = combined;
                    topk[pos].label = lbl;
                }
            }

            if (best_combined_label == y) combined_correct_1nn++;

            /* k-NN rank-weighted vote. */
            int cvotes[N_CLASSES] = {0};
            for (int i = 0; i < n_topk; i++)
                cvotes[topk[i].label] += (KNN_K - i);
            int kpred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cvotes[c] > cvotes[kpred]) kpred = c;
            if (kpred == y) combined_correct_knn++;
        }
    }
    double t_sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    /* Report. */
    printf("Sweep: %.1fs for %d queries.\n\n", t_sweep_sec, ds.n_test);

    printf("=== Multi-resolution combined scoring ===\n");
    printf("  combined 1-NN:     %6.2f%%\n", 100.0 * combined_correct_1nn / ds.n_test);
    printf("  combined k=%d-NN:  %6.2f%%\n", KNN_K, 100.0 * combined_correct_knn / ds.n_test);
    printf("\n");

    printf("=== Per-stage standalone accuracy ===\n");
    printf("  stage  N_PROJ  M   sig_bytes  accuracy   mean_margin\n");
    for (int s = 0; s < N_STAGES; s++) {
        int M_s = (s == 0) ? M_filter : M_rr;
        printf("   %d     %4d   %2d     %3d     %6.2f%%    %8.1f\n",
               s, stage_nproj[s], M_s, rr_sig_bytes[s],
               100.0 * stage_correct[s] / ds.n_test,
               (double)stage_margin_sum[s] / ds.n_test);
    }
    printf("\n");

    /* Cascade analysis: accept at the first stage whose margin ≥ T. */
    printf("=== Cascade accuracy (accept at first stage with margin >= T) ===\n");
    int thresholds[] = {0, 1, 2, 4, 8, 16, 32, 64, 128};
    int n_thresh = (int)(sizeof(thresholds) / sizeof(thresholds[0]));
    printf("     T   accuracy  %%@stg0  %%@stg1  %%@stg2  %%@stg3  %%@stg4  %%@stg5  %%@stg6\n");
    for (int ti = 0; ti < n_thresh; ti++) {
        int T = thresholds[ti];
        int correct = 0;
        int accepted_at[N_STAGES] = {0};
        for (int qi = 0; qi < ds.n_test; qi++) {
            int accepted = 0;
            for (int s = 0; s < N_STAGES; s++) {
                if (q_margin[qi * N_STAGES + s] >= T || s == N_STAGES - 1) {
                    correct += q_correct[qi * N_STAGES + s];
                    accepted_at[s]++;
                    accepted = 1;
                    break;
                }
            }
            if (!accepted) {
                correct += q_correct[qi * N_STAGES + (N_STAGES - 1)];
                accepted_at[N_STAGES - 1]++;
            }
        }
        printf("  %4d   %6.2f%%", T, 100.0 * correct / ds.n_test);
        for (int s = 0; s < N_STAGES; s++)
            printf("  %5.1f%%", 100.0 * accepted_at[s] / ds.n_test);
        printf("\n");
    }
    printf("\n");

    /* Per-stage "newly correct": queries wrong at all prior stages,
     * correct at this stage. Shows the rescue rate per escalation. */
    printf("=== Per-stage rescue rate ===\n");
    printf("  stage  N_PROJ  newly_correct  newly_wrong  net\n");
    for (int s = 0; s < N_STAGES; s++) {
        int rescued = 0, lost = 0;
        for (int qi = 0; qi < ds.n_test; qi++) {
            int prev_best = 0;
            for (int p = 0; p < s; p++)
                if (q_correct[qi * N_STAGES + p]) { prev_best = 1; break; }
            int curr = q_correct[qi * N_STAGES + s];
            if (!prev_best && curr) rescued++;
            if (s > 0) {
                int prev = q_correct[qi * N_STAGES + (s - 1)];
                if (prev && !curr) lost++;
            }
        }
        printf("   %d     %4d      %5d         %5d    %+d\n",
               s, stage_nproj[s], rescued, lost, rescued - lost);
    }
    printf("\n");

    /* Cleanup. */
    free(q_pred); free(q_correct); free(q_margin);
    for (int s = 0; s < N_STAGES; s++) {
        int M_s = (s == 0) ? M_filter : M_rr;
        for (int m = 0; m < M_s; m++) free(rr_q_bufs[s][m]);
        free(rr_q_bufs[s]); free(rr_q_ptrs[s]); free(rr_masks[s]);
        for (int m = 0; m < M_s; m++) {
            glyph_sig_builder_free(&rr_builders[s][m]);
            free(rr_train[s][m]);
        }
        free(rr_builders[s]); free(rr_train[s]);
    }
    free(st.votes); free(st.hit_list); free(fq_ptrs);
    for (int m = 0; m < M_filter; m++) {
        glyph_sig_builder_free(&filter_builders[m]);
        glyph_bucket_table_free(&filter_tables[m]);
        free(filter_train[m]); free(filter_test[m]);
    }
    free(filter_builders); free(filter_train); free(filter_test); free(filter_tables);
    glyph_dataset_free(&ds);
    return 0;
}
