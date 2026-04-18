/*
 * mnist_routed_bucket_multi.c — multi-table routed bucket LSH.
 *
 * Thin CLI consumer of libglyph. Every piece of infrastructure —
 * dataset loading, RNG, signature building, bucket index, ternary
 * multi-probe, resolver variants, CLI parsing — lives in libglyph
 * (see src/glyph_*.{h,c}). This file only orchestrates the sweep.
 *
 * Usage:
 *   mnist_routed_bucket_multi [options]
 *
 * See `--help` for the full option list. Defaults reproduce the
 * Axis 6 measurement (N_PROJ=16, density=0.33, M_MAX=64, r≤2,
 * MIN_CANDS=50, base_seed=42,123,456,789, mode=oracle).
 *
 * Full docs: journal/break_97_nproj16_phase3_results.md,
 *            docs/FINDINGS.md Axis 6.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_rng.h"
#include "glyph_sig.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include "glyph_resolver.h"

#include "m4t_trit_pack.h"
#include "m4t_route.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10

/* Per-query candidate-union state, reused across queries. Lazy zeroing
 * via hit_list[] keeps reset cost proportional to union size, not to
 * n_train. */
typedef struct {
    uint16_t* votes;        /* [n_train] vote count per prototype       */
    uint8_t*  min_radius;   /* [n_train] smallest multi-probe radius at
                              which any table found this prototype in
                              the query's neighborhood (0, 1, 2).
                              Undefined for proto_idxs not in hit_list. */
    int32_t*  hit_list;     /* [max_union] proto indices in current union */
    int       n_hit;
    int       max_union;
    int       n_probes;
    int       per_table_cands;
} probe_state_t;

typedef struct {
    const glyph_bucket_table_t* table;
    probe_state_t* state;
    int current_radius;     /* which multi-probe radius the enumerate
                              call is currently at (0, 1, or 2). Used
                              to record min_radius for newly-found or
                              re-visited candidates. */
} probe_ctx_t;

static int probe_cb(const uint8_t* probe_sig, void* vctx) {
    probe_ctx_t* pc = (probe_ctx_t*)vctx;
    probe_state_t* st = pc->state;
    const glyph_bucket_table_t* bt = pc->table;
    uint8_t cur_r = (uint8_t)pc->current_radius;

    st->n_probes++;
    uint32_t key = glyph_sig_to_key_u32(probe_sig);
    int lb = glyph_bucket_lower_bound(bt, key);
    if (lb >= bt->n_entries || bt->entries[lb].key != key) return 0;

    for (int i = lb; i < bt->n_entries && bt->entries[i].key == key; i++) {
        int idx = bt->entries[i].proto_idx;
        if (st->votes[idx] == 0) {
            if (st->n_hit >= st->max_union) return 1;   /* cap */
            st->hit_list[st->n_hit++] = idx;
            st->min_radius[idx] = cur_r;
        } else if (cur_r < st->min_radius[idx]) {
            st->min_radius[idx] = cur_r;
        }
        st->votes[idx]++;
        st->per_table_cands++;
        if (st->n_hit >= st->max_union) return 1;
    }
    return 0;
}

static void probe_state_reset(probe_state_t* st) {
    /* Lazy-zero votes AND min_radius at once. Keeping min_radius
     * zeroed between queries isn't strictly required (the cb
     * overwrites it on votes==0) but simplifies reasoning. */
    for (int j = 0; j < st->n_hit; j++) {
        int idx = st->hit_list[j];
        st->votes[idx] = 0;
        st->min_radius[idx] = 0;
    }
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
    probe_ctx_t pc = { bt, st, 0 };
    st->per_table_cands = 0;
    for (int r = 0; r <= max_radius; r++) {
        if (st->per_table_cands >= min_cands && r > 0) break;
        pc.current_radius = r;
        glyph_multiprobe_enumerate(q_sig, n_proj, sig_bytes, r, scratch, probe_cb, &pc);
        if (st->n_hit >= st->max_union) break;
    }
}

/* Derive a seed quadruple for table m. Table 0 uses the base seed
 * verbatim; tables ≥ 1 use a fixed derivation independent of base so
 * that a default run reproduces Phase 3 exactly. Changing --base_seed
 * therefore only shifts table 0's projection; the M ≥ 1 tables are
 * determined by m alone. This matches the inlined Phase 3 sweep tool
 * byte-for-byte.
 */
static void derive_seed(uint32_t m, const uint32_t base[4], uint32_t out[4]) {
    if (m == 0) { out[0]=base[0]; out[1]=base[1]; out[2]=base[2]; out[3]=base[3]; return; }
    out[0] = 2654435761u * m + 1013904223u;
    out[1] = 1597334677u * m + 2246822519u;
    out[2] = 3266489917u * m +  668265263u;
    out[3] =  374761393u * m + 3266489917u;
}

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    /* Load dataset. */
    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) {
        fprintf(stderr, "failed to load MNIST-format dataset from %s\n", cfg.data_dir);
        return 1;
    }
    if (!cfg.no_deskew) {
        glyph_dataset_deskew(&ds);
    }
    if (cfg.normalize) {
        glyph_dataset_normalize(&ds);
    }

    const int mixed_mode = (strcmp(cfg.density_schedule, "mixed") == 0);

    printf("mnist_routed_bucket_multi (libglyph CLI consumer)\n");
    printf("  data_dir=%s\n", cfg.data_dir);
    printf("  n_proj=%d  density=%.2f  m_max=%d  max_radius=%d  min_cands=%d  max_union=%d\n",
           cfg.n_proj, cfg.density, cfg.m_max, cfg.max_radius, cfg.min_cands, cfg.max_union);
    printf("  base_seed=%u,%u,%u,%u  mode=%s  deskew=%s  resolver_sum=%s\n",
           cfg.base_seed[0], cfg.base_seed[1], cfg.base_seed[2], cfg.base_seed[3],
           cfg.mode, cfg.no_deskew ? "off" : "on", cfg.resolver_sum);
    if (mixed_mode) {
        printf("  density_schedule=mixed  density_triple=%.2f,%.2f,%.2f\n",
               cfg.density_triple[0], cfg.density_triple[1], cfg.density_triple[2]);
    } else {
        printf("  density_schedule=fixed\n");
    }
    printf("  n_train=%d  n_test=%d  input_dim=%d\n\n",
           ds.n_train, ds.n_test, ds.input_dim);

    int full_mode = (strcmp(cfg.mode, "full") == 0);
    int sig_bytes = M4T_TRIT_PACKED_BYTES(cfg.n_proj);
    if (sig_bytes != 4) {
        fprintf(stderr, "this tool currently supports N_PROJ=16 (4-byte sigs) only; "
                        "got n_proj=%d sig_bytes=%d\n", cfg.n_proj, sig_bytes);
        glyph_dataset_free(&ds);
        return 1;
    }

    /* Calibration subset: first 1000 training vectors. */
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    /* Build M signature builders, one per table. */
    glyph_sig_builder_t* builders = calloc((size_t)cfg.m_max, sizeof(glyph_sig_builder_t));
    uint8_t** train_sigs = calloc((size_t)cfg.m_max, sizeof(uint8_t*));
    uint8_t** test_sigs  = calloc((size_t)cfg.m_max, sizeof(uint8_t*));
    glyph_bucket_table_t* tables = calloc((size_t)cfg.m_max, sizeof(glyph_bucket_table_t));

    clock_t t_build = clock();
    for (int m = 0; m < cfg.m_max; m++) {
        uint32_t seeds[4];
        derive_seed((uint32_t)m, cfg.base_seed, seeds);
        double table_density = mixed_mode ? cfg.density_triple[m % 3] : cfg.density;
        if (glyph_sig_builder_init(&builders[m], cfg.n_proj, ds.input_dim, table_density,
                                    seeds[0], seeds[1], seeds[2], seeds[3],
                                    ds.x_train, n_calib) != 0) {
            fprintf(stderr, "sig builder init failed for table %d\n", m);
            return 1;
        }
        train_sigs[m] = calloc((size_t)ds.n_train * sig_bytes, 1);
        test_sigs[m]  = calloc((size_t)ds.n_test  * sig_bytes, 1);
        glyph_sig_encode_batch(&builders[m], ds.x_train, ds.n_train, train_sigs[m]);
        glyph_sig_encode_batch(&builders[m], ds.x_test,  ds.n_test,  test_sigs[m]);
        if (glyph_bucket_build(&tables[m], train_sigs[m], ds.n_train, sig_bytes) != 0) {
            fprintf(stderr, "bucket build failed for table %d\n", m);
            return 1;
        }
    }
    /* Re-rank pass: build M wider signature encoders at N_PROJ=32.
     * These are used only for re-scoring the Stage-1 union — no
     * bucket index, no probing. Only built in full mode. */
    const int rr_n_proj = 32;
    const int rr_sig_bytes = M4T_TRIT_PACKED_BYTES(rr_n_proj);  /* 8 */
    const int M_rr = cfg.m_max;
    glyph_sig_builder_t* rr_builders = NULL;
    uint8_t** rr_train_sigs = NULL;

    if (full_mode) {
        rr_builders = calloc((size_t)M_rr, sizeof(glyph_sig_builder_t));
        rr_train_sigs = calloc((size_t)M_rr, sizeof(uint8_t*));
        for (int m = 0; m < M_rr; m++) {
            uint32_t seeds[4];
            derive_seed((uint32_t)m, cfg.base_seed, seeds);
            double td = mixed_mode ? cfg.density_triple[m % 3] : cfg.density;
            if (glyph_sig_builder_init(&rr_builders[m], rr_n_proj, ds.input_dim, td,
                                        seeds[0], seeds[1], seeds[2], seeds[3],
                                        ds.x_train, n_calib) != 0) {
                fprintf(stderr, "re-rank builder init failed for table %d\n", m);
                return 1;
            }
            rr_train_sigs[m] = calloc((size_t)ds.n_train * rr_sig_bytes, 1);
            glyph_sig_encode_batch(&rr_builders[m], ds.x_train, ds.n_train, rr_train_sigs[m]);
        }
    }

    double t_build_sec = (double)(clock() - t_build) / CLOCKS_PER_SEC;
    printf("Built %d tables", cfg.m_max);
    if (full_mode) printf(" + %d re-rank encoders (N_PROJ=%d)", M_rr, rr_n_proj);
    printf(" in %.2fs.\n", t_build_sec);

    /* Sanity: distinct buckets on table 0 (matches 37906 for canonical seed). */
    if (cfg.verbose) {
        printf("Table 0 distinct buckets: %d\n",
               glyph_bucket_count_distinct(&tables[0]));
    }
    printf("\n");

    /* M sweep setup. */
    int full_m_sweep[] = {1, 2, 4, 8, 16, 32, 64};
    int n_full = (int)(sizeof(full_m_sweep)/sizeof(full_m_sweep[0]));
    int m_values_buf[8];
    int* m_values;
    int n_M;
    if (cfg.single_m > 0) {
        m_values_buf[0] = cfg.single_m;
        m_values = m_values_buf;
        n_M = 1;
    } else {
        /* Clamp to cfg.m_max. */
        int k = 0;
        for (int i = 0; i < n_full; i++) {
            if (full_m_sweep[i] <= cfg.m_max) m_values_buf[k++] = full_m_sweep[i];
        }
        m_values = m_values_buf;
        n_M = k;
    }

    /* Per-query state (reused). */
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.min_radius = calloc((size_t)ds.n_train, sizeof(uint8_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    st.n_hit = 0;

    /* Scratch buffer for ternary multi-probe neighbor enumeration.
     * sig_bytes is hard-enforced to 4 above (the bucket index currently
     * supports only uint32 keys); enlarge this when the library gains
     * uint64 keys for the fused-filter variant. */
    uint8_t scratch[4];
    uint8_t* mask = malloc(sig_bytes);
    memset(mask, 0xFF, sig_bytes);

    /* Per-M metrics. */
    int* oracle_correct = calloc((size_t)n_M, sizeof(int));
    int* vote_correct   = calloc((size_t)n_M, sizeof(int));
    int* sum_correct    = calloc((size_t)n_M, sizeof(int));
    int* rr_correct     = calloc((size_t)n_M, sizeof(int));
    int* ptm_correct    = calloc((size_t)n_M, sizeof(int));
    long* total_union   = calloc((size_t)n_M, sizeof(long));
    long* total_probes  = calloc((size_t)n_M, sizeof(long));

    /* Per-class confusion at the FINAL M checkpoint under the SUM
     * resolver (whichever variant is selected). Used to diagnose
     * whether the resolver gap is concentrated in specific class
     * pairs. Tracked only at the last M to keep reporting compact. */
    int  per_class_total[N_CLASSES]  = {0};
    int  per_class_correct[N_CLASSES] = {0};
    int  final_confusion[N_CLASSES][N_CLASSES] = {{0}};
    int  final_M = m_values[n_M - 1];

    /* Per-query array of query-sig pointers (one per table). Reused
     * across queries; only the pointer targets change. train_sigs
     * itself is already a uint8_t**, so we pass it directly to the
     * resolvers — no redundant parallel copy. */
    const uint8_t** q_sigs_p = calloc((size_t)cfg.m_max, sizeof(uint8_t*));

    /* Re-rank query-sig scratch: encode one query at N_PROJ=32 on the fly. */
    uint8_t** rr_q_bufs = NULL;
    const uint8_t** rr_q_ptrs = NULL;
    uint8_t* rr_mask = NULL;
    if (full_mode) {
        rr_q_bufs = calloc((size_t)M_rr, sizeof(uint8_t*));
        rr_q_ptrs = calloc((size_t)M_rr, sizeof(const uint8_t*));
        for (int m = 0; m < M_rr; m++) {
            rr_q_bufs[m] = malloc(rr_sig_bytes);
            rr_q_ptrs[m] = rr_q_bufs[m];
        }
        rr_mask = malloc(rr_sig_bytes);
        memset(rr_mask, 0xFF, rr_sig_bytes);
    }

    glyph_union_t u = {0};
    u.y_train = ds.y_train;
    u.n_classes = N_CLASSES;

    clock_t t_sweep = clock();
    for (int s = 0; s < ds.n_test; s++) {
        int y = ds.y_test[s];
        /* Defensive bound on the per-class counter — the tool is
         * hardcoded for N_CLASSES=10 but a future caller pointing
         * --data at a larger-class dataset (e.g. EMNIST with 47
         * classes) would OOB without this guard. */
        if (y >= 0 && y < N_CLASSES) per_class_total[y]++;
        for (int m = 0; m < cfg.m_max; m++)
            q_sigs_p[m] = test_sigs[m] + (size_t)s * sig_bytes;

        probe_state_reset(&st);

        int prev_M = 0;
        for (int mi = 0; mi < n_M; mi++) {
            int M_target = m_values[mi];
            for (int m = prev_M; m < M_target; m++) {
                probe_table(&tables[m], q_sigs_p[m], cfg.n_proj, sig_bytes,
                            cfg.max_radius, cfg.min_cands, &st, scratch);
            }

            /* Oracle check. */
            for (int j = 0; j < st.n_hit; j++) {
                if (ds.y_train[st.hit_list[j]] == y) { oracle_correct[mi]++; break; }
            }
            total_union[mi]  += st.n_hit;
            total_probes[mi] += st.n_probes;

            if (full_mode) {
                u.hit_list = st.hit_list;
                u.n_hit    = st.n_hit;
                u.votes    = st.votes;

                int pred_v = glyph_resolver_vote(&u);
                if (pred_v == y) vote_correct[mi]++;

                int pred_s;
                if (strcmp(cfg.resolver_sum, "neon4") == 0 && sig_bytes == 4) {
                    pred_s = glyph_resolver_sum_neon4(&u, M_target, sig_bytes,
                                                      train_sigs, q_sigs_p, mask);
                } else if (strcmp(cfg.resolver_sum, "voteweighted") == 0) {
                    pred_s = glyph_resolver_sum_voteweighted(&u, M_target, sig_bytes,
                                                              train_sigs, q_sigs_p, mask);
                } else if (strcmp(cfg.resolver_sum, "radiusaware") == 0) {
                    pred_s = glyph_resolver_sum_radiusaware(&u, M_target, sig_bytes,
                                                             train_sigs, q_sigs_p, mask,
                                                             st.min_radius, cfg.radius_lambda);
                } else if (strcmp(cfg.resolver_sum, "knn") == 0) {
                    pred_s = glyph_resolver_sum_knn(&u, M_target, sig_bytes,
                                                     train_sigs, q_sigs_p, mask,
                                                     cfg.knn_k);
                } else {
                    pred_s = glyph_resolver_sum(&u, M_target, sig_bytes,
                                                train_sigs, q_sigs_p, mask);
                }
                if (pred_s == y) sum_correct[mi]++;

                /* Re-rank: re-score the Stage-1 union at N_PROJ=32. */
                {
                    const m4t_mtfp_t* qvec = ds.x_test + (size_t)s * ds.input_dim;
                    for (int m = 0; m < M_rr; m++)
                        glyph_sig_encode(&rr_builders[m], qvec, rr_q_bufs[m]);
                    int pred_rr;
                    if (strcmp(cfg.resolver_sum, "knn") == 0) {
                        pred_rr = glyph_resolver_sum_knn(
                            &u, M_rr, rr_sig_bytes,
                            rr_train_sigs, rr_q_ptrs, rr_mask, cfg.knn_k);
                    } else {
                        pred_rr = glyph_resolver_sum(
                            &u, M_rr, rr_sig_bytes,
                            rr_train_sigs, rr_q_ptrs, rr_mask);
                    }
                    if (pred_rr == y) rr_correct[mi]++;
                }

                /* Record per-class and confusion at the last M value.
                 * Guards match the per_class_total bounds check. */
                if (M_target == final_M && y >= 0 && y < N_CLASSES) {
                    if (pred_s == y) per_class_correct[y]++;
                    if (pred_s >= 0 && pred_s < N_CLASSES)
                        final_confusion[y][pred_s]++;
                }

                int pred_p = glyph_resolver_per_table_majority(&u, M_target, sig_bytes,
                                                train_sigs, q_sigs_p, mask);
                if (pred_p == y) ptm_correct[mi]++;
            }

            prev_M = M_target;
        }
    }
    double t_sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    /* Report. */
    printf("Total wall: %.2fs for %d queries (mode=%s).\n\n",
           t_sweep_sec, ds.n_test, cfg.mode);

    printf("Oracle ceiling:\n");
    printf("   M    oracle     avg_union  avg_probes\n");
    for (int mi = 0; mi < n_M; mi++) {
        printf("  %3d  %6.2f%%    %8.1f  %10.1f\n",
               m_values[mi],
               100.0 * oracle_correct[mi] / ds.n_test,
               (double)total_union[mi] / ds.n_test,
               (double)total_probes[mi] / ds.n_test);
    }
    printf("\n");

    if (full_mode) {
        printf("Resolver sweep:\n");
        printf("   M      VOTE    SUM_16   SUM_32_RR    PTM      oracle\n");
        for (int mi = 0; mi < n_M; mi++) {
            printf("  %3d   %6.2f%%  %6.2f%%   %6.2f%%  %6.2f%%   %6.2f%%\n",
                   m_values[mi],
                   100.0 * vote_correct[mi] / ds.n_test,
                   100.0 * sum_correct[mi]  / ds.n_test,
                   100.0 * rr_correct[mi]   / ds.n_test,
                   100.0 * ptm_correct[mi]  / ds.n_test,
                   100.0 * oracle_correct[mi] / ds.n_test);
        }
        printf("\n");

        /* Per-class accuracy at the final M checkpoint under the SUM
         * resolver. Diagnostic for whether the resolver gap is
         * concentrated in specific classes or uniform. */
        printf("Per-class SUM accuracy at M=%d (resolver_sum=%s):\n",
               final_M, cfg.resolver_sum);
        printf("  class   count   correct   accuracy\n");
        for (int c = 0; c < N_CLASSES; c++) {
            if (per_class_total[c] == 0) continue;
            printf("   %2d    %5d   %5d     %6.2f%%\n",
                   c, per_class_total[c], per_class_correct[c],
                   100.0 * per_class_correct[c] / per_class_total[c]);
        }
        printf("\n");

        /* Top confusion pairs (true class → predicted class, excluding
         * the diagonal). Shows which class pairs dominate the error
         * budget. */
        printf("Top off-diagonal confusions at M=%d (true → pred):\n", final_M);
        typedef struct { int t, p, n; } cf_t;
        cf_t pairs[N_CLASSES * N_CLASSES];
        int n_pairs = 0;
        for (int t = 0; t < N_CLASSES; t++)
            for (int p = 0; p < N_CLASSES; p++) {
                if (t == p) continue;
                if (final_confusion[t][p] > 0) {
                    pairs[n_pairs].t = t;
                    pairs[n_pairs].p = p;
                    pairs[n_pairs].n = final_confusion[t][p];
                    n_pairs++;
                }
            }
        /* Sort descending by count (insertion sort; n_pairs is small). */
        for (int i = 1; i < n_pairs; i++) {
            cf_t v = pairs[i];
            int j = i - 1;
            while (j >= 0 && pairs[j].n < v.n) { pairs[j+1] = pairs[j]; j--; }
            pairs[j+1] = v;
        }
        int shown = (n_pairs < 10) ? n_pairs : 10;
        printf("  true  pred  count\n");
        for (int i = 0; i < shown; i++) {
            printf("   %2d    %2d   %5d\n",
                   pairs[i].t, pairs[i].p, pairs[i].n);
        }
        printf("\n");
    }

    /* Cleanup. */
    free(mask);
    free(st.votes); free(st.min_radius); free(st.hit_list);
    free(oracle_correct); free(vote_correct); free(sum_correct);
    free(rr_correct); free(ptm_correct);
    free(total_union); free(total_probes);
    free(q_sigs_p);

    if (rr_q_bufs) {
        for (int m = 0; m < M_rr; m++) free(rr_q_bufs[m]);
        free(rr_q_bufs); free(rr_q_ptrs); free(rr_mask);
    }
    if (rr_builders) {
        for (int m = 0; m < M_rr; m++) {
            glyph_sig_builder_free(&rr_builders[m]);
            free(rr_train_sigs[m]);
        }
        free(rr_builders); free(rr_train_sigs);
    }

    for (int m = 0; m < cfg.m_max; m++) {
        glyph_sig_builder_free(&builders[m]);
        glyph_bucket_table_free(&tables[m]);
        free(train_sigs[m]);
        free(test_sigs[m]);
    }
    free(builders); free(train_sigs); free(test_sigs); free(tables);

    glyph_dataset_free(&ds);
    return 0;
}
