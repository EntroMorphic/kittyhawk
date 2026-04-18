/*
 * centroid_routed.c — class-centroid-relative Trit Lattice LSH.
 *
 * Instead of projecting raw pixels, projects (query - centroid_c)
 * for each class c. The difference removes common pixel structure
 * and amplifies class-distinctive features. Class centroids are
 * computed from training data via MTFP integer arithmetic.
 *
 * The projection and everything downstream (packing, bucket index,
 * multi-probe, k-NN resolve) is standard Trit Lattice LSH. Only
 * the INPUT to the projection changes: differences instead of
 * absolutes.
 *
 * Fully routing-native: centroid computation is integer sums,
 * differencing is MTFP subtraction, projection is ternary matmul.
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
#define KNN_K 5

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

static void probe_table_fn(const glyph_bucket_table_t* bt, const uint8_t* q_sig,
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

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    const int M = cfg.m_max;
    const int N_PROJ = 16;
    const int SB = M4T_TRIT_PACKED_BYTES(N_PROJ);
    const int D = ds.input_dim;
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    printf("centroid_routed: class-centroid-relative Trit Lattice LSH\n");
    printf("  data=%s  deskew=%s  density=%.2f  M=%d  N_PROJ=%d\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on", cfg.density, M, N_PROJ);
    printf("  n_train=%d  n_test=%d  input_dim=%d  knn_k=%d\n\n",
           ds.n_train, ds.n_test, D, KNN_K);

    /* ============================================================
     * Compute class centroids from training data (integer mean).
     * ============================================================ */
    printf("Computing class centroids...\n");
    int64_t* class_sums = calloc((size_t)N_CLASSES * D, sizeof(int64_t));
    int class_counts[N_CLASSES] = {0};
    for (int i = 0; i < ds.n_train; i++) {
        int c = ds.y_train[i];
        if (c < 0 || c >= N_CLASSES) continue;
        class_counts[c]++;
        for (int d = 0; d < D; d++)
            class_sums[(size_t)c * D + d] += ds.x_train[(size_t)i * D + d];
    }
    m4t_mtfp_t* centroids = malloc((size_t)N_CLASSES * D * sizeof(m4t_mtfp_t));
    for (int c = 0; c < N_CLASSES; c++) {
        for (int d = 0; d < D; d++) {
            centroids[(size_t)c * D + d] = (class_counts[c] > 0)
                ? (m4t_mtfp_t)(class_sums[(size_t)c * D + d] / class_counts[c])
                : 0;
        }
    }
    free(class_sums);

    /* ============================================================
     * Transform training data: compute (x - centroid_y) for each
     * training image, where y is its label. This is the centroid-
     * relative representation.
     * ============================================================ */
    printf("Computing centroid-relative training data...\n");
    clock_t t0 = clock();
    m4t_mtfp_t* train_rel = malloc((size_t)ds.n_train * D * sizeof(m4t_mtfp_t));
    for (int i = 0; i < ds.n_train; i++) {
        int c = ds.y_train[i];
        for (int d = 0; d < D; d++)
            train_rel[(size_t)i * D + d] =
                ds.x_train[(size_t)i * D + d] - centroids[(size_t)c * D + d];
    }

    /* ============================================================
     * Build LSH on centroid-relative data.
     * ============================================================ */
    printf("Building LSH on centroid-relative data...\n");
    glyph_sig_builder_t* builders = calloc((size_t)M, sizeof(glyph_sig_builder_t));
    uint8_t** train_sigs = calloc((size_t)M, sizeof(uint8_t*));
    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));

    for (int m = 0; m < M; m++) {
        uint32_t seeds[4]; derive_seed((uint32_t)m, cfg.base_seed, seeds);
        glyph_sig_builder_init(&builders[m], N_PROJ, D, cfg.density,
                                seeds[0], seeds[1], seeds[2], seeds[3],
                                train_rel, n_calib);
        train_sigs[m] = calloc((size_t)ds.n_train * SB, 1);
        glyph_sig_encode_batch(&builders[m], train_rel, ds.n_train, train_sigs[m]);
        glyph_bucket_build(&tables[m], train_sigs[m], ds.n_train, SB);
    }
    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Built %d tables in %.1fs.\n\n", M, build_sec);

    /* ============================================================
     * Classify: for each test query, try each class centroid.
     * Compute (query - centroid_c), project, probe, score the
     * union. The class c whose centroid-relative representation
     * finds the nearest neighbor wins.
     * ============================================================ */
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union; st.n_hit = 0;
    uint8_t scratch[4];
    uint8_t mask[4]; memset(mask, 0xFF, SB);
    const uint8_t** q_ptrs = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** q_sigs = calloc((size_t)M, sizeof(uint8_t*));
    for (int m = 0; m < M; m++) q_sigs[m] = malloc(SB);
    m4t_mtfp_t* q_rel = malloc((size_t)D * sizeof(m4t_mtfp_t));

    glyph_union_t u = {0};
    u.y_train = ds.y_train; u.n_classes = N_CLASSES;

    int correct_best = 0;

    printf("Classifying %d test queries (%d classes × probe)...\n", ds.n_test, N_CLASSES);
    clock_t t_sweep = clock();

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        const m4t_mtfp_t* qvec = ds.x_test + (size_t)qi * D;

        /* For each class c: compute (query - centroid_c), project,
         * probe, find the nearest neighbor and its distance. */
        int32_t best_class_dist = INT32_MAX;
        int best_class = -1;
        int class_knn_votes[N_CLASSES] = {0};

        for (int c = 0; c < N_CLASSES; c++) {
            /* Compute centroid-relative query. */
            for (int d = 0; d < D; d++)
                q_rel[d] = qvec[d] - centroids[(size_t)c * D + d];

            /* Encode through each table. */
            for (int m = 0; m < M; m++) {
                glyph_sig_encode(&builders[m], q_rel, q_sigs[m]);
                q_ptrs[m] = q_sigs[m];
            }

            /* Probe and find the nearest neighbor. */
            probe_state_reset(&st);
            for (int m = 0; m < M; m++)
                probe_table_fn(&tables[m], q_ptrs[m], N_PROJ, SB,
                            cfg.max_radius, cfg.min_cands, &st, scratch);

            if (st.n_hit == 0) continue;

            /* SUM distance to the 1-NN in the union. */
            u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;
            int32_t nn_dist = INT32_MAX;
            for (int j = 0; j < st.n_hit; j++) {
                int idx = st.hit_list[j];
                int32_t d = 0;
                for (int m = 0; m < M; m++)
                    d += m4t_popcount_dist(q_ptrs[m],
                        train_sigs[m] + (size_t)idx * SB, mask, SB);
                if (d < nn_dist) nn_dist = d;
            }

            if (nn_dist < best_class_dist) {
                best_class_dist = nn_dist;
                best_class = c;
            }
            class_knn_votes[c] = (int)(1000000 - nn_dist);
        }

        if (best_class == y) correct_best++;

        if ((qi + 1) % 500 == 0 || qi == ds.n_test - 1) {
            printf("  %d/%d  best_class=%.2f%%\n",
                   qi + 1, ds.n_test,
                   100.0 * correct_best / (qi + 1));
            fflush(stdout);
        }
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("\n=== Results ===\n");
    printf("  Centroid-relative 1-NN:  %6.2f%%\n", 100.0 * correct_best / ds.n_test);
    printf("  Sweep: %.1fs\n\n", sweep_sec);

    /* Cleanup. */
    free(q_rel);
    for (int m = 0; m < M; m++) free(q_sigs[m]);
    free(q_sigs); free(q_ptrs);
    free(st.votes); free(st.hit_list);
    free(train_rel); free(centroids);
    for (int m = 0; m < M; m++) {
        glyph_sig_builder_free(&builders[m]);
        glyph_bucket_table_free(&tables[m]);
        free(train_sigs[m]);
    }
    free(builders); free(tables); free(train_sigs);
    glyph_dataset_free(&ds);
    return 0;
}
