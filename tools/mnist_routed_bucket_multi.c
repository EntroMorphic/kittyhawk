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
    int32_t*  hit_list;     /* [max_union] proto indices in current union */
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
            if (st->n_hit >= st->max_union) return 1;   /* cap */
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
    if (glyph_dataset_load_mnist(&ds, cfg.data_dir) != 0) {
        fprintf(stderr, "failed to load MNIST from %s\n", cfg.data_dir);
        return 1;
    }
    glyph_dataset_deskew(&ds);

    printf("mnist_routed_bucket_multi (libglyph CLI consumer)\n");
    printf("  data_dir=%s\n", cfg.data_dir);
    printf("  n_proj=%d  density=%.2f  m_max=%d  max_radius=%d  min_cands=%d  max_union=%d\n",
           cfg.n_proj, cfg.density, cfg.m_max, cfg.max_radius, cfg.min_cands, cfg.max_union);
    printf("  base_seed=%u,%u,%u,%u  mode=%s\n",
           cfg.base_seed[0], cfg.base_seed[1], cfg.base_seed[2], cfg.base_seed[3], cfg.mode);
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
        if (glyph_sig_builder_init(&builders[m], cfg.n_proj, ds.input_dim, cfg.density,
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
    double t_build_sec = (double)(clock() - t_build) / CLOCKS_PER_SEC;
    printf("Built %d tables in %.2fs.\n", cfg.m_max, t_build_sec);

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
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    st.n_hit = 0;

    uint8_t scratch[8];          /* sig_bytes<=8 for foreseeable N_PROJ */
    uint8_t* mask = malloc(sig_bytes);
    memset(mask, 0xFF, sig_bytes);

    /* Per-M metrics. */
    int* oracle_correct = calloc((size_t)n_M, sizeof(int));
    int* vote_correct   = calloc((size_t)n_M, sizeof(int));
    int* sum_correct    = calloc((size_t)n_M, sizeof(int));
    int* ptm_correct    = calloc((size_t)n_M, sizeof(int));
    long* total_union   = calloc((size_t)n_M, sizeof(long));
    long* total_probes  = calloc((size_t)n_M, sizeof(long));

    /* Per-query: fixed-size pointer arrays for table sigs and query sigs. */
    uint8_t**       train_sigs_p = calloc((size_t)cfg.m_max, sizeof(uint8_t*));
    const uint8_t** q_sigs_p     = calloc((size_t)cfg.m_max, sizeof(uint8_t*));
    for (int m = 0; m < cfg.m_max; m++) train_sigs_p[m] = train_sigs[m];

    glyph_union_t u = {0};
    u.y_train = ds.y_train;
    u.n_classes = N_CLASSES;

    clock_t t_sweep = clock();
    for (int s = 0; s < ds.n_test; s++) {
        int y = ds.y_test[s];
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

                int pred_s = glyph_resolver_sum(&u, M_target, sig_bytes,
                                                train_sigs_p, q_sigs_p, mask);
                if (pred_s == y) sum_correct[mi]++;

                int pred_p = glyph_resolver_per_table_majority(&u, M_target, sig_bytes,
                                                train_sigs_p, q_sigs_p, mask);
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
        printf("   M      VOTE      SUM       PTM      oracle\n");
        for (int mi = 0; mi < n_M; mi++) {
            printf("  %3d   %6.2f%%  %6.2f%%  %6.2f%%   %6.2f%%\n",
                   m_values[mi],
                   100.0 * vote_correct[mi] / ds.n_test,
                   100.0 * sum_correct[mi]  / ds.n_test,
                   100.0 * ptm_correct[mi]  / ds.n_test,
                   100.0 * oracle_correct[mi] / ds.n_test);
        }
        printf("\n");
    }

    /* Cleanup. */
    free(mask);
    free(st.votes); free(st.hit_list);
    free(oracle_correct); free(vote_correct); free(sum_correct); free(ptm_correct);
    free(total_union); free(total_probes);
    free(train_sigs_p); free(q_sigs_p);

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
