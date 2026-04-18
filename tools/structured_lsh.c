/*
 * structured_lsh.c — Trit Lattice LSH with structured spatial projections.
 *
 * Instead of random ternary projection weights, each projection
 * direction is a spatial feature: horizontal gradient, vertical
 * gradient, or intensity. All are ternary by construction:
 *   gradient: [-1, +1] on adjacent pixels, 0 elsewhere
 *   intensity: [+1] on one pixel, 0 elsewhere
 *
 * Plugs directly into the existing pipeline: m4t_ternary_matmul,
 * packed-trit signatures, bucket index, multi-probe, k-NN resolve.
 * Only the projection matrix contents change.
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

/* Generate structured spatial projection directions for a
 * width × height × n_channels image. Directions are stored
 * as dense int8 arrays of length input_dim.
 *
 * Three types of directions:
 *   H-gradient: +1 at (x+1,y,c), -1 at (x,y,c)
 *   V-gradient: +1 at (x,y+1,c), -1 at (x,y,c)
 *   Intensity:  +1 at (x,y,c)
 *
 * Layout assumption: pixels stored as [C0 row-major, C1 row-major, ...]
 * i.e., R[0..W*H-1], G[W*H..2*W*H-1], B[2*W*H..3*W*H-1].
 *
 * Returns the number of directions generated. Caller provides
 * dirs[max_dirs][input_dim].
 */
static int generate_structured_directions(
    int8_t** dirs, int max_dirs, int width, int height, int n_channels, int input_dim)
{
    int n = 0;
    int ppc = width * height;  /* pixels per channel */

    /* Horizontal gradients: (x+1,y) - (x,y) per channel. */
    for (int ch = 0; ch < n_channels && n < max_dirs; ch++) {
        for (int y = 0; y < height && n < max_dirs; y++) {
            for (int x = 0; x < width - 1 && n < max_dirs; x++) {
                memset(dirs[n], 0, input_dim);
                dirs[n][ch * ppc + y * width + x]     = -1;
                dirs[n][ch * ppc + y * width + x + 1]  = +1;
                n++;
            }
        }
    }

    /* Vertical gradients: (x,y+1) - (x,y) per channel. */
    for (int ch = 0; ch < n_channels && n < max_dirs; ch++) {
        for (int y = 0; y < height - 1 && n < max_dirs; y++) {
            for (int x = 0; x < width && n < max_dirs; x++) {
                memset(dirs[n], 0, input_dim);
                dirs[n][ch * ppc + y * width + x]           = -1;
                dirs[n][ch * ppc + (y + 1) * width + x]     = +1;
                n++;
            }
        }
    }

    /* Intensity: single pixel per channel. */
    for (int ch = 0; ch < n_channels && n < max_dirs; ch++) {
        for (int y = 0; y < height && n < max_dirs; y++) {
            for (int x = 0; x < width && n < max_dirs; x++) {
                memset(dirs[n], 0, input_dim);
                dirs[n][ch * ppc + y * width + x] = +1;
                n++;
            }
        }
    }

    return n;
}

/* Encode a single vector using a structured projection matrix.
 * proj_dirs[n_proj] are dense int8 direction arrays.
 * Computes dot product, thresholds, packs to trits. */
static void structured_encode(
    const int8_t** proj_dirs, int n_proj, int input_dim,
    int64_t tau, const m4t_mtfp_t* x, uint8_t* out_sig, int sig_bytes)
{
    memset(out_sig, 0, sig_bytes);
    for (int p = 0; p < n_proj; p++) {
        int64_t dot = 0;
        for (int d = 0; d < input_dim; d++)
            dot += (int64_t)proj_dirs[p][d] * (int64_t)x[d];
        int8_t trit = 0;
        if (dot > tau) trit = +1;
        else if (dot < -tau) trit = -1;
        glyph_write_trit(out_sig, p, trit);
    }
}


int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);

    /* Generate ALL structured directions. */
    int max_dirs = n_ch * (img_w - 1) * img_h   /* H-grad */
                 + n_ch * img_w * (img_h - 1)   /* V-grad */
                 + n_ch * img_w * img_h;         /* intensity */
    int8_t** all_dirs = malloc((size_t)max_dirs * sizeof(int8_t*));
    for (int i = 0; i < max_dirs; i++)
        all_dirs[i] = calloc(ds.input_dim, 1);
    int n_dirs = generate_structured_directions(
        all_dirs, max_dirs, img_w, img_h, n_ch, ds.input_dim);

    printf("structured_lsh: spatial gradient + intensity projections\n");
    printf("  data=%s  deskew=%s  density=%.2f\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on", cfg.density);
    printf("  image: %dx%dx%d = %d dims\n", img_w, img_h, n_ch, ds.input_dim);
    printf("  structured directions: %d (H-grad + V-grad + intensity)\n", n_dirs);

    const int M = cfg.m_max;
    const int N_PROJ = 16;
    const int SB = M4T_TRIT_PACKED_BYTES(N_PROJ);

    printf("  M=%d tables, N_PROJ=%d per table (%d total)\n",
           M, N_PROJ, M * N_PROJ);
    printf("  n_train=%d  n_test=%d  knn_k=%d\n\n",
           ds.n_train, ds.n_test, KNN_K);

    if (M * N_PROJ > n_dirs) {
        printf("  WARNING: M*N_PROJ=%d > n_dirs=%d; directions will repeat.\n\n",
               M * N_PROJ, n_dirs);
    }

    /* Assign directions to tables. Shuffle all directions, then
     * assign sequential blocks of N_PROJ to each table. */
    clock_t t0 = clock();
    glyph_rng_t rng;
    glyph_rng_seed(&rng, cfg.base_seed[0], cfg.base_seed[1],
                    cfg.base_seed[2], cfg.base_seed[3]);
    int* dir_order = malloc((size_t)n_dirs * sizeof(int));
    for (int i = 0; i < n_dirs; i++) dir_order[i] = i;
    for (int i = n_dirs - 1; i > 0; i--) {
        int j = (int)(glyph_rng_next(&rng) % (uint32_t)(i + 1));
        int tmp = dir_order[i]; dir_order[i] = dir_order[j]; dir_order[j] = tmp;
    }

    /* Build tables with structured projections. */
    const int8_t*** table_dirs = malloc((size_t)M * sizeof(const int8_t**));
    int64_t* table_tau = malloc((size_t)M * sizeof(int64_t));
    uint8_t** train_sigs = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** test_sigs  = calloc((size_t)M, sizeof(uint8_t*));
    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));
    for (int m = 0; m < M; m++) {
        table_dirs[m] = malloc((size_t)N_PROJ * sizeof(const int8_t*));
        for (int p = 0; p < N_PROJ; p++) {
            int di = (m * N_PROJ + p) % n_dirs;
            table_dirs[m][p] = all_dirs[dir_order[di]];
        }
        /* For structured 2-weight directions, tau should be small
         * enough that real gradients produce ±1 but large enough
         * that noise (tiny differences) maps to 0 (hidden).
         * MTFP scale: pixel value 0-255 maps to 0-59049. A
         * gradient of ~10 pixel levels ≈ 10*59049/255 ≈ 2316.
         * Use that as tau so gradients below ~10/255 are hidden. */
        table_tau[m] = (int64_t)(10.0 * M4T_MTFP_SCALE / 255.0);

        train_sigs[m] = calloc((size_t)ds.n_train * SB, 1);
        test_sigs[m]  = calloc((size_t)ds.n_test  * SB, 1);
        for (int i = 0; i < ds.n_train; i++)
            structured_encode(table_dirs[m], N_PROJ, ds.input_dim, table_tau[m],
                              ds.x_train + (size_t)i * ds.input_dim,
                              train_sigs[m] + (size_t)i * SB, SB);
        for (int i = 0; i < ds.n_test; i++)
            structured_encode(table_dirs[m], N_PROJ, ds.input_dim, table_tau[m],
                              ds.x_test + (size_t)i * ds.input_dim,
                              test_sigs[m] + (size_t)i * SB, SB);
        glyph_bucket_build(&tables[m], train_sigs[m], ds.n_train, SB);
    }
    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Built %d structured tables in %.1fs.\n", M, build_sec);
    if (cfg.verbose)
        printf("Table 0 distinct buckets: %d\n", glyph_bucket_count_distinct(&tables[0]));
    printf("\n");

    /* Sweep M values. */
    int m_sweep[] = {1, 2, 4, 8, 16, 32, 64};
    int n_sweep = 0;
    for (int i = 0; i < 7; i++) if (m_sweep[i] <= M) n_sweep = i + 1;

    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union; st.n_hit = 0;
    uint8_t scratch[4];
    uint8_t mask[4]; memset(mask, 0xFF, SB);
    const uint8_t** q_ptrs = calloc((size_t)M, sizeof(uint8_t*));

    glyph_union_t u = {0};
    u.y_train = ds.y_train; u.n_classes = N_CLASSES;

    int oracle_c[7] = {0}, sum_c[7] = {0}, knn_c[7] = {0};

    clock_t t_sweep = clock();
    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        for (int m = 0; m < M; m++)
            q_ptrs[m] = test_sigs[m] + (size_t)qi * SB;
        probe_state_reset(&st);

        int prev = 0;
        for (int si = 0; si < n_sweep; si++) {
            int Mt = m_sweep[si];
            for (int m = prev; m < Mt; m++)
                probe_table(&tables[m], q_ptrs[m], N_PROJ, SB,
                            cfg.max_radius, cfg.min_cands, &st, scratch);
            for (int j = 0; j < st.n_hit; j++)
                if (ds.y_train[st.hit_list[j]] == y) { oracle_c[si]++; break; }
            u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;
            int ps = glyph_resolver_sum(&u, Mt, SB, train_sigs, q_ptrs, mask);
            if (ps == y) sum_c[si]++;
            int pk = glyph_resolver_sum_knn(&u, Mt, SB, train_sigs, q_ptrs, mask, KNN_K);
            if (pk == y) knn_c[si]++;
            prev = Mt;
        }
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("Sweep: %.1fs\n\n", sweep_sec);
    printf("   M    oracle    SUM_1NN    k=%d-NN\n", KNN_K);
    for (int si = 0; si < n_sweep; si++) {
        printf("  %3d   %6.2f%%   %6.2f%%   %6.2f%%\n",
               m_sweep[si],
               100.0 * oracle_c[si] / ds.n_test,
               100.0 * sum_c[si] / ds.n_test,
               100.0 * knn_c[si] / ds.n_test);
    }
    printf("\n");

    /* Per-class at max M. */
    int pc_total[N_CLASSES] = {0}, pc_correct[N_CLASSES] = {0};
    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        if (y < 0 || y >= N_CLASSES) continue;
        pc_total[y]++;
        for (int m = 0; m < M; m++)
            q_ptrs[m] = test_sigs[m] + (size_t)qi * SB;
        probe_state_reset(&st);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], q_ptrs[m], N_PROJ, SB,
                        cfg.max_radius, cfg.min_cands, &st, scratch);
        u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;
        int pk = glyph_resolver_sum_knn(&u, M, SB, train_sigs, q_ptrs, mask, KNN_K);
        if (pk == y) pc_correct[y]++;
    }
    printf("Per-class k=%d accuracy at M=%d:\n", KNN_K, M);
    printf("  class   count   correct   accuracy\n");
    for (int c = 0; c < N_CLASSES; c++) {
        if (pc_total[c] == 0) continue;
        printf("   %2d    %5d   %5d     %6.2f%%\n",
               c, pc_total[c], pc_correct[c],
               100.0 * pc_correct[c] / pc_total[c]);
    }

    /* Cleanup. */
    free(st.votes); free(st.hit_list); free(q_ptrs);
    for (int m = 0; m < M; m++) {
        free((void*)table_dirs[m]);
        glyph_bucket_table_free(&tables[m]);
        free(train_sigs[m]); free(test_sigs[m]);
    }
    free(table_dirs); free(table_tau); free(train_sigs); free(test_sigs); free(tables);
    free(dir_order);
    for (int i = 0; i < max_dirs; i++) free(all_dirs[i]);
    free(all_dirs);
    glyph_dataset_free(&ds);
    return 0;
}
