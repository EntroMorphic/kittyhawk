/*
 * subsetted_multi.c — dimension-subsetted multi-table routed k-NN.
 *
 * Each table projects a random subset of D input dimensions instead
 * of all input_dim dimensions. The per-table compression ratio drops
 * from 1:192 (CIFAR-10 full) to 1:16 (D=256), matching the density
 * that works on MNIST. Everything downstream — bucket index, multi-
 * probe, union, resolve — is unchanged.
 *
 * The dimension subsets are RNG-generated (Fisher-Yates shuffle,
 * take first D indices). Routing-native: no spatial knowledge, no
 * pixel-space computation, no float.
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

/* Generate a random subset of D indices from [0, input_dim) using
 * Fisher-Yates partial shuffle, then sort for cache-friendly access. */
static void generate_subset_random(int* subset, int D, int input_dim, glyph_rng_t* rng) {
    int* perm = malloc((size_t)input_dim * sizeof(int));
    for (int i = 0; i < input_dim; i++) perm[i] = i;
    for (int i = 0; i < D; i++) {
        int j = i + (int)(glyph_rng_next(rng) % (uint32_t)(input_dim - i));
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    memcpy(subset, perm, (size_t)D * sizeof(int));
    free(perm);
    for (int i = 1; i < D; i++) {
        int v = subset[i]; int j = i - 1;
        while (j >= 0 && subset[j] > v) { subset[j+1] = subset[j]; j--; }
        subset[j+1] = v;
    }
}

/* Generate a spatial block subset for a 32×32×3 image.
 * block_idx selects which spatial region; all 3 channels included.
 * Layout assumed: R[0..1023], G[1024..2047], B[2048..3071]
 * each in row-major 32×32 order. */
static int generate_subset_spatial(int* subset, int block_idx,
                                    int img_w, int img_h, int n_channels,
                                    int block_w, int block_h) {
    int blocks_per_row = img_w / block_w;
    int bx = (block_idx % blocks_per_row) * block_w;
    int by = (block_idx / blocks_per_row) * block_h;
    int D = 0;
    int pixels_per_channel = img_w * img_h;
    for (int ch = 0; ch < n_channels; ch++) {
        int ch_off = ch * pixels_per_channel;
        for (int dy = 0; dy < block_h; dy++) {
            int row = by + dy;
            if (row >= img_h) continue;
            for (int dx = 0; dx < block_w; dx++) {
                int col = bx + dx;
                if (col >= img_w) continue;
                subset[D++] = ch_off + row * img_w + col;
            }
        }
    }
    return D;
}

/* Extract a D-dim subset from a full input_dim vector. */
static void extract_subset(m4t_mtfp_t* dst, const m4t_mtfp_t* src,
                           const int* subset, int D) {
    for (int d = 0; d < D; d++) dst[d] = src[subset[d]];
}

int main(int argc, char** argv) {
    /* Strip --spatial before glyph_config sees it. */
    int spatial_mode = 0;
    int new_argc = 0;
    char** new_argv = malloc((size_t)argc * sizeof(char*));
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--spatial") == 0) { spatial_mode = 1; continue; }
        new_argv[new_argc++] = argv[i];
    }

    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, new_argc, new_argv);
    free(new_argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) {
        fprintf(stderr, "failed to load dataset from %s\n", cfg.data_dir);
        return 1;
    }
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);

    const int block_w = 8, block_h = 8;
    const int n_channels = (ds.input_dim > 784) ? 3 : 1;
    const int D_spatial = block_w * block_h * n_channels;
    const int D = spatial_mode ? D_spatial : 256;
    const int M = cfg.m_max;
    const int n_proj = cfg.n_proj;
    const int sig_bytes = M4T_TRIT_PACKED_BYTES(n_proj);
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    if (sig_bytes != 4) {
        fprintf(stderr, "n_proj must be 16 (sig_bytes=4)\n");
        return 1;
    }
    if (D >= ds.input_dim) {
        fprintf(stderr, "D=%d must be < input_dim=%d\n", D, ds.input_dim);
        return 1;
    }

    printf("subsetted_multi: dimension-subsetted routed k-NN\n");
    printf("  data_dir=%s  deskew=%s  mode=%s\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on",
           spatial_mode ? "spatial" : "random");
    printf("  D=%d  input_dim=%d  compression=1:%.0f\n",
           D, ds.input_dim, (double)D / n_proj);
    printf("  n_proj=%d  M=%d  density=%.2f  max_radius=%d  min_cands=%d\n",
           n_proj, M, cfg.density, cfg.max_radius, cfg.min_cands);
    printf("  n_train=%d  n_test=%d\n\n", ds.n_train, ds.n_test);

    clock_t t_build = clock();

    /* Per-table: generate subset, extract training data, build sigs + index. */
    int**                subsets       = calloc((size_t)M, sizeof(int*));
    m4t_mtfp_t**         sub_train    = calloc((size_t)M, sizeof(m4t_mtfp_t*));
    glyph_sig_builder_t* builders     = calloc((size_t)M, sizeof(glyph_sig_builder_t));
    uint8_t**            train_sigs   = calloc((size_t)M, sizeof(uint8_t*));
    glyph_bucket_table_t* tables      = calloc((size_t)M, sizeof(glyph_bucket_table_t));

    for (int m = 0; m < M; m++) {
        uint32_t seeds[4]; derive_seed((uint32_t)m, cfg.base_seed, seeds);
        glyph_rng_t rng; glyph_rng_seed(&rng, seeds[0], seeds[1], seeds[2], seeds[3]);

        /* Generate dimension subset. */
        subsets[m] = malloc((size_t)D * sizeof(int));
        if (spatial_mode) {
            int img_w = ds.img_w > 0 ? ds.img_w : 32;
            int img_h = ds.img_h > 0 ? ds.img_h : 32;
            int n_blocks = (img_w / block_w) * (img_h / block_h);
            int block_idx = m % n_blocks;
            generate_subset_spatial(subsets[m], block_idx, img_w, img_h,
                                    n_channels, block_w, block_h);
        } else {
            generate_subset_random(subsets[m], D, ds.input_dim, &rng);
        }

        /* Extract D-dim subset of all training vectors. */
        sub_train[m] = malloc((size_t)ds.n_train * D * sizeof(m4t_mtfp_t));
        for (int i = 0; i < ds.n_train; i++)
            extract_subset(sub_train[m] + (size_t)i * D,
                           ds.x_train + (size_t)i * ds.input_dim,
                           subsets[m], D);

        /* Build sig_builder on the D-dim subset. */
        if (glyph_sig_builder_init(&builders[m], n_proj, D, cfg.density,
                                    seeds[0], seeds[1], seeds[2], seeds[3],
                                    sub_train[m], (n_calib < ds.n_train ? n_calib : ds.n_train)) != 0) {
            fprintf(stderr, "builder init failed for table %d\n", m);
            return 1;
        }

        /* Encode training sigs. */
        train_sigs[m] = calloc((size_t)ds.n_train * sig_bytes, 1);
        glyph_sig_encode_batch(&builders[m], sub_train[m], ds.n_train, train_sigs[m]);

        /* Build bucket index. */
        if (glyph_bucket_build(&tables[m], train_sigs[m], ds.n_train, sig_bytes) != 0) {
            fprintf(stderr, "bucket build failed for table %d\n", m);
            return 1;
        }
    }
    double t_build_sec = (double)(clock() - t_build) / CLOCKS_PER_SEC;
    printf("Built %d subsetted tables in %.2fs.\n\n", M, t_build_sec);

    /* Query-time state. */
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union;
    st.n_hit = 0;
    uint8_t scratch[4];
    uint8_t mask[4]; memset(mask, 0xFF, 4);

    /* Per-query subset extraction buffer. */
    m4t_mtfp_t* q_sub = malloc((size_t)D * sizeof(m4t_mtfp_t));
    const uint8_t** q_sigs_p = calloc((size_t)M, sizeof(uint8_t*));

    /* Pre-encode test sigs per table. */
    uint8_t** test_sigs = calloc((size_t)M, sizeof(uint8_t*));
    for (int m = 0; m < M; m++) {
        test_sigs[m] = calloc((size_t)ds.n_test * sig_bytes, 1);
        for (int i = 0; i < ds.n_test; i++) {
            extract_subset(q_sub, ds.x_test + (size_t)i * ds.input_dim, subsets[m], D);
            glyph_sig_encode(&builders[m], q_sub, test_sigs[m] + (size_t)i * sig_bytes);
        }
    }

    glyph_union_t u = {0};
    u.y_train = ds.y_train;
    u.n_classes = N_CLASSES;

    int oracle_correct = 0;
    int sum_correct = 0;
    int knn_correct = 0;
    long total_union = 0;

    clock_t t_sweep = clock();
    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];

        for (int m = 0; m < M; m++)
            q_sigs_p[m] = test_sigs[m] + (size_t)qi * sig_bytes;

        probe_state_reset(&st);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], q_sigs_p[m], n_proj, sig_bytes,
                        cfg.max_radius, cfg.min_cands, &st, scratch);

        total_union += st.n_hit;

        /* Oracle. */
        for (int j = 0; j < st.n_hit; j++)
            if (ds.y_train[st.hit_list[j]] == y) { oracle_correct++; break; }

        u.hit_list = st.hit_list;
        u.n_hit    = st.n_hit;
        u.votes    = st.votes;

        /* SUM 1-NN. */
        int pred_s = glyph_resolver_sum(&u, M, sig_bytes, train_sigs, q_sigs_p, mask);
        if (pred_s == y) sum_correct++;

        /* k-NN k=5. */
        int pred_k = glyph_resolver_sum_knn(&u, M, sig_bytes, train_sigs, q_sigs_p, mask, 5);
        if (pred_k == y) knn_correct++;
    }
    double t_sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("Sweep: %.2fs for %d queries.\n\n", t_sweep_sec, ds.n_test);
    printf("  oracle:    %6.2f%%\n", 100.0 * oracle_correct / ds.n_test);
    printf("  SUM 1-NN:  %6.2f%%\n", 100.0 * sum_correct / ds.n_test);
    printf("  k=5-NN:    %6.2f%%\n", 100.0 * knn_correct / ds.n_test);
    printf("  avg_union: %.1f\n\n", (double)total_union / ds.n_test);

    /* Per-class accuracy under k=5. */
    /* Recount with per-class tracking. */
    int per_class_total[N_CLASSES] = {0};
    int per_class_correct[N_CLASSES] = {0};
    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        if (y < 0 || y >= N_CLASSES) continue;
        per_class_total[y]++;
        for (int m = 0; m < M; m++)
            q_sigs_p[m] = test_sigs[m] + (size_t)qi * sig_bytes;
        probe_state_reset(&st);
        for (int m = 0; m < M; m++)
            probe_table(&tables[m], q_sigs_p[m], n_proj, sig_bytes,
                        cfg.max_radius, cfg.min_cands, &st, scratch);
        u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;
        int pred = glyph_resolver_sum_knn(&u, M, sig_bytes, train_sigs, q_sigs_p, mask, 5);
        if (pred == y) per_class_correct[y]++;
    }
    printf("Per-class k=5 accuracy:\n");
    printf("  class   count   correct   accuracy\n");
    for (int c = 0; c < N_CLASSES; c++) {
        if (per_class_total[c] == 0) continue;
        printf("   %2d    %5d   %5d     %6.2f%%\n",
               c, per_class_total[c], per_class_correct[c],
               100.0 * per_class_correct[c] / per_class_total[c]);
    }

    /* Cleanup. */
    free(q_sub); free(q_sigs_p);
    free(st.votes); free(st.hit_list);
    for (int m = 0; m < M; m++) {
        free(subsets[m]); free(sub_train[m]);
        glyph_sig_builder_free(&builders[m]);
        glyph_bucket_table_free(&tables[m]);
        free(train_sigs[m]); free(test_sigs[m]);
    }
    free(subsets); free(sub_train); free(builders);
    free(train_sigs); free(test_sigs); free(tables);
    glyph_dataset_free(&ds);
    return 0;
}
