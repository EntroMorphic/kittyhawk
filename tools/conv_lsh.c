/*
 * conv_lsh.c — routed convolution + Trit Lattice LSH.
 *
 * Applies K ternary 3×3 kernels at every position of the image,
 * producing K feature maps. The feature maps are flattened and
 * used as the input to the standard LSH pipeline (random ternary
 * projection → bucket index → multi-probe → k-NN resolve).
 *
 * The convolution is routing-native:
 *   - Kernel weights are ternary ({-1, 0, +1})
 *   - The dot product is integer multiply-accumulate
 *   - The output is ternary (sign-extracted via threshold)
 *   - The structural zero in each kernel IS spatial attention
 *
 * Two modes:
 *   1. Conv features → random projection → LSH (conv as preprocessing)
 *   2. Conv features flattened AS signatures (conv as projection)
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
#define KERN_SIZE 3
#define MAX_KERNELS 64

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

/* Generate K random ternary 3×3 kernels. Each kernel is applied
 * per-channel (so it's 3×3, not 3×3×3). */
static void generate_kernels(int8_t kernels[][KERN_SIZE * KERN_SIZE],
                             int n_kernels,
                             const uint32_t base_seed[4]) {
    glyph_rng_t rng;
    glyph_rng_seed(&rng, base_seed[0], base_seed[1],
                    base_seed[2], base_seed[3]);
    for (int k = 0; k < n_kernels; k++) {
        for (int i = 0; i < KERN_SIZE * KERN_SIZE; i++) {
            uint32_t r = glyph_rng_next(&rng) % 3;
            kernels[k][i] = (int8_t)r - 1;  /* {-1, 0, +1} */
        }
    }
}

/* Apply a single 3×3 ternary kernel to one channel of an image.
 * Produces a (H-2)×(W-2) MTFP feature map. */
static void apply_kernel(const m4t_mtfp_t* channel, int W,
                         const int8_t kernel[KERN_SIZE * KERN_SIZE],
                         m4t_mtfp_t* out, int out_W, int out_H) {
    for (int y = 0; y < out_H; y++) {
        for (int x = 0; x < out_W; x++) {
            int64_t sum = 0;
            for (int ky = 0; ky < KERN_SIZE; ky++) {
                for (int kx = 0; kx < KERN_SIZE; kx++) {
                    sum += (int64_t)kernel[ky * KERN_SIZE + kx] *
                           (int64_t)channel[(y + ky) * W + (x + kx)];
                }
            }
            out[y * out_W + x] = (m4t_mtfp_t)sum;
        }
    }
}

/* Compute convolutional features for one image.
 * n_kernels kernels × n_channels channels → n_kernels × n_channels
 * feature maps, each (H-2)×(W-2). All concatenated into out_features. */
static void compute_conv_features(
    const m4t_mtfp_t* image, int W, int H, int n_channels,
    const int8_t kernels[][KERN_SIZE * KERN_SIZE], int n_kernels,
    m4t_mtfp_t* out_features)
{
    int out_W = W - KERN_SIZE + 1;
    int out_H = H - KERN_SIZE + 1;
    int map_size = out_W * out_H;
    int ppc = W * H;

    int idx = 0;
    for (int k = 0; k < n_kernels; k++) {
        for (int ch = 0; ch < n_channels; ch++) {
            apply_kernel(image + ch * ppc, W,
                         kernels[k], out_features + idx, out_W, out_H);
            idx += map_size;
        }
    }
}

int main(int argc, char** argv) {
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);
    if (cfg.normalize) glyph_dataset_normalize(&ds);

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);
    int out_w = img_w - KERN_SIZE + 1;
    int out_h = img_h - KERN_SIZE + 1;
    int map_size = out_w * out_h;

    const int N_KERNELS = 8;
    int feat_dim = N_KERNELS * n_ch * map_size;

    printf("conv_lsh: routed convolution + Trit Lattice LSH\n");
    printf("  data=%s  deskew=%s  normalize=%s  density=%.2f\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on",
           cfg.normalize ? "on" : "off", cfg.density);
    printf("  image: %dx%dx%d  kernels: %d × %dx%d  feature_dim: %d\n",
           img_w, img_h, n_ch, N_KERNELS, KERN_SIZE, KERN_SIZE, feat_dim);

    const int M = cfg.m_max;
    const int N_PROJ = 16;
    const int SB = M4T_TRIT_PACKED_BYTES(N_PROJ);
    const int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    printf("  M=%d  N_PROJ=%d  knn_k=%d\n", M, N_PROJ, KNN_K);
    printf("  n_train=%d  n_test=%d\n\n", ds.n_train, ds.n_test);

    /* Generate random ternary kernels. */
    int8_t kernels[MAX_KERNELS][KERN_SIZE * KERN_SIZE];
    generate_kernels(kernels, N_KERNELS, cfg.base_seed);

    printf("Kernels:\n");
    for (int k = 0; k < N_KERNELS; k++) {
        printf("  K%d: [", k);
        for (int i = 0; i < KERN_SIZE * KERN_SIZE; i++)
            printf("%+d%s", kernels[k][i], i < 8 ? "," : "");
        printf("]\n");
    }
    printf("\n");

    /* Compute convolutional features for all training and test images. */
    clock_t t0 = clock();
    printf("Computing conv features...\n");

    m4t_mtfp_t* train_feat = malloc((size_t)ds.n_train * feat_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_feat  = malloc((size_t)ds.n_test  * feat_dim * sizeof(m4t_mtfp_t));

    for (int i = 0; i < ds.n_train; i++)
        compute_conv_features(ds.x_train + (size_t)i * ds.input_dim,
                              img_w, img_h, n_ch, kernels, N_KERNELS,
                              train_feat + (size_t)i * feat_dim);
    for (int i = 0; i < ds.n_test; i++)
        compute_conv_features(ds.x_test + (size_t)i * ds.input_dim,
                              img_w, img_h, n_ch, kernels, N_KERNELS,
                              test_feat + (size_t)i * feat_dim);

    printf("  Conv features computed in %.1fs.\n",
           (double)(clock() - t0) / CLOCKS_PER_SEC);

    /* Build LSH on convolutional features. */
    printf("Building LSH on conv features (dim=%d)...\n", feat_dim);

    glyph_sig_builder_t* builders = calloc((size_t)M, sizeof(glyph_sig_builder_t));
    uint8_t** train_sigs = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** test_sigs  = calloc((size_t)M, sizeof(uint8_t*));
    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));

    for (int m = 0; m < M; m++) {
        uint32_t seeds[4]; derive_seed((uint32_t)(100 + m), cfg.base_seed, seeds);
        glyph_sig_builder_init(&builders[m], N_PROJ, feat_dim, cfg.density,
                                seeds[0], seeds[1], seeds[2], seeds[3],
                                train_feat, n_calib);
        train_sigs[m] = calloc((size_t)ds.n_train * SB, 1);
        test_sigs[m]  = calloc((size_t)ds.n_test  * SB, 1);
        glyph_sig_encode_batch(&builders[m], train_feat, ds.n_train, train_sigs[m]);
        glyph_sig_encode_batch(&builders[m], test_feat,  ds.n_test,  test_sigs[m]);
        glyph_bucket_build(&tables[m], train_sigs[m], ds.n_train, SB);
    }

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Built in %.1fs.\n\n", build_sec);

    /* Sweep. */
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union; st.n_hit = 0;
    uint8_t scratch[4];
    uint8_t mask[4]; memset(mask, 0xFF, SB);
    const uint8_t** q_ptrs = calloc((size_t)M, sizeof(uint8_t*));

    glyph_union_t u = {0};
    u.y_train = ds.y_train; u.n_classes = N_CLASSES;

    int m_sweep[] = {1, 2, 4, 8, 16, 32, 64};
    int n_sweep = 0;
    for (int i = 0; i < 7; i++) if (m_sweep[i] <= M) n_sweep = i + 1;

    int oracle_c[7]={0}, sum_c[7]={0}, knn_c[7]={0};
    long union_sum[7]={0};

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
            union_sum[si] += st.n_hit;
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
    printf("   M    oracle    avg_union   SUM_1NN    k=%d-NN\n", KNN_K);
    for (int si = 0; si < n_sweep; si++) {
        printf("  %3d   %6.2f%%   %7.1f    %6.2f%%   %6.2f%%\n",
               m_sweep[si],
               100.0 * oracle_c[si] / ds.n_test,
               (double)union_sum[si] / ds.n_test,
               100.0 * sum_c[si] / ds.n_test,
               100.0 * knn_c[si] / ds.n_test);
    }
    printf("\n");

    /* Per-class at max M. */
    int pc_total[N_CLASSES]={0}, pc_correct[N_CLASSES]={0};
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
    printf("Per-class k=%d at M=%d:\n", KNN_K, M);
    printf("  class   count   correct   accuracy\n");
    for (int c = 0; c < N_CLASSES; c++) {
        if (pc_total[c] == 0) continue;
        printf("   %2d    %5d   %5d     %6.2f%%\n",
               c, pc_total[c], pc_correct[c],
               100.0 * pc_correct[c] / pc_total[c]);
    }

    /* Cleanup. */
    free(train_feat); free(test_feat);
    free(st.votes); free(st.hit_list); free(q_ptrs);
    for (int m = 0; m < M; m++) {
        glyph_sig_builder_free(&builders[m]);
        glyph_bucket_table_free(&tables[m]);
        free(train_sigs[m]); free(test_sigs[m]);
    }
    free(builders); free(tables); free(train_sigs); free(test_sigs);
    glyph_dataset_free(&ds);
    return 0;
}
