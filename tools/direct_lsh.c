/*
 * direct_lsh.c — direct ternary quantization + Trit Lattice LSH.
 *
 * Each trit represents a SPECIFIC input dimension (pixel or gradient),
 * not a random mixture. Normalized pixels are quantized to {-1, 0, +1}
 * via per-value thresholding. Optionally appends horizontal and vertical
 * gradients as additional trit channels.
 *
 * The quantized image IS the signature. The LSH infrastructure (bucket
 * index, multi-probe, k-NN resolve) operates on these direct signatures.
 *
 * NO RANDOM PROJECTIONS. Each trit = one pixel or one gradient.
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
#define TRITS_PER_VOTE 4

static const int8_t vote_trits[10][TRITS_PER_VOTE] = {
    {-1,-1,-1,-1}, {-1,-1,-1, 0}, {-1,-1,-1,+1}, {-1,-1, 0,-1},
    {-1,-1, 0, 0}, {-1,-1, 0,+1}, {-1,-1,+1,-1}, {-1,-1,+1, 0},
    {-1,-1,+1,+1}, {-1, 0,-1,-1},
};

static void encode_gsh_sig(const int* labels, int n_tables,
                           uint8_t* out, int gsh_sb) {
    memset(out, 0, gsh_sb);
    for (int m = 0; m < n_tables; m++) {
        int lbl = labels[m];
        if (lbl < 0 || lbl >= N_CLASSES) lbl = 0;
        for (int t = 0; t < TRITS_PER_VOTE; t++)
            glyph_write_trit(out, m * TRITS_PER_VOTE + t, vote_trits[lbl][t]);
    }
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

/* Extract top-M nearest labels from the union by full-sig Hamming
 * distance. With direct quantization there's one signature per image,
 * not M per-table sigs — so we take the top-M nearest by distance
 * and use their labels as the M-dim routing pattern. */
static void union_top_m_labels(
    const probe_state_t* st, int M_labels, int sig_bytes,
    const uint8_t* train_sigs, const uint8_t* q_sig,
    const uint8_t* mask, const int* y_train,
    int exclude_idx, int* out_labels)
{
    typedef struct { int32_t d; int label; } dl_t;
    dl_t topk[256];
    int ntk = 0;
    int mlim = (M_labels < 256) ? M_labels : 256;

    for (int j = 0; j < st->n_hit; j++) {
        int idx = st->hit_list[j];
        if (idx == exclude_idx) continue;
        int32_t d = m4t_popcount_dist(
            q_sig, train_sigs + (size_t)idx * sig_bytes, mask, sig_bytes);
        int lbl = y_train[idx];
        if (ntk < mlim) {
            int pos = ntk;
            while (pos > 0 && topk[pos-1].d > d) { topk[pos]=topk[pos-1]; pos--; }
            topk[pos].d = d; topk[pos].label = lbl; ntk++;
        } else if (d < topk[mlim-1].d) {
            int pos = mlim - 1;
            while (pos > 0 && topk[pos-1].d > d) { topk[pos]=topk[pos-1]; pos--; }
            topk[pos].d = d; topk[pos].label = lbl;
        }
    }
    for (int i = 0; i < mlim; i++)
        out_labels[i] = (i < ntk) ? topk[i].label : 0;
}

/* NOTE: trit-native transitions (quantize first, gradients second)
 * were tested and HURT accuracy on all three datasets (CIFAR −10.67pp,
 * Fashion −3.53pp, MNIST −1.37pp). The transitions produce 81-91%
 * zeros because adjacent pixels often share the same ternary state.
 * Hamming distance treats those zeros as dead weight.
 *
 * The float gradient with separate tau calibration (below) preserves
 * continuous magnitude information and is quantized to keep 90%
 * non-zero. This is the correct design for Hamming-distance scoring.
 * SSTT's quantize-first approach works with IG-weighted inverted
 * index scoring but not with uniform Hamming. */
static void compute_gradients(const m4t_mtfp_t* img, int W, int H, int n_ch,
                              m4t_mtfp_t* hgrad, m4t_mtfp_t* vgrad) {
    int ppc = W * H;
    int idx_h = 0, idx_v = 0;
    for (int ch = 0; ch < n_ch; ch++) {
        const m4t_mtfp_t* c = img + ch * ppc;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W - 1; x++)
                hgrad[idx_h++] = c[y * W + x + 1] - c[y * W + x];
        for (int y = 0; y < H - 1; y++)
            for (int x = 0; x < W; x++)
                vgrad[idx_v++] = c[(y + 1) * W + x] - c[y * W + x];
    }
}

int main(int argc, char** argv) {
    /* Strip --gradients before glyph_config sees it. */
    int use_gradients = 0;
    int new_argc = 0;
    char** new_argv = malloc((size_t)argc * sizeof(char*));
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--gradients") == 0) { use_gradients = 1; continue; }
        new_argv[new_argc++] = argv[i];
    }

    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, new_argc, new_argv);
    free(new_argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew) glyph_dataset_deskew(&ds);
    glyph_dataset_normalize(&ds);

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);

    int intensity_dim = ds.input_dim;
    int hgrad_dim = n_ch * img_h * (img_w - 1);
    int vgrad_dim = n_ch * (img_h - 1) * img_w;
    int total_dim = intensity_dim + (use_gradients ? (hgrad_dim + vgrad_dim) : 0);
    int sig_bytes = M4T_TRIT_PACKED_BYTES(total_dim);

    printf("direct_lsh: direct ternary quantization + Trit Lattice LSH\n");
    printf("  data=%s  deskew=%s  gradients=%s\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on",
           use_gradients ? "on" : "off");
    printf("  image: %dx%dx%d  intensity_dim=%d\n", img_w, img_h, n_ch, intensity_dim);
    if (use_gradients)
        printf("  hgrad_dim=%d  vgrad_dim=%d  total_dim=%d\n",
               hgrad_dim, vgrad_dim, total_dim);
    /* NOTE: --density for direct quantization means "fraction of pixel
     * values that map to zero (structural zero)." This is different from
     * the random-projection meaning ("fraction of projection WEIGHTS
     * that are zero"). For normalized CIFAR-10, --density 0.395
     * produces tau≈0.6×SCALE which matches the empirically optimal
     * threshold. */
    printf("  sig_bytes=%d  density=%.3f (%.1f%% of intensity trits will be zero)\n",
           sig_bytes, cfg.density, cfg.density * 100.0);

    /* Multi-table: each table uses a DIFFERENT permutation of the
     * first 16 trits as its bucket key. The full signature is shared;
     * only the key subset differs per table. */
    const int M = cfg.m_max;
    const int KEY_TRITS = 16;
    printf("  M=%d tables (different 16-trit key subsets)  knn_k=%d\n",
           M, KNN_K);
    printf("  n_train=%d  n_test=%d\n\n", ds.n_train, ds.n_test);

    clock_t t0 = clock();

    /* Build feature vectors: intensity + optional float gradients. */
    printf("Building feature vectors...\n");
    m4t_mtfp_t* train_feat = malloc((size_t)ds.n_train * total_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_feat  = malloc((size_t)ds.n_test  * total_dim * sizeof(m4t_mtfp_t));

    if (use_gradients) {
        m4t_mtfp_t* hg = malloc((size_t)hgrad_dim * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* vg = malloc((size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        for (int i = 0; i < ds.n_train; i++) {
            const m4t_mtfp_t* img = ds.x_train + (size_t)i * ds.input_dim;
            m4t_mtfp_t* out = train_feat + (size_t)i * total_dim;
            memcpy(out, img, (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            compute_gradients(img, img_w, img_h, n_ch, hg, vg);
            memcpy(out + intensity_dim, hg, (size_t)hgrad_dim * sizeof(m4t_mtfp_t));
            memcpy(out + intensity_dim + hgrad_dim, vg, (size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        }
        for (int i = 0; i < ds.n_test; i++) {
            const m4t_mtfp_t* img = ds.x_test + (size_t)i * ds.input_dim;
            m4t_mtfp_t* out = test_feat + (size_t)i * total_dim;
            memcpy(out, img, (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            compute_gradients(img, img_w, img_h, n_ch, hg, vg);
            memcpy(out + intensity_dim, hg, (size_t)hgrad_dim * sizeof(m4t_mtfp_t));
            memcpy(out + intensity_dim + hgrad_dim, vg, (size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        }
        free(hg); free(vg);
    } else {
        memcpy(train_feat, ds.x_train, (size_t)ds.n_train * intensity_dim * sizeof(m4t_mtfp_t));
        memcpy(test_feat, ds.x_test, (size_t)ds.n_test * intensity_dim * sizeof(m4t_mtfp_t));
    }

    /* Calibrate tau: separate thresholds for intensity and gradients.
     * Extract into contiguous buffers for correct stride. */
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    m4t_mtfp_t* intensity_sample = malloc((size_t)n_calib * intensity_dim * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_calib; i++)
        memcpy(intensity_sample + (size_t)i * intensity_dim,
               train_feat + (size_t)i * total_dim,
               (size_t)intensity_dim * sizeof(m4t_mtfp_t));
    int64_t tau_intensity = glyph_sig_quantize_tau(
        intensity_sample, n_calib, intensity_dim, cfg.density);
    free(intensity_sample);

    int64_t tau_gradient = 0;
    if (use_gradients) {
        int grad_dim = hgrad_dim + vgrad_dim;
        m4t_mtfp_t* grad_sample = malloc((size_t)n_calib * grad_dim * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_calib; i++)
            memcpy(grad_sample + (size_t)i * grad_dim,
                   train_feat + (size_t)i * total_dim + intensity_dim,
                   (size_t)grad_dim * sizeof(m4t_mtfp_t));
        tau_gradient = glyph_sig_quantize_tau(grad_sample, n_calib, grad_dim, 0.10);
        free(grad_sample);
    }
    printf("  tau_intensity=%lld (%.3f × SCALE)  tau_gradient=%lld (%.3f × SCALE)\n",
           (long long)tau_intensity, (double)tau_intensity / M4T_MTFP_SCALE,
           (long long)tau_gradient, (double)tau_gradient / M4T_MTFP_SCALE);

    /* Quantize all images to trit signatures. */
    printf("Quantizing signatures (%d trits = %d bytes)...\n", total_dim, sig_bytes);
    uint8_t* train_sigs = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* test_sigs  = calloc((size_t)ds.n_test  * sig_bytes, 1);

    for (int pass = 0; pass < 2; pass++) {
        int n_imgs = (pass == 0) ? ds.n_train : ds.n_test;
        const m4t_mtfp_t* feat = (pass == 0) ? train_feat : test_feat;
        uint8_t* sigs = (pass == 0) ? train_sigs : test_sigs;
        for (int i = 0; i < n_imgs; i++) {
            const m4t_mtfp_t* f = feat + (size_t)i * total_dim;
            uint8_t* sig = sigs + (size_t)i * sig_bytes;
            for (int d = 0; d < intensity_dim; d++) {
                int64_t v = (int64_t)f[d];
                if (v > tau_intensity) glyph_write_trit(sig, d, +1);
                else if (v < -tau_intensity) glyph_write_trit(sig, d, -1);
            }
            if (use_gradients) {
                for (int d = 0; d < hgrad_dim + vgrad_dim; d++) {
                    int64_t v = (int64_t)f[intensity_dim + d];
                    int pos = intensity_dim + d;
                    if (v > tau_gradient) glyph_write_trit(sig, pos, +1);
                    else if (v < -tau_gradient) glyph_write_trit(sig, pos, -1);
                }
            }
        }
    }
    free(train_feat); free(test_feat);

    /* Hierarchical Trit Lattice LSH: spatial pooling builds the bucket
     * key by reducing blocks of trits via majority vote.
     *
     * For CIFAR-10 (32×32×3 = 3072 intensity trits):
     *   Level 0: 4×4×n_ch summary trits from 8×8 spatial blocks
     * For MNIST (28×28 = 784 intensity trits):
     *   Level 0: 7×7 summary trits from 4×4 spatial blocks
     *
     * Each summary trit = majority of the block's trits:
     *   more +1 than -1 → +1, more -1 → -1, balanced → 0.
     *
     * The bucket key is the first 16 of the summary trits. Each
     * table uses a different PERMUTATION of the summary trits so
     * different tables key on different spatial regions.
     *
     * No random projections — the key is a spatial summary of
     * directly quantized pixels. */

    /* Compute block summary dimensions. */
    /* Block size for hierarchical summary. Smaller blocks = more
     * summary trits = finer spatial key = better filtering. */
    int blk_w, blk_h;
    if (img_w == 32) { blk_w = 4; blk_h = 4; }      /* CIFAR: 8×8×3=192 summary */
    else             { blk_w = 2; blk_h = 2; }       /* MNIST: 14×14=196 summary */
    int sum_w = img_w / blk_w;
    int sum_h = img_h / blk_h;
    int summary_dim = sum_w * sum_h * n_ch;
    printf("Hierarchical key: %dx%d blocks → %dx%dx%d = %d summary trits\n",
           blk_w, blk_h, sum_w, sum_h, n_ch, summary_dim);

    /* Compute summary trits for all images by majority-voting
     * within each spatial block of the intensity channel. */
    int summary_bytes = M4T_TRIT_PACKED_BYTES(summary_dim);
    uint8_t* train_summary = calloc((size_t)ds.n_train * summary_bytes, 1);
    uint8_t* test_summary  = calloc((size_t)ds.n_test  * summary_bytes, 1);
    int ppc = img_w * img_h;

    for (int pass = 0; pass < 2; pass++) {
        int n_imgs = (pass == 0) ? ds.n_train : ds.n_test;
        const uint8_t* sigs = (pass == 0) ? train_sigs : test_sigs;
        uint8_t* summaries = (pass == 0) ? train_summary : test_summary;

        for (int i = 0; i < n_imgs; i++) {
            const uint8_t* sig = sigs + (size_t)i * sig_bytes;
            uint8_t* sum_sig = summaries + (size_t)i * summary_bytes;
            int si = 0;
            for (int ch = 0; ch < n_ch; ch++) {
                for (int by = 0; by < sum_h; by++) {
                    for (int bx = 0; bx < sum_w; bx++) {
                        int pos_count = 0, neg_count = 0;
                        for (int dy = 0; dy < blk_h; dy++) {
                            for (int dx = 0; dx < blk_w; dx++) {
                                int px = bx * blk_w + dx;
                                int py = by * blk_h + dy;
                                if (px >= img_w || py >= img_h) continue;
                                int trit_pos = ch * ppc + py * img_w + px;
                                int8_t t = glyph_read_trit(sig, trit_pos);
                                if (t > 0) pos_count++;
                                else if (t < 0) neg_count++;
                            }
                        }
                        int8_t summary_trit = 0;
                        if (pos_count > neg_count) summary_trit = +1;
                        else if (neg_count > pos_count) summary_trit = -1;
                        glyph_write_trit(sum_sig, si, summary_trit);
                        si++;
                    }
                }
            }
        }
    }

    /* Emission coverage diagnostic: measure trit distribution. */
    {
        long n_pos = 0, n_neg = 0, n_zero = 0;
        int sample_n = (ds.n_train < 1000) ? ds.n_train : 1000;
        for (int i = 0; i < sample_n; i++) {
            const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
            for (int d = 0; d < total_dim; d++) {
                int8_t t = glyph_read_trit(sig, d);
                if (t > 0) n_pos++;
                else if (t < 0) n_neg++;
                else n_zero++;
            }
        }
        long sampled = (long)sample_n * total_dim;
        printf("  Emission coverage (first %d images):\n", sample_n);
        printf("    +1: %.1f%%  0: %.1f%%  -1: %.1f%%\n",
               100.0 * n_pos / sampled,
               100.0 * n_zero / sampled,
               100.0 * n_neg / sampled);
        if (use_gradients) {
            long ip = 0, iz = 0, in_ = 0, gp = 0, gz = 0, gn = 0;
            for (int i = 0; i < sample_n; i++) {
                const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
                for (int d = 0; d < intensity_dim; d++) {
                    int8_t t = glyph_read_trit(sig, d);
                    if (t > 0) ip++; else if (t < 0) in_++; else iz++;
                }
                for (int d = intensity_dim; d < total_dim; d++) {
                    int8_t t = glyph_read_trit(sig, d);
                    if (t > 0) gp++; else if (t < 0) gn++; else gz++;
                }
            }
            long it = (long)sample_n * intensity_dim;
            long gt = (long)sample_n * (total_dim - intensity_dim);
            printf("    intensity: +1=%.1f%% 0=%.1f%% -1=%.1f%%\n",
                   100.0*ip/it, 100.0*iz/it, 100.0*in_/it);
            printf("    gradient:  +1=%.1f%% 0=%.1f%% -1=%.1f%%\n",
                   100.0*gp/gt, 100.0*gz/gt, 100.0*gn/gt);
        }
    }
    printf("\n");

    /* Build M bucket tables. Each table uses a different permutation
     * of the summary trits for its 16-trit key. */
    printf("Building %d bucket tables...\n", M);
    glyph_bucket_table_t* tables = calloc((size_t)M, sizeof(glyph_bucket_table_t));
    uint8_t** table_train_keys = calloc((size_t)M, sizeof(uint8_t*));
    uint8_t** table_test_keys  = calloc((size_t)M, sizeof(uint8_t*));

    /* Generate a per-table permutation of summary trit indices. */
    int* perm = malloc((size_t)summary_dim * sizeof(int));

    for (int m = 0; m < M; m++) {
        /* Fisher-Yates shuffle seeded per table for diverse keys. */
        for (int t = 0; t < summary_dim; t++) perm[t] = t;
        glyph_rng_t prng;
        uint32_t ps[4];
        ps[0] = cfg.base_seed[0] + (uint32_t)m * 9973u;
        ps[1] = cfg.base_seed[1] + (uint32_t)m * 7919u;
        ps[2] = cfg.base_seed[2] + (uint32_t)m * 6271u;
        ps[3] = cfg.base_seed[3] + (uint32_t)m * 5381u;
        if ((ps[0]|ps[1]|ps[2]|ps[3]) == 0) ps[0] = 1;
        glyph_rng_seed(&prng, ps[0], ps[1], ps[2], ps[3]);
        for (int t = summary_dim - 1; t > 0; t--) {
            int j = (int)(glyph_rng_next(&prng) % (uint32_t)(t + 1));
            int tmp = perm[t]; perm[t] = perm[j]; perm[j] = tmp;
        }

        table_train_keys[m] = calloc((size_t)ds.n_train * 4, 1);
        table_test_keys[m]  = calloc((size_t)ds.n_test  * 4, 1);

        for (int i = 0; i < ds.n_train; i++) {
            const uint8_t* sum_sig = train_summary + (size_t)i * summary_bytes;
            uint8_t* key = table_train_keys[m] + (size_t)i * 4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(key, t, glyph_read_trit(sum_sig, perm[t]));
        }
        for (int i = 0; i < ds.n_test; i++) {
            const uint8_t* sum_sig = test_summary + (size_t)i * summary_bytes;
            uint8_t* key = table_test_keys[m] + (size_t)i * 4;
            for (int t = 0; t < KEY_TRITS && t < summary_dim; t++)
                glyph_write_trit(key, t, glyph_read_trit(sum_sig, perm[t]));
        }
        glyph_bucket_build(&tables[m], table_train_keys[m], ds.n_train, 4);
    }
    free(perm);

    printf("LSH tables built.\n");

    /* ============================================================
     * GSH: compute training routing signatures via LSH probing,
     * encode as multi-trit vote patterns, build GSH bucket index.
     * ============================================================ */
    const int GSH_NTRITS = M * TRITS_PER_VOTE;
    const int GSH_SB = M4T_TRIT_PACKED_BYTES(GSH_NTRITS);
    printf("Building GSH (%d trits = %d bytes)...\n", GSH_NTRITS, GSH_SB);

    probe_state_t gsh_build_st;
    gsh_build_st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gsh_build_st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gsh_build_st.max_union = cfg.max_union; gsh_build_st.n_hit = 0;
    int* vote_labels = malloc((size_t)M * sizeof(int));
    uint8_t* gsh_train = calloc((size_t)ds.n_train * GSH_SB, 1);
    uint8_t gsh_build_scratch[4];
    uint8_t* gsh_build_mask = malloc(sig_bytes);
    memset(gsh_build_mask, 0xFF, sig_bytes);

    for (int i = 0; i < ds.n_train; i++) {
        const uint8_t* q = train_sigs + (size_t)i * sig_bytes;
        probe_state_reset(&gsh_build_st);
        for (int m = 0; m < M; m++) {
            const uint8_t* qk = table_train_keys[m] + (size_t)i * 4;
            probe_table(&tables[m], qk, KEY_TRITS, 4,
                        cfg.max_radius, cfg.min_cands, &gsh_build_st, gsh_build_scratch);
        }
        union_top_m_labels(&gsh_build_st, M, sig_bytes,
                           train_sigs, q, gsh_build_mask,
                           ds.y_train, i, vote_labels);
        encode_gsh_sig(vote_labels, M, gsh_train + (size_t)i * GSH_SB, GSH_SB);

        if ((i + 1) % 10000 == 0)
            printf("  %d/%d training GSH sigs\n", i + 1, ds.n_train);
    }
    free(gsh_build_st.votes); free(gsh_build_st.hit_list);
    free(gsh_build_mask);

    glyph_bucket_table_t gsh_table;
    glyph_bucket_build(&gsh_table, gsh_train, ds.n_train, GSH_SB);
    printf("  GSH: %d distinct buckets.\n", glyph_bucket_count_distinct(&gsh_table));

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Total build: %.1fs.\n\n", build_sec);

    /* Classify. The resolver scores by Hamming distance on the FULL
     * signature (all total_dim trits), not on the 16-trit key. The
     * bucket key is for FILTERING only. */
    probe_state_t st;
    st.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    st.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    st.max_union = cfg.max_union; st.n_hit = 0;
    uint8_t key_scratch[4];
    uint8_t* full_mask = malloc(sig_bytes); memset(full_mask, 0xFF, sig_bytes);

    const uint8_t* qs_ptr;

    glyph_union_t u = {0};
    u.y_train = ds.y_train; u.n_classes = N_CLASSES;

    int m_sweep[] = {1, 2, 4, 8, 16, 32, 64};
    int n_sweep = 0;
    for (int i = 0; i < 7; i++) if (m_sweep[i] <= M) n_sweep = i + 1;

    int oracle_c[7]={0}, sum_c[7]={0}, knn_c[7]={0}, maj_c[7]={0};
    long union_sum[7]={0};
    int* final_pred = malloc((size_t)ds.n_test * sizeof(int));
    memset(final_pred, 0xFF, (size_t)ds.n_test * sizeof(int));

    /* GSH query-time state. */
    probe_state_t gst;
    gst.votes = calloc((size_t)ds.n_train, sizeof(uint16_t));
    gst.hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    gst.max_union = cfg.max_union; gst.n_hit = 0;
    uint8_t* q_gsh = calloc(GSH_SB, 1);
    uint8_t* gsh_mask = malloc(GSH_SB); memset(gsh_mask, 0xFF, GSH_SB);

    int lsh_total_correct = 0, gsh_total_correct = 0;
    int agree_count = 0, agree_correct = 0;
    int disagree_count = 0, disagree_lsh_correct = 0, disagree_gsh_correct = 0;

    printf("Classifying %d queries...\n", ds.n_test);
    clock_t t_sweep = clock();

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        qs_ptr = test_sigs + (size_t)qi * sig_bytes;

        probe_state_reset(&st);
        int prev = 0;
        for (int si = 0; si < n_sweep; si++) {
            int Mt = m_sweep[si];
            for (int m = prev; m < Mt; m++) {
                const uint8_t* q_key = table_test_keys[m] + (size_t)qi * 4;
                probe_table(&tables[m], q_key, KEY_TRITS, 4,
                            cfg.max_radius, cfg.min_cands, &st, key_scratch);
            }

            for (int j = 0; j < st.n_hit; j++)
                if (ds.y_train[st.hit_list[j]] == y) { oracle_c[si]++; break; }
            union_sum[si] += st.n_hit;

            u.hit_list = st.hit_list; u.n_hit = st.n_hit; u.votes = st.votes;

            /* Score by Hamming distance on FULL signature (1 "table"). */
            int32_t best_d = INT32_MAX; int best_l = -1;
            typedef struct { int32_t s; int l; } tk_t;
            tk_t topk[64]; int ntk = 0;
            for (int j = 0; j < st.n_hit; j++) {
                int idx = st.hit_list[j];
                int32_t d = m4t_popcount_dist(
                    qs_ptr, train_sigs + (size_t)idx * sig_bytes,
                    full_mask, sig_bytes);
                if (d < best_d) { best_d = d; best_l = ds.y_train[idx]; }
                int lbl = ds.y_train[idx];
                if (ntk < KNN_K) {
                    int pos = ntk;
                    while (pos > 0 && topk[pos-1].s > d) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = d; topk[pos].l = lbl; ntk++;
                } else if (d < topk[KNN_K-1].s) {
                    int pos = KNN_K-1;
                    while (pos > 0 && topk[pos-1].s > d) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = d; topk[pos].l = lbl;
                }
            }
            if (best_l == y) sum_c[si]++;
            /* Rank-weighted vote. */
            int cv[N_CLASSES] = {0};
            for (int i = 0; i < ntk; i++) cv[topk[i].l] += (KNN_K - i);
            int kpred = 0;
            for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[kpred]) kpred = c;
            if (kpred == y) knn_c[si]++;
            if (m_sweep[si] == M) final_pred[qi] = kpred;
            /* Majority vote (for comparison with brute-force baseline). */
            int mv[N_CLASSES] = {0};
            for (int i = 0; i < ntk; i++) mv[topk[i].l]++;
            int mpred = 0;
            for (int c = 1; c < N_CLASSES; c++) if (mv[c] > mv[mpred]) mpred = c;
            if (mpred == y) maj_c[si]++;

            prev = Mt;
        }

        /* GSH pass at max M. */
        int lsh_pred = final_pred[qi];
        if (lsh_pred == y) lsh_total_correct++;

        union_top_m_labels(&st, M, sig_bytes, train_sigs, qs_ptr,
                           full_mask, ds.y_train, -1, vote_labels);
        encode_gsh_sig(vote_labels, M, q_gsh, GSH_SB);

        probe_state_reset(&gst);
        probe_table(&gsh_table, q_gsh, KEY_TRITS, 4,
                    cfg.max_radius, cfg.min_cands, &gst, key_scratch);

        int gsh_pred = -1;
        {
            int32_t best = INT32_MAX;
            for (int j = 0; j < gst.n_hit; j++) {
                int idx = gst.hit_list[j];
                int32_t d = m4t_popcount_dist(
                    q_gsh, gsh_train + (size_t)idx * GSH_SB,
                    gsh_mask, GSH_SB);
                if (d < best) { best = d; gsh_pred = ds.y_train[idx]; }
            }
        }
        if (gsh_pred == y) gsh_total_correct++;

        if (lsh_pred == gsh_pred) {
            agree_count++;
            if (lsh_pred == y) agree_correct++;
        } else {
            disagree_count++;
            if (lsh_pred == y) disagree_lsh_correct++;
            if (gsh_pred == y) disagree_gsh_correct++;
        }
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("Sweep: %.1fs\n\n", sweep_sec);
    printf("   M    oracle    avg_union   1-NN      k=%d-maj   k=%d-rw\n", KNN_K, KNN_K);
    for (int si = 0; si < n_sweep; si++)
        printf("  %3d   %6.2f%%   %7.1f   %6.2f%%   %6.2f%%   %6.2f%%\n",
               m_sweep[si],
               100.0 * oracle_c[si] / ds.n_test,
               (double)union_sum[si] / ds.n_test,
               100.0 * sum_c[si] / ds.n_test,
               100.0 * maj_c[si] / ds.n_test,
               100.0 * knn_c[si] / ds.n_test);
    printf("\n");

    printf("=== LSH + GSH ===\n");
    printf("  LSH k=%d-rw:              %6.2f%%\n", KNN_K, 100.0 * lsh_total_correct / ds.n_test);
    printf("  GSH 1-NN:                 %6.2f%%\n", 100.0 * gsh_total_correct / ds.n_test);
    printf("  Agreement:                %6.2f%%  (%d / %d)\n",
           100.0 * agree_count / ds.n_test, agree_count, ds.n_test);
    printf("  P(correct | agree):       %6.2f%%  (%d / %d)\n",
           agree_count ? 100.0 * agree_correct / agree_count : 0.0,
           agree_correct, agree_count);
    printf("  P(LSH correct | disagree):%6.2f%%  (%d / %d)\n",
           disagree_count ? 100.0 * disagree_lsh_correct / disagree_count : 0.0,
           disagree_lsh_correct, disagree_count);
    printf("  P(GSH correct | disagree):%6.2f%%  (%d / %d)\n",
           disagree_count ? 100.0 * disagree_gsh_correct / disagree_count : 0.0,
           disagree_gsh_correct, disagree_count);
    printf("\n");

    /* Per-class at max M (from stored predictions — no double sweep). */
    int pc_t[N_CLASSES]={0}, pc_c[N_CLASSES]={0};
    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        if (y < 0 || y >= N_CLASSES) continue;
        pc_t[y]++;
        if (final_pred[qi] == y) pc_c[y]++;
    }
    printf("Per-class k=%d at M=%d:\n", KNN_K, M);
    printf("  class   count   correct   accuracy\n");
    for (int c = 0; c < N_CLASSES; c++)
        if (pc_t[c] > 0)
            printf("   %2d    %5d   %5d     %6.2f%%\n",
                   c, pc_t[c], pc_c[c], 100.0 * pc_c[c] / pc_t[c]);

    /* Cleanup. */
    free(full_mask); free(st.votes); free(st.hit_list);
    free(final_pred); free(vote_labels);
    free(q_gsh); free(gsh_mask); free(gsh_train);
    glyph_bucket_table_free(&gsh_table);
    free(gst.votes); free(gst.hit_list);
    free(train_sigs); free(test_sigs);
    free(train_summary); free(test_summary);
    for (int m = 0; m < M; m++) {
        glyph_bucket_table_free(&tables[m]);
        free(table_train_keys[m]); free(table_test_keys[m]);
    }
    free(tables); free(table_train_keys); free(table_test_keys);
    glyph_dataset_free(&ds);
    return 0;
}
