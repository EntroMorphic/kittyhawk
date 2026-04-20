/*
 * inverted_ig.c — block-encoded inverted index with IG-weighted scoring.
 *
 * SSTT's scoring mechanism on Glyph's direct-quantized signatures.
 * Groups 3 consecutive trits into 27-value blocks. Builds an inverted
 * index: (block_position, block_value) → [training indices]. Scores
 * queries by accumulating IG-weighted hits per candidate.
 *
 * The inverted index IS multi-table routing at block-level granularity:
 *   3008 tables (one per block position) × 27 keys per table.
 *   IG weighting is per-table contribution weighting.
 *
 * NO RANDOM PROJECTIONS. NO HAMMING DISTANCE.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_sig.h"
#include "glyph_multiprobe.h"
#include "m4t_trit_pack.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10
#define KNN_K 5
#define N_BLOCK_VALS 27
#define BG_BLOCK 13  /* (0+1)*9 + (0+1)*3 + (0+1) = all-zero block */

static void compute_gradients(const m4t_mtfp_t* img, int W, int H, int n_ch,
                              m4t_mtfp_t* hgrad, m4t_mtfp_t* vgrad) {
    int ppc = W * H;
    for (int ch = 0; ch < n_ch; ch++) {
        const m4t_mtfp_t* c = img + ch * ppc;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W - 1; x++)
                *hgrad++ = c[y * W + x + 1] - c[y * W + x];
        for (int y = 0; y < H - 1; y++)
            for (int x = 0; x < W; x++)
                *vgrad++ = c[(y + 1) * W + x] - c[y * W + x];
    }
}

static inline uint8_t encode_block(int8_t a, int8_t b, int8_t c) {
    return (uint8_t)((a + 1) * 9 + (b + 1) * 3 + (c + 1));
}

int main(int argc, char** argv) {
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
    int n_blocks = total_dim / 3;
    if (total_dim % 3 != 0) n_blocks++;

    printf("inverted_ig: block-encoded inverted index + IG scoring\n");
    printf("  data=%s  gradients=%s  density=%.3f\n",
           cfg.data_dir, use_gradients ? "on" : "off", cfg.density);
    printf("  total_dim=%d  n_blocks=%d  sig_bytes=%d\n", total_dim, n_blocks, sig_bytes);
    printf("  n_train=%d  n_test=%d  knn_k=%d\n\n", ds.n_train, ds.n_test, KNN_K);

    clock_t t0 = clock();

    /* ============================================================
     * Quantize training and test signatures (same as direct_lsh).
     * ============================================================ */
    printf("Quantizing...\n");
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    int64_t tau_i = glyph_sig_quantize_tau(ds.x_train, n_calib, intensity_dim, cfg.density);
    int64_t tau_g = 0;

    m4t_mtfp_t* train_feat = NULL;
    if (use_gradients) {
        train_feat = malloc((size_t)ds.n_train * total_dim * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* hg = malloc((size_t)hgrad_dim * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* vg = malloc((size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        for (int i = 0; i < ds.n_train; i++) {
            m4t_mtfp_t* out = train_feat + (size_t)i * total_dim;
            memcpy(out, ds.x_train + (size_t)i * intensity_dim,
                   (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            compute_gradients(ds.x_train + (size_t)i * intensity_dim,
                              img_w, img_h, n_ch, hg, vg);
            memcpy(out + intensity_dim, hg, (size_t)hgrad_dim * sizeof(m4t_mtfp_t));
            memcpy(out + intensity_dim + hgrad_dim, vg, (size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        }
        /* Calibrate gradient tau. */
        m4t_mtfp_t* gs = malloc((size_t)n_calib * (hgrad_dim + vgrad_dim) * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_calib; i++)
            memcpy(gs + (size_t)i * (hgrad_dim + vgrad_dim),
                   train_feat + (size_t)i * total_dim + intensity_dim,
                   (size_t)(hgrad_dim + vgrad_dim) * sizeof(m4t_mtfp_t));
        tau_g = glyph_sig_quantize_tau(gs, n_calib, hgrad_dim + vgrad_dim, 0.10);
        free(gs); free(hg); free(vg);
    }
    printf("  tau_i=%lld (%.3f×S)  tau_g=%lld (%.3f×S)\n",
           (long long)tau_i, (double)tau_i/M4T_MTFP_SCALE,
           (long long)tau_g, (double)tau_g/M4T_MTFP_SCALE);

    /* Quantize to packed trit sigs. */
    uint8_t* train_sigs = calloc((size_t)ds.n_train * sig_bytes, 1);
    for (int i = 0; i < ds.n_train; i++) {
        const m4t_mtfp_t* f = use_gradients
            ? (train_feat + (size_t)i * total_dim)
            : (ds.x_train + (size_t)i * intensity_dim);
        uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        for (int d = 0; d < intensity_dim; d++) {
            int64_t v = (int64_t)f[d];
            if (v > tau_i) glyph_write_trit(sig, d, +1);
            else if (v < -tau_i) glyph_write_trit(sig, d, -1);
        }
        if (use_gradients) {
            for (int d = 0; d < hgrad_dim + vgrad_dim; d++) {
                int64_t v = (int64_t)f[intensity_dim + d];
                if (v > tau_g) glyph_write_trit(sig, intensity_dim + d, +1);
                else if (v < -tau_g) glyph_write_trit(sig, intensity_dim + d, -1);
            }
        }
    }
    free(train_feat);

    /* ============================================================
     * Block-encode training signatures.
     * ============================================================ */
    printf("Block encoding (%d blocks of 3 trits)...\n", n_blocks);
    uint8_t* train_blocks = malloc((size_t)ds.n_train * n_blocks);
    for (int i = 0; i < ds.n_train; i++) {
        const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        uint8_t* blk = train_blocks + (size_t)i * n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            int pos = b * 3;
            int8_t t0b = (pos < total_dim) ? glyph_read_trit(sig, pos) : 0;
            int8_t t1 = (pos+1 < total_dim) ? glyph_read_trit(sig, pos+1) : 0;
            int8_t t2 = (pos+2 < total_dim) ? glyph_read_trit(sig, pos+2) : 0;
            blk[b] = encode_block(t0b, t1, t2);
        }
    }

    /* ============================================================
     * Build inverted index: for each (position, block_value),
     * store the list of training image indices.
     * ============================================================ */
    printf("Building inverted index...\n");

    /* Count entries per (pos, val). */
    uint32_t* list_sz = calloc((size_t)n_blocks * N_BLOCK_VALS, sizeof(uint32_t));
    for (int i = 0; i < ds.n_train; i++) {
        const uint8_t* blk = train_blocks + (size_t)i * n_blocks;
        for (int b = 0; b < n_blocks; b++)
            if (blk[b] != BG_BLOCK)
                list_sz[(size_t)b * N_BLOCK_VALS + blk[b]]++;
    }

    /* Allocate posting lists and fill. */
    uint32_t total_entries = 0;
    uint32_t* list_off = malloc((size_t)n_blocks * N_BLOCK_VALS * sizeof(uint32_t));
    for (int k = 0; k < n_blocks * N_BLOCK_VALS; k++) {
        list_off[k] = total_entries;
        total_entries += list_sz[k];
    }
    uint32_t* pool = malloc((size_t)total_entries * sizeof(uint32_t));
    uint32_t* wp = malloc((size_t)n_blocks * N_BLOCK_VALS * sizeof(uint32_t));
    memcpy(wp, list_off, (size_t)n_blocks * N_BLOCK_VALS * sizeof(uint32_t));

    for (int i = 0; i < ds.n_train; i++) {
        const uint8_t* blk = train_blocks + (size_t)i * n_blocks;
        for (int b = 0; b < n_blocks; b++)
            if (blk[b] != BG_BLOCK)
                pool[wp[(size_t)b * N_BLOCK_VALS + blk[b]]++] = (uint32_t)i;
    }
    free(wp);
    printf("  Index: %u entries (%.1f MB)\n",
           total_entries, (double)total_entries * 4 / (1024*1024));

    /* ============================================================
     * Compute per-block-position IG on 27-value block distributions.
     * ============================================================ */
    printf("Computing block-level IG...\n");

    /* hot[b][v][c] = count of class c images with block value v at pos b. */
    uint16_t* hot = calloc((size_t)n_blocks * N_BLOCK_VALS * N_CLASSES, sizeof(uint16_t));
    #define BHOT(b, v, c) hot[(size_t)(b) * N_BLOCK_VALS * N_CLASSES + (size_t)(v) * N_CLASSES + (c)]

    for (int i = 0; i < ds.n_train; i++) {
        int lbl = ds.y_train[i];
        const uint8_t* blk = train_blocks + (size_t)i * n_blocks;
        for (int b = 0; b < n_blocks; b++)
            BHOT(b, blk[b], lbl)++;
    }

    int cc[N_CLASSES] = {0};
    for (int i = 0; i < ds.n_train; i++) cc[ds.y_train[i]]++;
    double h_class = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)cc[c] / ds.n_train;
        if (p > 0) h_class -= p * log2(p);
    }

    uint16_t* big = malloc((size_t)n_blocks * sizeof(uint16_t));
    double max_ig = 0;
    for (int b = 0; b < n_blocks; b++) {
        double hc = 0;
        for (int v = 0; v < N_BLOCK_VALS; v++) {
            if (v == BG_BLOCK) continue;
            int vt = 0;
            for (int c = 0; c < N_CLASSES; c++) vt += BHOT(b, v, c);
            if (!vt) continue;
            double pv = (double)vt / ds.n_train, hv = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)BHOT(b, v, c) / vt;
                if (pc > 0) hv -= pc * log2(pc);
            }
            hc += pv * hv;
        }
        double igv = h_class - hc;
        if (igv < 0) igv = 0;
        if (igv > max_ig) max_ig = igv;
        big[b] = 0; /* placeholder, normalize after */
    }
    /* Recompute to store. */
    int nonzero_ig = 0;
    for (int b = 0; b < n_blocks; b++) {
        double hc = 0;
        for (int v = 0; v < N_BLOCK_VALS; v++) {
            if (v == BG_BLOCK) continue;
            int vt = 0;
            for (int c = 0; c < N_CLASSES; c++) vt += BHOT(b, v, c);
            if (!vt) continue;
            double pv = (double)vt / ds.n_train, hv = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)BHOT(b, v, c) / vt;
                if (pc > 0) hv -= pc * log2(pc);
            }
            hc += pv * hv;
        }
        double igv = h_class - hc;
        if (igv < 0) igv = 0;
        big[b] = max_ig > 0 ? (uint16_t)(igv / max_ig * 1000 + 1) : 1;
        if (igv > 0.001) nonzero_ig++;
    }
    printf("  H(class)=%.3f  max_ig=%.4f  IG>0.001: %d/%d\n",
           h_class, max_ig, nonzero_ig, n_blocks);

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Build: %.1fs\n\n", build_sec);

    /* ============================================================
     * Classify via inverted index lookup.
     * ============================================================ */
    printf("Classifying %d queries...\n", ds.n_test);
    clock_t t_sweep = clock();

    /* Per-candidate score accumulator. */
    uint32_t* cand_score = calloc((size_t)ds.n_train, sizeof(uint32_t));

    int correct_knn = 0, correct_direct = 0;
    int pc_t[N_CLASSES]={0}, pc_k[N_CLASSES]={0}, pc_d[N_CLASSES]={0};

    m4t_mtfp_t* qfeat = NULL;
    if (use_gradients) qfeat = malloc((size_t)total_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* hg = use_gradients ? malloc((size_t)hgrad_dim * sizeof(m4t_mtfp_t)) : NULL;
    m4t_mtfp_t* vg = use_gradients ? malloc((size_t)vgrad_dim * sizeof(m4t_mtfp_t)) : NULL;

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        if (y >= 0 && y < N_CLASSES) pc_t[y]++;

        /* Quantize query. */
        uint8_t* qsig = calloc(sig_bytes, 1);
        const m4t_mtfp_t* qx = ds.x_test + (size_t)qi * ds.input_dim;
        for (int d = 0; d < intensity_dim; d++) {
            int64_t v = (int64_t)qx[d];
            if (v > tau_i) glyph_write_trit(qsig, d, +1);
            else if (v < -tau_i) glyph_write_trit(qsig, d, -1);
        }
        if (use_gradients) {
            compute_gradients(qx, img_w, img_h, n_ch, hg, vg);
            for (int d = 0; d < hgrad_dim; d++)
                if (hg[d] > tau_g) glyph_write_trit(qsig, intensity_dim + d, +1);
                else if (hg[d] < -tau_g) glyph_write_trit(qsig, intensity_dim + d, -1);
            for (int d = 0; d < vgrad_dim; d++)
                if (vg[d] > tau_g) glyph_write_trit(qsig, intensity_dim + hgrad_dim + d, +1);
                else if (vg[d] < -tau_g) glyph_write_trit(qsig, intensity_dim + hgrad_dim + d, -1);
        }

        /* Block-encode query. */
        uint8_t qblk[16384];
        for (int b = 0; b < n_blocks; b++) {
            int pos = b * 3;
            int8_t t0b = (pos < total_dim) ? glyph_read_trit(qsig, pos) : 0;
            int8_t t1 = (pos+1 < total_dim) ? glyph_read_trit(qsig, pos+1) : 0;
            int8_t t2 = (pos+2 < total_dim) ? glyph_read_trit(qsig, pos+2) : 0;
            qblk[b] = encode_block(t0b, t1, t2);
        }
        free(qsig);

        /* Inverted index lookup: accumulate IG-weighted hits. */
        int n_touched = 0;
        int touched[65536];

        for (int b = 0; b < n_blocks; b++) {
            uint8_t bv = qblk[b];
            if (bv == BG_BLOCK) continue;
            uint32_t off = list_off[(size_t)b * N_BLOCK_VALS + bv];
            uint32_t sz = list_sz[(size_t)b * N_BLOCK_VALS + bv];
            uint16_t w = big[b];
            for (uint32_t j = 0; j < sz; j++) {
                uint32_t idx = pool[off + j];
                if (cand_score[idx] == 0 && n_touched < 65536)
                    touched[n_touched++] = (int)idx;
                cand_score[idx] += w;
            }
        }

        /* k-NN on top-K by score. */
        typedef struct { uint32_t s; int l; } tk_t;
        tk_t topk[64]; int ntk = 0;
        for (int ti = 0; ti < n_touched; ti++) {
            int idx = touched[ti];
            uint32_t sc = cand_score[idx];
            int lbl = ds.y_train[idx];
            if (ntk < KNN_K) {
                int pos = ntk;
                while (pos > 0 && topk[pos-1].s < sc) { topk[pos]=topk[pos-1]; pos--; }
                topk[pos].s = sc; topk[pos].l = lbl; ntk++;
            } else if (sc > topk[KNN_K-1].s) {
                int pos = KNN_K-1;
                while (pos > 0 && topk[pos-1].s < sc) { topk[pos]=topk[pos-1]; pos--; }
                topk[pos].s = sc; topk[pos].l = lbl;
            }
        }
        int cv[N_CLASSES] = {0};
        for (int i = 0; i < ntk; i++) cv[topk[i].l] += (KNN_K - i);
        int kpred = 0;
        for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[kpred]) kpred = c;
        if (kpred == y) { correct_knn++; if (y < N_CLASSES) pc_k[y]++; }

        /* Direct class scoring: Σ_b ig[b] × P(class | block[b], pos=b). */
        double class_sc[N_CLASSES] = {0};
        for (int b = 0; b < n_blocks; b++) {
            uint8_t bv = qblk[b];
            if (bv == BG_BLOCK) continue;
            int vt = 0;
            for (int c = 0; c < N_CLASSES; c++) vt += BHOT(b, bv, c);
            if (!vt) continue;
            for (int c = 0; c < N_CLASSES; c++)
                class_sc[c] += (double)big[b] * (double)BHOT(b, bv, c) / vt;
        }
        int dpred = 0;
        for (int c = 1; c < N_CLASSES; c++)
            if (class_sc[c] > class_sc[dpred]) dpred = c;
        if (dpred == y) { correct_direct++; if (y < N_CLASSES) pc_d[y]++; }

        /* Reset candidate scores. */
        for (int ti = 0; ti < n_touched; ti++) cand_score[touched[ti]] = 0;

        if ((qi + 1) % 1000 == 0)
            printf("  %d/%d  knn=%.2f%%  direct=%.2f%%\n",
                   qi + 1, ds.n_test,
                   100.0 * correct_knn / (qi + 1),
                   100.0 * correct_direct / (qi + 1));
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("\n=== Results ===\n");
    printf("  k=%d-NN (IG-inverted):  %6.2f%%\n", KNN_K, 100.0 * correct_knn / ds.n_test);
    printf("  Direct class scoring:   %6.2f%%\n", 100.0 * correct_direct / ds.n_test);
    printf("  Sweep: %.1fs\n\n", sweep_sec);

    printf("Per-class:\n");
    printf("  class  count   knn    direct\n");
    for (int c = 0; c < N_CLASSES; c++)
        if (pc_t[c])
            printf("   %2d   %5d  %5.1f%%  %5.1f%%\n",
                   c, pc_t[c], 100.0*pc_k[c]/pc_t[c], 100.0*pc_d[c]/pc_t[c]);

    /* Cleanup. */
    free(hg); free(vg); free(qfeat);
    free(cand_score);
    free(hot); free(big);
    free(pool); free(list_off); free(list_sz);
    free(train_blocks); free(train_sigs);
    glyph_dataset_free(&ds);
    return 0;
}
