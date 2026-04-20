/*
 * block_distance.c — brute-force ceiling measurement for block-encoded
 * pattern distance vs per-trit Hamming.
 *
 * Groups consecutive trits into 3-trit blocks (27 symbols each).
 * Computes per-block-position IG from training data. Scores by
 * block-level IG-weighted distance. Compares to per-trit Hamming
 * and per-trit IG as baselines.
 *
 * This is a brute-force measurement (O(N_train) per query) to
 * establish whether block-level correlations carry classification
 * signal beyond what per-trit distance captures. If the ceiling
 * doesn't beat Hamming, there's no point integrating into LSH.
 *
 * NO RANDOM PROJECTIONS. Direct ternary quantization only.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_sig.h"
#include "glyph_multiprobe.h"
#include "m4t_trit_pack.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10
#define BLOCK_K 3
#define N_SYMBOLS 27   /* 3^BLOCK_K */
#define KNN_K 5

static inline int trit_to_sym3(int8_t t0, int8_t t1, int8_t t2) {
    return (t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1);
}

typedef struct { int32_t score; int label; } tk_t;

static inline void tk_insert(tk_t* tk, int* n, int k, int32_t s, int l) {
    if (*n < k) {
        int p = *n;
        while (p > 0 && tk[p-1].score > s) { tk[p] = tk[p-1]; p--; }
        tk[p].score = s; tk[p].label = l; (*n)++;
    } else if (s < tk[k-1].score) {
        int p = k - 1;
        while (p > 0 && tk[p-1].score > s) { tk[p] = tk[p-1]; p--; }
        tk[p].score = s; tk[p].label = l;
    }
}

static int tk_vote(const tk_t* tk, int n, int k) {
    int cv[N_CLASSES] = {0};
    for (int i = 0; i < n; i++) cv[tk[i].label] += (k - i);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[best]) best = c;
    return best;
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
    if (cfg.normalize) glyph_dataset_normalize(&ds);

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);

    int intensity_dim = ds.input_dim;
    int hgrad_dim = n_ch * img_h * (img_w - 1);
    int vgrad_dim = n_ch * (img_h - 1) * img_w;
    int total_dim = intensity_dim + (use_gradients ? (hgrad_dim + vgrad_dim) : 0);
    int sig_bytes = M4T_TRIT_PACKED_BYTES(total_dim);

    printf("block_distance: brute-force block-IG vs Hamming ceiling\n");
    printf("  data=%s  gradients=%s  total_dim=%d  block_k=%d\n",
           cfg.data_dir, use_gradients ? "on" : "off", total_dim, BLOCK_K);

    /* Build feature vectors + quantize (same as direct_lsh). */
    m4t_mtfp_t* train_feat = malloc((size_t)ds.n_train * total_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_feat  = malloc((size_t)ds.n_test  * total_dim * sizeof(m4t_mtfp_t));
    if (!train_feat || !test_feat) { fprintf(stderr, "OOM\n"); return 1; }

    for (int pass = 0; pass < 2; pass++) {
        int n_imgs = (pass == 0) ? ds.n_train : ds.n_test;
        const m4t_mtfp_t* src = (pass == 0) ? ds.x_train : ds.x_test;
        m4t_mtfp_t* feat = (pass == 0) ? train_feat : test_feat;
        for (int i = 0; i < n_imgs; i++) {
            memcpy(feat + (size_t)i * total_dim, src + (size_t)i * intensity_dim,
                   (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            if (use_gradients) {
                m4t_mtfp_t* hg = feat + (size_t)i * total_dim + intensity_dim;
                m4t_mtfp_t* vg = hg + hgrad_dim;
                glyph_dataset_gradients(src + (size_t)i * intensity_dim,
                                        img_w, img_h, n_ch, hg, vg);
            }
        }
    }

    /* Calibrate tau. */
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    int64_t tau_intensity = glyph_sig_quantize_tau(
        train_feat, n_calib, total_dim, cfg.density);
    int64_t tau_gradient = 0;
    if (use_gradients) {
        m4t_mtfp_t* grad_sample = malloc((size_t)n_calib * (hgrad_dim + vgrad_dim) * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_calib; i++)
            memcpy(grad_sample + (size_t)i * (hgrad_dim + vgrad_dim),
                   train_feat + (size_t)i * total_dim + intensity_dim,
                   (size_t)(hgrad_dim + vgrad_dim) * sizeof(m4t_mtfp_t));
        tau_gradient = glyph_sig_quantize_tau(
            grad_sample, n_calib, hgrad_dim + vgrad_dim, 0.10);
        free(grad_sample);
    }

    /* Quantize. */
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
                    if (v > tau_gradient) glyph_write_trit(sig, intensity_dim + d, +1);
                    else if (v < -tau_gradient) glyph_write_trit(sig, intensity_dim + d, -1);
                }
            }
        }
    }
    free(train_feat); free(test_feat);

    int n_blocks = total_dim / BLOCK_K;
    printf("  n_blocks=%d (trits %d..%d used, %d trailing trits unused)\n",
           n_blocks, 0, n_blocks * BLOCK_K - 1, total_dim - n_blocks * BLOCK_K);

    /* Compute per-block-position, per-class symbol histograms. */
    printf("Computing block-level IG weights...\n");
    uint16_t* blk_hist = calloc((size_t)n_blocks * N_SYMBOLS * N_CLASSES, sizeof(uint16_t));
    #define BH(b, sym, c) blk_hist[(size_t)(b)*N_SYMBOLS*N_CLASSES + (size_t)(sym)*N_CLASSES + (c)]
    for (int i = 0; i < ds.n_train; i++) {
        int lbl = ds.y_train[i];
        const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        for (int b = 0; b < n_blocks; b++) {
            int base = b * BLOCK_K;
            int sym = trit_to_sym3(
                glyph_read_trit(sig, base),
                glyph_read_trit(sig, base + 1),
                glyph_read_trit(sig, base + 2));
            BH(b, sym, lbl)++;
        }
    }

    /* Per-class-pair block-IG weights. For each pair (a,b) and block
     * position, compute IG of the 27-symbol distribution restricted
     * to classes a and b. Quantize to uint8 [1..16]. */
    int ig_cc[N_CLASSES] = {0};
    for (int i = 0; i < ds.n_train; i++) ig_cc[ds.y_train[i]]++;

    int n_pairs = N_CLASSES * (N_CLASSES - 1) / 2;
    uint8_t** pair_blk_ig = malloc((size_t)N_CLASSES * N_CLASSES * sizeof(uint8_t*));
    double* ig_tmp = malloc((size_t)n_blocks * sizeof(double));
    for (int a = 0; a < N_CLASSES; a++) {
        for (int b = a + 1; b < N_CLASSES; b++) {
            uint8_t* pw = malloc((size_t)n_blocks);
            int n_ab = ig_cc[a] + ig_cc[b];
            double pa = (double)ig_cc[a] / n_ab;
            double pb = (double)ig_cc[b] / n_ab;
            double h_ab = 0;
            if (pa > 0) h_ab -= pa * log2(pa);
            if (pb > 0) h_ab -= pb * log2(pb);
            double pmx = 0;
            for (int bl = 0; bl < n_blocks; bl++) {
                double hc = 0;
                for (int sym = 0; sym < N_SYMBOLS; sym++) {
                    int va = BH(bl, sym, a);
                    int vb = BH(bl, sym, b);
                    int vt = va + vb;
                    if (!vt) continue;
                    double pv = (double)vt / n_ab;
                    double ha = (double)va / vt;
                    double hb = (double)vb / vt;
                    double hv = 0;
                    if (ha > 0) hv -= ha * log2(ha);
                    if (hb > 0) hv -= hb * log2(hb);
                    hc += pv * hv;
                }
                ig_tmp[bl] = h_ab - hc;
                if (ig_tmp[bl] < 0) ig_tmp[bl] = 0;
                if (ig_tmp[bl] > pmx) pmx = ig_tmp[bl];
            }
            for (int bl = 0; bl < n_blocks; bl++)
                pw[bl] = pmx > 0 ? (uint8_t)(ig_tmp[bl] / pmx * 15.0 + 1.0) : 1;
            pair_blk_ig[a * N_CLASSES + b] = pw;
            pair_blk_ig[b * N_CLASSES + a] = pw;
        }
        pair_blk_ig[a * N_CLASSES + a] = NULL;
    }
    free(ig_tmp); free(blk_hist);
    printf("  Block-IG weights computed for %d pairs × %d blocks.\n", n_pairs, n_blocks);

    /* Brute-force classification: for each test query, score all
     * training images by three metrics and compare. */
    printf("Brute-force scoring %d test × %d train...\n", ds.n_test, ds.n_train);

    uint8_t* full_mask = malloc(sig_bytes);
    memset(full_mask, 0xFF, sig_bytes);

    int hamming_correct = 0, hamming_knn_correct = 0;
    int blk_match_correct = 0, blk_match_knn_correct = 0;
    int blk_ig_correct = 0, blk_ig_knn_correct = 0;
    int combined_knn_correct = 0;

    clock_t t0 = clock();
    for (int qi = 0; qi < ds.n_test; qi++) {
        if (qi % 1000 == 0 && qi > 0)
            printf("  %d/%d (%.1fs)...\n", qi, ds.n_test,
                   (double)(clock() - t0) / CLOCKS_PER_SEC);

        int y = ds.y_test[qi];
        const uint8_t* qs = test_sigs + (size_t)qi * sig_bytes;

        int32_t best_ham = INT32_MAX, best_ham_lbl = -1;
        int32_t best_blk = INT32_MAX, best_blk_lbl = -1;
        tk_t tk_ham[64]; int nk_ham = 0;
        tk_t tk_blk[64]; int nk_blk = 0;
        tk_t tk_big[64]; int nk_big = 0;
        tk_t tk_comb[64]; int nk_comb = 0;

        for (int ti = 0; ti < ds.n_train; ti++) {
            const uint8_t* ts = train_sigs + (size_t)ti * sig_bytes;
            int lbl = ds.y_train[ti];

            /* Metric 1: per-trit Hamming distance. */
            int32_t d_ham = m4t_popcount_dist(qs, ts, full_mask, sig_bytes);
            if (d_ham < best_ham) { best_ham = d_ham; best_ham_lbl = lbl; }
            tk_insert(tk_ham, &nk_ham, KNN_K, d_ham, lbl);

            /* Metric 2: block-match distance (count mismatching blocks). */
            int32_t d_blk = 0;
            for (int b = 0; b < n_blocks; b++) {
                int base = b * BLOCK_K;
                int sq = trit_to_sym3(
                    glyph_read_trit(qs, base),
                    glyph_read_trit(qs, base + 1),
                    glyph_read_trit(qs, base + 2));
                int st = trit_to_sym3(
                    glyph_read_trit(ts, base),
                    glyph_read_trit(ts, base + 1),
                    glyph_read_trit(ts, base + 2));
                if (sq != st) d_blk++;
            }
            if (d_blk < best_blk) { best_blk = d_blk; best_blk_lbl = lbl; }
            tk_insert(tk_blk, &nk_blk, KNN_K, d_blk, lbl);

            /* Metric 3: block-IG weighted distance.
             * Use the top-2 Hamming classes to select pair weights. */
            /* Deferred to k-NN pass below — need top-2 first. */

            /* Metric 4: combined = Hamming + block_match (normalized). */
            int32_t d_comb = d_ham + d_blk * (int32_t)(2 * BLOCK_K);
            tk_insert(tk_comb, &nk_comb, KNN_K, d_comb, lbl);
        }

        if (best_ham_lbl == y) hamming_correct++;
        hamming_knn_correct += (tk_vote(tk_ham, nk_ham, KNN_K) == y);
        if (best_blk_lbl == y) blk_match_correct++;
        blk_match_knn_correct += (tk_vote(tk_blk, nk_blk, KNN_K) == y);
        combined_knn_correct += (tk_vote(tk_comb, nk_comb, KNN_K) == y);

        /* Block-IG re-rank: identify top-2 from Hamming k-NN, then
         * re-score all training images with pair-specific block IG. */
        int c1 = tk_vote(tk_ham, nk_ham, KNN_K);
        int cv2[N_CLASSES] = {0};
        for (int i = 0; i < nk_ham; i++) cv2[tk_ham[i].label]++;
        int c2 = -1;
        for (int c = 0; c < N_CLASSES; c++)
            if (c != c1 && (c2 < 0 || cv2[c] > cv2[c2])) c2 = c;
        if (c2 < 0) c2 = (c1 + 1) % N_CLASSES;
        const uint8_t* pw = pair_blk_ig[c1 * N_CLASSES + c2];

        if (pw) {
            int32_t best_big = INT32_MAX, best_big_lbl = -1;
            nk_big = 0;
            for (int ti = 0; ti < ds.n_train; ti++) {
                const uint8_t* ts = train_sigs + (size_t)ti * sig_bytes;
                int lbl = ds.y_train[ti];
                int32_t d = 0;
                for (int b = 0; b < n_blocks; b++) {
                    int base = b * BLOCK_K;
                    int sq = trit_to_sym3(
                        glyph_read_trit(qs, base),
                        glyph_read_trit(qs, base + 1),
                        glyph_read_trit(qs, base + 2));
                    int st_sym = trit_to_sym3(
                        glyph_read_trit(ts, base),
                        glyph_read_trit(ts, base + 1),
                        glyph_read_trit(ts, base + 2));
                    if (sq != st_sym) d += pw[b];
                }
                if (d < best_big) { best_big = d; best_big_lbl = lbl; }
                tk_insert(tk_big, &nk_big, KNN_K, d, lbl);
            }
            if (best_big_lbl == y) blk_ig_correct++;
            blk_ig_knn_correct += (tk_vote(tk_big, nk_big, KNN_K) == y);
        }
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("\nResults (brute-force, %d test × %d train, %.1fs):\n",
           ds.n_test, ds.n_train, elapsed);
    printf("  Metric                  1-NN      k=%d-rw\n", KNN_K);
    printf("  Per-trit Hamming       %6.2f%%   %6.2f%%\n",
           100.0 * hamming_correct / ds.n_test,
           100.0 * hamming_knn_correct / ds.n_test);
    printf("  Block-match (k=3)      %6.2f%%   %6.2f%%\n",
           100.0 * blk_match_correct / ds.n_test,
           100.0 * blk_match_knn_correct / ds.n_test);
    printf("  Block-IG (pair, k=3)   %6.2f%%   %6.2f%%\n",
           100.0 * blk_ig_correct / ds.n_test,
           100.0 * blk_ig_knn_correct / ds.n_test);
    printf("  Combined (Ham+Block)   %6s   %6.2f%%\n",
           "—", 100.0 * combined_knn_correct / ds.n_test);

    double ham_pct = 100.0 * hamming_knn_correct / ds.n_test;
    double big_pct = 100.0 * blk_ig_knn_correct / ds.n_test;
    printf("\n  Block-IG vs Hamming k-NN: %+.2fpp\n", big_pct - ham_pct);
    printf("  Verdict: %s\n",
           big_pct > ham_pct + 0.5
               ? "BLOCK DISTANCE CARRIES INDEPENDENT SIGNAL"
               : big_pct > ham_pct
                   ? "Marginal improvement"
                   : "Block distance does not help");

    /* Cleanup. */
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = a + 1; b < N_CLASSES; b++)
            free(pair_blk_ig[a * N_CLASSES + b]);
    free(pair_blk_ig);
    free(full_mask);
    free(train_sigs); free(test_sigs);
    glyph_dataset_free(&ds);
    return 0;
}
