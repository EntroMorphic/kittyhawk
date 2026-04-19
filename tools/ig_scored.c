/*
 * ig_scored.c — IG-weighted scoring on direct ternary signatures.
 *
 * Uses the SAME direct-quantized trit signatures as direct_lsh.c
 * (normalized intensity + float gradients, density-calibrated tau).
 * Changes ONLY the scoring: instead of uniform Hamming distance,
 * scores by information-gain-weighted per-position contribution.
 *
 * For each trit position d and each value v ∈ {-1,0,+1}, precompute
 * the class distribution P(class | pos=d, val=v) from training data.
 * The IG weight of position d is the entropy reduction from knowing
 * the value at d. At query time, accumulate IG-weighted class scores
 * from each non-zero trit position.
 *
 * This is SSTT's inverted-index scoring principle applied to Glyph's
 * direct-quantized signatures. Tests whether the scoring mechanism
 * (not the features) explains the 8.3pp gap to SSTT's 53%.
 *
 * NO RANDOM PROJECTIONS. Same trit signatures as direct_lsh.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_rng.h"
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

    printf("ig_scored: IG-weighted scoring on direct trit signatures\n");
    printf("  data=%s  deskew=%s  gradients=%s  density=%.3f\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on",
           use_gradients ? "on" : "off", cfg.density);
    printf("  total_dim=%d  sig_bytes=%d\n", total_dim, sig_bytes);
    printf("  n_train=%d  n_test=%d\n\n", ds.n_train, ds.n_test);

    clock_t t0 = clock();

    /* Build feature vectors + quantize (same as direct_lsh). */
    printf("Building and quantizing signatures...\n");
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;

    /* For non-gradient case, features = raw pixels. */
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
        free(hg); free(vg);
    }

    /* Calibrate tau. */
    const m4t_mtfp_t* calib_src = use_gradients ? train_feat : ds.x_train;
    m4t_mtfp_t* int_sample = malloc((size_t)n_calib * intensity_dim * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_calib; i++)
        memcpy(int_sample + (size_t)i * intensity_dim,
               calib_src + (size_t)i * (use_gradients ? total_dim : intensity_dim),
               (size_t)intensity_dim * sizeof(m4t_mtfp_t));
    int64_t tau_i = glyph_sig_quantize_tau(int_sample, n_calib, intensity_dim, cfg.density);
    free(int_sample);

    int64_t tau_g = 0;
    if (use_gradients) {
        int gd = hgrad_dim + vgrad_dim;
        m4t_mtfp_t* gs = malloc((size_t)n_calib * gd * sizeof(m4t_mtfp_t));
        for (int i = 0; i < n_calib; i++)
            memcpy(gs + (size_t)i * gd,
                   train_feat + (size_t)i * total_dim + intensity_dim,
                   (size_t)gd * sizeof(m4t_mtfp_t));
        tau_g = glyph_sig_quantize_tau(gs, n_calib, gd, 0.10);
        free(gs);
    }
    printf("  tau_i=%lld (%.3f×S)  tau_g=%lld (%.3f×S)\n",
           (long long)tau_i, (double)tau_i/M4T_MTFP_SCALE,
           (long long)tau_g, (double)tau_g/M4T_MTFP_SCALE);

    /* Quantize training signatures. */
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
     * Build IG weights: for each position d, measure how much
     * knowing the trit value reduces class entropy.
     *
     * hot[d][v][c] = count of training images where position d has
     * value v and label is c. v: 0=neg, 1=zero, 2=pos.
     * ============================================================ */
    printf("Computing IG weights (%d positions)...\n", total_dim);

    /* Use uint16 counts to save memory (max 50K per cell). */
    uint16_t* hot = calloc((size_t)total_dim * 3 * N_CLASSES, sizeof(uint16_t));
    #define HOT(d, v, c) hot[(size_t)(d) * 3 * N_CLASSES + (size_t)(v) * N_CLASSES + (c)]

    for (int i = 0; i < ds.n_train; i++) {
        int lbl = ds.y_train[i];
        if (lbl < 0 || lbl >= N_CLASSES) continue;
        const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        for (int d = 0; d < total_dim; d++) {
            int8_t t = glyph_read_trit(sig, d);
            int v = (t < 0) ? 0 : (t == 0) ? 1 : 2;
            HOT(d, v, lbl)++;
        }
    }

    /* Compute per-position IG. */
    int cc[N_CLASSES] = {0};
    for (int i = 0; i < ds.n_train; i++) cc[ds.y_train[i]]++;
    double h_class = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)cc[c] / ds.n_train;
        if (p > 0) h_class -= p * log2(p);
    }

    double* ig = malloc((size_t)total_dim * sizeof(double));
    double max_ig = 0;
    for (int d = 0; d < total_dim; d++) {
        double h_cond = 0;
        for (int v = 0; v < 3; v++) {
            int vt = 0;
            for (int c = 0; c < N_CLASSES; c++) vt += HOT(d, v, c);
            if (!vt) continue;
            double pv = (double)vt / ds.n_train;
            double hv = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)HOT(d, v, c) / vt;
                if (pc > 0) hv -= pc * log2(pc);
            }
            h_cond += pv * hv;
        }
        ig[d] = h_class - h_cond;
        if (ig[d] > max_ig) max_ig = ig[d];
    }

    /* Normalize IG to integer weights [1, 16]. */
    uint8_t* ig_w = malloc((size_t)total_dim);
    int nonzero_ig = 0;
    for (int d = 0; d < total_dim; d++) {
        ig_w[d] = max_ig > 0 ? (uint8_t)(ig[d] / max_ig * 15.0 + 1.0) : 1;
        if (ig[d] > 0.001) nonzero_ig++;
    }
    printf("  H(class)=%.3f  max_ig=%.4f  positions with IG>0.001: %d/%d\n",
           h_class, max_ig, nonzero_ig, total_dim);

    /* IG stats by channel. */
    if (use_gradients) {
        double sum_i = 0, sum_g = 0;
        for (int d = 0; d < intensity_dim; d++) sum_i += ig[d];
        for (int d = intensity_dim; d < total_dim; d++) sum_g += ig[d];
        printf("  Avg IG: intensity=%.5f  gradient=%.5f  ratio=%.1fx\n",
               sum_i / intensity_dim, sum_g / (total_dim - intensity_dim),
               intensity_dim > 0 && (total_dim - intensity_dim) > 0
               ? (sum_i / intensity_dim) / (sum_g / (total_dim - intensity_dim)) : 0);
    }

    double build_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Build: %.1fs\n\n", build_sec);

    /* ============================================================
     * Classify: brute-force with IG-weighted scoring.
     *
     * Three scoring modes compared:
     *   1. Uniform Hamming (baseline, same as direct_lsh)
     *   2. IG-weighted Hamming: Σ_d ig_w[d] × (trit_q[d] != trit_t[d])
     *   3. Bayesian: Π_d P(class | pos=d, val=query[d])
     * ============================================================ */
    /* ============================================================
     * Item 5+8: Per-class-PAIR IG weights.
     * For each (class_a, class_b) pair, compute IG restricted to
     * training images of those two classes only. This produces
     * pair-specific weights where position d's weight reflects how
     * well it distinguishes class_a from class_b specifically.
     * ============================================================ */
    printf("Computing per-pair IG weights (%d pairs)...\n", N_CLASSES * (N_CLASSES - 1) / 2);

    /* pair_ig_w[a][b][d] = IG weight for position d when distinguishing a vs b.
     * Stored as uint8 [1..16]. Symmetric: pair_ig_w[a][b] == pair_ig_w[b][a]. */
    uint8_t** pair_ig_w = malloc((size_t)N_CLASSES * N_CLASSES * sizeof(uint8_t*));
    for (int a = 0; a < N_CLASSES; a++) {
        for (int b = a + 1; b < N_CLASSES; b++) {
            uint8_t* pw = malloc((size_t)total_dim);
            int n_ab = cc[a] + cc[b];
            double h_ab = 0;
            { double pa = (double)cc[a]/n_ab, pb = (double)cc[b]/n_ab;
              if (pa > 0) h_ab -= pa * log2(pa);
              if (pb > 0) h_ab -= pb * log2(pb); }
            double pmx = 0;
            for (int d = 0; d < total_dim; d++) {
                double hc = 0;
                for (int v = 0; v < 3; v++) {
                    int va = HOT(d, v, a), vb = HOT(d, v, b);
                    int vt = va + vb;
                    if (!vt) continue;
                    double pv = (double)vt / n_ab;
                    double ha = (double)va / vt, hb = (double)vb / vt;
                    double hv = 0;
                    if (ha > 0) hv -= ha * log2(ha);
                    if (hb > 0) hv -= hb * log2(hb);
                    hc += pv * hv;
                }
                double pig = h_ab - hc;
                if (pig < 0) pig = 0;
                if (pig > pmx) pmx = pig;
                /* Store raw in pw temporarily as scaled int. */
                pw[d] = 0; /* placeholder */
                /* Re-store after we know pmx. For now just track. */
                /* Actually need two passes. Use ig array. */
                ig[d] = pig;
            }
            for (int d = 0; d < total_dim; d++)
                pw[d] = pmx > 0 ? (uint8_t)(ig[d] / pmx * 15.0 + 1.0) : 1;
            pair_ig_w[a * N_CLASSES + b] = pw;
            pair_ig_w[b * N_CLASSES + a] = pw; /* symmetric alias */
        }
        pair_ig_w[a * N_CLASSES + a] = ig_w; /* self-pair uses global */
    }

    printf("Classifying %d queries (brute-force)...\n", ds.n_test);
    clock_t t_sweep = clock();

    int correct_hamming = 0, correct_ig = 0, correct_bayes = 0;
    int correct_pair_ig = 0;
    int top100_oracle = 0;
    int pc_t[N_CLASSES]={0}, pc_h[N_CLASSES]={0}, pc_i[N_CLASSES]={0}, pc_b[N_CLASSES]={0};
    int pc_p[N_CLASSES] = {0};

    for (int qi = 0; qi < ds.n_test; qi++) {
        int y = ds.y_test[qi];
        if (y >= 0 && y < N_CLASSES) pc_t[y]++;

        /* Quantize test query. */
        const m4t_mtfp_t* qx = ds.x_test + (size_t)qi * ds.input_dim;
        uint8_t* qsig = calloc(sig_bytes, 1);
        for (int d = 0; d < intensity_dim; d++) {
            int64_t v = (int64_t)qx[d];
            if (v > tau_i) glyph_write_trit(qsig, d, +1);
            else if (v < -tau_i) glyph_write_trit(qsig, d, -1);
        }
        if (use_gradients) {
            m4t_mtfp_t* hg = malloc((size_t)hgrad_dim * sizeof(m4t_mtfp_t));
            m4t_mtfp_t* vg = malloc((size_t)vgrad_dim * sizeof(m4t_mtfp_t));
            compute_gradients(qx, img_w, img_h, n_ch, hg, vg);
            for (int d = 0; d < hgrad_dim; d++) {
                if (hg[d] > tau_g) glyph_write_trit(qsig, intensity_dim + d, +1);
                else if (hg[d] < -tau_g) glyph_write_trit(qsig, intensity_dim + d, -1);
            }
            for (int d = 0; d < vgrad_dim; d++) {
                if (vg[d] > tau_g) glyph_write_trit(qsig, intensity_dim + hgrad_dim + d, +1);
                else if (vg[d] < -tau_g) glyph_write_trit(qsig, intensity_dim + hgrad_dim + d, -1);
            }
            free(hg); free(vg);
        }

        /* Step 1: Hamming brute-force to find top-100 candidates. */
        #define TOP_N 100
        typedef struct { int32_t s; int idx; } cand_t;
        cand_t cands[TOP_N];
        int ncand = 0;
        uint8_t* mask = malloc(sig_bytes); memset(mask, 0xFF, sig_bytes);

        for (int ti = 0; ti < ds.n_train; ti++) {
            int32_t dh = m4t_popcount_dist(qsig,
                train_sigs + (size_t)ti * sig_bytes, mask, sig_bytes);
            if (ncand < TOP_N) {
                int pos = ncand;
                while (pos > 0 && cands[pos-1].s > dh) { cands[pos]=cands[pos-1]; pos--; }
                cands[pos].s = dh; cands[pos].idx = ti; ncand++;
            } else if (dh < cands[TOP_N-1].s) {
                int pos = TOP_N-1;
                while (pos > 0 && cands[pos-1].s > dh) { cands[pos]=cands[pos-1]; pos--; }
                cands[pos].s = dh; cands[pos].idx = ti;
            }
        }
        free(mask);

        /* Hamming k-NN from top-5 of the 100 candidates. */
        {
            int cv[N_CLASSES]={0};
            for (int i = 0; i < KNN_K && i < ncand; i++)
                cv[ds.y_train[cands[i].idx]] += (KNN_K - i);
            int pred = 0;
            for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[pred]) pred = c;
            if (pred == y) { correct_hamming++; if (y < N_CLASSES) pc_h[y]++; }
        }

        /* Step 2: IG-weighted re-rank of top-100 candidates. */
        {
            typedef struct { int32_t s; int l; } tk_t;
            tk_t topk[64]; int ntk = 0;
            for (int ci = 0; ci < ncand; ci++) {
                int ti = cands[ci].idx;
                const uint8_t* tsig = train_sigs + (size_t)ti * sig_bytes;
                int32_t dig = 0;
                for (int d = 0; d < total_dim; d++) {
                    int8_t a = glyph_read_trit(qsig, d);
                    int8_t b = glyph_read_trit(tsig, d);
                    if (a != b) dig += ig_w[d];
                }
                int lbl = ds.y_train[ti];
                if (ntk < KNN_K) {
                    int pos = ntk;
                    while (pos > 0 && topk[pos-1].s > dig) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = dig; topk[pos].l = lbl; ntk++;
                } else if (dig < topk[KNN_K-1].s) {
                    int pos = KNN_K-1;
                    while (pos > 0 && topk[pos-1].s > dig) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = dig; topk[pos].l = lbl;
                }
            }
            int cv[N_CLASSES]={0};
            for (int i = 0; i < ntk; i++) cv[topk[i].l] += (KNN_K - i);
            int pred = 0;
            for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[pred]) pred = c;
            if (pred == y) { correct_ig++; if (y < N_CLASSES) pc_i[y]++; }
        }

        /* Item 1: top-100 oracle. */
        {
            int found = 0;
            for (int ci = 0; ci < ncand; ci++)
                if (ds.y_train[cands[ci].idx] == y) { found = 1; break; }
            if (found) top100_oracle++;
        }

        /* Step 3: Bayesian re-rank of top-100 (item 6 — fixed to per-class). */
        {
            double log_p[N_CLASSES];
            memset(log_p, 0, sizeof(log_p));
            for (int d = 0; d < total_dim; d++) {
                int8_t qt = glyph_read_trit(qsig, d);
                if (qt == 0) continue;
                int qv = (qt < 0) ? 0 : 2;
                for (int c = 0; c < N_CLASSES; c++)
                    log_p[c] += log((double)(HOT(d, qv, c) + 0.5) /
                                    (cc[c] + 1.5));
            }
            int bayes_pred = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (log_p[c] > log_p[bayes_pred]) bayes_pred = c;
            if (bayes_pred == y) { correct_bayes++; if (y < N_CLASSES) pc_b[y]++; }
        }

        /* Step 4: class-pair IG re-rank (items 5+8).
         * Use Hamming top-2 to identify the likely confusion pair.
         * Re-rank the top-100 candidates with pair-specific IG. */
        {
            /* Identify the top-2 classes from Hamming k=5. */
            int cv_tmp[N_CLASSES] = {0};
            for (int i = 0; i < KNN_K && i < ncand; i++)
                cv_tmp[ds.y_train[cands[i].idx]]++;
            int c1 = 0, c2 = -1;
            for (int c = 1; c < N_CLASSES; c++)
                if (cv_tmp[c] > cv_tmp[c1]) c1 = c;
            for (int c = 0; c < N_CLASSES; c++) {
                if (c == c1) continue;
                if (c2 < 0 || cv_tmp[c] > cv_tmp[c2]) c2 = c;
            }
            if (c2 < 0) c2 = (c1 + 1) % N_CLASSES;

            const uint8_t* pw = pair_ig_w[c1 * N_CLASSES + c2];
            typedef struct { int32_t s; int l; } tk_t;
            tk_t topk[64]; int ntk = 0;
            for (int ci = 0; ci < ncand; ci++) {
                int ti = cands[ci].idx;
                const uint8_t* tsig = train_sigs + (size_t)ti * sig_bytes;
                int32_t dig = 0;
                for (int d = 0; d < total_dim; d++) {
                    int8_t a = glyph_read_trit(qsig, d);
                    int8_t b = glyph_read_trit(tsig, d);
                    if (a != b) dig += pw[d];
                }
                int lbl = ds.y_train[ti];
                if (ntk < KNN_K) {
                    int pos = ntk;
                    while (pos > 0 && topk[pos-1].s > dig) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = dig; topk[pos].l = lbl; ntk++;
                } else if (dig < topk[KNN_K-1].s) {
                    int pos = KNN_K-1;
                    while (pos > 0 && topk[pos-1].s > dig) { topk[pos]=topk[pos-1]; pos--; }
                    topk[pos].s = dig; topk[pos].l = lbl;
                }
            }
            int cv[N_CLASSES] = {0};
            for (int i = 0; i < ntk; i++) cv[topk[i].l] += (KNN_K - i);
            int pred = 0;
            for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[pred]) pred = c;
            if (pred == y) { correct_pair_ig++; if (y < N_CLASSES) pc_p[y]++; }
        }

        free(qsig);

        if ((qi + 1) % 1000 == 0)
            printf("  %d/%d  hamming=%.2f%%  ig=%.2f%%  pair=%.2f%%  bayes=%.2f%%\n",
                   qi + 1, ds.n_test,
                   100.0 * correct_hamming / (qi + 1),
                   100.0 * correct_ig / (qi + 1),
                   100.0 * correct_pair_ig / (qi + 1),
                   100.0 * correct_bayes / (qi + 1));
    }
    double sweep_sec = (double)(clock() - t_sweep) / CLOCKS_PER_SEC;

    printf("\n=== Results (brute-force, k=%d rank-weighted) ===\n", KNN_K);
    printf("  Top-100 oracle:      %6.2f%%\n", 100.0 * top100_oracle / ds.n_test);
    printf("  Uniform Hamming:     %6.2f%%\n", 100.0 * correct_hamming / ds.n_test);
    printf("  IG-weighted Hamming: %6.2f%%\n", 100.0 * correct_ig / ds.n_test);
    printf("  Pair-IG re-rank:     %6.2f%%\n", 100.0 * correct_pair_ig / ds.n_test);
    printf("  Bayesian (direct):   %6.2f%%\n", 100.0 * correct_bayes / ds.n_test);
    printf("  Sweep: %.1fs\n\n", sweep_sec);

    printf("Per-class:\n");
    printf("  class  count  hamming   ig-wt   pair-ig   bayes\n");
    for (int c = 0; c < N_CLASSES; c++)
        if (pc_t[c])
            printf("   %2d   %5d   %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%\n",
                   c, pc_t[c],
                   100.0*pc_h[c]/pc_t[c],
                   100.0*pc_i[c]/pc_t[c],
                   100.0*pc_p[c]/pc_t[c],
                   100.0*pc_b[c]/pc_t[c]);

    /* Cleanup. */
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = a + 1; b < N_CLASSES; b++)
            free(pair_ig_w[a * N_CLASSES + b]);
    free(pair_ig_w);
    free(hot); free(ig); free(ig_w);
    free(train_sigs);
    glyph_dataset_free(&ds);
    return 0;
}
