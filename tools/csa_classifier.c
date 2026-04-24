/*
 * csa_classifier.c — Class-Signature Argmax classifier.
 *
 * One ternary prototype W_c ∈ {-1, 0, +1}^D per class, scored against
 * the query's direct-quantized signature via m4t_popcount_dist.
 * Prediction: argmin_c Hamming(sig, W_c).
 *
 * Training (E1): integer class centroid + sign-threshold. For each
 * class c, accumulate per-dim sum of training-sample trits; sign the
 * result to produce W_c. Pure integer arithmetic, one pass, one-shot.
 *
 * Training (E2, --train_mode perceptron): perceptron-style integer
 * updates on top of the centroid init. For each misclassified training
 * sample, increment the correct class's centroid by the sample's trits
 * and decrement the predicted class's centroid. Re-sign at epoch end.
 *
 * No random projections. No random weights. No binary float at runtime.
 * Weights are data-derived and carry a legible derivation story
 * (centroid majority direction ± margin). See journal/lr_scaffold_*.md
 * for the LMM cycle that motivated this tool.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_sig.h"
#include "glyph_multiprobe.h"
#include "m4t_trit_pack.h"
#include <limits.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10

static int8_t sign_trit(int32_t v, int32_t tau) {
    if (v >  tau) return  1;
    if (v < -tau) return -1;
    return 0;
}

/* Accumulate per-class, per-dim centroid sums from packed-trit
 * training signatures. Sign-threshold with per-class margin (median
 * absolute centroid value × frac). */
static void build_centroid_prototypes(
    const uint8_t* train_sigs, const int* y_train,
    int n_train, int total_dim, int sig_bytes,
    int32_t margin_frac_num, int32_t margin_frac_den,
    uint8_t* prototypes_packed)
{
    int32_t* cents = calloc((size_t)N_CLASSES * total_dim, sizeof(int32_t));
    int class_count[N_CLASSES] = {0};

    for (int i = 0; i < n_train; i++) {
        int lbl = y_train[i];
        class_count[lbl]++;
        const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        int32_t* c_row = cents + (size_t)lbl * total_dim;
        for (int d = 0; d < total_dim; d++) {
            int8_t t = glyph_read_trit(sig, d);
            c_row[d] += t;
        }
    }

    /* Per-class margin: require |centroid| > margin_frac × class_count
     * before emitting a non-zero trit. At margin_frac = 0 we emit sign
     * whenever the centroid is non-zero (coarsest prototype). Higher
     * fractions demand cleaner class agreement and produce sparser
     * prototypes. */
    for (int c = 0; c < N_CLASSES; c++) {
        int32_t tau = (int32_t)((int64_t)class_count[c]
                                * margin_frac_num / margin_frac_den);
        const int32_t* c_row = cents + (size_t)c * total_dim;
        uint8_t* w_packed = prototypes_packed + (size_t)c * sig_bytes;
        memset(w_packed, 0, sig_bytes);
        for (int d = 0; d < total_dim; d++) {
            glyph_write_trit(w_packed, d, sign_trit(c_row[d], tau));
        }
    }

    free(cents);
}

static int predict_argmin(
    const uint8_t* q_sig, const uint8_t* prototypes_packed,
    const uint8_t* mask, int sig_bytes)
{
    int32_t best = INT32_MAX;
    int best_c = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        int32_t d = m4t_popcount_dist(
            q_sig, prototypes_packed + (size_t)c * sig_bytes,
            mask, sig_bytes);
        if (d < best) { best = d; best_c = c; }
    }
    return best_c;
}

/* Multi-prototype predict: argmin over (class × prototype) pairs.
 * prototypes_packed is [N_CLASSES × k × sig_bytes]. Returns the class
 * of the closest prototype. */
static int predict_argmin_multi(
    const uint8_t* q_sig, const uint8_t* prototypes_packed,
    const uint8_t* mask, int sig_bytes, int k_per_class)
{
    int32_t best = INT32_MAX;
    int best_c = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        for (int kk = 0; kk < k_per_class; kk++) {
            int32_t d = m4t_popcount_dist(
                q_sig,
                prototypes_packed + ((size_t)c * k_per_class + kk) * sig_bytes,
                mask, sig_bytes);
            if (d < best) { best = d; best_c = c; }
        }
    }
    return best_c;
}

/* k-NN predict over multi-prototype set: find the top_k closest prototypes
 * across all (class, prototype) pairs, then rank-weighted majority vote
 * over their class labels. Rank weight = top_k − rank, ties broken by
 * lowest class index. */
#define CSA_MAX_TOPK 64
static int predict_knn_multi(
    const uint8_t* q_sig, const uint8_t* prototypes_packed,
    const uint8_t* mask, int sig_bytes, int k_per_class, int top_k)
{
    if (top_k < 1) top_k = 1;
    if (top_k > CSA_MAX_TOPK) top_k = CSA_MAX_TOPK;

    int32_t top_d[CSA_MAX_TOPK];
    int     top_c[CSA_MAX_TOPK];
    int ntk = 0;

    for (int c = 0; c < N_CLASSES; c++) {
        for (int kk = 0; kk < k_per_class; kk++) {
            int32_t d = m4t_popcount_dist(
                q_sig,
                prototypes_packed + ((size_t)c * k_per_class + kk) * sig_bytes,
                mask, sig_bytes);
            if (ntk < top_k) {
                int pos = ntk;
                while (pos > 0 && top_d[pos-1] > d) {
                    top_d[pos] = top_d[pos-1]; top_c[pos] = top_c[pos-1]; pos--;
                }
                top_d[pos] = d; top_c[pos] = c; ntk++;
            } else if (d < top_d[top_k - 1]) {
                int pos = top_k - 1;
                while (pos > 0 && top_d[pos-1] > d) {
                    top_d[pos] = top_d[pos-1]; top_c[pos] = top_c[pos-1]; pos--;
                }
                top_d[pos] = d; top_c[pos] = c;
            }
        }
    }

    int votes[N_CLASSES] = {0};
    for (int i = 0; i < ntk; i++) votes[top_c[i]] += (top_k - i);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
    return best;
}

/* Perceptron epoch: scan training set, for each misclassified example
 * update centroid sums (+sig on correct class, -sig on predicted).
 * Re-sign at end of epoch to refresh prototypes. Returns number of
 * misclassifications this epoch. */
static int perceptron_epoch(
    const uint8_t* train_sigs, const int* y_train, int n_train,
    int total_dim, int sig_bytes,
    int32_t margin_frac_num, int32_t margin_frac_den,
    int32_t* cents, int* class_count,
    uint8_t* prototypes_packed, const uint8_t* mask)
{
    int miss = 0;
    for (int i = 0; i < n_train; i++) {
        const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
        int y = y_train[i];
        int pred = predict_argmin(sig, prototypes_packed, mask, sig_bytes);
        if (pred == y) continue;
        miss++;

        int32_t* c_pos = cents + (size_t)y * total_dim;
        int32_t* c_neg = cents + (size_t)pred * total_dim;
        for (int d = 0; d < total_dim; d++) {
            int8_t t = glyph_read_trit(sig, d);
            c_pos[d] += t;
            c_neg[d] -= t;
        }
    }

    /* Re-sign prototypes with the updated centroid margins. */
    for (int c = 0; c < N_CLASSES; c++) {
        int32_t tau = (int32_t)((int64_t)class_count[c]
                                * margin_frac_num / margin_frac_den);
        const int32_t* c_row = cents + (size_t)c * total_dim;
        uint8_t* w_packed = prototypes_packed + (size_t)c * sig_bytes;
        memset(w_packed, 0, sig_bytes);
        for (int d = 0; d < total_dim; d++) {
            glyph_write_trit(w_packed, d, sign_trit(c_row[d], tau));
        }
    }

    return miss;
}

int main(int argc, char** argv) {
    /* Parse CSA-specific flags before handing off to glyph_config. */
    int use_gradients = 0;
    int use_perceptron = 0;
    int perceptron_epochs = 3;
    int32_t margin_num = 0, margin_den = 100;   /* default: no margin */
    int prototypes_per_class = 1;                /* E1 default: 1 centroid; k>1 = multi-prototype */
    int top_k = 1;                                /* 1 = argmin; >1 = rank-weighted k-NN vote */
    int new_argc = 0;
    char** new_argv = malloc((size_t)argc * sizeof(char*));
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--gradients") == 0) { use_gradients = 1; continue; }
        if (strcmp(argv[i], "--train_mode") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "perceptron") == 0) use_perceptron = 1;
            i++; continue;
        }
        if (strcmp(argv[i], "--perceptron_epochs") == 0 && i + 1 < argc) {
            perceptron_epochs = atoi(argv[i+1]); i++; continue;
        }
        if (strcmp(argv[i], "--margin") == 0 && i + 1 < argc) {
            /* --margin 0.05 → margin_num=5, margin_den=100 */
            double f = atof(argv[i+1]);
            if (f < 0.0) f = 0.0;
            if (f > 1.0) f = 1.0;
            margin_num = (int32_t)(f * 100.0);
            i++; continue;
        }
        if (strcmp(argv[i], "--prototypes_per_class") == 0 && i + 1 < argc) {
            prototypes_per_class = atoi(argv[i+1]);
            if (prototypes_per_class < 1) prototypes_per_class = 1;
            i++; continue;
        }
        if (strcmp(argv[i], "--top_k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[i+1]);
            if (top_k < 1) top_k = 1;
            i++; continue;
        }
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

    printf("csa_classifier: ternary class prototypes scored via popcount_dist\n");
    printf("  data=%s  deskew=%s  gradients=%s  normalize=%s\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on",
           use_gradients ? "on" : "off", cfg.normalize ? "on" : "off");
    printf("  image: %dx%dx%d  total_dim=%d  sig_bytes=%d\n",
           img_w, img_h, n_ch, total_dim, sig_bytes);
    printf("  train_mode=%s  margin=%.2f%s\n",
           use_perceptron ? "centroid+perceptron" : "centroid",
           (double)margin_num / margin_den,
           use_perceptron ? " (per-epoch re-sign)" : "");
    printf("  n_train=%d  n_test=%d\n\n", ds.n_train, ds.n_test);

    /* Build feature vectors: intensity + optional gradients. */
    m4t_mtfp_t* train_feat = malloc((size_t)ds.n_train * total_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_feat  = malloc((size_t)ds.n_test  * total_dim * sizeof(m4t_mtfp_t));

    if (use_gradients) {
        m4t_mtfp_t* hg = malloc((size_t)hgrad_dim * sizeof(m4t_mtfp_t));
        m4t_mtfp_t* vg = malloc((size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        for (int i = 0; i < ds.n_train; i++) {
            const m4t_mtfp_t* img = ds.x_train + (size_t)i * ds.input_dim;
            m4t_mtfp_t* out = train_feat + (size_t)i * total_dim;
            memcpy(out, img, (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            glyph_dataset_gradients(img, img_w, img_h, n_ch, hg, vg);
            memcpy(out + intensity_dim, hg, (size_t)hgrad_dim * sizeof(m4t_mtfp_t));
            memcpy(out + intensity_dim + hgrad_dim, vg, (size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        }
        for (int i = 0; i < ds.n_test; i++) {
            const m4t_mtfp_t* img = ds.x_test + (size_t)i * ds.input_dim;
            m4t_mtfp_t* out = test_feat + (size_t)i * total_dim;
            memcpy(out, img, (size_t)intensity_dim * sizeof(m4t_mtfp_t));
            glyph_dataset_gradients(img, img_w, img_h, n_ch, hg, vg);
            memcpy(out + intensity_dim, hg, (size_t)hgrad_dim * sizeof(m4t_mtfp_t));
            memcpy(out + intensity_dim + hgrad_dim, vg, (size_t)vgrad_dim * sizeof(m4t_mtfp_t));
        }
        free(hg); free(vg);
    } else {
        memcpy(train_feat, ds.x_train, (size_t)ds.n_train * intensity_dim * sizeof(m4t_mtfp_t));
        memcpy(test_feat,  ds.x_test,  (size_t)ds.n_test  * intensity_dim * sizeof(m4t_mtfp_t));
    }

    /* Tau calibration: separate for intensity and gradients (mirrors direct_lsh). */
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

    /* Quantize signatures. Matches direct_lsh.c. */
    printf("Quantizing signatures...\n");
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

    /* Train: per-class ternary prototypes.
     * k=1: single centroid per class (classical E1 shape).
     * k>1: select the first k training signatures per class as prototypes.
     *      No training rule needed — the prototypes ARE training samples.
     *      Inference argmin is then k-NN restricted to a class-balanced
     *      10·k subset of the training set. */
    clock_t t_train = clock();
    int k = prototypes_per_class;
    uint8_t* prototypes = calloc((size_t)N_CLASSES * k * sig_bytes, 1);
    uint8_t* mask = malloc(sig_bytes);
    memset(mask, 0xFF, sig_bytes);

    if (k == 1) {
        build_centroid_prototypes(train_sigs, ds.y_train, ds.n_train,
                                  total_dim, sig_bytes, margin_num, margin_den,
                                  prototypes);
    } else {
        /* Pick first k training signatures per class. Deterministic order;
         * the training set is already shuffled at dataset build time. */
        int filled[N_CLASSES] = {0};
        for (int i = 0; i < ds.n_train && 1; i++) {
            int lbl = ds.y_train[i];
            if (filled[lbl] >= k) continue;
            memcpy(prototypes + ((size_t)lbl * k + filled[lbl]) * sig_bytes,
                   train_sigs + (size_t)i * sig_bytes, sig_bytes);
            filled[lbl]++;
        }
        int short_class = -1;
        for (int c = 0; c < N_CLASSES; c++)
            if (filled[c] < k) { short_class = c; break; }
        if (short_class >= 0) {
            fprintf(stderr, "warning: class %d has only %d training samples, need %d\n",
                    short_class, filled[short_class], k);
        }
    }
    double t_centroid = (double)(clock() - t_train) / CLOCKS_PER_SEC;
    if (k == 1)
        printf("Centroid training: %.2fs\n", t_centroid);
    else
        printf("Multi-prototype build (k=%d per class): %.2fs\n", k, t_centroid);

    /* Training-set accuracy after centroid init (sanity). */
    int train_correct = 0;
    for (int i = 0; i < ds.n_train; i++) {
        int pred;
        if (k == 1) {
            pred = predict_argmin(train_sigs + (size_t)i * sig_bytes, prototypes, mask, sig_bytes);
        } else if (top_k == 1) {
            pred = predict_argmin_multi(train_sigs + (size_t)i * sig_bytes, prototypes, mask, sig_bytes, k);
        } else {
            pred = predict_knn_multi(train_sigs + (size_t)i * sig_bytes, prototypes, mask, sig_bytes, k, top_k);
        }
        if (pred == ds.y_train[i]) train_correct++;
    }
    printf("  Training-set accuracy (centroid):  %.2f%% (%d/%d)\n",
           100.0 * train_correct / ds.n_train, train_correct, ds.n_train);

    if (use_perceptron && k > 1) {
        printf("  (skipping perceptron refinement — only defined for k=1 centroid prototypes)\n");
    } else if (use_perceptron) {
        /* Need centroid state for perceptron updates; rebuild once so we
         * have both the cents buffer and the class counts. */
        int32_t* cents = calloc((size_t)N_CLASSES * total_dim, sizeof(int32_t));
        int class_count[N_CLASSES] = {0};
        for (int i = 0; i < ds.n_train; i++) {
            int lbl = ds.y_train[i];
            class_count[lbl]++;
            const uint8_t* sig = train_sigs + (size_t)i * sig_bytes;
            int32_t* c_row = cents + (size_t)lbl * total_dim;
            for (int d = 0; d < total_dim; d++) {
                int8_t t = glyph_read_trit(sig, d);
                c_row[d] += t;
            }
        }

        /* Per-epoch test-set evaluation so we see trajectory, not just
         * final. No algorithm change — strictly an observability hook. */
        int best_epoch = 0;
        int best_test = 0;
        for (int e = 0; e < perceptron_epochs; e++) {
            int miss = perceptron_epoch(
                train_sigs, ds.y_train, ds.n_train,
                total_dim, sig_bytes, margin_num, margin_den,
                cents, class_count, prototypes, mask);
            int epoch_test_correct = 0;
            for (int i = 0; i < ds.n_test; i++) {
                int pred = predict_argmin(
                    test_sigs + (size_t)i * sig_bytes,
                    prototypes, mask, sig_bytes);
                if (pred == ds.y_test[i]) epoch_test_correct++;
            }
            if (epoch_test_correct > best_test) {
                best_test = epoch_test_correct;
                best_epoch = e + 1;
            }
            printf("  Perceptron epoch %3d: %6d miss (%.2f%% train err)  test=%.2f%%\n",
                   e + 1, miss,
                   100.0 * miss / ds.n_train,
                   100.0 * epoch_test_correct / ds.n_test);
            if (miss == 0) break;
        }
        printf("  Best epoch: %d (test accuracy %.2f%%)\n",
               best_epoch, 100.0 * best_test / ds.n_test);
        free(cents);
    }

    /* Test-set accuracy. */
    clock_t t_test = clock();
    int test_correct = 0;
    int per_class_total[N_CLASSES] = {0};
    int per_class_correct[N_CLASSES] = {0};
    int confusion[N_CLASSES][N_CLASSES] = {{0}};

    for (int i = 0; i < ds.n_test; i++) {
        int y = ds.y_test[i];
        int pred;
        if (k == 1) {
            pred = predict_argmin(test_sigs + (size_t)i * sig_bytes, prototypes, mask, sig_bytes);
        } else if (top_k == 1) {
            pred = predict_argmin_multi(test_sigs + (size_t)i * sig_bytes, prototypes, mask, sig_bytes, k);
        } else {
            pred = predict_knn_multi(test_sigs + (size_t)i * sig_bytes, prototypes, mask, sig_bytes, k, top_k);
        }
        per_class_total[y]++;
        confusion[y][pred]++;
        if (pred == y) { test_correct++; per_class_correct[y]++; }
    }
    double t_infer = (double)(clock() - t_test) / CLOCKS_PER_SEC;

    printf("\n=== CSA results ===\n");
    printf("  Test accuracy:           %.2f%% (%d/%d)\n",
           100.0 * test_correct / ds.n_test, test_correct, ds.n_test);
    printf("  Inference time:          %.2fs (%.1f μs/query)\n",
           t_infer, 1e6 * t_infer / ds.n_test);

    printf("\n  Per-class breakdown:\n");
    printf("    class  count  correct  accuracy\n");
    for (int c = 0; c < N_CLASSES; c++)
        printf("    %3d   %5d  %5d    %.2f%%\n",
               c, per_class_total[c], per_class_correct[c],
               per_class_total[c] > 0
                   ? 100.0 * per_class_correct[c] / per_class_total[c]
                   : 0.0);

    /* Prototype sparsity diagnostic — only meaningful for k=1 centroid
     * prototypes. For k>1 the prototypes are raw training signatures and
     * their sparsity distribution just reflects the dataset. */
    if (k == 1) {
        printf("\n  Prototype sparsity (emission: +1 / 0 / -1):\n");
        for (int c = 0; c < N_CLASSES; c++) {
            int np = 0, nz = 0, nn = 0;
            const uint8_t* w = prototypes + (size_t)c * sig_bytes;
            for (int d = 0; d < total_dim; d++) {
                int8_t t = glyph_read_trit(w, d);
                if (t > 0) np++;
                else if (t < 0) nn++;
                else nz++;
            }
            printf("    class %d:  +%.1f%% / 0=%.1f%% / -%.1f%%\n",
                   c, 100.0*np/total_dim, 100.0*nz/total_dim, 100.0*nn/total_dim);
        }
    }

    free(train_sigs); free(test_sigs); free(prototypes); free(mask);
    glyph_dataset_free(&ds);
    return 0;
}
