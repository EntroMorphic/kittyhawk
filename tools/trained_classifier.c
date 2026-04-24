/*
 * trained_classifier.c — minimal learned ternary linear classifier.
 *
 * Tests the "learned weights patch the representation tax" hypothesis:
 * takes the same direct-quantization signatures `direct_lsh` uses,
 * trains a dense ternary linear layer via libtrain's tlinear primitives
 * (forward + backward_dW + hysteresis requantization), reports test
 * accuracy.
 *
 * Motivation:
 *   - libtrain ships with passing gradient checks and a 96.5% convergence
 *     result on a 2-class linear-separable toy but has never been applied
 *     to a real dataset.
 *   - CIFAR-10 has a measured representation tax (direct_lsh k-NN 46.63%
 *     vs SSTT 53%). k-NN can't move off the Hamming-of-raw-quantized
 *     manifold; learned weights might.
 *   - `csa_classifier` (centroid + perceptron) exists but uses integer
 *     sign-updates without LR control, hysteresis, or STE — its
 *     perceptron mode made CIFAR *worse* (29% → 22%). Principled SGD
 *     with hysteresis requant is the untested shape.
 *
 * Architecture:
 *   Signature = direct ternary quantization of (intensity + gradients)
 *   → unpacked to float X ∈ [-1, 0, +1].
 *   Classifier = dense ternary linear layer W ∈ trit^{N_CLASSES × dim}.
 *   Loss = MSE against one-hot target.
 *   Training = SGD on W_latent (float shadow), periodic hysteresis
 *   re-quantization to W.
 *
 * Scope guardrails:
 *   - ONE dense ternary layer only. No routing, no multi-tile dispatch.
 *     The routed path hit expert collapse on the 10-class toy; this is
 *     the non-routed baseline.
 *   - Features identical to direct_lsh's rule-compliant path (intensity +
 *     optional gradients + optional normalize). Substrate-aligned
 *     comparison.
 *   - Pocket snapshot: best-train-accuracy weights kept, evaluated on
 *     test at end. Same pattern as test_toy_convergence.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_sig.h"
#include "glyph_multiprobe.h"
#include "m4t_trit_pack.h"

#include "backward_linear.h"
#include "requantize.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10

static uint32_t rng_state = 0xdeadbeefu;
static uint32_t xor32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x; return x;
}
static float gauss(void) {
    float u1 = ((xor32() >> 8) + 1) * (1.0f / (float)(1u << 24));
    float u2 = ((xor32() >> 8) + 1) * (1.0f / (float)(1u << 24));
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* Unpack a packed-trit signature into a float vector {-1.0, 0.0, +1.0}.
 * Caller-provided output buffer must have n_dims floats. */
static void unpack_trits_to_float(const uint8_t* sig, int n_dims, float* out) {
    for (int d = 0; d < n_dims; d++) {
        int8_t t = glyph_read_trit(sig, d);
        out[d] = (float)t;
    }
}

int main(int argc, char** argv) {
    /* Parse extra flags (--epochs, --batch, --lr) before glyph_config. */
    int epochs = 20;
    int batch_size = 128;
    float learning_rate = 5e-4f;
    int requant_every_steps = 0;  /* 0 → auto: once per epoch */

    int use_gradients = 0;
    int new_argc = 0;
    char** new_argv = malloc((size_t)argc * sizeof(char*));
    new_argv[new_argc++] = argv[0];
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--epochs") && i + 1 < argc) {
            epochs = atoi(argv[++i]); continue;
        }
        if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch_size = atoi(argv[++i]); continue;
        }
        if (!strcmp(argv[i], "--lr") && i + 1 < argc) {
            learning_rate = (float)atof(argv[++i]); continue;
        }
        if (!strcmp(argv[i], "--requant_every") && i + 1 < argc) {
            requant_every_steps = atoi(argv[++i]); continue;
        }
        if (!strcmp(argv[i], "--gradients")) {
            use_gradients = 1; continue;
        }
        new_argv[new_argc++] = argv[i];
    }

    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, new_argc, new_argv);
    free(new_argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, cfg.data_dir) != 0) return 1;
    if (!cfg.no_deskew)  glyph_dataset_deskew(&ds);
    if (cfg.normalize)   glyph_dataset_normalize(&ds);

    int n_ch = (ds.input_dim > 784) ? 3 : 1;
    int img_w = ds.img_w > 0 ? ds.img_w : (n_ch == 3 ? 32 : 28);
    int img_h = ds.img_h > 0 ? ds.img_h : (n_ch == 3 ? 32 : 28);
    int intensity_dim = ds.input_dim;
    int hgrad_dim = n_ch * img_h * (img_w - 1);
    int vgrad_dim = n_ch * (img_h - 1) * img_w;
    int total_dim = intensity_dim + (use_gradients ? (hgrad_dim + vgrad_dim) : 0);
    int sig_bytes = M4T_TRIT_PACKED_BYTES(total_dim);

    printf("trained_classifier (libtrain tlinear + hysteresis requant)\n");
    printf("  data=%s  deskew=%s  normalize=%s  gradients=%s\n",
           cfg.data_dir, cfg.no_deskew ? "off" : "on",
           cfg.normalize ? "on" : "off", use_gradients ? "on" : "off");
    printf("  image=%dx%dx%d  intensity_dim=%d  total_dim=%d  sig_bytes=%d\n",
           img_w, img_h, n_ch, intensity_dim, total_dim, sig_bytes);
    printf("  epochs=%d batch=%d lr=%.4g density=%.3f\n",
           epochs, batch_size, learning_rate, cfg.density);

    /* Build full feature vectors (intensity + optional gradients). */
    m4t_mtfp_t* train_feat = malloc((size_t)ds.n_train * total_dim * sizeof(m4t_mtfp_t));
    m4t_mtfp_t* test_feat  = malloc((size_t)ds.n_test  * total_dim * sizeof(m4t_mtfp_t));
    clock_t t0 = clock();
    for (int pass = 0; pass < 2; pass++) {
        const m4t_mtfp_t* src = (pass == 0) ? ds.x_train : ds.x_test;
        m4t_mtfp_t* dst = (pass == 0) ? train_feat : test_feat;
        int n_imgs = (pass == 0) ? ds.n_train : ds.n_test;
        for (int i = 0; i < n_imgs; i++) {
            m4t_mtfp_t* out = dst + (size_t)i * total_dim;
            const m4t_mtfp_t* img = src + (size_t)i * ds.input_dim;
            memcpy(out, img, (size_t)ds.input_dim * sizeof(m4t_mtfp_t));
            if (use_gradients) {
                glyph_dataset_gradients(img, img_w, img_h, n_ch,
                    out + intensity_dim,
                    out + intensity_dim + hgrad_dim);
            }
        }
    }
    double t_feat = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("features built in %.2fs\n", t_feat);

    /* Calibrate tau and quantize. */
    int64_t tau = glyph_sig_quantize_tau(train_feat, ds.n_train, total_dim, cfg.density);
    uint8_t* train_sigs = calloc((size_t)ds.n_train, sig_bytes);
    uint8_t* test_sigs  = calloc((size_t)ds.n_test,  sig_bytes);
    glyph_sig_quantize_batch(train_feat, ds.n_train, total_dim, tau, train_sigs);
    glyph_sig_quantize_batch(test_feat,  ds.n_test,  total_dim, tau, test_sigs);
    free(train_feat); free(test_feat);
    printf("tau=%lld sig_bytes=%d\n", (long long)tau, sig_bytes);

    /* Unpack signatures into float arrays [N × total_dim]. */
    float* X_train = malloc((size_t)ds.n_train * total_dim * sizeof(float));
    float* X_test  = malloc((size_t)ds.n_test  * total_dim * sizeof(float));
    for (int i = 0; i < ds.n_train; i++)
        unpack_trits_to_float(train_sigs + (size_t)i * sig_bytes,
                              total_dim, X_train + (size_t)i * total_dim);
    for (int i = 0; i < ds.n_test; i++)
        unpack_trits_to_float(test_sigs + (size_t)i * sig_bytes,
                              total_dim, X_test + (size_t)i * total_dim);
    free(train_sigs); free(test_sigs);

    /* One-hot encode train labels to {-1, +1} — matches test_toy pattern. */
    float* Y_train = malloc((size_t)ds.n_train * N_CLASSES * sizeof(float));
    for (int i = 0; i < ds.n_train; i++) {
        int c = ds.y_train[i];
        for (int n = 0; n < N_CLASSES; n++)
            Y_train[(size_t)i * N_CLASSES + n] = (n == c) ? 1.0f : -1.0f;
    }

    /* Weight init. W ∈ [N_CLASSES × total_dim] per the tlinear_forward
     * convention (Y[m,n] = Σ_k X[m,k] · W[n,k]). */
    size_t W_sz = (size_t)N_CLASSES * total_dim;
    float*  W_latent = calloc(W_sz, sizeof(float));
    int8_t* W        = calloc(W_sz, 1);
    int8_t* W_best   = calloc(W_sz, 1);
    for (size_t i = 0; i < W_sz; i++) W_latent[i] = 0.05f * gauss();
    requantize_hysteresis(W, W_latent, (int)W_sz, cfg.density, 0.10);

    int requant_every = requant_every_steps > 0
                        ? requant_every_steps
                        : (ds.n_train / batch_size);

    /* Training scratch. */
    float*   Y_pred = malloc((size_t)batch_size * N_CLASSES * sizeof(float));
    float*   dY     = malloc((size_t)batch_size * N_CLASSES * sizeof(float));
    float*   dW     = malloc(W_sz * sizeof(float));
    int*     order  = malloc((size_t)ds.n_train * sizeof(int));
    for (int i = 0; i < ds.n_train; i++) order[i] = i;

    /* Shuffled batch inputs — re-packed contiguously each batch. */
    float* X_batch = malloc((size_t)batch_size * total_dim * sizeof(float));
    float* Y_batch = malloc((size_t)batch_size * N_CLASSES * sizeof(float));

    double best_train_acc = -1.0;

    printf("\ntraining %d epochs (%d batches/epoch, requant every %d steps)...\n",
           epochs, ds.n_train / batch_size, requant_every);
    int step = 0;
    for (int e = 0; e < epochs; e++) {
        /* Fisher-Yates shuffle. */
        for (int i = ds.n_train - 1; i > 0; i--) {
            int j = (int)(xor32() % (uint32_t)(i + 1));
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        double epoch_loss = 0.0;
        int n_batches = ds.n_train / batch_size;
        for (int b = 0; b < n_batches; b++) {
            for (int m = 0; m < batch_size; m++) {
                int src = order[b * batch_size + m];
                memcpy(X_batch + (size_t)m * total_dim,
                       X_train + (size_t)src * total_dim,
                       (size_t)total_dim * sizeof(float));
                memcpy(Y_batch + (size_t)m * N_CLASSES,
                       Y_train + (size_t)src * N_CLASSES,
                       (size_t)N_CLASSES * sizeof(float));
            }

            tlinear_forward(Y_pred, X_batch, W, batch_size, total_dim, N_CLASSES);
            /* Scale Y to unit-ish range so MSE against {-1, +1} targets
             * is well-posed. Without this, Y magnitudes ~sqrt(K) dominate
             * the target scale and the gradient signal for class
             * discrimination is dwarfed by "shrink Y magnitude" pressure.
             * The scale factor sqrt(K) is chosen so Y_scaled has O(1) std
             * under random trit W. Chain rule back through the division:
             * dY_raw = dY_scaled / sqrt(K). */
            float scale = sqrtf((float)total_dim);
            double batch_loss = 0.0;
            for (int i = 0; i < batch_size * N_CLASSES; i++) {
                float y_scaled = Y_pred[i] / scale;
                float e_ = y_scaled - Y_batch[i];
                batch_loss += 0.5 * (double)e_ * (double)e_;
                dY[i] = (e_ / scale) / (float)batch_size;
            }
            epoch_loss += batch_loss / batch_size;

            memset(dW, 0, W_sz * sizeof(float));
            /* tlinear_backward_dW signature: dW[n,k] += Σ_m dY[m,n] * X[m,k].
             * Expects dW as [K×N] layout? Check — the toy passes dW as a flat
             * N×K accumulator; see train/src/backward_linear.c. Shape matches
             * our W_latent layout (row-major [N_CLASSES × total_dim]).
             */
            tlinear_backward_dW(dW, dY, X_batch, batch_size, total_dim, N_CLASSES);
            for (size_t i = 0; i < W_sz; i++) {
                if (fabsf(W_latent[i]) < 1.0f) W_latent[i] -= learning_rate * dW[i];
            }
            step++;
            if ((step % requant_every) == 0) {
                requantize_hysteresis(W, W_latent, (int)W_sz, cfg.density, 0.10);
            }
        }
        int flips = requantize_hysteresis(W, W_latent, (int)W_sz, cfg.density, 0.10);

        /* Evaluate on TRAIN subset for pocket snapshot. Stride-sampled
         * for speed (1/10 of train). */
        int train_correct = 0, train_seen = 0;
        for (int i = 0; i < ds.n_train; i += 10) {
            tlinear_forward(Y_pred, X_train + (size_t)i * total_dim, W, 1, total_dim, N_CLASSES);
            int argmax = 0;
            for (int n = 1; n < N_CLASSES; n++)
                if (Y_pred[n] > Y_pred[argmax]) argmax = n;
            if (argmax == ds.y_train[i]) train_correct++;
            train_seen++;
        }
        double train_acc = 100.0 * train_correct / train_seen;

        printf("  epoch %2d  loss=%.4f  train_acc=%.2f%%  flips=%d%s\n",
               e, epoch_loss / n_batches, train_acc, flips,
               (train_acc > best_train_acc) ? "  [pocket]" : "");
        if (train_acc > best_train_acc) {
            best_train_acc = train_acc;
            memcpy(W_best, W, W_sz);
        }
    }

    /* Use pocket snapshot for test eval. */
    memcpy(W, W_best, W_sz);

    int test_correct = 0;
    for (int i = 0; i < ds.n_test; i++) {
        tlinear_forward(Y_pred, X_test + (size_t)i * total_dim, W, 1, total_dim, N_CLASSES);
        int argmax = 0;
        for (int n = 1; n < N_CLASSES; n++)
            if (Y_pred[n] > Y_pred[argmax]) argmax = n;
        if (argmax == ds.y_test[i]) test_correct++;
    }
    double test_acc = 100.0 * test_correct / ds.n_test;
    printf("\nTEST accuracy (pocket): %.2f%% (%d/%d)  best_train=%.2f%%\n",
           test_acc, test_correct, ds.n_test, best_train_acc);

    free(X_train); free(X_test); free(Y_train);
    free(W_latent); free(W); free(W_best);
    free(Y_pred); free(dY); free(dW);
    free(order); free(X_batch); free(Y_batch);
    glyph_dataset_free(&ds);
    return 0;
}
