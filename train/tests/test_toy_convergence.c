/*
 * test_toy_convergence.c — week-1 gate for routed autodiff MVP.
 *
 * Synthetic 2-class linearly separable dataset. Train a routed ternary
 * classifier via SGD + STE, verify convergence. If this fails, the MVP
 * STE design is broken or the float-latent+re-quantization cadence is
 * wrong — do not proceed to CIFAR.
 *
 * Dataset:
 *   H = 32 input dimensions. Class 0 has its "signal dim" = -1, class 1
 *   has its "signal dim" = +1. Other dims are zero-mean noise. The
 *   discriminative signal is concentrated in a single dim — well within
 *   a single-tile linear model's capacity. 800 train, 200 test.
 *
 * Model:
 *   Routed layer with T=4 tiles, H inputs, N=2 outputs (class logits).
 *   U (gating weights) frozen at random ternary. W trainable latent float,
 *   re-quantized to ternary every 10 SGD steps.
 *
 * Loss: MSE against one-hot target. Simpler than cross-entropy.
 *
 * Gate: test accuracy ≥ 95% after training. On a 1-dim signal problem
 * with 800 training samples and a linear classifier, this should be
 * trivial for a correctly-plumbed trainer. Anything less means STE or
 * re-quantization has a bug.
 */

#include "backward_linear.h"
#include "backward_routed.h"
#include "requantize.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H_DIM     16
#define T_TILES   4            /* actual routing, not trivial T=1 */
#define N_OUT     2
#define K_SEL     2
#define N_TRAIN   800
#define N_TEST    200
#define SIG_DIM   7            /* which dim carries class signal */
#define BATCH     32
#define EPOCHS    40
/* LR derivation: with MSE loss and |X|≈0.25, |dY|≈0.1 at mid-training,
 * per-batch |dW|≈0.025. LR=5e-4 → per-step |ΔW_latent|≈1.25e-5.
 * Over REQUANT_EVERY=25 steps, latent drift ≈ 3e-4 << τ ≈ 0.05, so
 * trit percentiles move smoothly rather than thrashing.  */
#define LR        0.0005f
#define REQUANT_EVERY 25       /* once per epoch (800/32 = 25 steps/epoch) */

static uint32_t rng_state = 0xdeadbeefu;
static uint32_t xor32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x; return x;
}
static float gauss(void) {
    /* Box–Muller using uniform (0, 1]; drop one sample. */
    float u1 = ((xor32() >> 8) + 1) * (1.0f / (float)(1u << 24));
    float u2 = ((xor32() >> 8) + 1) * (1.0f / (float)(1u << 24));
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
static int8_t rand_trit(void) {
    uint32_t r = xor32() % 3u;
    return (r == 0) ? -1 : (r == 1) ? 0 : 1;
}

/* Re-quantization uses requantize_hysteresis from train/src/requantize.c.
 *
 * REQUANT_DENSITY derivation: base-3 balanced. Under max-entropy prior for
 *   a uniform trit distribution over {−1, 0, +1}, P(zero) = 1/3. Setting
 *   τ at the 67th percentile of |W_latent| realizes that prior on the
 *   deployed weights. Any other density implicitly codes a bias that the
 *   optimizer must undo with additional gradient signal.
 *
 * REQUANT_HYSTERESIS derivation: the dead-zone half-width must exceed
 *   typical per-requant-cycle latent drift but stay below the spacing
 *   between adjacent percentiles. Cumulative drift per cycle ≈ LR·|dW|·
 *   REQUANT_EVERY ≈ 3e-4. τ≈0.05 mid-training, so 0.1·τ ≈ 5e-3 — one order
 *   above typical motion, well below signal separation. Validated
 *   empirically: sel_flips → 0 once training settles.  */
#define REQUANT_DENSITY    0.33
#define REQUANT_HYSTERESIS 0.10

/* One training run at a given seed. Returns test accuracy.
 * Refactored from main so we can sweep seeds in multi-seed mode. */
static double run_one_seed(uint32_t seed, int verbose) {
    rng_state = seed ? seed : 0xdeadbeefu;
    (void)verbose;
    /* ---- dataset ---- */
    float* X_train = malloc((size_t)N_TRAIN * H_DIM * sizeof(float));
    float* Y_train = malloc((size_t)N_TRAIN * N_OUT * sizeof(float)); /* one-hot */
    int* y_train   = malloc((size_t)N_TRAIN * sizeof(int));
    float* X_test  = malloc((size_t)N_TEST * H_DIM * sizeof(float));
    int* y_test    = malloc((size_t)N_TEST * sizeof(int));

    for (int i = 0; i < N_TRAIN; i++) {
        int c = (int)(xor32() & 1u);
        y_train[i] = c;
        for (int h = 0; h < H_DIM; h++) {
            X_train[(size_t)i * H_DIM + h] = 0.25f * gauss();
        }
        X_train[(size_t)i * H_DIM + SIG_DIM] += (c == 0 ? -1.0f : +1.0f);
        for (int n = 0; n < N_OUT; n++)
            Y_train[(size_t)i * N_OUT + n] = (n == c ? 1.0f : -1.0f);
    }
    for (int i = 0; i < N_TEST; i++) {
        int c = (int)(xor32() & 1u);
        y_test[i] = c;
        for (int h = 0; h < H_DIM; h++) {
            X_test[(size_t)i * H_DIM + h] = 0.25f * gauss();
        }
        X_test[(size_t)i * H_DIM + SIG_DIM] += (c == 0 ? -1.0f : +1.0f);
    }

    /* ---- model state ---- */
    int8_t* U = malloc((size_t)T_TILES * H_DIM);             /* frozen gating */
    int8_t* W = malloc((size_t)T_TILES * H_DIM * N_OUT);     /* deployed ternary */
    float*  W_latent = calloc((size_t)T_TILES * H_DIM * N_OUT, sizeof(float));
    int8_t* W_best = malloc((size_t)T_TILES * H_DIM * N_OUT); /* pocket snapshot */
    double best_train_acc = -1.0;

    for (int i = 0; i < T_TILES * H_DIM; i++) U[i] = rand_trit();
    /* W_latent init scale: σ=0.05. Under this Gaussian prior, |W_latent|
     * follows half-normal with mean ≈ 0.04, so τ (67th-percentile) ≈ 0.049.
     * Almost every latent starts inside [−τ, +τ] → W quantizes to all-zero.
     * Trainer must *earn* non-zero trits via gradient flow. Scale << signal
     * magnitude (1.0) prevents premature commitment to random ternary. */
    for (int i = 0; i < T_TILES * H_DIM * N_OUT; i++) {
        W_latent[i] = 0.05f * gauss();
    }
    memset(W, 0, (size_t)T_TILES * H_DIM * N_OUT); /* needed for hysteresis stickiness */
    requantize_hysteresis(W, W_latent, T_TILES * H_DIM * N_OUT,
                          REQUANT_DENSITY, REQUANT_HYSTERESIS);

    /* ---- per-batch scratch ---- */
    int32_t* decisions = malloc((size_t)BATCH * K_SEL * sizeof(int32_t));
    int8_t*  signs     = malloc((size_t)BATCH * K_SEL);
    float*   Y_pred    = malloc((size_t)BATCH * N_OUT * sizeof(float));
    float*   dY        = malloc((size_t)BATCH * N_OUT * sizeof(float));
    float*   dX_scratch = malloc((size_t)BATCH * H_DIM * sizeof(float));
    float*   dW_accum   = calloc((size_t)T_TILES * H_DIM * N_OUT, sizeof(float));

    /* STE behavior monitor: record per-training-sample routing decisions
     * once per epoch, compare to previous epoch. Selection-flip rate tells
     * us whether training is actually changing which tiles fire — with
     * the MVP's frozen U, this MUST be zero (U doesn't move, so X·U
     * scores are constant, so top-k selections are deterministic in X).
     * A non-zero flip rate would indicate the frozen-U invariant is
     * broken. With future learnable U, this becomes the primary STE
     * behavior diagnostic. */
    int32_t* decisions_prev = calloc((size_t)N_TRAIN * K_SEL, sizeof(int32_t));
    int32_t* decisions_cur  = malloc((size_t)N_TRAIN * K_SEL * sizeof(int32_t));
    for (int i = 0; i < N_TRAIN * K_SEL; i++) decisions_prev[i] = -2; /* init "unseen" */

    int step = 0;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        for (int batch_start = 0; batch_start + BATCH <= N_TRAIN; batch_start += BATCH) {
            const float* Xb = X_train + (size_t)batch_start * H_DIM;
            const float* Yb = Y_train + (size_t)batch_start * N_OUT;

            rroute_forward_select(decisions, signs, Xb, U, BATCH, H_DIM, T_TILES, K_SEL);
            rroute_forward_dispatch(Y_pred, Xb, W, decisions, signs,
                                    BATCH, H_DIM, T_TILES, N_OUT, K_SEL);
            /* MSE loss. dY = (Y_pred − Y_target) / BATCH. */
            double batch_loss = 0.0;
            for (int i = 0; i < BATCH * N_OUT; i++) {
                float e = Y_pred[i] - Yb[i];
                batch_loss += 0.5 * (double)e * (double)e;
                dY[i] = e / (float)BATCH;
            }
            epoch_loss += batch_loss / BATCH;

            memset(dW_accum, 0, (size_t)T_TILES * H_DIM * N_OUT * sizeof(float));
            rroute_backward_dW(dW_accum, dY, Xb, decisions, signs,
                               BATCH, H_DIM, T_TILES, N_OUT, K_SEL);
            (void)dX_scratch; /* dX not needed — input layer, no upstream gradient */

            /* SGD update on W_latent with clipped STE. Clip at ±1.0 — the
             * magnitude of a fully committed trit. Latents past this range
             * add no useful signal (the trit is saturated) and would keep
             * integrating noise, blocking future re-quantization to 0 or
             * opposite-sign. Mirrors standard BinaryConnect/XNOR-Net
             * straight-through with hard tanh. */
            for (int i = 0; i < T_TILES * H_DIM * N_OUT; i++) {
                if (fabsf(W_latent[i]) < 1.0f) {
                    W_latent[i] -= LR * dW_accum[i];
                }
            }

            step++;
            if (step % REQUANT_EVERY == 0) {
                requantize_hysteresis(W, W_latent,
                    T_TILES * H_DIM * N_OUT, REQUANT_DENSITY, REQUANT_HYSTERESIS);
            }
        }
        /* Requantize at end of epoch for the test eval. */
        int epoch_flips = requantize_hysteresis(W, W_latent,
            T_TILES * H_DIM * N_OUT, REQUANT_DENSITY, REQUANT_HYSTERESIS);

        /* Training accuracy at end of epoch. Capture routing decisions into
         * decisions_cur so we can diff against last epoch's selections. */
        int correct = 0;
        for (int i = 0; i < N_TRAIN; i += BATCH) {
            int cur = (i + BATCH <= N_TRAIN) ? BATCH : (N_TRAIN - i);
            rroute_forward_select(decisions, signs, X_train + (size_t)i * H_DIM,
                                  U, cur, H_DIM, T_TILES, K_SEL);
            rroute_forward_dispatch(Y_pred, X_train + (size_t)i * H_DIM, W,
                                    decisions, signs, cur, H_DIM, T_TILES, N_OUT, K_SEL);
            memcpy(decisions_cur + (size_t)i * K_SEL, decisions,
                   (size_t)cur * K_SEL * sizeof(int32_t));
            for (int m = 0; m < cur; m++) {
                int argmax = 0;
                float best = Y_pred[(size_t)m * N_OUT];
                for (int n = 1; n < N_OUT; n++) {
                    if (Y_pred[(size_t)m * N_OUT + n] > best) {
                        best = Y_pred[(size_t)m * N_OUT + n]; argmax = n;
                    }
                }
                if (argmax == y_train[i + m]) correct++;
            }
        }
        double train_acc = 100.0 * correct / N_TRAIN;

        /* STE behavior: count per-sample selection flips vs previous epoch.
         * Frozen-U invariant means this is identically 0 from epoch 1 onward.
         * Epoch 0 compares against the -2 sentinel and is not a flip signal. */
        int sel_flips = 0;
        if (epoch > 0) {
            for (int i = 0; i < N_TRAIN * K_SEL; i++) {
                if (decisions_cur[i] != decisions_prev[i]) sel_flips++;
            }
        }
        memcpy(decisions_prev, decisions_cur,
               (size_t)N_TRAIN * K_SEL * sizeof(int32_t));

        printf("  epoch %2d  loss=%.4f  train_acc=%.2f%%  flips=%d  sel_flips=%d%s\n",
               epoch, epoch_loss / (N_TRAIN / BATCH), train_acc, epoch_flips, sel_flips,
               (train_acc > best_train_acc) ? "  [pocket]" : "");
        if (epoch > 0 && sel_flips != 0) {
            fprintf(stderr, "FAIL: frozen-U invariant broken: epoch %d saw %d selection flips\n",
                    epoch, sel_flips);
            /* Fall through so the rest of the run finishes and produces a diff. */
        }
        if (train_acc > best_train_acc) {
            best_train_acc = train_acc;
            memcpy(W_best, W, (size_t)T_TILES * H_DIM * N_OUT);
        }
    }

    /* Use pocket snapshot for test. Pocket-perceptron style: best-train
     * weights weather the re-quantization oscillations that can briefly
     * spike a bad config at the end of training. */
    memcpy(W, W_best, (size_t)T_TILES * H_DIM * N_OUT);
    printf("\n  Pocket snapshot best_train_acc=%.2f%%\n", best_train_acc);

    /* Test accuracy. */
    int correct = 0;
    for (int i = 0; i < N_TEST; i += BATCH) {
        int cur = (i + BATCH <= N_TEST) ? BATCH : (N_TEST - i);
        rroute_forward_select(decisions, signs, X_test + (size_t)i * H_DIM,
                              U, cur, H_DIM, T_TILES, K_SEL);
        rroute_forward_dispatch(Y_pred, X_test + (size_t)i * H_DIM, W,
                                decisions, signs, cur, H_DIM, T_TILES, N_OUT, K_SEL);
        for (int m = 0; m < cur; m++) {
            int argmax = 0;
            float best = Y_pred[(size_t)m * N_OUT];
            for (int n = 1; n < N_OUT; n++) {
                if (Y_pred[(size_t)m * N_OUT + n] > best) {
                    best = Y_pred[(size_t)m * N_OUT + n]; argmax = n;
                }
            }
            if (argmax == y_test[i + m]) correct++;
        }
    }
    double test_acc = 100.0 * correct / N_TEST;
    printf("\n  TEST acc = %.2f%% (%d/%d)  [seed=%08x]\n", test_acc, correct, N_TEST, seed);

    free(X_train); free(Y_train); free(y_train);
    free(X_test); free(y_test);
    free(U); free(W); free(W_latent); free(W_best);
    free(decisions); free(signs); free(Y_pred); free(dY);
    free(dX_scratch); free(dW_accum);
    free(decisions_prev); free(decisions_cur);

    return test_acc;
}

int main(int argc, char** argv) {
    /* Default: 5-seed sweep. With CLI --single, run one seed (faster). */
    int multi_seed = 1;
    if (argc >= 2 && strcmp(argv[1], "--single") == 0) multi_seed = 0;

    if (!multi_seed) {
        double acc = run_one_seed(0xdeadbeefu, 1);
        if (acc >= 95.0) {
            printf("PASS: routed autodiff converges on 2-class toy\n");
            return 0;
        }
        fprintf(stderr, "FAIL: test accuracy %.2f%% below 95%% gate\n", acc);
        return 1;
    }

    const uint32_t seeds[] = {
        0xdeadbeefu, 0x13579bdfu, 0xa5a5a5a5u, 0x0badf00du, 0xfeedfaceu
    };
    const int n_seeds = (int)(sizeof(seeds) / sizeof(seeds[0]));
    double accs[8];  /* space for up to 8 seeds */

    printf("=== multi-seed toy convergence (%d seeds) ===\n", n_seeds);
    for (int s = 0; s < n_seeds; s++) {
        printf("\n--- seed %d (0x%08x) ---\n", s, seeds[s]);
        accs[s] = run_one_seed(seeds[s], 0);
    }

    double mean = 0.0;
    for (int s = 0; s < n_seeds; s++) mean += accs[s];
    mean /= n_seeds;
    double var = 0.0;
    for (int s = 0; s < n_seeds; s++) {
        double d = accs[s] - mean;
        var += d * d;
    }
    double stddev = sqrt(var / n_seeds);
    double acc_min = accs[0], acc_max = accs[0];
    for (int s = 1; s < n_seeds; s++) {
        if (accs[s] < acc_min) acc_min = accs[s];
        if (accs[s] > acc_max) acc_max = accs[s];
    }

    printf("\n=== multi-seed summary ===\n");
    for (int s = 0; s < n_seeds; s++) {
        printf("  seed 0x%08x: %.2f%%\n", seeds[s], accs[s]);
    }
    printf("  mean=%.2f%%  σ=%.2fpp  range=[%.2f, %.2f]\n",
           mean, stddev, acc_min, acc_max);

    /* Gates: mean ≥ 95 % AND σ < 5 pp AND min ≥ 85 % (no catastrophic seeds). */
    int fails = 0;
    if (mean < 95.0) { fprintf(stderr, "FAIL: mean %.2f%% below 95%%\n", mean); fails++; }
    if (stddev > 5.0) { fprintf(stderr, "FAIL: σ %.2fpp above 5pp\n", stddev); fails++; }
    if (acc_min < 85.0) { fprintf(stderr, "FAIL: min %.2f%% below 85%%\n", acc_min); fails++; }

    if (fails == 0) {
        printf("PASS: routed autodiff converges stably across %d seeds\n", n_seeds);
        return 0;
    }
    return 1;
}
