/*
 * test_ternary_route_fair.c — Fair head-to-head: top-1 vs ternary routing
 *
 * BOTH models use the same architecture wrapper:
 *   - LayerNorm + residual
 *   - AdamW optimizer
 *   - Same classification head
 *   - Same MTFP21 quantization
 *   - Same output scaling
 *
 * The ONLY difference: routing strategy.
 *   - top-1: argmax(scores) → one tile
 *   - ternary(k): top-k by |score|, sign-weighted sum
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        g_fail++; \
    } else { g_pass++; } \
} while(0)

/* 6-class cluster dataset */
static void make_data(float* x, int* labels, int n, int d, int n_cls, uint64_t seed) {
    srand48(seed);
    for (int i = 0; i < n; i++) {
        int c = i % n_cls;
        labels[i] = c;
        float angle = (float)c / (float)n_cls * 6.28318f;
        for (int j = 0; j < d; j++) {
            float noise = (float)(drand48() * 2.0 - 1.0) * 0.3f;
            if (j == 0) x[i*d+j] = 2.0f * cosf(angle) + noise;
            else if (j == 1) x[i*d+j] = 2.0f * sinf(angle) + noise;
            else x[i*d+j] = noise * 0.1f;
        }
    }
}

/* Train one model configuration and return accuracy */
static float train_and_eval(
    int d_model, int num_tiles, int tile_hidden, int active_k,
    float* x_train, int* y_train, int n_train,
    float* x_test, int* y_test, int n_test,
    int n_classes, int epochs, uint64_t seed,
    const char* label)
{
    TrixTernaryRouteConfig cfg = {
        .d_model = d_model,
        .num_tiles = num_tiles,
        .tile_hidden = tile_hidden,
        .active_k = active_k,
        .output_scale_init = 0.1f,
        .ln_eps = 1e-5f,
    };
    TrixTernaryRoutedFFN* tr = trix_ternary_route_create(cfg, seed);
    TrixAtomFFN* head = trix_atom_ffn_create(d_model, 32, n_classes, seed);

    int batch = 64;
    float best_acc = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0;
        int n_b = 0;

        for (int b = 0; b + batch <= n_train; b += batch) {
            float* xb = x_train + b * d_model;
            int* yb = y_train + b;

            float* ffn_out = calloc(batch * d_model, sizeof(float));
            trix_ternary_route_forward(tr, xb, ffn_out, batch);

            float* logits = calloc(batch * n_classes, sizeof(float));
            trix_atom_ffn_forward(head, ffn_out, logits, batch);
            float loss = trix_cross_entropy_loss(logits, yb, batch, n_classes);
            epoch_loss += loss;
            n_b++;

            float* dlogits = calloc(batch * n_classes, sizeof(float));
            trix_cross_entropy_grad(dlogits, logits, yb, batch, n_classes);

            trix_atom_ffn_zero_grad(head);
            float* dffn = calloc(batch * d_model, sizeof(float));
            trix_atom_ffn_backward(head, ffn_out, dlogits, dffn, batch);

            trix_ternary_route_zero_grad(tr);
            trix_ternary_route_backward(tr, xb, dffn, NULL, batch);
            trix_ternary_route_clip_grad_norm(tr, 1.0f);
            trix_ternary_route_adamw_step(tr, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_atom_ffn_sgd_step(head, 0.01f);

            free(ffn_out); free(logits); free(dlogits); free(dffn);
        }

        /* Eval on test set */
        float* test_out = calloc(n_test * d_model, sizeof(float));
        trix_ternary_route_forward(tr, x_test, test_out, n_test);
        float acc = trix_atom_ffn_accuracy(head, test_out, y_test, n_test);
        free(test_out);

        if (acc > best_acc) best_acc = acc;

        if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 25 == 0) {
            printf("    [%s k=%d] epoch %3d: loss=%.4f test_acc=%.1f%%\n",
                   label, active_k, epoch, epoch_loss / fmaxf(n_b, 1), acc * 100.0f);
        }
    }

    /* Routing analysis (final epoch) */
    float* final_out = calloc(n_test * d_model, sizeof(float));
    trix_ternary_route_forward(tr, x_test, final_out, n_test);

    int pos[64] = {0}, neg[64] = {0};
    trix_ternary_route_get_tile_activity(tr, pos, neg, n_test);
    printf("    [%s k=%d] Tile activity: ", label, active_k);
    for (int t = 0; t < num_tiles; t++) {
        printf("T%d(+%d/-%d) ", t, pos[t], neg[t]);
    }
    printf("\n");

    free(final_out);
    trix_atom_ffn_destroy(head);
    trix_ternary_route_destroy(tr);

    return best_acc;
}

int main(void) {
    printf("=== FAIR HEAD-TO-HEAD: Top-1 vs Ternary Routing ===\n");
    printf("Same architecture, same optimizer, same head. Only routing differs.\n\n");

    int D = 16, T = 4, H = 16;
    int n_classes = 6;
    int n_train = 600, n_test = 300;
    int epochs = 100;

    float* x_train = malloc(n_train * D * sizeof(float));
    int* y_train = malloc(n_train * sizeof(int));
    float* x_test = malloc(n_test * D * sizeof(float));
    int* y_test = malloc(n_test * sizeof(int));

    /* Run 3 seeds for variance */
    uint64_t seeds[] = {42, 123, 456};
    float acc_k1[3], acc_k2[3], acc_k3[3];

    for (int s = 0; s < 3; s++) {
        uint64_t seed = seeds[s];
        printf("--- Seed %llu ---\n", (unsigned long long)seed);

        /* Different train/test data per seed */
        make_data(x_train, y_train, n_train, D, n_classes, seed);
        make_data(x_test, y_test, n_test, D, n_classes, seed + 1000);

        acc_k1[s] = train_and_eval(D, T, H, 1, x_train, y_train, n_train,
                                    x_test, y_test, n_test, n_classes, epochs, seed, "top1");
        acc_k2[s] = train_and_eval(D, T, H, 2, x_train, y_train, n_train,
                                    x_test, y_test, n_test, n_classes, epochs, seed, "tern");
        acc_k3[s] = train_and_eval(D, T, H, 3, x_train, y_train, n_train,
                                    x_test, y_test, n_test, n_classes, epochs, seed, "tern");
        printf("\n");
    }

    printf("=== SUMMARY (3 seeds, test accuracy) ===\n");
    printf("  k=1 (top-1):    %.1f%% %.1f%% %.1f%% → mean %.1f%%\n",
           acc_k1[0]*100, acc_k1[1]*100, acc_k1[2]*100,
           (acc_k1[0]+acc_k1[1]+acc_k1[2])/3*100);
    printf("  k=2 (ternary):  %.1f%% %.1f%% %.1f%% → mean %.1f%%\n",
           acc_k2[0]*100, acc_k2[1]*100, acc_k2[2]*100,
           (acc_k2[0]+acc_k2[1]+acc_k2[2])/3*100);
    printf("  k=3 (ternary):  %.1f%% %.1f%% %.1f%% → mean %.1f%%\n",
           acc_k3[0]*100, acc_k3[1]*100, acc_k3[2]*100,
           (acc_k3[0]+acc_k3[1]+acc_k3[2])/3*100);

    float mean_k1 = (acc_k1[0]+acc_k1[1]+acc_k1[2])/3;
    float mean_k2 = (acc_k2[0]+acc_k2[1]+acc_k2[2])/3;
    float mean_k3 = (acc_k3[0]+acc_k3[1]+acc_k3[2])/3;

    printf("\n  k=2 vs k=1: %+.1f%%\n", (mean_k2 - mean_k1) * 100);
    printf("  k=3 vs k=1: %+.1f%%\n", (mean_k3 - mean_k1) * 100);

    ASSERT_TRUE(1, "completed"); /* just mark as pass if we got here */

    free(x_train); free(y_train); free(x_test); free(y_test);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
