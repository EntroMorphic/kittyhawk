/*
 * test_ternary_mnist_seeds.c — Multi-seed ternary routing on MNIST
 *
 * 3 seeds × 4 k values at LR=0.01 and LR=0.02 (the two best from sweep).
 * 64 epochs each. Reports mean ± std for each k.
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── Data loading (identical to sweep) ── */
static uint32_t read_u32_be(FILE* f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) { fprintf(stderr, "read error\n"); exit(1); }
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
}
static float* load_images(const char* path, int* n) {
    FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = (int)read_u32_be(f); read_u32_be(f); read_u32_be(f);
    size_t total = (size_t)(*n) * 784;
    uint8_t* raw = malloc(total); fread(raw, 1, total, f); fclose(f);
    float* data = malloc(total * sizeof(float));
    for (size_t i = 0; i < total; i++) data[i] = (float)raw[i] / 255.0f;
    free(raw); return data;
}
static int* load_labels(const char* path, int* n) {
    FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = (int)read_u32_be(f);
    uint8_t* raw = malloc(*n); fread(raw, 1, *n, f); fclose(f);
    int* labels = malloc(*n * sizeof(int));
    for (int i = 0; i < *n; i++) labels[i] = (int)raw[i];
    free(raw); return labels;
}
static void shuffle_idx(int* idx, int n, uint64_t seed) {
    srand(seed);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

/* ── Single run ── */
static float run_once(
    int D, int T, int H, int K, float lr,
    float* x_train, int* y_train, int n_train,
    float* x_test, int* y_test, int n_test,
    int epochs, int batch, uint64_t seed,
    float* epoch0_acc_out)
{
    TrixTernaryRouteConfig cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f,
    };
    TrixTernaryRoutedFFN* tr = trix_ternary_route_create(cfg, seed);
    TrixAtomFFN* proj = trix_atom_ffn_create(784, D * 2, D, seed);
    TrixAtomFFN* head = trix_atom_ffn_create(D, 64, 10, seed + 1);

    int* indices = malloc(n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) indices[i] = i;

    float* proj_out = malloc(batch * D * sizeof(float));
    float* ffn_out  = malloc(batch * D * sizeof(float));
    float* logits   = malloc(batch * 10 * sizeof(float));
    float* dlogits  = malloc(batch * 10 * sizeof(float));
    float* dffn     = malloc(batch * D * sizeof(float));
    float* dproj    = malloc(batch * D * sizeof(float));

    float best_acc = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_idx(indices, n_train, seed + epoch * 1000);

        for (int b = 0; b + batch <= n_train; b += batch) {
            float* xb = malloc(batch * 784 * sizeof(float));
            int* yb = malloc(batch * sizeof(int));
            for (int i = 0; i < batch; i++) {
                memcpy(xb + i * 784, x_train + indices[b+i] * 784, 784 * sizeof(float));
                yb[i] = y_train[indices[b+i]];
            }

            trix_atom_ffn_forward(proj, xb, proj_out, batch);
            trix_ternary_route_forward(tr, proj_out, ffn_out, batch);
            trix_atom_ffn_forward(head, ffn_out, logits, batch);
            trix_cross_entropy_grad(dlogits, logits, yb, batch, 10);

            trix_atom_ffn_zero_grad(head);
            trix_atom_ffn_backward(head, ffn_out, dlogits, dffn, batch);
            trix_ternary_route_zero_grad(tr);
            trix_ternary_route_backward(tr, proj_out, dffn, dproj, batch);
            trix_atom_ffn_zero_grad(proj);
            trix_atom_ffn_backward(proj, xb, dproj, NULL, batch);

            trix_ternary_route_clip_grad_norm(tr, 1.0f);
            trix_ternary_route_adamw_step(tr, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_atom_ffn_sgd_step(head, lr);
            trix_atom_ffn_sgd_step(proj, lr);

            free(xb); free(yb);
        }

        /* Eval */
        int correct = 0, chunk = 500;
        for (int off = 0; off < n_test; off += chunk) {
            int n = (off + chunk <= n_test) ? chunk : n_test - off;
            float* po = malloc(n * D * sizeof(float));
            float* fo = malloc(n * D * sizeof(float));
            float* lo = malloc(n * 10 * sizeof(float));
            trix_atom_ffn_forward(proj, x_test + off * 784, po, n);
            trix_ternary_route_forward(tr, po, fo, n);
            trix_atom_ffn_forward(head, fo, lo, n);
            int* pred = malloc(n * sizeof(int));
            trix_argmax(pred, lo, n, 10);
            for (int i = 0; i < n; i++)
                if (pred[i] == y_test[off+i]) correct++;
            free(po); free(fo); free(lo); free(pred);
        }
        float acc = (float)correct / (float)n_test;
        if (epoch == 0 && epoch0_acc_out) *epoch0_acc_out = acc;
        if (acc > best_acc) best_acc = acc;
    }

    free(proj_out); free(ffn_out); free(logits);
    free(dlogits); free(dffn); free(dproj); free(indices);
    trix_atom_ffn_destroy(proj);
    trix_atom_ffn_destroy(head);
    trix_ternary_route_destroy(tr);
    return best_acc;
}

int main(int argc, char** argv) {
    const char* data_dir = "data/mnist";
    if (argc > 1) data_dir = argv[1];

    char path[512];
    int n_train, n_test;
    printf("Loading MNIST...\n");
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", data_dir); float* x_train = load_images(path, &n_train);
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", data_dir); int* y_train = load_labels(path, &n_train);
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", data_dir);  float* x_test = load_images(path, &n_test);
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", data_dir);  int* y_test = load_labels(path, &n_test);

    int D = 32, T = 4, H = 32;
    int epochs = 64, batch = 128;
    float lrs[] = {0.01f, 0.02f};
    int n_lrs = 2;
    uint64_t seeds[] = {42, 123, 456};
    int n_seeds = 3;
    int ks[] = {1, 2, 3, 4};
    int n_ks = 4;

    printf("\n=== MULTI-SEED TERNARY ROUTING (MNIST, %d epochs, %d seeds) ===\n\n", epochs, n_seeds);

    /* results[k][lr][seed] */
    float results[4][2][3];
    float epoch0[4][2][3];

    for (int ki = 0; ki < n_ks; ki++) {
        for (int li = 0; li < n_lrs; li++) {
            for (int si = 0; si < n_seeds; si++) {
                printf("  k=%d lr=%.3f seed=%llu ... ", ks[ki], lrs[li], (unsigned long long)seeds[si]);
                fflush(stdout);
                float e0 = 0;
                float acc = run_once(D, T, H, ks[ki], lrs[li],
                                     x_train, y_train, n_train,
                                     x_test, y_test, n_test,
                                     epochs, batch, seeds[si], &e0);
                results[ki][li][si] = acc;
                epoch0[ki][li][si] = e0;
                printf("%.2f%% (epoch0: %.2f%%)\n", acc * 100, e0 * 100);
            }
        }
    }

    printf("\n=== RESULTS BY LR ===\n\n");
    for (int li = 0; li < n_lrs; li++) {
        printf("LR = %.3f:\n", lrs[li]);
        printf("  %-6s  %8s %8s %8s  %8s ± %5s   %8s ± %5s\n",
               "k", "seed42", "seed123", "seed456", "mean", "std", "ep0_mean", "std");
        for (int ki = 0; ki < n_ks; ki++) {
            float* r = results[ki][li];
            float* e = epoch0[ki][li];
            float mean_r = (r[0] + r[1] + r[2]) / 3.0f;
            float mean_e = (e[0] + e[1] + e[2]) / 3.0f;
            float var_r = 0, var_e = 0;
            for (int s = 0; s < 3; s++) {
                var_r += (r[s] - mean_r) * (r[s] - mean_r);
                var_e += (e[s] - mean_e) * (e[s] - mean_e);
            }
            float std_r = sqrtf(var_r / 2.0f); /* sample std, ddof=1 */
            float std_e = sqrtf(var_e / 2.0f);
            printf("  k=%-4d  %7.2f%% %7.2f%% %7.2f%%  %7.2f%% ± %.2f%%  %7.2f%% ± %.2f%%\n",
                   ks[ki], r[0]*100, r[1]*100, r[2]*100,
                   mean_r*100, std_r*100, mean_e*100, std_e*100);
        }
        printf("\n");
    }

    /* Best LR per k, then final comparison */
    printf("=== FINAL COMPARISON (best LR per k) ===\n\n");
    for (int ki = 0; ki < n_ks; ki++) {
        float best_mean = 0;
        int best_li = 0;
        for (int li = 0; li < n_lrs; li++) {
            float* r = results[ki][li];
            float mean = (r[0] + r[1] + r[2]) / 3.0f;
            if (mean > best_mean) { best_mean = mean; best_li = li; }
        }
        float* r = results[ki][best_li];
        float var = 0;
        for (int s = 0; s < 3; s++) var += (r[s] - best_mean) * (r[s] - best_mean);
        float std = sqrtf(var / 2.0f);
        printf("  k=%d: %.2f%% ± %.2f%% (lr=%.3f)\n",
               ks[ki], best_mean * 100, std * 100, lrs[best_li]);
    }

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
