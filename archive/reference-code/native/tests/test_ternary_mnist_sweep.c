/*
 * test_ternary_mnist_sweep.c — LR sweep × k sweep on MNIST
 *
 * Tests all combinations of:
 *   k ∈ {1, 2, 3, 4}
 *   lr ∈ {0.001, 0.003, 0.005, 0.01, 0.02}
 *
 * Same architecture for all: TrixTernaryRoutedFFN + projection + head.
 * Reports best LR per k, then compares at each model's best LR.
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── Data loading ── */

static uint32_t read_u32_be(FILE* f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) { fprintf(stderr, "read error\n"); exit(1); }
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
}

static float* load_images(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = (int)read_u32_be(f);
    read_u32_be(f); read_u32_be(f);
    size_t total = (size_t)(*n) * 784;
    uint8_t* raw = malloc(total);
    fread(raw, 1, total, f); fclose(f);
    float* data = malloc(total * sizeof(float));
    for (size_t i = 0; i < total; i++) data[i] = (float)raw[i] / 255.0f;
    free(raw);
    return data;
}

static int* load_labels(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = (int)read_u32_be(f);
    uint8_t* raw = malloc(*n);
    fread(raw, 1, *n, f); fclose(f);
    int* labels = malloc(*n * sizeof(int));
    for (int i = 0; i < *n; i++) labels[i] = (int)raw[i];
    free(raw);
    return labels;
}

static void shuffle(int* idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

/* ── Single training run ── */

static float run_once(
    int D, int T, int H, int K, float lr,
    float* x_train, int* y_train, int n_train,
    float* x_test, int* y_test, int n_test,
    int epochs, int batch, uint64_t seed)
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
    srand(seed);

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices, n_train);

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
        int correct = 0;
        int chunk = 500;
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
        if (acc > best_acc) best_acc = acc;
    }

    free(proj_out); free(ffn_out); free(logits);
    free(dlogits); free(dffn); free(dproj); free(indices);
    trix_atom_ffn_destroy(proj);
    trix_atom_ffn_destroy(head);
    trix_ternary_route_destroy(tr);

    return best_acc;
}

/* ── Main ── */

int main(int argc, char** argv) {
    const char* data_dir = "data/mnist";
    if (argc > 1) data_dir = argv[1];

    char path[512];
    int n_train, n_test;

    printf("Loading MNIST...\n");
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", data_dir);
    float* x_train = load_images(path, &n_train);
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", data_dir);
    int* y_train = load_labels(path, &n_train);
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", data_dir);
    float* x_test = load_images(path, &n_test);
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", data_dir);
    int* y_test = load_labels(path, &n_test);

    int D = 32, T = 4, H = 32;
    int epochs = 64, batch = 128;

    float lrs[] = {0.001f, 0.003f, 0.005f, 0.01f, 0.02f};
    int n_lrs = 5;
    int ks[] = {1, 2, 3, 4};
    int n_ks = 4;

    printf("\n=== LR SWEEP × K SWEEP (MNIST, D=%d, T=%d, H=%d, %d epochs) ===\n\n", D, T, H, epochs);

    /* results[k_idx][lr_idx] = best test accuracy */
    float results[4][5];

    printf("%-6s", "k\\lr");
    for (int l = 0; l < n_lrs; l++) printf(" %8.4f", lrs[l]);
    printf("    best_lr  best_acc\n");
    printf("------");
    for (int l = 0; l < n_lrs; l++) printf(" --------");
    printf("  --------  --------\n");

    for (int ki = 0; ki < n_ks; ki++) {
        int k = ks[ki];
        printf("k=%-4d", k);
        fflush(stdout);

        float best_acc_for_k = 0;
        float best_lr_for_k = 0;

        for (int li = 0; li < n_lrs; li++) {
            float acc = run_once(D, T, H, k, lrs[li],
                                 x_train, y_train, n_train,
                                 x_test, y_test, n_test,
                                 epochs, batch, 42);
            results[ki][li] = acc;
            printf(" %7.2f%%", acc * 100.0f);
            fflush(stdout);

            if (acc > best_acc_for_k) {
                best_acc_for_k = acc;
                best_lr_for_k = lrs[li];
            }
        }
        printf("  lr=%.4f  %6.2f%%\n", best_lr_for_k, best_acc_for_k * 100.0f);
    }

    printf("\n=== AT EACH MODEL'S BEST LR ===\n");
    for (int ki = 0; ki < n_ks; ki++) {
        float best = 0;
        int best_li = 0;
        for (int li = 0; li < n_lrs; li++) {
            if (results[ki][li] > best) { best = results[ki][li]; best_li = li; }
        }
        printf("  k=%d: %.2f%% (lr=%.4f)\n", ks[ki], best * 100.0f, lrs[best_li]);
    }

    /* Delta at best LR */
    float best_k1 = 0, best_k2 = 0, best_k3 = 0, best_k4 = 0;
    for (int li = 0; li < n_lrs; li++) {
        if (results[0][li] > best_k1) best_k1 = results[0][li];
        if (results[1][li] > best_k2) best_k2 = results[1][li];
        if (results[2][li] > best_k3) best_k3 = results[2][li];
        if (results[3][li] > best_k4) best_k4 = results[3][li];
    }
    printf("\n  k=2 vs k=1 (at each best LR): %+.2f%%\n", (best_k2 - best_k1) * 100);
    printf("  k=3 vs k=1 (at each best LR): %+.2f%%\n", (best_k3 - best_k1) * 100);
    printf("  k=4 vs k=1 (at each best LR): %+.2f%%\n", (best_k4 - best_k1) * 100);

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
