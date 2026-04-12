/*
 * test_ternary_mnist.c — Ternary routing on MNIST
 *
 * Fair comparison: k=1 (top-1) vs k=2 vs k=3 vs k=4
 * Same architecture, same optimizer, same head. Only routing differs.
 *
 * Expects MNIST data in data/mnist/ relative to working directory.
 *
 * Model: TrixTernaryRoutedFFN(784→D) + TrixAtomFFN(D→hidden→10)
 * ~50K params for D=32, T=4, H=32.
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── Data loading (same as test_mnist.c) ── */

static uint32_t read_u32_be(FILE* f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) { fprintf(stderr, "read error\n"); exit(1); }
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
}

static float* load_images(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    uint32_t magic = read_u32_be(f);
    if (magic != 2051) { fprintf(stderr, "Bad magic %u\n", magic); exit(1); }
    *n = (int)read_u32_be(f);
    read_u32_be(f); read_u32_be(f); /* rows, cols */
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
    read_u32_be(f); /* magic */
    *n = (int)read_u32_be(f);
    uint8_t* raw = malloc(*n);
    fread(raw, 1, *n, f); fclose(f);
    int* labels = malloc(*n * sizeof(int));
    for (int i = 0; i < *n; i++) labels[i] = (int)raw[i];
    free(raw);
    return labels;
}

/* Fisher-Yates shuffle */
static void shuffle(int* idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

/* ── Training loop for one configuration ── */

typedef struct {
    float test_acc;
    float train_loss;
    float epoch0_acc;
    double wall_secs;
} RunResult;

static RunResult train_config(
    int D, int T, int H, int K,
    float* x_train, int* y_train, int n_train,
    float* x_test, int* y_test, int n_test,
    int epochs, int batch, float lr, uint64_t seed)
{
    TrixTernaryRouteConfig cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f,
    };
    TrixTernaryRoutedFFN* tr = trix_ternary_route_create(cfg, seed);

    /* Project 784→D, then classification head D→64→10 */
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

    RunResult result = {0};
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    srand(seed);

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices, n_train);
        float epoch_loss = 0;
        int n_b = 0;

        for (int b = 0; b + batch <= n_train; b += batch) {
            /* Gather batch */
            float* xb = malloc(batch * 784 * sizeof(float));
            int* yb = malloc(batch * sizeof(int));
            for (int i = 0; i < batch; i++) {
                memcpy(xb + i * 784, x_train + indices[b + i] * 784, 784 * sizeof(float));
                yb[i] = y_train[indices[b + i]];
            }

            /* Forward: project → ternary FFN → classify */
            trix_atom_ffn_forward(proj, xb, proj_out, batch);
            trix_ternary_route_forward(tr, proj_out, ffn_out, batch);
            trix_atom_ffn_forward(head, ffn_out, logits, batch);

            float loss = trix_cross_entropy_loss(logits, yb, batch, 10);
            epoch_loss += loss;
            n_b++;

            /* Backward */
            trix_cross_entropy_grad(dlogits, logits, yb, batch, 10);

            trix_atom_ffn_zero_grad(head);
            trix_atom_ffn_backward(head, ffn_out, dlogits, dffn, batch);

            trix_ternary_route_zero_grad(tr);
            trix_ternary_route_backward(tr, proj_out, dffn, dproj, batch);

            trix_atom_ffn_zero_grad(proj);
            trix_atom_ffn_backward(proj, xb, dproj, NULL, batch);

            /* Optimizer */
            trix_ternary_route_clip_grad_norm(tr, 1.0f);
            trix_ternary_route_adamw_step(tr, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_atom_ffn_sgd_step(head, lr);
            trix_atom_ffn_sgd_step(proj, lr);

            free(xb); free(yb);
        }

        /* Eval on test set (chunked to avoid huge allocs) */
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
            for (int i = 0; i < n; i++) {
                if (pred[i] == y_test[off + i]) correct++;
            }
            free(po); free(fo); free(lo); free(pred);
        }
        float acc = (float)correct / (float)n_test;

        if (epoch == 0) result.epoch0_acc = acc;

        float avg_loss = epoch_loss / fmaxf(n_b, 1);
        if (epoch % 5 == 0 || epoch == epochs - 1) {
            printf("    [k=%d] epoch %2d: loss=%.4f test_acc=%.2f%%\n",
                   K, epoch, avg_loss, acc * 100.0f);
        }

        result.test_acc = acc;
        result.train_loss = avg_loss;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    result.wall_secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);

    /* Print routing patterns */
    float* test_proj = malloc(n_test * D * sizeof(float));
    float* test_ffn = malloc(n_test * D * sizeof(float));
    trix_atom_ffn_forward(proj, x_test, test_proj, n_test);
    trix_ternary_route_forward(tr, test_proj, test_ffn, n_test);

    int pos[64] = {0}, neg[64] = {0};
    trix_ternary_route_get_tile_activity(tr, pos, neg, n_test);
    printf("    [k=%d] Tiles: ", K);
    for (int t = 0; t < T; t++) printf("T%d(+%d/-%d) ", t, pos[t], neg[t]);
    printf("\n");

    free(test_proj); free(test_ffn);
    free(proj_out); free(ffn_out); free(logits);
    free(dlogits); free(dffn); free(dproj);
    free(indices);
    trix_atom_ffn_destroy(proj);
    trix_atom_ffn_destroy(head);
    trix_ternary_route_destroy(tr);

    return result;
}

/* ── Main ── */

int main(int argc, char** argv) {
    const char* data_dir = "data/mnist";
    if (argc > 1) data_dir = argv[1];

    char path[512];

    printf("Loading MNIST from %s...\n", data_dir);

    int n_train, n_test;
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", data_dir);
    float* x_train = load_images(path, &n_train);
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", data_dir);
    int* y_train = load_labels(path, &n_train);
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", data_dir);
    float* x_test = load_images(path, &n_test);
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", data_dir);
    int* y_test = load_labels(path, &n_test);

    printf("Train: %d, Test: %d\n\n", n_train, n_test);

    /* Config: ~50K params
     * D=32, T=4, H=32, proj=784→64→32 (~50K), head=32→64→10 (~3K)
     * Ternary route: 4 tiles × (32*32 + 32 + 32*32 + 32) = 4 × 2112 = 8448 */
    int D = 32, T = 4, H = 32;
    int epochs = 20;
    int batch = 128;
    float lr = 0.005f;

    printf("=== MNIST TERNARY ROUTING: k-sweep ===\n");
    printf("D=%d, T=%d, H=%d, epochs=%d, batch=%d, lr=%.4f\n\n", D, T, H, epochs, batch, lr);

    RunResult results[4];

    for (int k = 1; k <= T; k++) {
        printf("--- k=%d ---\n", k);
        results[k-1] = train_config(D, T, H, k,
                                     x_train, y_train, n_train,
                                     x_test, y_test, n_test,
                                     epochs, batch, lr, 42);
        printf("\n");
    }

    printf("=== SUMMARY ===\n");
    printf("  %-8s %10s %10s %10s %10s\n", "k", "test_acc", "epoch0_acc", "loss", "wall_s");
    for (int k = 1; k <= T; k++) {
        RunResult r = results[k-1];
        printf("  k=%-6d %9.2f%% %9.2f%% %10.4f %9.1fs\n",
               k, r.test_acc * 100, r.epoch0_acc * 100, r.train_loss, r.wall_secs);
    }

    printf("\n  k=2 vs k=1: %+.2f%% test accuracy\n",
           (results[1].test_acc - results[0].test_acc) * 100);
    printf("  k=3 vs k=1: %+.2f%% test accuracy\n",
           (results[2].test_acc - results[0].test_acc) * 100);
    printf("  k=4 vs k=1: %+.2f%% test accuracy\n",
           (results[3].test_acc - results[0].test_acc) * 100);

    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
