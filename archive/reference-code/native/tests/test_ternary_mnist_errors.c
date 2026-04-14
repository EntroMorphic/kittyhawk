/*
 * test_ternary_mnist_errors.c — Error analysis of the last 2.5%
 *
 * Trains the best config (k=4, LR=0.02, 64 epochs), then:
 *   1. Identifies every misclassified test image
 *   2. Per-digit confusion matrix
 *   3. Per-digit accuracy breakdown
 *   4. Confidence analysis (softmax entropy of errors vs correct)
 *   5. Routing pattern analysis: do errors route differently?
 *   6. Hardest images: which test indices are misclassified by all 3 seeds?
 */

#include "trix_ternary_route.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── Data loading ── */
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

/* Softmax for confidence analysis */
static void softmax10(float* out, const float* logits) {
    float mx = logits[0];
    for (int j = 1; j < 10; j++) if (logits[j] > mx) mx = logits[j];
    float sum = 0;
    for (int j = 0; j < 10; j++) { out[j] = expf(logits[j] - mx); sum += out[j]; }
    for (int j = 0; j < 10; j++) out[j] /= sum;
}

/* Shannon entropy of a probability distribution */
static float entropy10(const float* probs) {
    float h = 0;
    for (int j = 0; j < 10; j++) {
        if (probs[j] > 1e-10f) h -= probs[j] * logf(probs[j]);
    }
    return h;
}

/* ── Train and return model components for analysis ── */
typedef struct {
    TrixTernaryRoutedFFN* tr;
    TrixAtomFFN* proj;
    TrixAtomFFN* head;
} TrainedModel;

static TrainedModel train_model(
    int D, int T, int H, int K, float lr,
    float* x_train, int* y_train, int n_train,
    int epochs, int batch, uint64_t seed)
{
    TrixTernaryRouteConfig cfg = {
        .d_model = D, .num_tiles = T, .tile_hidden = H,
        .active_k = K, .output_scale_init = 0.1f, .ln_eps = 1e-5f,
    };
    TrainedModel m;
    m.tr = trix_ternary_route_create(cfg, seed);
    m.proj = trix_atom_ffn_create(784, D * 2, D, seed);
    m.head = trix_atom_ffn_create(D, 64, 10, seed + 1);

    int* indices = malloc(n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) indices[i] = i;

    float* proj_out = malloc(batch * D * sizeof(float));
    float* ffn_out  = malloc(batch * D * sizeof(float));
    float* logits   = malloc(batch * 10 * sizeof(float));
    float* dlogits  = malloc(batch * 10 * sizeof(float));
    float* dffn     = malloc(batch * D * sizeof(float));
    float* dproj    = malloc(batch * D * sizeof(float));

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_idx(indices, n_train, seed + epoch * 1000);
        for (int b = 0; b + batch <= n_train; b += batch) {
            float* xb = malloc(batch * 784 * sizeof(float));
            int* yb = malloc(batch * sizeof(int));
            for (int i = 0; i < batch; i++) {
                memcpy(xb + i * 784, x_train + indices[b+i] * 784, 784 * sizeof(float));
                yb[i] = y_train[indices[b+i]];
            }
            trix_atom_ffn_forward(m.proj, xb, proj_out, batch);
            trix_ternary_route_forward(m.tr, proj_out, ffn_out, batch);
            trix_atom_ffn_forward(m.head, ffn_out, logits, batch);
            trix_cross_entropy_grad(dlogits, logits, yb, batch, 10);

            trix_atom_ffn_zero_grad(m.head);
            trix_atom_ffn_backward(m.head, ffn_out, dlogits, dffn, batch);
            trix_ternary_route_zero_grad(m.tr);
            trix_ternary_route_backward(m.tr, proj_out, dffn, dproj, batch);
            trix_atom_ffn_zero_grad(m.proj);
            trix_atom_ffn_backward(m.proj, xb, dproj, NULL, batch);

            trix_ternary_route_clip_grad_norm(m.tr, 1.0f);
            trix_ternary_route_adamw_step(m.tr, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
            trix_atom_ffn_sgd_step(m.head, lr);
            trix_atom_ffn_sgd_step(m.proj, lr);
            free(xb); free(yb);
        }
    }
    free(proj_out); free(ffn_out); free(logits);
    free(dlogits); free(dffn); free(dproj); free(indices);
    return m;
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

    int D = 32, T = 4, H = 32, K = 4;
    float lr = 0.02f;
    int epochs = 64, batch = 128;

    /* Train 3 models for consensus analysis */
    uint64_t seeds[] = {42, 123, 456};
    int n_seeds = 3;
    int* wrong_by[3]; /* per-seed: 1 if wrong, 0 if correct */
    int* pred_by[3];  /* per-seed: predicted label */

    printf("\n=== TRAINING 3 MODELS (k=%d, lr=%.3f, %d epochs) ===\n\n", K, lr, epochs);

    for (int s = 0; s < n_seeds; s++) {
        printf("Seed %llu...\n", (unsigned long long)seeds[s]);
        TrainedModel m = train_model(D, T, H, K, lr, x_train, y_train, n_train,
                                      epochs, batch, seeds[s]);

        wrong_by[s] = calloc(n_test, sizeof(int));
        pred_by[s] = calloc(n_test, sizeof(int));

        /* Eval all test images one by one for detailed analysis */
        float* po = malloc(D * sizeof(float));
        float* fo = malloc(D * sizeof(float));
        float* lo = malloc(10 * sizeof(float));

        for (int i = 0; i < n_test; i++) {
            trix_atom_ffn_forward(m.proj, x_test + i * 784, po, 1);
            trix_ternary_route_forward(m.tr, po, fo, 1);
            trix_atom_ffn_forward(m.head, fo, lo, 1);

            int pred = 0;
            for (int j = 1; j < 10; j++) if (lo[j] > lo[pred]) pred = j;
            pred_by[s][i] = pred;
            wrong_by[s][i] = (pred != y_test[i]) ? 1 : 0;
        }

        int correct = 0;
        for (int i = 0; i < n_test; i++) correct += (1 - wrong_by[s][i]);
        printf("  Seed %llu: %d/10000 correct (%.2f%%)\n\n",
               (unsigned long long)seeds[s], correct, (float)correct / 100.0f);

        free(po); free(fo); free(lo);
        trix_atom_ffn_destroy(m.proj);
        trix_atom_ffn_destroy(m.head);
        trix_ternary_route_destroy(m.tr);
    }

    /* ── 1. Consensus errors: wrong by ALL 3 seeds ── */
    int n_consensus_wrong = 0;
    int* consensus_wrong = calloc(n_test, sizeof(int)); /* 1 if all 3 wrong */
    for (int i = 0; i < n_test; i++) {
        if (wrong_by[0][i] && wrong_by[1][i] && wrong_by[2][i]) {
            consensus_wrong[i] = 1;
            n_consensus_wrong++;
        }
    }

    int n_any_wrong = 0;
    for (int i = 0; i < n_test; i++) {
        if (wrong_by[0][i] || wrong_by[1][i] || wrong_by[2][i]) n_any_wrong++;
    }

    printf("=== ERROR CONSENSUS ===\n");
    printf("  Wrong by at least 1 seed: %d images\n", n_any_wrong);
    printf("  Wrong by all 3 seeds:     %d images (%.2f%%)\n",
           n_consensus_wrong, n_consensus_wrong / 100.0f);
    printf("  Wrong by exactly 1 seed:  %d images\n",
           n_any_wrong - n_consensus_wrong -
           /* count wrong by exactly 2 */
           ({int c=0; for(int i=0;i<n_test;i++){int s=wrong_by[0][i]+wrong_by[1][i]+wrong_by[2][i]; if(s==2)c++;} c;}));

    /* Count by number of seeds wrong */
    int wrong_count[4] = {0};
    for (int i = 0; i < n_test; i++) {
        int s = wrong_by[0][i] + wrong_by[1][i] + wrong_by[2][i];
        wrong_count[s]++;
    }
    printf("\n  Wrong by 0 seeds: %d (%.2f%%) — always correct\n", wrong_count[0], wrong_count[0]/100.0f);
    printf("  Wrong by 1 seed:  %d (%.2f%%) — borderline\n", wrong_count[1], wrong_count[1]/100.0f);
    printf("  Wrong by 2 seeds: %d (%.2f%%) — likely hard\n", wrong_count[2], wrong_count[2]/100.0f);
    printf("  Wrong by 3 seeds: %d (%.2f%%) — truly hard\n", wrong_count[3], wrong_count[3]/100.0f);

    /* ── 2. Per-digit accuracy (using seed 42) ── */
    printf("\n=== PER-DIGIT ACCURACY (seed 42) ===\n");
    int digit_total[10] = {0}, digit_correct[10] = {0};
    for (int i = 0; i < n_test; i++) {
        digit_total[y_test[i]]++;
        if (!wrong_by[0][i]) digit_correct[y_test[i]]++;
    }
    printf("  Digit   Total  Correct  Accuracy  Errors\n");
    for (int d = 0; d < 10; d++) {
        float acc = (float)digit_correct[d] / (float)digit_total[d] * 100.0f;
        printf("    %d     %5d    %5d   %5.2f%%   %4d\n",
               d, digit_total[d], digit_correct[d], acc, digit_total[d] - digit_correct[d]);
    }

    /* ── 3. Confusion matrix (seed 42) ── */
    printf("\n=== CONFUSION MATRIX (seed 42, errors only) ===\n");
    int confusion[10][10] = {{0}};
    for (int i = 0; i < n_test; i++) {
        if (wrong_by[0][i]) {
            confusion[y_test[i]][pred_by[0][i]]++;
        }
    }
    printf("  True\\Pred ");
    for (int j = 0; j < 10; j++) printf(" %3d", j);
    printf("\n");
    for (int t = 0; t < 10; t++) {
        int row_sum = 0;
        for (int p = 0; p < 10; p++) row_sum += confusion[t][p];
        if (row_sum == 0) continue;
        printf("    %d      ", t);
        for (int p = 0; p < 10; p++) {
            if (confusion[t][p] > 0) printf(" %3d", confusion[t][p]);
            else printf("   .");
        }
        printf("  (%d errors)\n", row_sum);
    }

    /* ── 4. Top confusion pairs ── */
    printf("\n=== TOP CONFUSION PAIRS ===\n");
    typedef struct { int true_d; int pred_d; int count; } ConfPair;
    ConfPair pairs[100];
    int n_pairs = 0;
    for (int t = 0; t < 10; t++) {
        for (int p = 0; p < 10; p++) {
            if (t != p && confusion[t][p] > 0) {
                pairs[n_pairs++] = (ConfPair){t, p, confusion[t][p]};
            }
        }
    }
    /* Sort by count descending */
    for (int i = 0; i < n_pairs - 1; i++) {
        for (int j = i + 1; j < n_pairs; j++) {
            if (pairs[j].count > pairs[i].count) {
                ConfPair tmp = pairs[i]; pairs[i] = pairs[j]; pairs[j] = tmp;
            }
        }
    }
    for (int i = 0; i < 15 && i < n_pairs; i++) {
        printf("  %d → %d: %d errors\n", pairs[i].true_d, pairs[i].pred_d, pairs[i].count);
    }

    /* ── 5. Consensus hard images ── */
    printf("\n=== HARDEST IMAGES (wrong by all 3 seeds) ===\n");
    printf("  Index  True  Pred(s42) Pred(s123) Pred(s456)  Consensus\n");
    int shown = 0;
    for (int i = 0; i < n_test && shown < 30; i++) {
        if (!consensus_wrong[i]) continue;
        int agree = (pred_by[0][i] == pred_by[1][i] && pred_by[1][i] == pred_by[2][i]);
        printf("  %5d   %d      %d          %d          %d       %s\n",
               i, y_test[i], pred_by[0][i], pred_by[1][i], pred_by[2][i],
               agree ? "all agree" : "disagree");
        shown++;
    }

    /* ── 6. Pixel intensity stats for errors vs correct ── */
    printf("\n=== PIXEL STATISTICS: ERRORS vs CORRECT ===\n");
    float err_mean_intensity = 0, cor_mean_intensity = 0;
    float err_mean_nonzero = 0, cor_mean_nonzero = 0;
    int n_err = 0, n_cor = 0;
    for (int i = 0; i < n_test; i++) {
        float sum = 0; int nz = 0;
        for (int j = 0; j < 784; j++) {
            sum += x_test[i * 784 + j];
            if (x_test[i * 784 + j] > 0.01f) nz++;
        }
        if (wrong_by[0][i]) {
            err_mean_intensity += sum / 784.0f;
            err_mean_nonzero += nz;
            n_err++;
        } else {
            cor_mean_intensity += sum / 784.0f;
            cor_mean_nonzero += nz;
            n_cor++;
        }
    }
    printf("  Correct (%d): mean intensity=%.4f, mean nonzero pixels=%.1f\n",
           n_cor, cor_mean_intensity / n_cor, cor_mean_nonzero / n_cor);
    printf("  Errors  (%d): mean intensity=%.4f, mean nonzero pixels=%.1f\n",
           n_err, err_mean_intensity / n_err, err_mean_nonzero / n_err);

    for (int s = 0; s < n_seeds; s++) { free(wrong_by[s]); free(pred_by[s]); }
    free(consensus_wrong);
    free(x_train); free(y_train); free(x_test); free(y_test);
    return 0;
}
