/*
 * image_distance_probe.c — measure Hamming vs hamming_norm on image pipelines.
 *
 * This is the decisive gate for whether `hamming_norm` (the substrate
 * distance fix found on Go positions in the substrate_distance_refinement
 * cycle) is a substrate-wide primitive or a Go-specific refinement.
 *
 * Intentionally mirrors tools/go_probe.c in structure: load dataset,
 * build ternary signatures, brute-force k-NN classification at k ∈
 * {50, 100, 200}, report head-to-head between the two metrics.
 *
 * What this probe does:
 *   1. Load MNIST / Fashion-MNIST / CIFAR-10 via glyph_dataset_load_auto.
 *   2. Optional: deskew, per-image normalize (matches direct_lsh defaults).
 *   3. Calibrate τ on train sample at --density (default 0.33, balanced base-3).
 *   4. Quantize all train + test intensity to packed-trit signatures.
 *   5. For each metric, run brute-force k-NN class vote at k = 50/100/200
 *      over a capped test subset. Report accuracy.
 *
 * Scope guardrails:
 *   - Raw intensity only (no gradients, no multi-scale). Keeps the
 *     measurement apples-to-apples against go_probe's raw ternary Go
 *     positions. Gradients/MS4 are explicit downstream enrichments that
 *     already include density-decorrelation; measuring there would obscure
 *     the distance-fix signal.
 *   - Single dataset per run. Multiple runs for MNIST/Fashion/CIFAR.
 *
 * Decision rule (from substrate_distance_refinement_closeout §Updated
 * recommendations):
 *   - hamming_norm improves all three by ≥2pp at k=50 → substrate-wide
 *     fix. Retrofit into direct_lsh. Proceed to routed_go trainer (#41).
 *   - hamming_norm neutral (±1pp) on images → Go-specific. hamming_norm
 *     still useful for Go + other sparse-discrete domains. routed_go
 *     trainer proceeds, but substrate-wide claim dropped.
 *   - hamming_norm hurts on images → the Go finding is *inversely*
 *     useful: it shows raw Hamming already compensates implicitly for
 *     image pipelines' density decorrelation. Rare case; would need
 *     careful reinterpretation.
 */

#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_sig.h"
#include "m4t_trit_pack.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLASSES 10

/* Count nonzero trits in a packed-trit signature.
 *
 * Packing (per m4t/src/m4t_trit_pack.h):
 *   +1 → 0b01      0 → 0b00     −1 → 0b10
 * Each byte holds 4 trits as 2-bit pairs. A nonzero trit sets exactly
 * one of its two bits, so popcount(byte) = count of nonzero trits in
 * that byte's 4 trits. Density = sum of popcounts across all bytes. */
static int trit_density_packed(const uint8_t* sig, int n_bytes) {
    int d = 0;
    for (int i = 0; i < n_bytes; i++) d += __builtin_popcount(sig[i]);
    return d;
}

typedef enum { MET_HAMMING = 0, MET_HAMMING_NORM = 1 } metric_t;

/* Pair-score: smaller = closer. For hamming_norm, scale by inverse
 * sum-of-densities (fixed-point 1024 / denom) to match the go_probe
 * formula exactly. */
static int32_t score_pair(
    const uint8_t* a, int32_t h_raw, int da,
    const uint8_t* b, int db,
    const uint8_t* mask, int sig_bytes,
    metric_t metric)
{
    (void)a; (void)b; (void)mask; (void)sig_bytes;
    if (metric == MET_HAMMING) return h_raw;
    int denom = da + db + 1;
    return (h_raw * 1024) / denom;
}

typedef struct { int32_t d; int cls; } neighbor_t;

static int cmp_neigh(const void* a, const void* b) {
    return ((const neighbor_t*)a)->d - ((const neighbor_t*)b)->d;
}

static double knn_class_accuracy(
    const uint8_t* train_sigs, const int* y_train, const int* dens_train, int n_tr,
    const uint8_t* test_sigs,  const int* y_test,  const int* dens_test,  int n_te,
    const uint8_t* mask, int sig_bytes,
    int k, metric_t metric)
{
    neighbor_t* neigh = malloc((size_t)n_tr * sizeof(neighbor_t));
    int correct = 0;
    for (int t = 0; t < n_te; t++) {
        const uint8_t* q = test_sigs + (size_t)t * sig_bytes;
        int dq = dens_test[t];
        for (int i = 0; i < n_tr; i++) {
            int32_t h = m4t_popcount_dist(
                q, train_sigs + (size_t)i * sig_bytes, mask, sig_bytes);
            neigh[i].d = score_pair(q, h, dq,
                                    train_sigs + (size_t)i * sig_bytes, dens_train[i],
                                    mask, sig_bytes, metric);
            neigh[i].cls = y_train[i];
        }
        qsort(neigh, (size_t)n_tr, sizeof(neighbor_t), cmp_neigh);
        int votes[N_CLASSES] = {0};
        int kk = k < n_tr ? k : n_tr;
        for (int i = 0; i < kk; i++) votes[neigh[i].cls]++;
        int pred = 0;
        for (int c = 1; c < N_CLASSES; c++) if (votes[c] > votes[pred]) pred = c;
        if (pred == y_test[t]) correct++;
    }
    free(neigh);
    return (double)correct / (double)n_te;
}

int main(int argc, char** argv) {
    /* CLI: reuse glyph_config for --data/--density/--no_deskew/--normalize,
     * add --metric and --test_cap. */
    double density_param = 0.33;
    const char* metric_name = "hamming";
    int test_cap = 500;
    int no_deskew = 0;
    int normalize = 0;
    const char* data_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else if (!strcmp(argv[i], "--density") && i + 1 < argc) density_param = atof(argv[++i]);
        else if (!strcmp(argv[i], "--metric") && i + 1 < argc) metric_name = argv[++i];
        else if (!strcmp(argv[i], "--test_cap") && i + 1 < argc) test_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no_deskew")) no_deskew = 1;
        else if (!strcmp(argv[i], "--normalize")) normalize = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s --data DIR [--density D] [--metric {hamming,hamming_norm}]\n"
                   "           [--test_cap N] [--no_deskew] [--normalize]\n", argv[0]);
            return 0;
        }
    }
    if (!data_dir) {
        fprintf(stderr, "error: --data DIR is required\n");
        return 1;
    }

    metric_t metric;
    if (!strcmp(metric_name, "hamming")) metric = MET_HAMMING;
    else if (!strcmp(metric_name, "hamming_norm")) metric = MET_HAMMING_NORM;
    else { fprintf(stderr, "unknown metric: %s\n", metric_name); return 1; }

    printf("== image_distance_probe metric=%s density=%.3f data=%s ==\n",
           metric_name, density_param, data_dir);

    glyph_dataset_t ds;
    if (glyph_dataset_load_auto(&ds, data_dir) != 0) return 1;
    if (!no_deskew) glyph_dataset_deskew(&ds);
    if (normalize) glyph_dataset_normalize(&ds);
    printf("loaded: train=%d test=%d  dim=%d  img=%dx%d\n",
           ds.n_train, ds.n_test, ds.input_dim, ds.img_w, ds.img_h);

    /* Quantize train + test at calibrated τ. */
    int sig_bytes = M4T_TRIT_PACKED_BYTES(ds.input_dim);
    int64_t tau = glyph_sig_quantize_tau(ds.x_train, ds.n_train, ds.input_dim, density_param);
    printf("tau = %lld  (density=%.3f, sig_bytes=%d)\n",
           (long long)tau, density_param, sig_bytes);

    uint8_t* train_sigs = calloc((size_t)ds.n_train, sig_bytes);
    uint8_t* test_sigs  = calloc((size_t)ds.n_test,  sig_bytes);
    clock_t t0 = clock();
    glyph_sig_quantize_batch(ds.x_train, ds.n_train, ds.input_dim, tau, train_sigs);
    glyph_sig_quantize_batch(ds.x_test,  ds.n_test,  ds.input_dim, tau, test_sigs);
    double t_quant = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("quantize train+test in %.2fs\n", t_quant);

    /* Precompute densities and mean density for sanity. */
    int* dens_tr = malloc((size_t)ds.n_train * sizeof(int));
    int* dens_te = malloc((size_t)ds.n_test  * sizeof(int));
    double sum_tr = 0, sum_te = 0;
    for (int i = 0; i < ds.n_train; i++) {
        dens_tr[i] = trit_density_packed(train_sigs + (size_t)i * sig_bytes, sig_bytes);
        sum_tr += dens_tr[i];
    }
    for (int i = 0; i < ds.n_test; i++) {
        dens_te[i] = trit_density_packed(test_sigs + (size_t)i * sig_bytes, sig_bytes);
        sum_te += dens_te[i];
    }
    printf("mean density: train=%.1f  test=%.1f  (of %d trits → %.3f fraction)\n",
           sum_tr / ds.n_train, sum_te / ds.n_test, ds.input_dim,
           (sum_tr / ds.n_train) / ds.input_dim);

    /* Density distribution per class on train — diagnostic. */
    double class_density[N_CLASSES] = {0};
    int    class_count  [N_CLASSES] = {0};
    for (int i = 0; i < ds.n_train; i++) {
        int c = ds.y_train[i];
        if (c < 0 || c >= N_CLASSES) continue;
        class_density[c] += dens_tr[i];
        class_count  [c]++;
    }
    printf("per-class mean density:\n");
    for (int c = 0; c < N_CLASSES; c++) {
        if (class_count[c] == 0) continue;
        printf("  class %d: %.1f  (n=%d)\n",
               c, class_density[c] / class_count[c], class_count[c]);
    }

    /* Full-bits mask: every trit position participates. */
    uint8_t* mask = malloc((size_t)sig_bytes);
    memset(mask, 0xFF, sig_bytes);
    /* Last byte may have unused trit slots at the top; zero those bits
     * so popcount_dist doesn't count trash. A trit uses 2 bits; n_dims
     * mod 4 trits sit in the low 2·(n_dims % 4) bits of the last byte. */
    int trailing = ds.input_dim % 4;
    if (trailing != 0) {
        int valid_bits = trailing * 2;
        uint8_t last_mask = (uint8_t)((1u << valid_bits) - 1u);
        mask[sig_bytes - 1] = last_mask;
    }

    int n_te = ds.n_test < test_cap ? ds.n_test : test_cap;
    printf("running brute-force k-NN: train=%d test=%d  k ∈ {50, 100, 200}\n",
           ds.n_train, n_te);

    int ks[] = {50, 100, 200};
    for (int ki = 0; ki < 3; ki++) {
        int k = ks[ki];
        clock_t kt0 = clock();
        double acc = knn_class_accuracy(
            train_sigs, ds.y_train, dens_tr, ds.n_train,
            test_sigs,  ds.y_test,  dens_te, n_te,
            mask, sig_bytes, k, metric);
        double dt = (double)(clock() - kt0) / CLOCKS_PER_SEC;
        printf("k=%3d  %s  acc = %.2f%%  (%.2fs)\n",
               k, metric_name, 100.0 * acc, dt);
    }

    free(train_sigs); free(test_sigs);
    free(dens_tr); free(dens_te);
    free(mask);
    glyph_dataset_free(&ds);
    return 0;
}
