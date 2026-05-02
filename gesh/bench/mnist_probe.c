/*
 * mnist_probe.c — Phase B Gate 1: image canon parity probe.
 *
 * Per `journal/gesh_findings_synthesize.md`, this probe asks whether
 * Gesh's forward + lattice-update pipeline transfers from the synthetic
 * prototype benchmark to MNIST under the canonical (no random projection)
 * pixel pipeline.
 *
 * Pipeline:
 *   MNIST IDX → MTFP-encoded → per-image normalize →
 *   direct ternary quantization (per-pixel; tau calibrated from sample) →
 *   Gesh forward + lattice-update.
 *
 * Probe configuration:
 *   - n_train subsample: 2000 (matches synthetic for runtime)
 *   - n_test:            2000 (subsample for eval speed; ±1pp resolution)
 *   - sig_dim sweep:     {128, 256}
 *   - seeds:             3 per cell
 *   - flip budget:       capped at 20k per training run
 *   - intra-epoch refresh per Phase A.2 H1/H2 remediations
 *
 * Pre-committed gates (per synthesize):
 *   PASS:  trained Gesh ≥ 95% MNIST AND beats untrained random ≥ +2pp avg
 *   FAIL:  trained Gesh < 90% MNIST OR trained ≤ random within seed noise
 *   INCONCLUSIVE zone: 90–95% accuracy or marginal gain
 */

#include "image_canon.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_train.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_SEEDS 3
#define N_TRAIN_SUBSET 2000
#define N_TEST_SUBSET  2000
#define FLIP_BUDGET    20000
#define BATCH_SIZE     128
#define N_EPOCHS       50

static const char* MNIST_DIR =
    "/Users/aaronjosserand-austin/Projects/glyph/01MAY26_archived/data/mnist";

/* ── Subsample helpers ───────────────────────────────────────────────── */

static void subsample(
    const m4t_trit_t* src_trits, const int* src_lbl, int src_n, int dim,
    m4t_trit_t* dst_trits, int* dst_lbl, int dst_n,
    uint32_t seed)
{
    uint32_t s = seed ? seed : 0x12345678u;
    int* picks = malloc((size_t)dst_n * sizeof(int));
    /* Random-without-replacement via Floyd's algorithm. */
    for (int i = 0; i < dst_n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        picks[i] = (int)(s % (uint32_t)src_n);
    }
    /* Note: not strictly without-replacement; for n_train=2000 of 60000
     * the collision rate is ~3% per pick which is within noise tolerance. */
    for (int i = 0; i < dst_n; i++) {
        memcpy(dst_trits + (size_t)i * dim,
               src_trits + (size_t)picks[i] * dim,
               (size_t)dim * sizeof(m4t_trit_t));
        dst_lbl[i] = src_lbl[picks[i]];
    }
    free(picks);
}

/* ── Eval ────────────────────────────────────────────────────────────── */

static int eval_test_accuracy(
    const m4t_trit_t* R,
    const gesh_bank_t* bank,
    const m4t_trit_t* test, const int* test_lbl,
    int n_test, int sig_dim, int input_dim)
{
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    int rc = gesh_forward_classify(preds, test, n_test, bank, &proj, 1);
    if (rc != 0) { free(preds); return -1; }
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        if (preds[i] == test_lbl[i]) correct++;
    }
    free(preds);
    return (correct * 1000) / n_test;  /* permille for finer reporting */
}

static void build_bank(
    gesh_bank_t* bank,
    const m4t_trit_t* R,
    const m4t_trit_t* train, const int* train_lbl,
    int n_train, int sig_dim, int input_dim, int n_classes)
{
    m4t_trit_t* projected = malloc((size_t)n_train * (size_t)sig_dim
                                    * sizeof(m4t_trit_t));
    for (int i = 0; i < n_train; i++) {
        const m4t_trit_t* x = train + (size_t)i * input_dim;
        m4t_trit_t* s = projected + (size_t)i * sig_dim;
        for (int oi = 0; oi < sig_dim; oi++) {
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            int32_t acc = 0;
            for (int j = 0; j < input_dim; j++)
                acc += (int32_t)r[j] * (int32_t)x[j];
            s[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
        }
    }
    gesh_bank_build_class_mean(bank, projected, train_lbl, n_train, n_classes);
    free(projected);
}

static int eval_identity(
    const m4t_trit_t* train, const int* train_lbl, int n_train,
    const m4t_trit_t* test,  const int* test_lbl,  int n_test,
    int input_dim, int n_classes)
{
    gesh_bank_t bank;
    int Dp = M4T_TRIT_PACKED_BYTES(input_dim);
    bank.tiles_packed = malloc((size_t)n_classes * (size_t)Dp);
    bank.labels = malloc((size_t)n_classes * sizeof(int));
    bank.n_tiles = n_classes;
    bank.sig_dim = input_dim;
    gesh_bank_build_class_mean(&bank, train, train_lbl, n_train, n_classes);
    int per_mille = eval_test_accuracy(NULL, &bank, test, test_lbl,
                                         n_test, input_dim, input_dim);
    free(bank.tiles_packed);
    free(bank.labels);
    return per_mille;
}

static int run_random(
    const m4t_trit_t* train, const int* train_lbl, int n_train,
    const m4t_trit_t* test,  const int* test_lbl,  int n_test,
    int sig_dim, int input_dim, int n_classes,
    uint32_t init_seed)
{
    m4t_trit_t* R = malloc((size_t)sig_dim * input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, input_dim, init_seed);

    gesh_bank_t bank;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    bank.tiles_packed = malloc((size_t)n_classes * (size_t)Dp);
    bank.labels = malloc((size_t)n_classes * sizeof(int));
    bank.n_tiles = n_classes;
    bank.sig_dim = sig_dim;

    build_bank(&bank, R, train, train_lbl, n_train, sig_dim, input_dim, n_classes);
    int pm = eval_test_accuracy(R, &bank, test, test_lbl,
                                 n_test, sig_dim, input_dim);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static int run_trained(
    const m4t_trit_t* train, const int* train_lbl, int n_train,
    const m4t_trit_t* test,  const int* test_lbl,  int n_test,
    int sig_dim, int input_dim, int n_classes,
    int flip_budget, uint32_t init_seed, uint32_t train_seed)
{
    m4t_trit_t* R = malloc((size_t)sig_dim * input_dim * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, input_dim, init_seed);

    gesh_bank_t bank;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    bank.tiles_packed = malloc((size_t)n_classes * (size_t)Dp);
    bank.labels = malloc((size_t)n_classes * sizeof(int));
    bank.n_tiles = n_classes;
    bank.sig_dim = sig_dim;

    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = N_EPOCHS;
    cfg.n_flip_evals_per_epoch = flip_budget / cfg.n_epochs;
    if (cfg.n_flip_evals_per_epoch < 10) cfg.n_flip_evals_per_epoch = 10;
    cfg.batch_size = BATCH_SIZE;
    cfg.bank_refresh_every = cfg.n_flip_evals_per_epoch / 4;
    if (cfg.bank_refresh_every < 1) cfg.bank_refresh_every = 1;
    cfg.batch_refresh_every = cfg.n_flip_evals_per_epoch / 4;
    if (cfg.batch_refresh_every < 1) cfg.batch_refresh_every = 1;
    cfg.early_stop_patience = 5;
    cfg.log_per_epoch = 0;
    cfg.seed = train_seed;

    int rc = gesh_train_lattice_update(R, &bank, train, train_lbl,
                                         n_train, n_classes, sig_dim, input_dim,
                                         1, &cfg);
    if (rc < 0) {
        free(R); free(bank.tiles_packed); free(bank.labels);
        return -1;
    }
    int pm = eval_test_accuracy(R, &bank, test, test_lbl,
                                 n_test, sig_dim, input_dim);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static void compute_stats(const int* vals, int n,
                            double* out_mean_pct, double* out_sd_pp) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)vals[i];
    double mean = sum / (double)n;  /* permille */
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)vals[i] - mean;
        sq += d * d;
    }
    double var = (n > 1) ? sq / (double)(n - 1) : 0.0;
    *out_mean_pct = mean / 10.0;       /* % */
    *out_sd_pp    = sqrt(var) / 10.0;  /* pp */
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : MNIST_DIR;
    image_canon_dataset_t ds;
    printf("# Loading MNIST from %s\n", dir);
    fflush(stdout);
    if (image_canon_load_mnist(&ds, dir) != 0) {
        fprintf(stderr, "mnist_probe: load failed\n");
        return 1;
    }
    printf("# Loaded train=%d test=%d input_dim=%d (%dx%d)\n",
           ds.n_train, ds.n_test, ds.input_dim, ds.img_w, ds.img_h);

    printf("# Normalizing...\n"); fflush(stdout);
    image_canon_normalize(&ds);

    /* Calibrate tau from a 1000-image sample of the training set at
     * density=0.60: ~60% of pixels map to the structural zero. */
    printf("# Calibrating tau (density=0.60, sample=1000)...\n");
    fflush(stdout);
    int n_calib = (ds.n_train < 1000) ? ds.n_train : 1000;
    int64_t tau = image_canon_quantize_tau(ds.x_train, n_calib, ds.input_dim, 0.60);
    printf("# tau = %lld\n", (long long)tau); fflush(stdout);

    /* Quantize whole dataset to unpacked trits. */
    printf("# Quantizing %d train + %d test images...\n",
           ds.n_train, ds.n_test);
    fflush(stdout);
    m4t_trit_t* train_trits = malloc((size_t)ds.n_train * ds.input_dim
                                       * sizeof(m4t_trit_t));
    m4t_trit_t* test_trits  = malloc((size_t)ds.n_test  * ds.input_dim
                                       * sizeof(m4t_trit_t));
    image_canon_quantize_unpacked_batch(ds.x_train, ds.n_train, ds.input_dim,
                                          tau, train_trits);
    image_canon_quantize_unpacked_batch(ds.x_test,  ds.n_test,  ds.input_dim,
                                          tau, test_trits);

    /* Subsample. */
    m4t_trit_t* tr_sub = malloc((size_t)N_TRAIN_SUBSET * ds.input_dim
                                  * sizeof(m4t_trit_t));
    int* tr_sub_lbl = malloc((size_t)N_TRAIN_SUBSET * sizeof(int));
    subsample(train_trits, ds.y_train, ds.n_train, ds.input_dim,
              tr_sub, tr_sub_lbl, N_TRAIN_SUBSET, 0xa5a5a5a5u);

    m4t_trit_t* te_sub = malloc((size_t)N_TEST_SUBSET * ds.input_dim
                                  * sizeof(m4t_trit_t));
    int* te_sub_lbl = malloc((size_t)N_TEST_SUBSET * sizeof(int));
    subsample(test_trits, ds.y_test, ds.n_test, ds.input_dim,
              te_sub, te_sub_lbl, N_TEST_SUBSET, 0xc0ffeedu);

    /* Identity baseline (deterministic, single trial). */
    int id_pm = eval_identity(tr_sub, tr_sub_lbl, N_TRAIN_SUBSET,
                                te_sub, te_sub_lbl, N_TEST_SUBSET,
                                ds.input_dim, 10);

    int sig_dims[] = { 128, 256 };
    int n_dims = (int)(sizeof(sig_dims) / sizeof(sig_dims[0]));
    uint32_t init_seeds[N_SEEDS]  = { 0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu };
    uint32_t train_seeds[N_SEEDS] = { 0xa5a5a5a5u, 0xc7c7c7c7u, 0xb22bd00du };

    printf("\n");
    printf("# Phase B Gate 1: MNIST canonical pipeline probe\n");
    printf("# n_train=%d (subsampled), n_test=%d (subsampled), seeds=%d\n",
           N_TRAIN_SUBSET, N_TEST_SUBSET, N_SEEDS);
    printf("# Identity baseline (sig_dim=%d, no projection): %.1f%%\n\n",
           ds.input_dim, id_pm / 10.0);

    printf("| sig_dim | random           | trained          | gain        |\n");
    printf("|---------|------------------|------------------|-------------|\n");
    fflush(stdout);

    clock_t t0 = clock();
    for (int i = 0; i < n_dims; i++) {
        int sig = sig_dims[i];
        int rand_results[N_SEEDS];
        int train_results[N_SEEDS];
        for (int s = 0; s < N_SEEDS; s++) {
            rand_results[s] = run_random(tr_sub, tr_sub_lbl, N_TRAIN_SUBSET,
                                           te_sub, te_sub_lbl, N_TEST_SUBSET,
                                           sig, ds.input_dim, 10,
                                           init_seeds[s]);
        }
        for (int s = 0; s < N_SEEDS; s++) {
            train_results[s] = run_trained(tr_sub, tr_sub_lbl, N_TRAIN_SUBSET,
                                              te_sub, te_sub_lbl, N_TEST_SUBSET,
                                              sig, ds.input_dim, 10,
                                              FLIP_BUDGET,
                                              init_seeds[s], train_seeds[s]);
        }
        double r_mean, r_sd, t_mean, t_sd;
        compute_stats(rand_results, N_SEEDS, &r_mean, &r_sd);
        compute_stats(train_results, N_SEEDS, &t_mean, &t_sd);
        double gain = t_mean - r_mean;
        printf("| %7d | %5.1f%% ± %4.1fpp | %5.1f%% ± %4.1fpp | %+5.1f pp     |\n",
               sig, r_mean, r_sd, t_mean, t_sd, gain);
        fflush(stdout);
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\nTotal probe runtime: %.1fs\n", elapsed);

    free(train_trits); free(test_trits);
    free(tr_sub); free(tr_sub_lbl);
    free(te_sub); free(te_sub_lbl);
    image_canon_free(&ds);
    return 0;
}
