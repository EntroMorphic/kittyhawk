/*
 * mnist_probe.c — Phase B Gate 1 + remediation ablation.
 *
 * The original Gate 1 probe (commit 500ddaf) tested ONE configuration
 * (n_train=2000, budget=20K) at sig_dim ∈ {128, 256} and reported
 * trained ≤ random within seed noise. The Phase B red-team caught that
 * this single-config measurement cannot causally support the closeout's
 * "consumer architecture is the bottleneck" narrative — undertraining
 * (H1) or starved sample size (H2) are equally consistent with the data.
 *
 * This rewrite runs an ablation that disambiguates:
 *
 *   Cell A — baseline (matched to original probe):
 *            n_train=2000, flip_budget=20K
 *
 *   Cell B — 10× flip budget (isolates undertraining):
 *            n_train=2000, flip_budget=200K
 *
 *   Cell C — 10× n_train (isolates bank-sample-size starvation):
 *            n_train=20000, flip_budget=20K
 *
 *   Cell D — both (interaction):
 *            n_train=20000, flip_budget=200K
 *
 *   Cell E — C2 faithful regime (random R at sig_dim = D = 784):
 *            n_train=2000, no training
 *
 *   sig_dim multi-config sweep (random R only, no training; C3+H6):
 *            sig_dims ∈ {64, 128, 256, 512, 784}, n_train=2000
 *
 *   Identity baseline:
 *            sig_dim=784, n_train=2000 (deterministic, single trial)
 *
 * Cells A–D fix the same sig_dim (128, where the original probe ran)
 * to isolate the budget × n_train factors. Cell E and the sig_dim sweep
 * test C2 across configurations.
 *
 * Pre-committed reading of the ablation:
 *   - If Cell B trained ≫ Cell A trained: undertraining was the cause;
 *     architecture-bottleneck claim weakens.
 *   - If Cell C trained ≫ Cell A trained: sample-size starvation was;
 *     architecture-bottleneck claim weakens.
 *   - If Cells B and C both ≈ Cell A: undertraining and sample-size
 *     ruled out; architecture-bottleneck survives.
 *   - If Cell D ≈ Cell A: both ruled out; architecture-bottleneck
 *     strongly supported (consumer cap appears regardless of training
 *     compute or data).
 *
 * Multi-seed: 3 seeds per cell with independent (init_R, train_batch)
 * seed pairs. Per the H3 limit, multi-seed varies (init, train) but
 * still uses one (subsample_train, subsample_test) draw per cell —
 * data-realization variance is unsampled.
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
#define SIG_DIM_PRIMARY 128
#define BATCH_SIZE      128
#define N_EPOCHS        50

static const char* DEFAULT_MNIST_DIR =
    "/Users/aaronjosserand-austin/Projects/glyph/01MAY26_archived/data/mnist";

/* Uniform sampling with replacement. For n_pick = 2000 of pool 60000
 * the expected duplicate rate is ~1.6% (negligible at probe granularity).
 * Not Floyd's; not without-replacement. Documented honestly per M1. */
static void subsample_with_replacement(
    const m4t_trit_t* src_trits, const int* src_lbl, int src_n, int dim,
    m4t_trit_t* dst_trits, int* dst_lbl, int dst_n,
    uint32_t seed)
{
    uint32_t s = seed ? seed : 0x12345678u;
    int* picks = malloc((size_t)dst_n * sizeof(int));
    for (int i = 0; i < dst_n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        picks[i] = (int)(s % (uint32_t)src_n);
    }
    for (int i = 0; i < dst_n; i++) {
        memcpy(dst_trits + (size_t)i * dim,
               src_trits + (size_t)picks[i] * dim,
               (size_t)dim * sizeof(m4t_trit_t));
        dst_lbl[i] = src_lbl[picks[i]];
    }
    free(picks);
}

/* ── Eval ────────────────────────────────────────────────────────────── */

static int eval_test_accuracy_pm(  /* returns permille */
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
    return (correct * 1000) / n_test;
}

static void build_bank_with_R(
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

static int eval_identity_pm(
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
    int pm = eval_test_accuracy_pm(NULL, &bank, test, test_lbl,
                                     n_test, input_dim, input_dim);
    free(bank.tiles_packed);
    free(bank.labels);
    return pm;
}

static int run_random_one(
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

    build_bank_with_R(&bank, R, train, train_lbl, n_train, sig_dim,
                        input_dim, n_classes);
    int pm = eval_test_accuracy_pm(R, &bank, test, test_lbl,
                                     n_test, sig_dim, input_dim);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static int run_trained_one(
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
    int pm = eval_test_accuracy_pm(R, &bank, test, test_lbl,
                                     n_test, sig_dim, input_dim);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static void compute_stats_pm(const int* vals, int n,
                                double* out_mean_pct, double* out_sd_pp) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)vals[i];
    double mean = sum / (double)n;
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)vals[i] - mean;
        sq += d * d;
    }
    double var = (n > 1) ? sq / (double)(n - 1) : 0.0;
    *out_mean_pct = mean / 10.0;
    *out_sd_pp    = sqrt(var) / 10.0;
}

/* ── Probe driver ────────────────────────────────────────────────────── */

typedef struct {
    m4t_trit_t* tr;
    int* tr_lbl;
    m4t_trit_t* te;
    int* te_lbl;
    int n_tr, n_te, dim;
} subset_t;

static void make_subset(
    subset_t* s,
    const m4t_trit_t* tr_full, const int* tr_full_lbl, int n_tr_full,
    const m4t_trit_t* te_full, const int* te_full_lbl, int n_te_full,
    int dim, int n_tr, int n_te,
    uint32_t tr_seed, uint32_t te_seed)
{
    s->tr = malloc((size_t)n_tr * (size_t)dim * sizeof(m4t_trit_t));
    s->tr_lbl = malloc((size_t)n_tr * sizeof(int));
    s->te = malloc((size_t)n_te * (size_t)dim * sizeof(m4t_trit_t));
    s->te_lbl = malloc((size_t)n_te * sizeof(int));
    s->n_tr = n_tr;
    s->n_te = n_te;
    s->dim = dim;
    subsample_with_replacement(tr_full, tr_full_lbl, n_tr_full, dim,
                                  s->tr, s->tr_lbl, n_tr, tr_seed);
    subsample_with_replacement(te_full, te_full_lbl, n_te_full, dim,
                                  s->te, s->te_lbl, n_te, te_seed);
}

static void free_subset(subset_t* s) {
    free(s->tr); free(s->tr_lbl); free(s->te); free(s->te_lbl);
}

static void print_cell(
    const char* label, const subset_t* s,
    int sig_dim, int flip_budget,
    const uint32_t* init_seeds, const uint32_t* train_seeds,
    int n_classes)
{
    int rand_pm[N_SEEDS], train_pm[N_SEEDS];
    clock_t t0 = clock();
    for (int k = 0; k < N_SEEDS; k++)
        rand_pm[k] = run_random_one(s->tr, s->tr_lbl, s->n_tr,
                                       s->te, s->te_lbl, s->n_te,
                                       sig_dim, s->dim, n_classes,
                                       init_seeds[k]);
    for (int k = 0; k < N_SEEDS; k++)
        train_pm[k] = run_trained_one(s->tr, s->tr_lbl, s->n_tr,
                                          s->te, s->te_lbl, s->n_te,
                                          sig_dim, s->dim, n_classes,
                                          flip_budget,
                                          init_seeds[k], train_seeds[k]);
    double r_mean, r_sd, t_mean, t_sd;
    compute_stats_pm(rand_pm, N_SEEDS, &r_mean, &r_sd);
    compute_stats_pm(train_pm, N_SEEDS, &t_mean, &t_sd);
    double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("| %-20s | n_tr=%-6d budget=%-7d | "
           "rand %5.1f%% ± %4.1f | trained %5.1f%% ± %4.1f | gain %+4.1f | %5.1fs |\n",
           label, s->n_tr, flip_budget,
           r_mean, r_sd, t_mean, t_sd, t_mean - r_mean, dt);
    fflush(stdout);
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : DEFAULT_MNIST_DIR;
    image_canon_dataset_t ds;
    printf("# Loading MNIST from %s\n", dir); fflush(stdout);
    if (image_canon_load_mnist(&ds, dir) != 0) {
        fprintf(stderr, "mnist_probe: load failed\n");
        return 1;
    }
    printf("# Loaded train=%d test=%d input_dim=%d (%dx%d)\n",
           ds.n_train, ds.n_test, ds.input_dim, ds.img_w, ds.img_h);
    image_canon_normalize(&ds);
    int64_t tau = image_canon_quantize_tau(ds.x_train, 1000, ds.input_dim, 0.60);
    printf("# tau = %lld (density 0.60, sample 1000)\n", (long long)tau);

    m4t_trit_t* train_trits = malloc((size_t)ds.n_train * ds.input_dim
                                       * sizeof(m4t_trit_t));
    m4t_trit_t* test_trits  = malloc((size_t)ds.n_test  * ds.input_dim
                                       * sizeof(m4t_trit_t));
    image_canon_quantize_unpacked_batch(ds.x_train, ds.n_train, ds.input_dim,
                                          tau, train_trits);
    image_canon_quantize_unpacked_batch(ds.x_test,  ds.n_test,  ds.input_dim,
                                          tau, test_trits);

    /* Subsamples for the ablation. */
    subset_t s_2k_2k, s_20k_2k;
    make_subset(&s_2k_2k,
                  train_trits, ds.y_train, ds.n_train,
                  test_trits,  ds.y_test,  ds.n_test,
                  ds.input_dim, 2000, 2000,
                  0xa5a5a5a5u, 0xc0ffeedu);
    make_subset(&s_20k_2k,
                  train_trits, ds.y_train, ds.n_train,
                  test_trits,  ds.y_test,  ds.n_test,
                  ds.input_dim, 20000, 2000,
                  0xa5a5a5a5u, 0xc0ffeedu);

    /* Identity baseline at full-D = 784. */
    int id_pm = eval_identity_pm(s_2k_2k.tr, s_2k_2k.tr_lbl, 2000,
                                    s_2k_2k.te, s_2k_2k.te_lbl, 2000,
                                    ds.input_dim, 10);

    uint32_t init_seeds[N_SEEDS]  = { 0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu };
    uint32_t train_seeds[N_SEEDS] = { 0xa5a5a5a5u, 0xc7c7c7c7u, 0xb22bd00du };

    printf("\n");
    printf("# Phase B Gate 1 — remediation ablation (post Phase B red-team)\n");
    printf("# Identity baseline (sig_dim=%d, no projection): %.1f%%\n",
           ds.input_dim, id_pm / 10.0);
    printf("\n");
    printf("# Cells A–D fix sig_dim=%d to isolate budget × n_train.\n",
           SIG_DIM_PRIMARY);
    printf("# All cells use 3 seeds with independent (init_R, train) pairs.\n");
    printf("\n");
    printf("| cell                 | config                       | random           | trained          | gain  | runtime |\n");
    printf("|----------------------|------------------------------|------------------|------------------|-------|---------|\n");

    /* Cell A — baseline (matches original probe). */
    print_cell("A: baseline",         &s_2k_2k,  SIG_DIM_PRIMARY,    20000,
                 init_seeds, train_seeds, 10);

    /* Cell B — 10× budget, isolates undertraining (H1). */
    print_cell("B: 10x budget",       &s_2k_2k,  SIG_DIM_PRIMARY,   200000,
                 init_seeds, train_seeds, 10);

    /* Cell C — 10× n_train, isolates sample-size (H2). */
    print_cell("C: 10x n_train",      &s_20k_2k, SIG_DIM_PRIMARY,    20000,
                 init_seeds, train_seeds, 10);

    /* Cell D — both, tests interaction. */
    print_cell("D: 10x both",         &s_20k_2k, SIG_DIM_PRIMARY,   200000,
                 init_seeds, train_seeds, 10);

    /* sig_dim multi-config — random R only, no training (C2 + C3 + H6). */
    printf("\n");
    printf("# sig_dim multi-config (random R only, no training; n_train=2000)\n");
    printf("| sig_dim | random           | gap-vs-identity |\n");
    printf("|---------|------------------|-----------------|\n");
    int sig_dims[] = { 64, 128, 256, 512, 784 };
    for (int i = 0; i < 5; i++) {
        int sig = sig_dims[i];
        int rand_pm[N_SEEDS];
        for (int k = 0; k < N_SEEDS; k++)
            rand_pm[k] = run_random_one(s_2k_2k.tr, s_2k_2k.tr_lbl, 2000,
                                           s_2k_2k.te, s_2k_2k.te_lbl, 2000,
                                           sig, ds.input_dim, 10,
                                           init_seeds[k]);
        double r_mean, r_sd;
        compute_stats_pm(rand_pm, N_SEEDS, &r_mean, &r_sd);
        double gap = r_mean - (double)id_pm / 10.0;
        printf("| %7d | %5.1f%% ± %4.1fpp | %+5.1f pp        |\n",
               sig, r_mean, r_sd, gap);
        fflush(stdout);
    }

    free(train_trits); free(test_trits);
    free_subset(&s_2k_2k);
    free_subset(&s_20k_2k);
    image_canon_free(&ds);
    return 0;
}
