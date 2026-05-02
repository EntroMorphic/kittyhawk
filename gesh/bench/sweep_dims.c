/*
 * sweep_dims.c — Phase A.2 sig_dim sweep with multi-seed averaging.
 *
 * Sweeps sig_dim across a range of compression and expansion ratios,
 * comparing two variants (random R, trained R) plus a fixed identity
 * reference at sig_dim = D. **Each (sig_dim, variant) point is averaged
 * across N_SEEDS independent seeds** to surface variance and avoid the
 * single-seed-confound issue (C1 in the Phase A.2 red-team).
 *
 * Synthetic prototype-classification benchmark: D=64 (16 informative +
 * 48 noise), C=10 classes, 10% per-trit noise, n_train=2000, n_test=500.
 *
 * Training budget scales with R size: ~5 flip-evaluations per trit on
 * average, spread over 50 epochs. Intra-epoch bank/batch refresh per
 * the H1/H2 remediations.
 *
 * Deterministic via fixed seed lists. Output is a printable table with
 * mean and stddev per cell.
 */

#include "synth_proto.h"
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

#define N_SEEDS 5

/* ── Fixture ─────────────────────────────────────────────────────────────── */

typedef struct {
    m4t_trit_t* train;
    int* train_lbl;
    m4t_trit_t* test;
    int* test_lbl;
    int n_train, n_test, C, D;
} fixture_t;

static fixture_t make_fixture(void) {
    fixture_t f;
    synth_proto_config_t cfg = synth_proto_default();
    f.C = cfg.n_classes;
    f.D = cfg.input_dim;
    f.n_train = 2000;
    f.n_test = 500;

    m4t_trit_t* protos = malloc((size_t)f.C * f.D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);

    f.train = malloc((size_t)f.n_train * f.D * sizeof(m4t_trit_t));
    f.train_lbl = malloc((size_t)f.n_train * sizeof(int));
    synth_proto_generate_samples(f.train, f.train_lbl, f.n_train, protos, &cfg, 0x11111111u);

    f.test = malloc((size_t)f.n_test * f.D * sizeof(m4t_trit_t));
    f.test_lbl = malloc((size_t)f.n_test * sizeof(int));
    synth_proto_generate_samples(f.test, f.test_lbl, f.n_test, protos, &cfg, 0x22222222u);

    free(protos);
    return f;
}

static void free_fixture(fixture_t* f) {
    free(f->train); free(f->train_lbl); free(f->test); free(f->test_lbl);
}

/* ── Eval helpers ────────────────────────────────────────────────────────── */

static int eval_test_accuracy(
    const m4t_trit_t* R,
    const gesh_bank_t* bank,
    const m4t_trit_t* test, const int* test_lbl,
    int n_test, int sig_dim, int input_dim, int top_k)
{
    gesh_projection_t proj;
    proj.R = R;
    proj.input_dim = input_dim;
    proj.sig_dim = sig_dim;
    int* preds = malloc((size_t)n_test * sizeof(int));
    int rc = gesh_forward_classify(preds, test, n_test, bank, &proj, top_k);
    if (rc != 0) { free(preds); return -1; }
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        if (preds[i] == test_lbl[i]) correct++;
    }
    free(preds);
    return (correct * 100) / n_test;
}

static void build_bank_from_projection(
    gesh_bank_t* bank,
    const m4t_trit_t* R,
    const m4t_trit_t* train, const int* train_lbl,
    int n_train, int sig_dim, int input_dim, int n_classes)
{
    m4t_trit_t* projected = malloc((size_t)n_train * (size_t)sig_dim * sizeof(m4t_trit_t));
    for (int i = 0; i < n_train; i++) {
        const m4t_trit_t* x = train + (size_t)i * input_dim;
        m4t_trit_t* s = projected + (size_t)i * sig_dim;
        for (int oi = 0; oi < sig_dim; oi++) {
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            int32_t acc = 0;
            for (int j = 0; j < input_dim; j++) {
                acc += (int32_t)r[j] * (int32_t)x[j];
            }
            s[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
        }
    }
    gesh_bank_build_class_mean(bank, projected, train_lbl, n_train, n_classes);
    free(projected);
}

/* ── Variant runners (one seed each) ────────────────────────────────────── */

static int run_random_one_seed(const fixture_t* f, int sig_dim, uint32_t init_seed) {
    m4t_trit_t* R = malloc((size_t)sig_dim * f->D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, f->D, init_seed);

    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;

    build_bank_from_projection(&bank, R, f->train, f->train_lbl,
                                f->n_train, sig_dim, f->D, f->C);

    int pct = eval_test_accuracy(R, &bank, f->test, f->test_lbl,
                                   f->n_test, sig_dim, f->D, 1);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pct;
}

static int run_trained_one_seed(const fixture_t* f, int sig_dim,
                                 int flip_budget,
                                 uint32_t init_seed, uint32_t train_seed) {
    m4t_trit_t* R = malloc((size_t)sig_dim * f->D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, f->D, init_seed);

    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;

    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = 50;
    cfg.n_flip_evals_per_epoch = flip_budget / cfg.n_epochs;
    if (cfg.n_flip_evals_per_epoch < 10) cfg.n_flip_evals_per_epoch = 10;
    cfg.batch_size = 128;
    cfg.bank_refresh_every = cfg.n_flip_evals_per_epoch / 4;
    if (cfg.bank_refresh_every < 1) cfg.bank_refresh_every = 1;
    cfg.batch_refresh_every = cfg.n_flip_evals_per_epoch / 4;
    if (cfg.batch_refresh_every < 1) cfg.batch_refresh_every = 1;
    cfg.early_stop_patience = 5;
    cfg.log_per_epoch = 0;
    cfg.seed = train_seed;

    int rc = gesh_train_lattice_update(R, &bank, f->train, f->train_lbl,
                                         f->n_train, f->C, sig_dim, f->D,
                                         1, &cfg);
    if (rc < 0) {
        free(R); free(bank.tiles_packed); free(bank.labels);
        return -1;
    }
    int pct = eval_test_accuracy(R, &bank, f->test, f->test_lbl,
                                   f->n_test, sig_dim, f->D, 1);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pct;
}

static int run_identity(const fixture_t* f) {
    int sig_dim = f->D;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;
    gesh_bank_build_class_mean(&bank, f->train, f->train_lbl,
                                f->n_train, f->C);
    int pct = eval_test_accuracy(NULL, &bank, f->test, f->test_lbl,
                                   f->n_test, sig_dim, f->D, 1);
    free(bank.tiles_packed); free(bank.labels);
    return pct;
}

/* ── Stats over an array of percentages ─────────────────────────────────── */

static void compute_stats(const int* vals, int n, double* out_mean, double* out_stddev) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)vals[i];
    double mean = sum / (double)n;
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)vals[i] - mean;
        sq += d * d;
    }
    double var = (n > 1) ? sq / (double)(n - 1) : 0.0;
    *out_mean = mean;
    *out_stddev = sqrt(var);
}

/* ── Main sweep ─────────────────────────────────────────────────────────── */

int main(void) {
    fixture_t f = make_fixture();

    int dims[] = { 2, 4, 8, 16, 32, 64, 128, 256 };
    int n_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    /* Independent seed pairs per (variant, sig_dim, trial). */
    uint32_t init_seeds[N_SEEDS]  = { 0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu,
                                      0xb16b00b5u, 0x13579bdfu };
    uint32_t train_seeds[N_SEEDS] = { 0xa5a5a5a5u, 0xc7c7c7c7u, 0xb22bd00du,
                                      0xdeadc0deu, 0x0123abcdu };

    printf("# Gesh Phase A.2 sig_dim sweep (multi-seed)\n");
    printf("# D=%d (16 informative + %d noise), C=%d, 10%%%% noise, "
           "n_train=%d, n_test=%d, top_k=1.\n",
           f.D, f.D - 16, f.C, f.n_train, f.n_test);
    printf("# Each (sig_dim, variant) = mean ± stddev across %d seeds.\n",
           N_SEEDS);
    printf("# Trained budget: ~5 flip-evals/trit, intra-epoch refresh "
           "(bank+batch) every n_flips/4.\n\n");

    printf("| sig_dim | random           | trained          | gain        |  budget |\n");
    printf("|---------|------------------|------------------|-------------|---------|\n");

    clock_t t0 = clock();
    for (int i = 0; i < n_dims; i++) {
        int sig = dims[i];

        int rand_results[N_SEEDS];
        int train_results[N_SEEDS];
        for (int s = 0; s < N_SEEDS; s++) {
            rand_results[s] = run_random_one_seed(&f, sig, init_seeds[s]);
        }

        int budget = 5 * sig * f.D;
        if (budget < 500) budget = 500;
        for (int s = 0; s < N_SEEDS; s++) {
            train_results[s] = run_trained_one_seed(&f, sig, budget,
                                                      init_seeds[s],
                                                      train_seeds[s]);
        }

        double rand_mean, rand_sd, train_mean, train_sd;
        compute_stats(rand_results, N_SEEDS, &rand_mean, &rand_sd);
        compute_stats(train_results, N_SEEDS, &train_mean, &train_sd);
        double gain_mean = train_mean - rand_mean;

        printf("| %7d | %5.1f%% ± %4.1fpp | %5.1f%% ± %4.1fpp | %+5.1f pp     | %7d |\n",
               sig, rand_mean, rand_sd, train_mean, train_sd, gain_mean, budget);
        fflush(stdout);
    }

    int id_pct = run_identity(&f);
    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\n");
    printf("Identity (sig_dim=D=%d, no projection): %d%% (single trial; deterministic)\n",
           f.D, id_pct);
    printf("\nTotal sweep runtime: %.1fs (%d sig_dims × %d seeds × 2 variants)\n",
           total_s, n_dims, N_SEEDS);

    free_fixture(&f);
    return 0;
}
