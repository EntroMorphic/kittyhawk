/*
 * finding3_probe.c — high-seed-count probe for Finding 3
 * (capacity floor at sig_dim ≤ 4) from sweep_dims_results.md.
 *
 * Original sweep ran 5 seeds per cell. At sig_dim ∈ {2, 4} the seed
 * stddev is 1.6–3.1pp on a 15–27% point estimate — wide enough that
 * the "capacity floor" framing leans on a noisy measurement. This
 * probe runs N_SEEDS = 30 (6× the original) at sig_dim ∈ {2, 4, 8}
 * to harden the claim: tighter CI on the absolute number, and a
 * cleaner test of the ordering between cells.
 *
 * The capacity argument:
 *   sig_dim = 2 → 3² = 9 distinct ternary signatures. C = 10 classes.
 *   At least one class must collide with another in signature space.
 *   Information-theoretic ceiling on classification accuracy is bounded
 *   below 100%; the measurement asks how close we get.
 *
 * Output: mean ± stddev (5 seeds vs 30 seeds, side-by-side for the
 * cells the original sweep covered).
 */

#include "synth_proto.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_project.h"
#include "gesh_train.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_SEEDS_HIGH 30

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
    synth_proto_generate_samples(f.train, f.train_lbl, f.n_train,
                                   protos, &cfg, 0x11111111u);

    f.test = malloc((size_t)f.n_test * f.D * sizeof(m4t_trit_t));
    f.test_lbl = malloc((size_t)f.n_test * sizeof(int));
    synth_proto_generate_samples(f.test, f.test_lbl, f.n_test,
                                   protos, &cfg, 0x22222222u);

    free(protos);
    return f;
}

static void free_fixture(fixture_t* f) {
    free(f->train); free(f->train_lbl); free(f->test); free(f->test_lbl);
}

static int eval_accuracy_pm(
    const m4t_trit_t* R, const gesh_bank_t* bank,
    const m4t_trit_t* test, const int* test_lbl, int n_test,
    int sig_dim, int input_dim)
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
    return (correct * 1000) / n_test;  /* permille */
}

static void build_bank(
    gesh_bank_t* bank, const m4t_trit_t* R,
    const m4t_trit_t* train, const int* train_lbl,
    int n_train, int sig_dim, int input_dim, int n_classes)
{
    m4t_trit_t* projected = malloc((size_t)n_train * (size_t)sig_dim
                                     * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(projected, train, n_train,
                                  R, sig_dim, input_dim);
    gesh_bank_build_class_mean(bank, projected, train_lbl, n_train, n_classes);
    free(projected);
}

static int run_random_seed(const fixture_t* f, int sig_dim, uint32_t init_seed) {
    m4t_trit_t* R = malloc((size_t)sig_dim * f->D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, f->D, init_seed);

    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;

    build_bank(&bank, R, f->train, f->train_lbl, f->n_train, sig_dim,
                 f->D, f->C);
    int pm = eval_accuracy_pm(R, &bank, f->test, f->test_lbl, f->n_test,
                                 sig_dim, f->D);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static int run_trained_seed(const fixture_t* f, int sig_dim,
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
    int pm = eval_accuracy_pm(R, &bank, f->test, f->test_lbl, f->n_test,
                                 sig_dim, f->D);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

static void compute_stats_pm(const int* vals, int n,
                                double* out_mean_pct, double* out_sd_pp,
                                double* out_min_pct, double* out_max_pct) {
    double sum = 0.0;
    int vmin = vals[0], vmax = vals[0];
    for (int i = 0; i < n; i++) {
        sum += (double)vals[i];
        if (vals[i] < vmin) vmin = vals[i];
        if (vals[i] > vmax) vmax = vals[i];
    }
    double mean = sum / (double)n;
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)vals[i] - mean;
        sq += d * d;
    }
    double var = (n > 1) ? sq / (double)(n - 1) : 0.0;
    *out_mean_pct = mean / 10.0;
    *out_sd_pp    = sqrt(var) / 10.0;
    *out_min_pct  = vmin / 10.0;
    *out_max_pct  = vmax / 10.0;
}

int main(void) {
    fixture_t f = make_fixture();

    /* 30 independent (init, train) seed pairs. The first 5 match the
     * original sweep_dims seed list, so the 5-seed sub-mean of this
     * 30-seed run is bit-identical to the original sweep's cell mean. */
    uint32_t init_seeds[N_SEEDS_HIGH] = {
        /* original sweep_dims seeds (positions 0..4) */
        0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu, 0xb16b00b5u, 0x13579bdfu,
        /* 25 fresh seeds */
        0x1a2b3c4du, 0x5e6f7081u, 0x92a3b4c5u, 0xd6e7f809u, 0x1234abcdu,
        0xdeadbeefu, 0xcafebabeu, 0xbadc0deau, 0xf00dfaceu, 0x8badf00du,
        0x10203040u, 0x50607080u, 0x90a0b0c0u, 0xd0e0f000u, 0x11223344u,
        0x55667788u, 0x99aabbccu, 0xddeeff00u, 0x21436587u, 0xa9cbedffu,
        0x01234567u, 0x89abcdefu, 0xfedcba98u, 0x76543210u, 0x4242424au
    };
    uint32_t train_seeds[N_SEEDS_HIGH] = {
        /* original sweep_dims seeds (positions 0..4) */
        0xa5a5a5a5u, 0xc7c7c7c7u, 0xb22bd00du, 0xdeadc0deu, 0x0123abcdu,
        /* 25 fresh seeds */
        0x6789abcdu, 0xfedcba98u, 0x13572468u, 0x9bdf0246u, 0x12345678u,
        0xfeedfaceu, 0xbeefcafeu, 0xa0b0c0d0u, 0xe0f01020u, 0x30405060u,
        0x70809010u, 0x20304050u, 0x60708090u, 0xa0b0c0d1u, 0x11111111u,
        0x22222222u, 0x33333333u, 0x44444444u, 0x55555555u, 0x66666666u,
        0x77777777u, 0x88888888u, 0x99999999u, 0xaaaaaaaau, 0xbbbbbbbbu
    };

    int sig_dims[] = { 2, 4, 8 };
    int n_dims = (int)(sizeof(sig_dims) / sizeof(sig_dims[0]));

    printf("# Finding 3 high-seed probe — capacity floor at sig_dim ≤ 4\n");
    printf("# D=%d (16 informative + %d noise), C=%d, n_train=%d, n_test=%d\n",
           f.D, f.D - 16, f.C, f.n_train, f.n_test);
    printf("# 30 seeds per cell (vs original sweep's 5). First 5 seed positions\n");
    printf("# match the original sweep — the 5-seed sub-mean reproduces.\n\n");

    printf("| sig_dim | variant   | mean ± stddev      | min  | max  |   95%% CI    |\n");
    printf("|---------|-----------|--------------------|------|------|-------------|\n");

    clock_t t0 = clock();
    for (int i = 0; i < n_dims; i++) {
        int sig = sig_dims[i];
        int budget = 5 * sig * f.D;
        if (budget < 500) budget = 500;

        int rand_results[N_SEEDS_HIGH];
        int train_results[N_SEEDS_HIGH];
        for (int s = 0; s < N_SEEDS_HIGH; s++)
            rand_results[s] = run_random_seed(&f, sig, init_seeds[s]);
        for (int s = 0; s < N_SEEDS_HIGH; s++)
            train_results[s] = run_trained_seed(&f, sig, budget,
                                                   init_seeds[s], train_seeds[s]);

        double r_mean, r_sd, r_min, r_max;
        double t_mean, t_sd, t_min, t_max;
        compute_stats_pm(rand_results, N_SEEDS_HIGH,
                            &r_mean, &r_sd, &r_min, &r_max);
        compute_stats_pm(train_results, N_SEEDS_HIGH,
                            &t_mean, &t_sd, &t_min, &t_max);
        /* 95% CI for the mean ≈ ±1.96 × sd / sqrt(n). */
        double r_ci = 1.96 * r_sd / sqrt(N_SEEDS_HIGH);
        double t_ci = 1.96 * t_sd / sqrt(N_SEEDS_HIGH);

        printf("| %7d | random    | %5.1f%% ± %5.2fpp | %4.1f | %4.1f | ±%4.2fpp     |\n",
               sig, r_mean, r_sd, r_min, r_max, r_ci);
        printf("| %7d | trained   | %5.1f%% ± %5.2fpp | %4.1f | %4.1f | ±%4.2fpp     |\n",
               sig, t_mean, t_sd, t_min, t_max, t_ci);
        printf("|---------|-----------|--------------------|------|------|-------------|\n");
        fflush(stdout);

        /* Sub-mean over first 5 seeds for cross-check. */
        double r5_mean, r5_sd, r5_min, r5_max;
        double t5_mean, t5_sd, t5_min, t5_max;
        compute_stats_pm(rand_results, 5,
                            &r5_mean, &r5_sd, &r5_min, &r5_max);
        compute_stats_pm(train_results, 5,
                            &t5_mean, &t5_sd, &t5_min, &t5_max);
        printf("|   (5-seed sub-mean) %5.1f / %5.1f%%   "
               "vs original sweep_dims_results.md\n",
               r5_mean, t5_mean);
        printf("|---------|-----------|--------------------|------|------|-------------|\n");
    }

    /* Capacity-ceiling discussion. */
    printf("\nCapacity argument:\n");
    printf("  sig_dim = 2 → 3² = 9 distinct ternary signatures\n");
    printf("  C = 10 classes → at least one class shares signature space\n");
    printf("  Information-theoretic upper bound: < 100%% by construction\n");
    printf("  Trained accuracy at sig_dim = 2 should sit far below sig_dim = 8\n");
    printf("  (capacity floor) but well above C = 10%% chance.\n\n");

    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Total runtime: %.1fs (%d sig_dims × %d seeds × 2 variants)\n",
           total_s, n_dims, N_SEEDS_HIGH);

    free_fixture(&f);
    return 0;
}
