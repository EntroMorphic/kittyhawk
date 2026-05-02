/*
 * sweep_dims.c — Phase A.2 benchmark: sweeps sig_dim across a range of
 * compression and expansion ratios, comparing the three variants:
 *   1. Random R (untrained ternary projection, sig_dim out of D)
 *   2. Trained R (lattice-update from same random init)
 *   3. Identity (no projection, sig_dim = D, reference)
 *
 * Synthetic prototype-classification benchmark: D=64 (16 informative +
 * 48 noise), C=10 classes, 10% per-trit noise, n_train=2000, n_test=500.
 *
 * Training budget scales with R size: 5 flip-evaluations per trit on
 * average (so larger projections get proportionally more training).
 *
 * Deterministic via fixed seeds. Output is a printable table.
 */

#include "synth_proto.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_train.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    const m4t_trit_t* R,        /* nullable for identity */
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

/* Project all training samples through R, build bank from projections. */
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

/* ── Variant runners ────────────────────────────────────────────────────── */

static int run_random(const fixture_t* f, int sig_dim, double* out_sec) {
    clock_t t0 = clock();
    m4t_trit_t* R = malloc((size_t)sig_dim * f->D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, f->D, 0xc0ffeebbu);

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
    *out_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pct;
}

static int run_trained(const fixture_t* f, int sig_dim, int flip_budget,
                         double* out_sec)
{
    clock_t t0 = clock();
    m4t_trit_t* R = malloc((size_t)sig_dim * f->D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, sig_dim, f->D, 0xc0ffeebbu);

    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;

    /* Spread the budget over a fixed number of epochs so end-of-epoch
     * bank refresh frequency is comparable across sig_dims. */
    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = 50;
    cfg.n_flip_evals_per_epoch = flip_budget / cfg.n_epochs;
    if (cfg.n_flip_evals_per_epoch < 10) cfg.n_flip_evals_per_epoch = 10;
    cfg.batch_size = 128;
    cfg.log_per_epoch = 0;
    cfg.seed = 0xa5a5a5a5u;

    int rc = gesh_train_lattice_update(R, &bank, f->train, f->train_lbl,
                                         f->n_train, f->C, sig_dim, f->D,
                                         1, &cfg);
    if (rc < 0) {
        *out_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
        free(R); free(bank.tiles_packed); free(bank.labels);
        return -1;
    }

    int pct = eval_test_accuracy(R, &bank, f->test, f->test_lbl,
                                   f->n_test, sig_dim, f->D, 1);
    *out_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pct;
}

static int run_identity(const fixture_t* f, double* out_sec) {
    clock_t t0 = clock();
    int sig_dim = f->D;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;
    /* Identity: bank is built directly from samples (no projection). */
    gesh_bank_build_class_mean(&bank, f->train, f->train_lbl,
                                f->n_train, f->C);
    int pct = eval_test_accuracy(NULL, &bank, f->test, f->test_lbl,
                                   f->n_test, sig_dim, f->D, 1);
    *out_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    free(bank.tiles_packed); free(bank.labels);
    return pct;
}

/* ── Main sweep ─────────────────────────────────────────────────────────── */

int main(void) {
    fixture_t f = make_fixture();

    /* Sweep dims as requested + reference points at 32, 64. */
    int dims[] = { 2, 4, 8, 16, 32, 64, 128, 256 };
    int n_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    printf("# Gesh Phase A.2 sig_dim sweep\n");
    printf("# Synthetic prototype classification: D=%d (K=16 informative + "
           "%d noise), C=%d, 10%%%% noise, n_train=%d, n_test=%d.\n",
           f.D, f.D - 16, f.C, f.n_train, f.n_test);
    printf("# Trained budget: ~5 flip-evals per trit, spread over 50 epochs.\n");
    printf("\n");
    printf("| sig_dim | random | trained | gain | flip_budget |  rand_s | train_s |\n");
    printf("|---------|--------|---------|------|-------------|---------|---------|\n");

    for (int i = 0; i < n_dims; i++) {
        int sig = dims[i];
        double rand_t = 0.0, train_t = 0.0;
        int rand_pct = run_random(&f, sig, &rand_t);

        /* Budget = 5 trits * sig_dim * input_dim, with floor for very
         * small R to ensure adequate training. */
        int budget = 5 * sig * f.D;
        if (budget < 500) budget = 500;

        int trained_pct = run_trained(&f, sig, budget, &train_t);

        if (rand_pct < 0 || trained_pct < 0) {
            printf("| %7d | ERROR  | ERROR   | —    | %11d | %7.2f | %7.2f |\n",
                   sig, budget, rand_t, train_t);
            continue;
        }
        printf("| %7d | %5d%% |  %5d%% | %+3d  | %11d | %7.2f | %7.2f |\n",
               sig, rand_pct, trained_pct, trained_pct - rand_pct, budget,
               rand_t, train_t);
        fflush(stdout);
    }

    /* Identity reference (sig_dim = D, no projection). */
    double id_t = 0.0;
    int id_pct = run_identity(&f, &id_t);
    printf("\n");
    printf("Identity (sig_dim=D=%d, no projection): %d%%   (%.2fs)\n",
           f.D, id_pct, id_t);

    free_fixture(&f);
    return 0;
}
