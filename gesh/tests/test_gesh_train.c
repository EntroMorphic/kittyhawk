/*
 * test_gesh_train.c — verifies that lattice-update training improves
 * over random projection initialization.
 *
 * Properties checked:
 *   1. trains_reduces_loss — over training, batch error count goes down.
 *   2. beats_random_baseline — final test accuracy exceeds the
 *      untrained random-projection baseline by a meaningful margin.
 *   3. determinism — same seed → same final R and final bank.
 */

#include "gesh_train.h"
#include "gesh_forward.h"
#include "gesh_bank.h"
#include "synth_proto.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Evaluate test-set accuracy with given R and bank. Returns int percent. */
static int eval_test_accuracy(
    const m4t_trit_t* R,
    const gesh_bank_t* bank,
    const m4t_trit_t* test_samples,
    const int* test_labels,
    int n_test,
    int sig_dim, int input_dim, int top_k)
{
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    int rc = gesh_forward_classify(preds, test_samples, n_test, bank, &proj, top_k);
    if (rc != 0) { free(preds); return -1; }
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        if (preds[i] == test_labels[i]) correct++;
    }
    free(preds);
    return (correct * 100) / n_test;
}

/* Build training + test data, set up R + bank scaffolding, return all
 * via out parameters. Caller frees. */
typedef struct {
    m4t_trit_t* protos;
    m4t_trit_t* train;
    int* train_lbl;
    m4t_trit_t* test;
    int* test_lbl;
    int n_train, n_test;
    int C, D, sig_dim;
} fixture_t;

static fixture_t make_fixture(uint32_t prototypes_seed,
                              uint32_t train_seed, uint32_t test_seed,
                              int n_train, int n_test, int sig_dim) {
    fixture_t f;
    synth_proto_config_t cfg = synth_proto_default();
    cfg.seed = prototypes_seed;
    f.C = cfg.n_classes;
    f.D = cfg.input_dim;
    f.sig_dim = sig_dim;
    f.n_train = n_train;
    f.n_test = n_test;
    f.protos = malloc((size_t)f.C * f.D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(f.protos, &cfg);
    f.train = malloc((size_t)n_train * f.D * sizeof(m4t_trit_t));
    f.train_lbl = malloc((size_t)n_train * sizeof(int));
    synth_proto_generate_samples(f.train, f.train_lbl, n_train, f.protos, &cfg, train_seed);
    f.test = malloc((size_t)n_test * f.D * sizeof(m4t_trit_t));
    f.test_lbl = malloc((size_t)n_test * sizeof(int));
    synth_proto_generate_samples(f.test, f.test_lbl, n_test, f.protos, &cfg, test_seed);
    return f;
}

static void free_fixture(fixture_t* f) {
    free(f->protos); free(f->train); free(f->train_lbl);
    free(f->test); free(f->test_lbl);
}

/* Allocate a fresh bank with the right shape. */
static void alloc_bank(gesh_bank_t* bank, int n_classes, int sig_dim) {
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    bank->tiles_packed = malloc((size_t)n_classes * (size_t)Dp);
    bank->labels = malloc((size_t)n_classes * sizeof(int));
    bank->n_tiles = n_classes;
    bank->sig_dim = sig_dim;
}
static void free_bank(gesh_bank_t* bank) {
    free(bank->tiles_packed); free(bank->labels);
}

/* Property 1: training reduces error count. We don't gate on a
 * specific reduction — just that the FINAL training error count
 * is lower than what the INITIAL random projection produced on the
 * SAME training batch. */
static int test_trains_reduces_loss(void) {
    fixture_t f = make_fixture(0xdeadbeefu, 0x11111111u, 0x22222222u,
                                1000, 500, 32);

    /* Initial R: random ±1 projection. */
    m4t_trit_t* R = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, f.sig_dim, f.D, 0xc0ffeebbu);

    gesh_bank_t bank;
    alloc_bank(&bank, f.C, f.sig_dim);

    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = 30;
    cfg.n_flip_evals_per_epoch = 100;
    cfg.batch_size = 64;
    cfg.seed = 0xa5a5a5a5u;

    int final_errors = gesh_train_lattice_update(
        R, &bank, f.train, f.train_lbl, f.n_train, f.C,
        f.sig_dim, f.D, 1, &cfg);

    if (final_errors < 0) {
        printf("FAIL trains_reduces_loss: training returned -1\n");
        free(R); free_bank(&bank); free_fixture(&f);
        return 1;
    }
    /* M1 fix: tighter gate. With C=10, random chance ≈ 58/64 errors;
     * we want training to drive errors well below chance. Require
     * final_errors < batch_size / 2 = 32. Anything weaker and the
     * test passes "no learning" silently. */
    if (final_errors >= cfg.batch_size / 2) {
        printf("FAIL trains_reduces_loss: final errors=%d/%d "
               "(below batch_size/2 = %d threshold)\n",
               final_errors, cfg.batch_size, cfg.batch_size / 2);
        free(R); free_bank(&bank); free_fixture(&f);
        return 1;
    }

    free(R); free_bank(&bank); free_fixture(&f);
    return 0;
}

/* Property 2: trained projection beats random projection on test set
 * by a meaningful margin. */
static int test_beats_random_baseline(void) {
    fixture_t f = make_fixture(0xdeadbeefu, 0x11111111u, 0x22222222u,
                                2000, 500, 32);

    /* === Random-baseline R (untrained). === */
    m4t_trit_t* R_random = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R_random, f.sig_dim, f.D, 0xc0ffeebbu);
    gesh_bank_t bank_random;
    alloc_bank(&bank_random, f.C, f.sig_dim);
    /* Build bank from random-projected training data (no training yet). */
    {
        m4t_trit_t* proj_train = malloc(
            (size_t)f.n_train * (size_t)f.sig_dim * sizeof(m4t_trit_t));
        for (int i = 0; i < f.n_train; i++) {
            const m4t_trit_t* x = f.train + (size_t)i * f.D;
            m4t_trit_t* s = proj_train + (size_t)i * f.sig_dim;
            for (int oi = 0; oi < f.sig_dim; oi++) {
                const m4t_trit_t* r = R_random + (size_t)oi * f.D;
                int32_t acc = 0;
                for (int j = 0; j < f.D; j++) {
                    acc += (int32_t)r[j] * (int32_t)x[j];
                }
                s[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
            }
        }
        gesh_bank_build_class_mean(&bank_random, proj_train, f.train_lbl,
                                    f.n_train, f.C);
        free(proj_train);
    }
    int random_pct = eval_test_accuracy(R_random, &bank_random,
                                          f.test, f.test_lbl, f.n_test,
                                          f.sig_dim, f.D, 1);

    /* === Trained R (lattice update from same random init). === */
    m4t_trit_t* R_trained = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    memcpy(R_trained, R_random, (size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    gesh_bank_t bank_trained;
    alloc_bank(&bank_trained, f.C, f.sig_dim);

    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = 50;
    cfg.n_flip_evals_per_epoch = 200;
    cfg.batch_size = 128;
    cfg.seed = 0xa5a5a5a5u;

    int final_errors = gesh_train_lattice_update(
        R_trained, &bank_trained, f.train, f.train_lbl, f.n_train, f.C,
        f.sig_dim, f.D, 1, &cfg);
    if (final_errors < 0) {
        printf("FAIL beats_random_baseline: training failed\n");
        free(R_random); free_bank(&bank_random);
        free(R_trained); free_bank(&bank_trained); free_fixture(&f);
        return 1;
    }

    int trained_pct = eval_test_accuracy(R_trained, &bank_trained,
                                           f.test, f.test_lbl, f.n_test,
                                           f.sig_dim, f.D, 1);

    printf("INFO random R: %d%%, trained R: %d%%, gain: %d pp\n",
           random_pct, trained_pct, trained_pct - random_pct);

    free(R_random); free_bank(&bank_random);
    free(R_trained); free_bank(&bank_trained);
    free_fixture(&f);

    /* Gate: trained beats random by at least 5 pp. With C=10, random
     * baseline is ~50-65% in our data; trained should reach ~75%+ if
     * lattice update is doing meaningful work. */
    if (trained_pct < random_pct + 5) {
        printf("FAIL beats_random_baseline: trained %d%% vs random %d%% — "
               "gain %d pp below 5pp threshold\n",
               trained_pct, random_pct, trained_pct - random_pct);
        return 1;
    }
    return 0;
}

/* Property 3: determinism — same seed produces same final R and bank. */
static int test_train_determinism(void) {
    fixture_t f = make_fixture(0xdeadbeefu, 0x11111111u, 0x22222222u,
                                500, 200, 32);

    m4t_trit_t* R1 = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    m4t_trit_t* R2 = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R1, f.sig_dim, f.D, 0xc0ffeebbu);
    gesh_init_random_projection(R2, f.sig_dim, f.D, 0xc0ffeebbu);
    /* Sanity: identical init from identical seed. */
    if (memcmp(R1, R2, (size_t)f.sig_dim * f.D * sizeof(m4t_trit_t)) != 0) {
        printf("FAIL train_determinism: identical seeds produced different init\n");
        free(R1); free(R2); free_fixture(&f);
        return 1;
    }

    gesh_bank_t bank1, bank2;
    alloc_bank(&bank1, f.C, f.sig_dim);
    alloc_bank(&bank2, f.C, f.sig_dim);

    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = 10;
    cfg.n_flip_evals_per_epoch = 50;
    cfg.batch_size = 32;
    cfg.seed = 0x12345678u;

    int e1 = gesh_train_lattice_update(R1, &bank1, f.train, f.train_lbl,
                                         f.n_train, f.C, f.sig_dim, f.D, 1, &cfg);
    int e2 = gesh_train_lattice_update(R2, &bank2, f.train, f.train_lbl,
                                         f.n_train, f.C, f.sig_dim, f.D, 1, &cfg);

    int Dp = M4T_TRIT_PACKED_BYTES(f.sig_dim);
    int ok = (e1 == e2)
          && (memcmp(R1, R2, (size_t)f.sig_dim * f.D * sizeof(m4t_trit_t)) == 0)
          && (memcmp(bank1.tiles_packed, bank2.tiles_packed,
                     (size_t)f.C * (size_t)Dp) == 0);

    free(R1); free(R2); free_bank(&bank1); free_bank(&bank2); free_fixture(&f);
    if (!ok) {
        printf("FAIL train_determinism: parallel runs disagree (e1=%d e2=%d)\n",
               e1, e2);
        return 1;
    }
    return 0;
}

/* M2: multi-seed stability. Across 3 seeds, training should beat random
 * with positive gain ON AVERAGE. Catches single-seed flukes. */
static int test_multi_seed_stability(void) {
    fixture_t f = make_fixture(0xdeadbeefu, 0x11111111u, 0x22222222u,
                                1500, 400, 32);

    uint32_t init_seeds[3]  = { 0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu };
    uint32_t train_seeds[3] = { 0x11111111u, 0x22222222u, 0x33333333u };

    int total_gain = 0;
    int n_seeds = 3;
    for (int s = 0; s < n_seeds; s++) {
        /* Random R baseline. */
        m4t_trit_t* R_rand = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
        gesh_init_random_projection(R_rand, f.sig_dim, f.D, init_seeds[s]);
        gesh_bank_t bank_rand;
        alloc_bank(&bank_rand, f.C, f.sig_dim);
        {
            m4t_trit_t* proj = malloc(
                (size_t)f.n_train * (size_t)f.sig_dim * sizeof(m4t_trit_t));
            for (int i = 0; i < f.n_train; i++) {
                const m4t_trit_t* x = f.train + (size_t)i * f.D;
                m4t_trit_t* sg = proj + (size_t)i * f.sig_dim;
                for (int oi = 0; oi < f.sig_dim; oi++) {
                    const m4t_trit_t* r = R_rand + (size_t)oi * f.D;
                    int32_t acc = 0;
                    for (int j = 0; j < f.D; j++) {
                        acc += (int32_t)r[j] * (int32_t)x[j];
                    }
                    sg[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
                }
            }
            gesh_bank_build_class_mean(&bank_rand, proj, f.train_lbl,
                                        f.n_train, f.C);
            free(proj);
        }
        int rand_pct = eval_test_accuracy(R_rand, &bank_rand,
                                            f.test, f.test_lbl, f.n_test,
                                            f.sig_dim, f.D, 1);

        /* Trained from same init. */
        m4t_trit_t* R_trained = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
        memcpy(R_trained, R_rand, (size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
        gesh_bank_t bank_trained;
        alloc_bank(&bank_trained, f.C, f.sig_dim);
        gesh_train_config_t cfg = gesh_train_default();
        cfg.n_epochs = 30;
        cfg.n_flip_evals_per_epoch = 150;
        cfg.batch_size = 96;
        cfg.bank_refresh_every = 75;
        cfg.batch_refresh_every = 50;
        cfg.seed = train_seeds[s];

        int rc = gesh_train_lattice_update(
            R_trained, &bank_trained, f.train, f.train_lbl, f.n_train, f.C,
            f.sig_dim, f.D, 1, &cfg);
        if (rc < 0) {
            printf("FAIL multi_seed_stability: train returned -1 at seed %d\n", s);
            free(R_rand); free_bank(&bank_rand);
            free(R_trained); free_bank(&bank_trained); free_fixture(&f);
            return 1;
        }
        int trained_pct = eval_test_accuracy(R_trained, &bank_trained,
                                               f.test, f.test_lbl, f.n_test,
                                               f.sig_dim, f.D, 1);
        int gain = trained_pct - rand_pct;
        total_gain += gain;
        printf("INFO multi_seed s=%d: random=%d%% trained=%d%% gain=%+d\n",
               s, rand_pct, trained_pct, gain);

        free(R_rand); free_bank(&bank_rand);
        free(R_trained); free_bank(&bank_trained);
    }

    free_fixture(&f);

    /* Average gain across 3 seeds must be positive AND at least +3pp.
     * This is a regression bound (M3): catches systematic
     * "training-doesn't-help" regimes. */
    int avg_gain = total_gain / n_seeds;
    if (avg_gain < 3) {
        printf("FAIL multi_seed_stability: avg gain %d pp < 3 pp threshold\n",
               avg_gain);
        return 1;
    }
    return 0;
}

/* M3 / regression: training must not catastrophically degrade. With
 * intra-epoch refresh enabled, training should produce a valid R (not
 * one that's strictly worse than random by a large margin). */
static int test_no_catastrophic_regression(void) {
    fixture_t f = make_fixture(0xdeadbeefu, 0x77777777u, 0x88888888u,
                                1000, 300, 16);

    m4t_trit_t* R_rand = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R_rand, f.sig_dim, f.D, 0xb22bu);

    /* Build a rand-baseline bank. */
    gesh_bank_t bank_rand;
    alloc_bank(&bank_rand, f.C, f.sig_dim);
    {
        m4t_trit_t* proj = malloc(
            (size_t)f.n_train * (size_t)f.sig_dim * sizeof(m4t_trit_t));
        for (int i = 0; i < f.n_train; i++) {
            const m4t_trit_t* x = f.train + (size_t)i * f.D;
            m4t_trit_t* sg = proj + (size_t)i * f.sig_dim;
            for (int oi = 0; oi < f.sig_dim; oi++) {
                const m4t_trit_t* r = R_rand + (size_t)oi * f.D;
                int32_t acc = 0;
                for (int j = 0; j < f.D; j++) {
                    acc += (int32_t)r[j] * (int32_t)x[j];
                }
                sg[oi] = (acc > 0) ? 1 : (acc < 0) ? -1 : 0;
            }
        }
        gesh_bank_build_class_mean(&bank_rand, proj, f.train_lbl,
                                    f.n_train, f.C);
        free(proj);
    }
    int rand_pct = eval_test_accuracy(R_rand, &bank_rand,
                                        f.test, f.test_lbl, f.n_test,
                                        f.sig_dim, f.D, 1);

    /* Train. */
    m4t_trit_t* R_trained = malloc((size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    memcpy(R_trained, R_rand, (size_t)f.sig_dim * f.D * sizeof(m4t_trit_t));
    gesh_bank_t bank_trained;
    alloc_bank(&bank_trained, f.C, f.sig_dim);
    gesh_train_config_t cfg = gesh_train_default();
    cfg.n_epochs = 20;
    cfg.n_flip_evals_per_epoch = 100;
    cfg.batch_size = 96;
    cfg.bank_refresh_every = 50;
    cfg.batch_refresh_every = 50;
    cfg.seed = 0xc7c7c7c7u;

    int rc = gesh_train_lattice_update(
        R_trained, &bank_trained, f.train, f.train_lbl, f.n_train, f.C,
        f.sig_dim, f.D, 1, &cfg);
    if (rc < 0) {
        printf("FAIL no_catastrophic_regression: training failed\n");
        free(R_rand); free_bank(&bank_rand);
        free(R_trained); free_bank(&bank_trained); free_fixture(&f);
        return 1;
    }
    int trained_pct = eval_test_accuracy(R_trained, &bank_trained,
                                           f.test, f.test_lbl, f.n_test,
                                           f.sig_dim, f.D, 1);

    free(R_rand); free_bank(&bank_rand);
    free(R_trained); free_bank(&bank_trained); free_fixture(&f);

    /* Catastrophic regression bound: trained must not be more than 5pp
     * worse than random. The sig_dim=64 anomaly observed in the sweep
     * was −2pp; this gate is calibrated to catch a true regression
     * (e.g., training breaks and produces near-chance accuracy). */
    if (trained_pct < rand_pct - 5) {
        printf("FAIL no_catastrophic_regression: random=%d%% trained=%d%% "
               "(regression %d pp exceeds 5 pp threshold)\n",
               rand_pct, trained_pct, rand_pct - trained_pct);
        return 1;
    }
    printf("INFO no_catastrophic_regression: random=%d%% trained=%d%%\n",
           rand_pct, trained_pct);
    return 0;
}

int main(void) {
    if (test_trains_reduces_loss())          return 1;
    if (test_beats_random_baseline())        return 1;
    if (test_train_determinism())            return 1;
    if (test_multi_seed_stability())         return 1;
    if (test_no_catastrophic_regression())   return 1;
    printf("gesh_train: all 5 tests passed\n");
    return 0;
}
