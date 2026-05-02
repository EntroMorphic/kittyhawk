/*
 * finding3_probe.c — high-seed-count probe for Finding 3
 * (capacity floor at sig_dim ≤ 4) from sweep_dims_results.md.
 *
 * Layered against the SDOT-finding3 red-team's C3, H6, M1, M2:
 *
 *   C3 — capacity-floor MECHANISM probe: build the bank from trained R,
 *        count distinct tile signatures across C classes (predict ≤ 9
 *        at sig_dim=2 by pigeonhole), output per-class confusion matrix
 *        on the test set, identify any class pair whose tiles map to
 *        the same signature (collision pattern). Outcome → mechanism.
 *
 *   H6 — paired-difference CI for the gain (gain[s] = trained[s] -
 *        random[s] per seed; report the paired stddev; the correct CI
 *        is the paired one because (init, train) seeds are matched).
 *
 *   M2 — robust statistics: median + 10% trimmed mean alongside the
 *        arithmetic mean (per-cell range was 13–15pp; outliers can
 *        skew the mean).
 *
 *   M1 — unstructured random seeds (no byte-stride patterns).
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

/* Variant of run_trained_seed that returns the (R, bank) AND the eval
 * accuracy. Used for the mechanism probe at sig_dim=2 to inspect the
 * trained R and bank rather than discard them. */
static int run_trained_seed_keep(
    const fixture_t* f, int sig_dim,
    int flip_budget,
    uint32_t init_seed, uint32_t train_seed,
    m4t_trit_t* out_R, gesh_bank_t* out_bank)
{
    gesh_init_random_projection(out_R, sig_dim, f->D, init_seed);

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

    int rc = gesh_train_lattice_update(out_R, out_bank, f->train,
                                          f->train_lbl, f->n_train, f->C,
                                          sig_dim, f->D, 1, &cfg);
    if (rc < 0) return -1;
    return eval_accuracy_pm(out_R, out_bank, f->test, f->test_lbl,
                              f->n_test, sig_dim, f->D);
}

static int run_trained_seed(const fixture_t* f, int sig_dim,
                              int flip_budget,
                              uint32_t init_seed, uint32_t train_seed) {
    m4t_trit_t* R = malloc((size_t)sig_dim * f->D * sizeof(m4t_trit_t));
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;

    int pm = run_trained_seed_keep(f, sig_dim, flip_budget,
                                      init_seed, train_seed, R, &bank);
    free(R); free(bank.tiles_packed); free(bank.labels);
    return pm;
}

/* ── Robust statistics (M2) ──────────────────────────────────────────── */

static int cmp_int(const void* a, const void* b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* Permille-input statistics with arithmetic mean, median, 10% trimmed
 * mean, stddev, and 95% CI on mean. All output values are in percent
 * (mean) or pp (stddev/CI). */
typedef struct {
    double mean_pct, sd_pp, ci95_pp;
    double median_pct, trimmed_pct;
    double min_pct, max_pct;
} cell_stats_t;

static cell_stats_t compute_cell_stats(const int* vals, int n) {
    cell_stats_t s = {0};
    int* sorted = malloc((size_t)n * sizeof(int));
    memcpy(sorted, vals, (size_t)n * sizeof(int));
    qsort(sorted, n, sizeof(int), cmp_int);

    s.min_pct = sorted[0] / 10.0;
    s.max_pct = sorted[n - 1] / 10.0;

    /* Median. */
    if (n % 2 == 1) s.median_pct = sorted[n/2] / 10.0;
    else            s.median_pct = (sorted[n/2 - 1] + sorted[n/2]) / 20.0;

    /* 10%-trimmed mean: drop floor(0.1 * n) from each tail. */
    int trim = (int)(0.1 * (double)n);
    int n_kept = n - 2 * trim;
    double tsum = 0.0;
    for (int i = trim; i < n - trim; i++) tsum += (double)sorted[i];
    s.trimmed_pct = (n_kept > 0) ? tsum / n_kept / 10.0 : 0.0;

    /* Arithmetic mean + stddev. */
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)vals[i];
    double mean_pm = sum / (double)n;
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)vals[i] - mean_pm;
        sq += d * d;
    }
    double var = (n > 1) ? sq / (double)(n - 1) : 0.0;
    s.mean_pct = mean_pm / 10.0;
    s.sd_pp    = sqrt(var) / 10.0;
    s.ci95_pp  = 1.96 * s.sd_pp / sqrt((double)n);

    free(sorted);
    return s;
}

/* Paired-difference statistics (H6): per-seed gain[s] = trained[s] -
 * random[s]; the correct CI for the gain treats the (random, trained)
 * pair as matched, not independent. */
static cell_stats_t compute_paired_gain(const int* random_pm,
                                          const int* trained_pm, int n) {
    int* gains = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) gains[i] = trained_pm[i] - random_pm[i];
    cell_stats_t s = compute_cell_stats(gains, n);
    free(gains);
    return s;
}

/* ── Mechanism probe (C3) ────────────────────────────────────────────── */

/* Output for one trained R at sig_dim. The bank tile per class is a
 * packed-trit signature; we expand each tile to an integer key and
 * count distinct keys to detect pigeonhole collisions. Confusion
 * matrix tracks per-class predicted distribution on the test set. */
typedef struct {
    int n_distinct_signatures;  /* count of distinct class-tile keys */
    int collisions[16];          /* pairs of class indices that collide; -1 sentinel terminates */
    int n_collisions;            /* number of collision pairs found */
    int confusion[16][16];       /* per-class predicted counts on test set (max C=16) */
    int test_correct[16];        /* per-class correct counts */
    int test_count[16];          /* per-class test-sample counts */
} mechanism_t;

static int tile_key(const uint8_t* tile_packed, int sig_dim) {
    /* Convert a small packed-trit tile to a single int by treating each
     * trit as base-3 digit and accumulating. Works for sig_dim ≤ 9
     * within int range (3^9 ≈ 20K). */
    int key = 0;
    int pow3 = 1;
    for (int t = 0; t < sig_dim; t++) {
        int byte_idx = t >> 2;
        int bit_off  = (t & 3) * 2;
        uint8_t code = (uint8_t)((tile_packed[byte_idx] >> bit_off) & 0x3u);
        int trit = (code == 0x01u) ? 1
                  : (code == 0x02u) ? -1
                  : 0;
        key += (trit + 1) * pow3;  /* shift to {0,1,2} for ternary digit */
        pow3 *= 3;
    }
    return key;
}

static void run_mechanism_probe(
    const fixture_t* f, int sig_dim, int flip_budget,
    uint32_t init_seed, uint32_t train_seed,
    mechanism_t* out)
{
    memset(out, 0, sizeof(*out));
    out->n_collisions = 0;
    for (int i = 0; i < 16; i++) out->collisions[i] = -1;

    /* Train one R; capture both R and bank. */
    m4t_trit_t* R = malloc((size_t)sig_dim * f->D * sizeof(m4t_trit_t));
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)f->C * (size_t)Dp);
    bank.labels = malloc((size_t)f->C * sizeof(int));
    bank.n_tiles = f->C;
    bank.sig_dim = sig_dim;

    int rc = run_trained_seed_keep(f, sig_dim, flip_budget,
                                      init_seed, train_seed, R, &bank);
    if (rc < 0) {
        free(R); free(bank.tiles_packed); free(bank.labels);
        return;
    }

    /* Tile-key analysis: detect pigeonhole-forced collisions. */
    int keys[16];
    for (int c = 0; c < f->C; c++) {
        keys[c] = tile_key(bank.tiles_packed + (size_t)c * Dp, sig_dim);
    }
    /* Count distinct. */
    int distinct = 0;
    for (int c = 0; c < f->C; c++) {
        int seen = 0;
        for (int prev = 0; prev < c; prev++) {
            if (keys[prev] == keys[c]) { seen = 1; break; }
        }
        if (!seen) distinct++;
    }
    out->n_distinct_signatures = distinct;
    /* Record colliding class-pairs. */
    for (int c = 0; c < f->C && out->n_collisions < 8; c++) {
        for (int c2 = c + 1; c2 < f->C && out->n_collisions < 8; c2++) {
            if (keys[c] == keys[c2]) {
                out->collisions[out->n_collisions * 2]     = c;
                out->collisions[out->n_collisions * 2 + 1] = c2;
                out->n_collisions++;
            }
        }
    }

    /* Per-class confusion on test set. */
    gesh_projection_t proj = { .R = R, .input_dim = f->D, .sig_dim = sig_dim };
    int* preds = malloc((size_t)f->n_test * sizeof(int));
    rc = gesh_forward_classify(preds, f->test, f->n_test, &bank, &proj, 1);
    if (rc == 0) {
        for (int i = 0; i < f->n_test; i++) {
            int true_c = f->test_lbl[i];
            int pred_c = preds[i];
            if (true_c >= 0 && true_c < 16 && pred_c >= 0 && pred_c < 16) {
                out->confusion[true_c][pred_c]++;
                out->test_count[true_c]++;
                if (true_c == pred_c) out->test_correct[true_c]++;
            }
        }
    }
    free(preds);

    free(R); free(bank.tiles_packed); free(bank.labels);
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fixture_t f = make_fixture();

    /* Seed pairs. First 5 match sweep_dims for cross-check; remaining 25
     * are unstructured random hex (M1 fix — no byte-stride patterns). */
    uint32_t init_seeds[N_SEEDS_HIGH] = {
        /* original sweep_dims seeds (positions 0..4) — match for sub-mean check */
        0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu, 0xb16b00b5u, 0x13579bdfu,
        /* 25 unstructured random hex seeds */
        0x7d3a91c4u, 0x2f5e1b8au, 0xc8b9d6fau, 0x4a72e905u, 0x8e1d3c46u,
        0x9f6b2ad7u, 0x35e8c14bu, 0xb74af293u, 0x6c1e58dau, 0xa329b1f6u,
        0x52d4a780u, 0xe18c63b9u, 0x4f9ad271u, 0xc7e3805au, 0x9d2148f3u,
        0x6b8c5e9fu, 0xa4f72d1cu, 0x3e5b9c80u, 0x71d6a384u, 0xc92e7b15u,
        0x8a4f1d6bu, 0x57c39e02u, 0xd1b46a87u, 0x2e9853a4u, 0xf6c8d27bu
    };
    uint32_t train_seeds[N_SEEDS_HIGH] = {
        /* original sweep_dims seeds (positions 0..4) */
        0xa5a5a5a5u, 0xc7c7c7c7u, 0xb22bd00du, 0xdeadc0deu, 0x0123abcdu,
        /* 25 unstructured random hex seeds */
        0x4d8c1f3au, 0x9e2b7d05u, 0x36c4a91fu, 0xb78fe2d3u, 0x1a93c45eu,
        0x7e6d12c8u, 0xa509f3b6u, 0x4c3187a2u, 0xd2b9670fu, 0x6f8e4c15u,
        0x83a7d219u, 0xe43b5c87u, 0x29f1a608u, 0x5d8c4b73u, 0xc6921ea4u,
        0x71f3d258u, 0x4e9a3c1du, 0xb87126fau, 0x35e9d24cu, 0xa1d8754bu,
        0x6c2f9b30u, 0xe7a1d569u, 0x9b3478c2u, 0x521ae8d4u, 0xf09b2c87u
    };

    int sig_dims[] = { 2, 4, 8 };
    int n_dims = (int)(sizeof(sig_dims) / sizeof(sig_dims[0]));

    printf("# Finding 3 high-seed probe — capacity floor at sig_dim ≤ 4\n");
    printf("# D=%d (16 informative + %d noise), C=%d, n_train=%d, n_test=%d\n",
           f.D, f.D - 16, f.C, f.n_train, f.n_test);
    printf("# 30 seeds per cell; first 5 match sweep_dims for sub-mean check.\n\n");

    /* Outcome statistics (mean ± stddev, median, trimmed mean, paired CI). */
    printf("## Outcome statistics (30 seeds; permille precision)\n\n");
    printf("| sig_dim | variant   | mean ± stddev      | median | trim10 | min  | max  | 95%% CI |\n");
    printf("|---------|-----------|--------------------|--------|--------|------|------|----------|\n");

    /* Cache per-seed permille results for paired-CI computation later. */
    int* random_per_dim_seed[3];
    int* trained_per_dim_seed[3];

    clock_t t0 = clock();
    for (int i = 0; i < n_dims; i++) {
        int sig = sig_dims[i];
        int budget = 5 * sig * f.D;
        if (budget < 500) budget = 500;

        int* random_pm  = malloc((size_t)N_SEEDS_HIGH * sizeof(int));
        int* trained_pm = malloc((size_t)N_SEEDS_HIGH * sizeof(int));
        for (int s = 0; s < N_SEEDS_HIGH; s++) {
            random_pm[s]  = run_random_seed(&f, sig, init_seeds[s]);
            trained_pm[s] = run_trained_seed(&f, sig, budget,
                                                init_seeds[s], train_seeds[s]);
        }

        cell_stats_t r_stats = compute_cell_stats(random_pm,  N_SEEDS_HIGH);
        cell_stats_t t_stats = compute_cell_stats(trained_pm, N_SEEDS_HIGH);
        printf("| %7d | random    | %5.1f%% ± %5.2fpp | %5.1f%% | %5.1f%% | %4.1f | %4.1f | ±%4.2fpp |\n",
               sig, r_stats.mean_pct, r_stats.sd_pp, r_stats.median_pct,
               r_stats.trimmed_pct, r_stats.min_pct, r_stats.max_pct, r_stats.ci95_pp);
        printf("| %7d | trained   | %5.1f%% ± %5.2fpp | %5.1f%% | %5.1f%% | %4.1f | %4.1f | ±%4.2fpp |\n",
               sig, t_stats.mean_pct, t_stats.sd_pp, t_stats.median_pct,
               t_stats.trimmed_pct, t_stats.min_pct, t_stats.max_pct, t_stats.ci95_pp);
        printf("|---------|-----------|--------------------|--------|--------|------|------|----------|\n");
        fflush(stdout);

        random_per_dim_seed[i]  = random_pm;
        trained_per_dim_seed[i] = trained_pm;
    }

    /* Paired-difference CI on the gain (H6). */
    printf("\n## Paired-difference gain (correct CI for gain; not independent-samples)\n\n");
    printf("| sig_dim | gain mean | paired stddev | paired 95%% CI | lower bound > 0? |\n");
    printf("|---------|-----------|----------------|------------------|-------------------|\n");
    for (int i = 0; i < n_dims; i++) {
        int sig = sig_dims[i];
        cell_stats_t g = compute_paired_gain(random_per_dim_seed[i],
                                                trained_per_dim_seed[i],
                                                N_SEEDS_HIGH);
        double lower = g.mean_pct - g.ci95_pp;
        printf("| %7d | %+5.2fpp  | %5.2fpp       | ±%4.2fpp         | %s            |\n",
               sig, g.mean_pct, g.sd_pp, g.ci95_pp,
               (lower > 0.0) ? "YES" : "no");
    }

    /* Cross-check: 5-seed sub-mean should match sweep_dims (post-permille). */
    printf("\n## 5-seed sub-mean cross-check (positions 0..4 match sweep_dims seeds)\n\n");
    for (int i = 0; i < n_dims; i++) {
        cell_stats_t r5 = compute_cell_stats(random_per_dim_seed[i],  5);
        cell_stats_t t5 = compute_cell_stats(trained_per_dim_seed[i], 5);
        printf("  sig_dim = %d:  random %5.1f%%  trained %5.1f%%   "
               "(must match permille-corrected sweep_dims)\n",
               sig_dims[i], r5.mean_pct, t5.mean_pct);
    }

    /* Mechanism probe (C3): inspect trained R, bank, and confusion at
     * sig_dim=2 to verify the pigeonhole collision pattern. */
    printf("\n## Capacity-floor MECHANISM probe (C3)\n\n");
    for (int i = 0; i < n_dims; i++) {
        int sig = sig_dims[i];
        int budget = 5 * sig * f.D;
        if (budget < 500) budget = 500;
        /* One representative seed for mechanism inspection. */
        mechanism_t m;
        run_mechanism_probe(&f, sig, budget, init_seeds[0], train_seeds[0], &m);

        int max_signatures = 1;
        for (int p = 0; p < sig; p++) max_signatures *= 3;

        printf("### sig_dim = %d (max %d distinct signatures, C = %d classes)\n",
               sig, max_signatures, f.C);
        printf("  Distinct trained class-tile signatures: **%d** (of %d possible)\n",
               m.n_distinct_signatures, f.C);
        if (m.n_distinct_signatures < f.C) {
            printf("  Pigeonhole collisions (class pairs sharing a signature):\n");
            for (int p = 0; p < m.n_collisions; p++) {
                int c1 = m.collisions[p * 2];
                int c2 = m.collisions[p * 2 + 1];
                printf("    classes %d ↔ %d  share a tile signature\n", c1, c2);
            }
            printf("  Pigeonhole prediction: at sig_dim=%d, max=%d signatures < %d classes\n",
                   sig, max_signatures, f.C);
            if (max_signatures < f.C) {
                printf("    → forced collision is required by capacity. CONFIRMED.\n");
            } else {
                printf("    → collision possible but not forced; observed instance.\n");
            }
        } else {
            printf("  No collisions: each of %d classes maps to a distinct signature.\n",
                   f.C);
        }
        printf("  Per-class accuracy on test set:\n");
        for (int c = 0; c < f.C; c++) {
            double acc = (m.test_count[c] > 0)
                ? 100.0 * (double)m.test_correct[c] / m.test_count[c]
                : 0.0;
            printf("    class %d:  %3d/%3d correct  (%5.1f%%)\n",
                   c, m.test_correct[c], m.test_count[c], acc);
        }
        printf("\n");
    }

    /* Cleanup. */
    for (int i = 0; i < n_dims; i++) {
        free(random_per_dim_seed[i]);
        free(trained_per_dim_seed[i]);
    }

    /* Capacity argument: pigeonhole with sig_dim < log_3(C). */
    printf("## Capacity argument\n");
    printf("  sig_dim = 2 → 3^2 =  9 distinct ternary signatures.\n");
    printf("  sig_dim = 4 → 3^4 = 81 distinct ternary signatures.\n");
    printf("  sig_dim = 8 → 3^8 = 6561 distinct ternary signatures.\n");
    printf("  C = 10 classes.\n");
    printf("  Pigeonhole: collisions FORCED at sig_dim < log_3(C) ≈ 2.1.\n");
    printf("  At sig_dim = 2, AT LEAST one pair of classes must share a signature.\n");
    printf("  Mechanism probe above tests this prediction directly.\n\n");

    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Total runtime: %.1fs (%d sig_dims × %d seeds × 2 variants + mechanism probes)\n",
           total_s, n_dims, N_SEEDS_HIGH);

    free_fixture(&f);
    return 0;
}
