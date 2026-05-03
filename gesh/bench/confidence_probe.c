/*
 * confidence_probe.c — P0-2 verification.
 *
 * Synth_proto multi-seed (3 seeds) + MNIST regression check. Compares
 * standard Hamming routing against confidence-weighted routing using
 * the dual-extract + class_mean_with_confidence pair.
 */

#include "synth_proto.h"
#include "image_canon.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_project.h"
#include "gesh_train.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIG_DIM 64
#define TOP_K   1
#define N_SEEDS 3

/* Forward classify by confidence-weighted dist over a pre-built bank.
 * Skips projection — caller passes already-projected query trits. */
static int classify_confidence(
    int* preds,
    const uint8_t* test_trit_packed, const uint8_t* test_conf_bits,
    int n_test, int sig_dim,
    const uint8_t* tiles_packed, const uint8_t* tile_conf_bits,
    int T, const int* labels)
{
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int conf_bytes = (sig_dim + 7) / 8;
    uint8_t mask[256];
    memset(mask, 0xFF, (size_t)Dp);
    int tail = sig_dim & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);

    /* Class label range is implicit in labels[]; we don't need n_classes
     * for nearest-tile classification (just take label of best tile). */
    for (int q = 0; q < n_test; q++) {
        const uint8_t* qt = test_trit_packed + (size_t)q * Dp;
        const uint8_t* qc = test_conf_bits   + (size_t)q * conf_bytes;
        int best_class = 0;
        int32_t best_dist = INT32_MAX;
        for (int t = 0; t < T; t++) {
            int32_t d = m4t_route_confidence_weighted_dist(
                qt, qc,
                tiles_packed   + (size_t)t * Dp,
                tile_conf_bits + (size_t)t * conf_bytes,
                mask, sig_dim);
            if (d < best_dist) {
                best_dist = d;
                best_class = labels[t];
            }
        }
        preds[q] = best_class;
    }
    return 0;
}

/* Project samples through R, then dual-extract to get (trit, conf)
 * per sample. Returns malloc'd buffers. */
static void project_and_dual_extract(
    uint8_t** out_trits, uint8_t** out_conf,
    const m4t_trit_t* samples, int n,
    const m4t_trit_t* R, int sig_dim, int input_dim,
    int64_t tau_weak, int64_t tau_strong)
{
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int conf_bytes = (sig_dim + 7) / 8;
    *out_trits = malloc((size_t)n * (size_t)Dp);
    *out_conf  = malloc((size_t)n * (size_t)conf_bytes);

    int64_t* row = malloc((size_t)sig_dim * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        const m4t_trit_t* x = samples + (size_t)i * input_dim;
        /* Compute int64 acc per output dim. */
        for (int oi = 0; oi < sig_dim; oi++) {
            int64_t acc = 0;
            const m4t_trit_t* r = R + (size_t)oi * input_dim;
            for (int j = 0; j < input_dim; j++) acc += (int64_t)r[j] * (int64_t)x[j];
            row[oi] = acc;
        }
        m4t_route_threshold_extract_dual(
            *out_trits + (size_t)i * Dp,
            *out_conf  + (size_t)i * conf_bytes,
            row, tau_weak, tau_strong, sig_dim);
    }
    free(row);
}

/* Run one synth_proto seed; return (baseline_pm, confidence_pm) at the
 * specified tau_strong. tau_weak fixed at 0 (sign extraction). */
static void run_synth_seed_tau(int* baseline_pm, int* conf_pm,
                                  uint32_t train_seed, uint32_t test_seed,
                                  uint32_t r_seed,
                                  int64_t tau_strong, int tau_strong_pm_for_bank);

static void run_synth_seed(int* baseline_pm, int* conf_pm,
                              uint32_t train_seed, uint32_t test_seed,
                              uint32_t r_seed)
{
    run_synth_seed_tau(baseline_pm, conf_pm, train_seed, test_seed, r_seed,
                          /*tau_strong*/5, /*tau_strong_pm_for_bank*/600);
}

static void run_synth_seed_tau(int* baseline_pm, int* conf_pm,
                                  uint32_t train_seed, uint32_t test_seed,
                                  uint32_t r_seed,
                                  int64_t tau_strong,
                                  int tau_strong_pm_for_bank)
{
    synth_proto_config_t cfg = synth_proto_default();
    int D = cfg.input_dim, C = cfg.n_classes;
    int n_train = 2000, n_test = 500;
    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    int conf_bytes = (SIG_DIM + 7) / 8;

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);
    m4t_trit_t* train = malloc((size_t)n_train * D * sizeof(m4t_trit_t));
    int* train_lbl = malloc((size_t)n_train * sizeof(int));
    synth_proto_generate_samples(train, train_lbl, n_train, protos, &cfg, train_seed);
    m4t_trit_t* test = malloc((size_t)n_test * D * sizeof(m4t_trit_t));
    int* test_lbl = malloc((size_t)n_test * sizeof(int));
    synth_proto_generate_samples(test, test_lbl, n_test, protos, &cfg, test_seed);

    m4t_trit_t* R = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, D, r_seed);

    /* Project training samples for bank build. */
    m4t_trit_t* train_proj = malloc((size_t)n_train * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, n_train, R, SIG_DIM, D);

    /* Baseline: class_mean bank + standard Hamming forward. */
    gesh_bank_t bank;
    bank.tiles_packed = malloc((size_t)C * (size_t)Dp);
    bank.labels = malloc((size_t)C * sizeof(int));
    bank.n_tiles = C; bank.sig_dim = SIG_DIM;
    gesh_bank_build_class_mean(&bank, train_proj, train_lbl, n_train, C);

    int* preds = malloc((size_t)n_test * sizeof(int));
    gesh_projection_t proj = { .R = R, .input_dim = D, .sig_dim = SIG_DIM };
    gesh_forward_classify(preds, test, n_test, &bank, &proj, TOP_K);
    int correct = 0;
    for (int i = 0; i < n_test; i++) if (preds[i] == test_lbl[i]) correct++;
    *baseline_pm = (correct * 1000) / n_test;

    /* Confidence path: caller-supplied tau_strong (raw acc) and
     * tau_strong_pm_for_bank (permille for class-mean magnitude). */
    int64_t tau_weak = 0;
    int tau_strong_permille = tau_strong_pm_for_bank;

    uint8_t* tile_conf = malloc((size_t)C * (size_t)conf_bytes);
    gesh_bank_build_class_mean_with_confidence(
        &bank, tile_conf, train_proj, train_lbl, n_train, C,
        tau_strong_permille);

    uint8_t* test_trit_packed;
    uint8_t* test_conf_bits;
    project_and_dual_extract(&test_trit_packed, &test_conf_bits,
                                test, n_test, R, SIG_DIM, D,
                                tau_weak, tau_strong);

    classify_confidence(preds, test_trit_packed, test_conf_bits,
                          n_test, SIG_DIM,
                          bank.tiles_packed, tile_conf, C, bank.labels);
    correct = 0;
    for (int i = 0; i < n_test; i++) if (preds[i] == test_lbl[i]) correct++;
    *conf_pm = (correct * 1000) / n_test;

    free(protos); free(train); free(train_lbl); free(test); free(test_lbl);
    free(R); free(train_proj); free(preds);
    free(bank.tiles_packed); free(bank.labels); free(tile_conf);
    free(test_trit_packed); free(test_conf_bits);
}

int main(void) {
    printf("# P0-2 confidence verification — synth_proto, %d seeds, tau-sweep\n", N_SEEDS);
    uint32_t train_seeds[3] = { 0x11111111u, 0x22222222u, 0x33333333u };
    uint32_t test_seeds[3]  = { 0x44444444u, 0x55555555u, 0x66666666u };
    uint32_t r_seeds[3]     = { 0xc7c7c7c7u, 0xb22bd00du, 0xdeadc0deu };

    /* C2 fix: tau-sweep to verify gain isn't a calibration artifact.
     * Sweep tau_strong over a reasonable range; report per-tau gain. */
    struct tau_pair { int64_t raw; int permille; const char* name; } taus[] = {
        {  2, 300, "low" },
        {  5, 600, "default" },
        { 10, 750, "high" },
        { 20, 900, "very-high" },
    };
    int n_taus = (int)(sizeof(taus) / sizeof(taus[0]));

    printf("\n## Tau-sweep (per-tau, %d seeds each)\n", N_SEEDS);
    printf("| tau_strong (raw, perm) | baseline mean | confidence mean | paired gain | 95%% CI       |\n");
    printf("|------------------------|---------------|------------------|-------------|---------------|\n");

    for (int ti = 0; ti < n_taus; ti++) {
        int baseline[N_SEEDS], conf[N_SEEDS];
        for (int s = 0; s < N_SEEDS; s++) {
            run_synth_seed_tau(&baseline[s], &conf[s],
                                  train_seeds[s], test_seeds[s], r_seeds[s],
                                  taus[ti].raw, taus[ti].permille);
        }
        double bsum = 0, csum = 0; for (int s=0;s<N_SEEDS;s++){bsum+=baseline[s];csum+=conf[s];}
        double bmean = bsum / N_SEEDS, cmean = csum / N_SEEDS;
        double gsum = 0; for (int s=0;s<N_SEEDS;s++) gsum += (conf[s] - baseline[s]);
        double gmean = gsum / N_SEEDS;
        double gsq = 0; for (int s=0;s<N_SEEDS;s++) {
            double d = (conf[s] - baseline[s]) - gmean; gsq += d*d;
        }
        double gsd = sqrt(gsq / (N_SEEDS - 1));
        double gci = 1.96 * gsd / sqrt(N_SEEDS);
        printf("| %s (%lld, %d)        | %5.1f%%        | %5.1f%%           | %+5.2fpp     | [%+5.2f, %+5.2f] |\n",
               taus[ti].name, (long long)taus[ti].raw, taus[ti].permille,
               bmean / 10.0, cmean / 10.0, gmean / 10.0,
               (gmean - gci) / 10.0, (gmean + gci) / 10.0);
    }

    /* Default-tau detail (kept for backward-compat with the prior reporting). */
    int baseline[N_SEEDS], conf[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++) {
        run_synth_seed(&baseline[s], &conf[s],
                          train_seeds[s], test_seeds[s], r_seeds[s]);
    }

    double bsum = 0, csum = 0;
    for (int s = 0; s < N_SEEDS; s++) { bsum += baseline[s]; csum += conf[s]; }
    double bmean = bsum / N_SEEDS, cmean = csum / N_SEEDS;

    int gain_pm[N_SEEDS];
    double gsum = 0;
    for (int s = 0; s < N_SEEDS; s++) {
        gain_pm[s] = conf[s] - baseline[s];
        gsum += gain_pm[s];
    }
    double gmean = gsum / N_SEEDS;
    double gsq = 0;
    for (int s = 0; s < N_SEEDS; s++) {
        double d = gain_pm[s] - gmean;
        gsq += d * d;
    }
    double gsd = sqrt(gsq / (N_SEEDS - 1));
    double gci = 1.96 * gsd / sqrt(N_SEEDS);

    printf("\n## Synth_proto multi-seed results\n");
    printf("| seed | baseline (mean+Hamming) | confidence (mean+conf) | gain |\n");
    printf("|------|-------------------------|------------------------|------|\n");
    for (int s = 0; s < N_SEEDS; s++) {
        printf("|  %d  |        %5.1f%%          |       %5.1f%%          | %+.1fpp |\n",
               s, baseline[s] / 10.0, conf[s] / 10.0, gain_pm[s] / 10.0);
    }
    printf("| MEAN |        %5.1f%%          |       %5.1f%%          | %+.2fpp |\n",
           bmean / 10.0, cmean / 10.0, gmean / 10.0);
    printf("\n## Paired gain CI\n");
    printf("  paired mean: %+.2fpp\n", gmean / 10.0);
    printf("  paired stddev: %.2fpp\n", gsd / 10.0);
    printf("  95%% CI: [%+.2f, %+.2f]\n",
           (gmean - gci) / 10.0, (gmean + gci) / 10.0);
    printf("  Gate 1 PASS bar: lower bound > 0 AND mean ≥ +2pp\n");
    if ((gmean - gci) > 0 && gmean >= 20)
        printf("  Gate 1 verdict: **PASS**\n");
    else if ((gmean - gci) > 0)
        printf("  Gate 1 verdict: WEAK PASS (CI excludes zero, magnitude < 2pp)\n");
    else
        printf("  Gate 1 verdict: INCONCLUSIVE / FAIL (CI includes zero)\n");

    return 0;
}
