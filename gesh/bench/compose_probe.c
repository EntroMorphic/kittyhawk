/*
 * compose_probe.c — P0-4 verification, multi-seed.
 *
 * Compares three classifiers on synth_close_proto (close-prototype
 * benchmark from P0-3 remediation):
 *
 *   1. single-stage class-mean       (regular Hamming) — baseline
 *   2. single-stage wildcard bank    (P0-1) — the more permissive baseline
 *   3. two-stage hierarchical        (P0-4) — the new compositional thing
 *
 * Reports paired-CI on accuracy across N_SEEDS seeds. Passes Gate 1
 * iff (3 vs 2) lower CI bound > 0 with effect ≥ 1pp.
 */

#include "synth_close_proto.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_project.h"
#include "gesh_train.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_TRAIN 2000
#define N_TEST  500
#define SIG_DIM 64
#define TOP_K   1
#define WILDCARD_SNR_THRESHOLD_PERMILLE 200
#define N_SEEDS 10

static int eval_single(
    int* preds_out,
    const gesh_projection_t* proj,
    const gesh_bank_t* bank,
    const m4t_trit_t* test, int n_test, int top_k, int wildcard)
{
    return wildcard
        ? gesh_forward_classify_wildcard(preds_out, test, n_test, bank, proj, top_k)
        : gesh_forward_classify         (preds_out, test, n_test, bank, proj, top_k);
}

static int correct_pm(const int* preds, const int* labels, int n) {
    int c = 0; for (int i = 0; i < n; i++) if (preds[i] == labels[i]) c++;
    return (c * 1000) / n;
}

typedef struct {
    int classmean_pm;
    int wildcard_pm;
    int compose_pm;
} seed_result_t;

static void run_one_seed(seed_result_t* r,
                          uint32_t proto_seed, uint32_t train_seed,
                          uint32_t test_seed,  uint32_t r_seed) {
    synth_close_proto_config_t cfg = synth_close_proto_default();
    cfg.seed = proto_seed;

    int D = cfg.input_dim;
    int C = cfg.n_classes;
    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    int n_train = N_TRAIN;
    int n_test  = N_TEST;

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_close_proto_generate_prototypes(protos, &cfg);

    m4t_trit_t* train  = malloc((size_t)n_train * D * sizeof(m4t_trit_t));
    m4t_trit_t* test   = malloc((size_t)n_test  * D * sizeof(m4t_trit_t));
    int*        ytrain = malloc((size_t)n_train * sizeof(int));
    int*        ytest  = malloc((size_t)n_test  * sizeof(int));

    synth_close_proto_generate_samples(train, ytrain, n_train, protos, &cfg, train_seed);
    synth_close_proto_generate_samples(test,  ytest,  n_test,  protos, &cfg, test_seed);

    m4t_trit_t* R = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, D, r_seed);
    gesh_projection_t proj = { .R = R, .input_dim = D, .sig_dim = SIG_DIM };

    /* Project training samples once (for bank construction). */
    m4t_trit_t* train_proj = malloc((size_t)n_train * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, n_train, R, SIG_DIM, D);

    /* (1) classmean + Hamming. */
    gesh_bank_t b_mean = { .tiles_packed = malloc((size_t)C * Dp),
                            .labels = malloc((size_t)C * sizeof(int)),
                            .n_tiles = C, .sig_dim = SIG_DIM };
    gesh_bank_build_class_mean(&b_mean, train_proj, ytrain, n_train, C);
    int* preds = malloc((size_t)n_test * sizeof(int));
    eval_single(preds, &proj, &b_mean, test, n_test, TOP_K, /*wildcard=*/0);
    r->classmean_pm = correct_pm(preds, ytest, n_test);

    /* (2) wildcard bank + wildcard kernel. */
    gesh_bank_t b_wild = { .tiles_packed = malloc((size_t)C * Dp),
                            .labels = malloc((size_t)C * sizeof(int)),
                            .n_tiles = C, .sig_dim = SIG_DIM };
    gesh_bank_build_class_wildcard(&b_wild, train_proj, ytrain, n_train, C,
                                      WILDCARD_SNR_THRESHOLD_PERMILLE);
    eval_single(preds, &proj, &b_wild, test, n_test, TOP_K, /*wildcard=*/1);
    r->wildcard_pm = correct_pm(preds, ytest, n_test);

    /* (3) hierarchical (compositional). */
    gesh_bank_hier_t hbank;
    gesh_bank_hier_alloc(&hbank, C, SIG_DIM);
    gesh_bank_build_hierarchical(&hbank, train_proj, ytrain, n_train, C,
                                    WILDCARD_SNR_THRESHOLD_PERMILLE);
    gesh_forward_classify_hierarchical(preds, NULL, test, n_test, &hbank, &proj);
    r->compose_pm = correct_pm(preds, ytest, n_test);

    free(preds);
    free(b_mean.tiles_packed); free(b_mean.labels);
    free(b_wild.tiles_packed); free(b_wild.labels);
    gesh_bank_hier_free(&hbank);
    free(train_proj); free(R);
    free(train); free(test); free(ytrain); free(ytest);
    free(protos);
}

static void mean_ci(const double* deltas, int n, double* mean, double* ci) {
    double s = 0; for (int i = 0; i < n; i++) s += deltas[i];
    *mean = s / n;
    double var = 0;
    for (int i = 0; i < n; i++) { double d = deltas[i] - *mean; var += d*d; }
    var /= (n > 1 ? n - 1 : 1);
    double se = sqrt(var / n);
    *ci = 1.96 * se;
}

static const char* verdict_label(double mean, double ci) {
    double lo = mean - ci, hi = mean + ci;
    if (lo > 0  && mean >= 1.0) return "**PASS**";
    if (lo > 0)                  return "WEAK PASS";
    if (hi < 0  && mean <= -1.0) return "**FAIL**";
    if (hi < 0)                  return "WEAK FAIL";
    return "TIE";
}

int main(void) {
    printf("# P0-4 compositional routing verification\n");
    printf("# %d seeds, synth_close_proto (close prototypes)\n", N_SEEDS);
    printf("# n_train=%d n_test=%d sig_dim=%d top_k=%d snr_pm=%d\n\n",
           N_TRAIN, N_TEST, SIG_DIM, TOP_K, WILDCARD_SNR_THRESHOLD_PERMILLE);

    seed_result_t results[N_SEEDS];
    clock_t t0 = clock();
    for (int s = 0; s < N_SEEDS; s++) {
        uint32_t base = 0xC0DEC0DEu + (uint32_t)s * 17;
        run_one_seed(&results[s],
                      base, base ^ 0xA1, base ^ 0xB2, base ^ 0xC3);
        printf("seed %2d: classmean=%.1f%%  wildcard=%.1f%%  compose=%.1f%%\n",
               s, results[s].classmean_pm/10.0,
               results[s].wildcard_pm/10.0,
               results[s].compose_pm/10.0);
    }
    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Aggregate. */
    double mean_cm = 0, mean_wc = 0, mean_cp = 0;
    for (int s = 0; s < N_SEEDS; s++) {
        mean_cm += results[s].classmean_pm/10.0;
        mean_wc += results[s].wildcard_pm/10.0;
        mean_cp += results[s].compose_pm/10.0;
    }
    mean_cm /= N_SEEDS; mean_wc /= N_SEEDS; mean_cp /= N_SEEDS;

    /* Paired deltas: compose vs wildcard (the headline gate),
     * and compose vs classmean. */
    double d_vs_wc[N_SEEDS], d_vs_cm[N_SEEDS], d_wc_vs_cm[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++) {
        d_vs_wc[s]  = (results[s].compose_pm  - results[s].wildcard_pm) / 10.0;
        d_vs_cm[s]  = (results[s].compose_pm  - results[s].classmean_pm) / 10.0;
        d_wc_vs_cm[s] = (results[s].wildcard_pm - results[s].classmean_pm) / 10.0;
    }

    double m_cwc, ci_cwc, m_ccm, ci_ccm, m_wcm, ci_wcm;
    mean_ci(d_vs_wc, N_SEEDS, &m_cwc, &ci_cwc);
    mean_ci(d_vs_cm, N_SEEDS, &m_ccm, &ci_ccm);
    mean_ci(d_wc_vs_cm, N_SEEDS, &m_wcm, &ci_wcm);

    printf("\n## Means across %d seeds\n", N_SEEDS);
    printf("  classmean (Hamming) : %.2f%%\n", mean_cm);
    printf("  wildcard  (P0-1)    : %.2f%%\n", mean_wc);
    printf("  compose   (P0-4)    : %.2f%%\n", mean_cp);

    printf("\n## Paired comparisons (Δpp, 95%% CI from %d seeds)\n", N_SEEDS);
    printf("  wildcard vs classmean : %+.2fpp  CI [%+.2f, %+.2f]  → %s\n",
           m_wcm, m_wcm - ci_wcm, m_wcm + ci_wcm, verdict_label(m_wcm, ci_wcm));
    printf("  compose  vs classmean : %+.2fpp  CI [%+.2f, %+.2f]  → %s\n",
           m_ccm, m_ccm - ci_ccm, m_ccm + ci_ccm, verdict_label(m_ccm, ci_ccm));
    printf("  compose  vs wildcard  : %+.2fpp  CI [%+.2f, %+.2f]  → %s   <-- Gate 1 headline\n",
           m_cwc, m_cwc - ci_cwc, m_cwc + ci_cwc, verdict_label(m_cwc, ci_cwc));

    printf("\n## Total runtime: %.2fs\n", total_s);
    return 0;
}
