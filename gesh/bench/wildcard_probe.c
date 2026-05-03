/*
 * wildcard_probe.c — P0-1 verification, multi-seed.
 *
 * Three independent (proto, train, test, R) seed sets. Reports mean
 * ± stddev for each (bank, kernel) cell. The substrate-novel finding
 * is the wildcard BANK paired with standard Hamming kernel.
 */

#include "synth_wildcard.h"
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

#define N_TRAIN 2000
#define N_TEST  500
#define SIG_DIM 64
#define TOP_K   1
#define WILDCARD_SNR_THRESHOLD_PERMILLE 200
#define N_SEEDS 3

typedef enum { DIST_HAMMING = 0, DIST_WILDCARD = 1 } dist_kind_t;
typedef enum { BANK_MEAN = 0, BANK_WILDCARD = 1 } bank_kind_t;

static int eval_pm(
    const m4t_trit_t* R, const gesh_bank_t* bank,
    const m4t_trit_t* test, const int* test_lbl, int n_test,
    int sig_dim, int input_dim, dist_kind_t dist_kind)
{
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    int rc = (dist_kind == DIST_HAMMING)
        ? gesh_forward_classify(preds, test, n_test, bank, &proj, TOP_K)
        : gesh_forward_classify_wildcard(preds, test, n_test, bank, &proj, TOP_K);
    if (rc != 0) { free(preds); return -1; }
    int correct = 0;
    for (int i = 0; i < n_test; i++) if (preds[i] == test_lbl[i]) correct++;
    free(preds);
    return (correct * 1000) / n_test;
}

/* Run one seed set; populate the four (bank, kernel) cell results. */
static void run_one_seed(
    int cell_pm[2][2],
    uint32_t proto_seed, uint32_t train_seed, uint32_t test_seed, uint32_t r_seed)
{
    synth_wildcard_config_t cfg = synth_wildcard_default();
    cfg.proto_seed = proto_seed;
    int D = cfg.input_dim;
    int C = cfg.n_classes;
    int Dp_sig = M4T_TRIT_PACKED_BYTES(SIG_DIM);

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_wildcard_generate_prototypes(protos, &cfg);

    m4t_trit_t* train = malloc((size_t)N_TRAIN * D * sizeof(m4t_trit_t));
    int* train_lbl = malloc((size_t)N_TRAIN * sizeof(int));
    synth_wildcard_generate_samples(train, train_lbl, N_TRAIN, protos, &cfg, train_seed);

    m4t_trit_t* test = malloc((size_t)N_TEST * D * sizeof(m4t_trit_t));
    int* test_lbl = malloc((size_t)N_TEST * sizeof(int));
    synth_wildcard_generate_samples(test, test_lbl, N_TEST, protos, &cfg, test_seed);

    m4t_trit_t* R = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, D, r_seed);

    m4t_trit_t* train_proj = malloc((size_t)N_TRAIN * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, N_TRAIN, R, SIG_DIM, D);

    gesh_bank_t bm, bw;
    bm.tiles_packed = malloc((size_t)C * (size_t)Dp_sig);
    bm.labels = malloc((size_t)C * sizeof(int));
    bm.n_tiles = C; bm.sig_dim = SIG_DIM;
    gesh_bank_build_class_mean(&bm, train_proj, train_lbl, N_TRAIN, C);

    bw.tiles_packed = malloc((size_t)C * (size_t)Dp_sig);
    bw.labels = malloc((size_t)C * sizeof(int));
    bw.n_tiles = C; bw.sig_dim = SIG_DIM;
    gesh_bank_build_class_wildcard(&bw, train_proj, train_lbl, N_TRAIN, C,
                                       WILDCARD_SNR_THRESHOLD_PERMILLE);

    cell_pm[BANK_MEAN][DIST_HAMMING]      = eval_pm(R, &bm, test, test_lbl, N_TEST, SIG_DIM, D, DIST_HAMMING);
    cell_pm[BANK_MEAN][DIST_WILDCARD]     = eval_pm(R, &bm, test, test_lbl, N_TEST, SIG_DIM, D, DIST_WILDCARD);
    cell_pm[BANK_WILDCARD][DIST_HAMMING]  = eval_pm(R, &bw, test, test_lbl, N_TEST, SIG_DIM, D, DIST_HAMMING);
    cell_pm[BANK_WILDCARD][DIST_WILDCARD] = eval_pm(R, &bw, test, test_lbl, N_TEST, SIG_DIM, D, DIST_WILDCARD);

    free(protos); free(train); free(train_lbl); free(test); free(test_lbl);
    free(R); free(train_proj);
    free(bm.tiles_packed); free(bm.labels);
    free(bw.tiles_packed); free(bw.labels);
}

static void stats(const int* vals, int n, double* mean_pct, double* sd_pp) {
    double sum = 0; for (int i = 0; i < n; i++) sum += vals[i];
    double m = sum / n;
    double sq = 0; for (int i = 0; i < n; i++) { double d = vals[i] - m; sq += d*d; }
    double var = (n > 1) ? sq / (n - 1) : 0;
    *mean_pct = m / 10.0;
    *sd_pp = sqrt(var) / 10.0;
}

static const uint32_t PROTO_SEEDS[N_SEEDS]  = { 0xc0ffeebbu, 0xa5a5a5a5u, 0xfeedfaceu };
static const uint32_t TRAIN_SEEDS[N_SEEDS]  = { 0x11111111u, 0x22222222u, 0x33333333u };
static const uint32_t TEST_SEEDS[N_SEEDS]   = { 0x44444444u, 0x55555555u, 0x66666666u };
static const uint32_t R_SEEDS[N_SEEDS]      = { 0xc7c7c7c7u, 0xb22bd00du, 0xdeadc0deu };

int main(void) {
    printf("# P0-1 wildcard verification — %d seeds\n", N_SEEDS);
    printf("# synth_wildcard: C=10, D=64 (K=16+M=16+N=32), n_train=%d, n_test=%d, sig_dim=%d\n",
           N_TRAIN, N_TEST, SIG_DIM);
    printf("\n");

    int cells_per_seed[N_SEEDS][2][2];
    for (int s = 0; s < N_SEEDS; s++) {
        run_one_seed(cells_per_seed[s], PROTO_SEEDS[s], TRAIN_SEEDS[s], TEST_SEEDS[s], R_SEEDS[s]);
    }

    /* Aggregate. */
    int mean_ham[N_SEEDS], mean_wld[N_SEEDS], wld_ham[N_SEEDS], wld_wld[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++) {
        mean_ham[s] = cells_per_seed[s][BANK_MEAN][DIST_HAMMING];
        mean_wld[s] = cells_per_seed[s][BANK_MEAN][DIST_WILDCARD];
        wld_ham[s]  = cells_per_seed[s][BANK_WILDCARD][DIST_HAMMING];
        wld_wld[s]  = cells_per_seed[s][BANK_WILDCARD][DIST_WILDCARD];
    }
    double m_mh, sd_mh, m_mw, sd_mw, m_wh, sd_wh, m_ww, sd_ww;
    stats(mean_ham, N_SEEDS, &m_mh, &sd_mh);
    stats(mean_wld, N_SEEDS, &m_mw, &sd_mw);
    stats(wld_ham,  N_SEEDS, &m_wh, &sd_wh);
    stats(wld_wld,  N_SEEDS, &m_ww, &sd_ww);

    /* Paired gains (substrate-novel cell minus baseline cell, per seed). */
    int gain_bank_alone[N_SEEDS], gain_committed_pair[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++) {
        gain_bank_alone[s]      = wld_ham[s]  - mean_ham[s];
        gain_committed_pair[s]  = wld_wld[s]  - mean_ham[s];
    }
    double gba_mean, gba_sd, gcp_mean, gcp_sd;
    stats(gain_bank_alone,     N_SEEDS, &gba_mean, &gba_sd);
    stats(gain_committed_pair, N_SEEDS, &gcp_mean, &gcp_sd);

    printf("| bank          | kernel    | mean ± stddev    | min   | max   |\n");
    printf("|---------------|-----------|------------------|-------|-------|\n");
    int min_v, max_v;
    min_v = max_v = mean_ham[0]; for (int s=1;s<N_SEEDS;s++){if(mean_ham[s]<min_v)min_v=mean_ham[s];if(mean_ham[s]>max_v)max_v=mean_ham[s];}
    printf("| class_mean    | Hamming   | %5.1f%% ± %4.2fpp | %5.1f | %5.1f |\n", m_mh, sd_mh, min_v/10.0, max_v/10.0);
    min_v = max_v = mean_wld[0]; for (int s=1;s<N_SEEDS;s++){if(mean_wld[s]<min_v)min_v=mean_wld[s];if(mean_wld[s]>max_v)max_v=mean_wld[s];}
    printf("| class_mean    | Wildcard  | %5.1f%% ± %4.2fpp | %5.1f | %5.1f |\n", m_mw, sd_mw, min_v/10.0, max_v/10.0);
    min_v = max_v = wld_ham[0]; for (int s=1;s<N_SEEDS;s++){if(wld_ham[s]<min_v)min_v=wld_ham[s];if(wld_ham[s]>max_v)max_v=wld_ham[s];}
    printf("| class_wildcard| Hamming   | %5.1f%% ± %4.2fpp | %5.1f | %5.1f | <-- bank alone\n", m_wh, sd_wh, min_v/10.0, max_v/10.0);
    min_v = max_v = wld_wld[0]; for (int s=1;s<N_SEEDS;s++){if(wld_wld[s]<min_v)min_v=wld_wld[s];if(wld_wld[s]>max_v)max_v=wld_wld[s];}
    printf("| class_wildcard| Wildcard  | %5.1f%% ± %4.2fpp | %5.1f | %5.1f | <-- committed pair\n", m_ww, sd_ww, min_v/10.0, max_v/10.0);

    printf("\n");
    printf("## Paired gains vs baseline (class_mean + Hamming)\n");
    printf("  Bank alone (class_wildcard + Hamming): %+5.2fpp ± %4.2fpp paired stddev\n", gba_mean, gba_sd);
    printf("  Committed pair (class_wildcard + Wildcard): %+5.2fpp ± %4.2fpp paired stddev\n", gcp_mean, gcp_sd);
    printf("  95%% CI on bank-alone gain: [%+5.2f, %+5.2f] (mean ± 1.96*sd/sqrt(n))\n",
           gba_mean - 1.96*gba_sd/sqrt(N_SEEDS), gba_mean + 1.96*gba_sd/sqrt(N_SEEDS));

    return 0;
}
