/* P0-3 verification: geometric vs error-trained R on synth_proto.
 * 3 seeds, paired CI. Compares classification accuracy.
 */

#include "synth_proto.h"
#include "gesh_bank.h"
#include "gesh_forward.h"
#include "gesh_project.h"
#include "gesh_train.h"
#include "m4t_trit_pack.h"
#include "m4t_route.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_TRAIN 2000
#define N_TEST  500
#define SIG_DIM 64
#define TOP_K 1
#define N_SEEDS 3
#define N_EPOCHS 50
#define N_FLIPS_PER_EPOCH 200
#define BUDGET (N_EPOCHS * N_FLIPS_PER_EPOCH)

static int eval_pm(const m4t_trit_t* R, const gesh_bank_t* bank,
                      const m4t_trit_t* test, const int* test_lbl, int n_test,
                      int sig_dim, int input_dim) {
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    gesh_forward_classify(preds, test, n_test, bank, &proj, TOP_K);
    int correct = 0;
    for (int i = 0; i < n_test; i++) if (preds[i] == test_lbl[i]) correct++;
    free(preds);
    return (correct * 1000) / n_test;
}

static int32_t pairwise_sum(const gesh_bank_t* bank) {
    int Dp = M4T_TRIT_PACKED_BYTES(bank->sig_dim);
    uint8_t* mask = malloc((size_t)Dp);
    memset(mask, 0xFF, (size_t)Dp);
    int tail = bank->sig_dim & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
    int32_t v = m4t_route_pairwise_hamming_sum(bank->tiles_packed, mask,
                                                  bank->n_tiles, bank->sig_dim);
    free(mask);
    return v;
}

typedef struct { int random_pm; int error_pm; int geom_pm;
                  int32_t random_margin; int32_t error_margin; int32_t geom_margin; } seed_result_t;

static void run_seed(seed_result_t* r,
                       uint32_t train_seed, uint32_t test_seed, uint32_t r_seed) {
    synth_proto_config_t cfg = synth_proto_default();
    int D = cfg.input_dim, C = cfg.n_classes;
    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_proto_generate_prototypes(protos, &cfg);
    m4t_trit_t* train = malloc((size_t)N_TRAIN * D * sizeof(m4t_trit_t));
    int* train_lbl = malloc((size_t)N_TRAIN * sizeof(int));
    synth_proto_generate_samples(train, train_lbl, N_TRAIN, protos, &cfg, train_seed);
    m4t_trit_t* test = malloc((size_t)N_TEST * D * sizeof(m4t_trit_t));
    int* test_lbl = malloc((size_t)N_TEST * sizeof(int));
    synth_proto_generate_samples(test, test_lbl, N_TEST, protos, &cfg, test_seed);

    /* Three R variants — same init, different training. */
    m4t_trit_t* R_random = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    m4t_trit_t* R_error  = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    m4t_trit_t* R_geom   = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R_random, SIG_DIM, D, r_seed);
    memcpy(R_error, R_random, (size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    memcpy(R_geom,  R_random, (size_t)SIG_DIM * D * sizeof(m4t_trit_t));

    gesh_bank_t b_rand, b_err, b_geom;
    b_rand.tiles_packed = malloc((size_t)C * (size_t)Dp);
    b_rand.labels = malloc((size_t)C * sizeof(int));
    b_rand.n_tiles = C; b_rand.sig_dim = SIG_DIM;
    b_err = b_rand; b_err.tiles_packed = malloc((size_t)C * (size_t)Dp);
    b_err.labels = malloc((size_t)C * sizeof(int));
    b_geom = b_rand; b_geom.tiles_packed = malloc((size_t)C * (size_t)Dp);
    b_geom.labels = malloc((size_t)C * sizeof(int));

    /* Random R baseline: build bank from random R. */
    m4t_trit_t* train_proj = malloc((size_t)N_TRAIN * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, N_TRAIN, R_random, SIG_DIM, D);
    gesh_bank_build_class_mean(&b_rand, train_proj, train_lbl, N_TRAIN, C);
    r->random_pm = eval_pm(R_random, &b_rand, test, test_lbl, N_TEST, SIG_DIM, D);
    r->random_margin = pairwise_sum(&b_rand);

    /* Error-trained R. */
    gesh_train_config_t cfg_err = gesh_train_default();
    cfg_err.n_epochs = N_EPOCHS;
    cfg_err.n_flip_evals_per_epoch = N_FLIPS_PER_EPOCH;
    cfg_err.batch_size = 128;
    cfg_err.bank_refresh_every = N_FLIPS_PER_EPOCH / 4;
    cfg_err.batch_refresh_every = N_FLIPS_PER_EPOCH / 4;
    cfg_err.early_stop_patience = 5;
    cfg_err.log_per_epoch = 0;
    cfg_err.seed = r_seed ^ 0xa5a5a5a5u;
    gesh_train_lattice_update(R_error, &b_err, train, train_lbl,
                                 N_TRAIN, C, SIG_DIM, D, TOP_K, &cfg_err);
    r->error_pm = eval_pm(R_error, &b_err, test, test_lbl, N_TEST, SIG_DIM, D);
    r->error_margin = pairwise_sum(&b_err);

    /* Geometric-trained R. */
    gesh_train_config_t cfg_geom = gesh_train_default();
    cfg_geom.n_epochs = N_EPOCHS;
    cfg_geom.n_flip_evals_per_epoch = N_FLIPS_PER_EPOCH;
    cfg_geom.early_stop_patience = 5;
    cfg_geom.log_per_epoch = 0;
    cfg_geom.seed = r_seed ^ 0xb7b7b7b7u;
    gesh_train_lattice_update_geometric(R_geom, &b_geom, train, train_lbl,
                                           N_TRAIN, C, SIG_DIM, D, &cfg_geom);
    r->geom_pm = eval_pm(R_geom, &b_geom, test, test_lbl, N_TEST, SIG_DIM, D);
    r->geom_margin = pairwise_sum(&b_geom);

    free(protos); free(train); free(train_lbl); free(test); free(test_lbl);
    free(R_random); free(R_error); free(R_geom);
    free(b_rand.tiles_packed); free(b_rand.labels);
    free(b_err.tiles_packed);  free(b_err.labels);
    free(b_geom.tiles_packed); free(b_geom.labels);
    free(train_proj);
}

int main(void) {
    printf("# P0-3 geometric training verification — 3 seeds, synth_proto\n");
    printf("# n_train=%d, n_test=%d, sig_dim=%d, budget=%d flips\n",
           N_TRAIN, N_TEST, SIG_DIM, BUDGET);

    uint32_t train_seeds[3] = { 0x11111111u, 0x22222222u, 0x33333333u };
    uint32_t test_seeds[3]  = { 0x44444444u, 0x55555555u, 0x66666666u };
    uint32_t r_seeds[3]     = { 0xc7c7c7c7u, 0xb22bd00du, 0xdeadc0deu };

    seed_result_t results[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++) {
        run_seed(&results[s], train_seeds[s], test_seeds[s], r_seeds[s]);
        printf("  seed %d: random=%.1f%%, error=%.1f%%, geom=%.1f%% | margins random=%d error=%d geom=%d\n",
               s, results[s].random_pm/10.0, results[s].error_pm/10.0, results[s].geom_pm/10.0,
               results[s].random_margin, results[s].error_margin, results[s].geom_margin);
    }

    /* Aggregate. */
    int rs[N_SEEDS], es[N_SEEDS], gs[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++) { rs[s] = results[s].random_pm; es[s] = results[s].error_pm; gs[s] = results[s].geom_pm; }

    double rmean = 0, emean = 0, gmean = 0;
    for (int s = 0; s < N_SEEDS; s++) { rmean += rs[s]; emean += es[s]; gmean += gs[s]; }
    rmean /= N_SEEDS; emean /= N_SEEDS; gmean /= N_SEEDS;

    /* Paired gain: geom - error per seed. */
    double gain_sum = 0, gain_sq = 0;
    for (int s = 0; s < N_SEEDS; s++) gain_sum += (gs[s] - es[s]);
    double gain_mean = gain_sum / N_SEEDS;
    for (int s = 0; s < N_SEEDS; s++) {
        double d = (gs[s] - es[s]) - gain_mean;
        gain_sq += d * d;
    }
    double gain_sd = sqrt(gain_sq / (N_SEEDS - 1));
    double gain_ci = 1.96 * gain_sd / sqrt(N_SEEDS);

    printf("\n## Mean accuracy\n");
    printf("  random R:     %.1f%%\n", rmean / 10.0);
    printf("  error-trained: %.1f%%\n", emean / 10.0);
    printf("  geometric:    %.1f%%\n", gmean / 10.0);
    printf("\n## Paired gain (geometric - error-trained)\n");
    printf("  paired mean: %+.2fpp\n", gain_mean / 10.0);
    printf("  paired stddev: %.2fpp\n", gain_sd / 10.0);
    printf("  95%% CI: [%+.2f, %+.2f]\n", (gain_mean - gain_ci)/10.0, (gain_mean + gain_ci)/10.0);

    /* Gate 1: geometric ≥ error-trained, paired-CI lower bound > 0 (ideal) or
     * gain_mean ≥ 0pp (not regressing). */
    if ((gain_mean - gain_ci) > 0)
        printf("  Gate 1 verdict: **PASS** (CI excludes zero, geometric beats error-trained)\n");
    else if (gain_mean >= 0)
        printf("  Gate 1 verdict: TIE (geometric ≥ error-trained on average; CI includes zero)\n");
    else
        printf("  Gate 1 verdict: **FAIL** (geometric trains worse than error-trained)\n");

    /* Gate 2 by construction: did geometric optimize the margin? */
    int geom_margin_higher = 1;
    for (int s = 0; s < N_SEEDS; s++) {
        if (results[s].geom_margin <= results[s].error_margin) { geom_margin_higher = 0; break; }
    }
    printf("\n## Gate 2 — geometric R has higher pairwise margin than error-trained R\n");
    printf("  Gate 2 verdict: %s (geom margin > error margin in all seeds: %s)\n",
           geom_margin_higher ? "**PASS**" : "FAIL",
           geom_margin_higher ? "yes" : "no");

    return 0;
}
