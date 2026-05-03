/* P0-3 verification probe (red-team remediated).
 *
 * Addresses:
 *  C1 — Replaces tautological "geom margin > error margin" Gate with a
 *       meaningful substrate-novelty test on synth_close_proto where
 *       classes are intentionally near-degenerate.
 *  C2 — Adds synth_close_proto benchmark.
 *  C3 — 10 seeds (was 3).
 *  H4 — Timing reported per phase.
 *  H5 — Verdict labels: PASS / WEAK PASS / TIE / WEAK FAIL / FAIL.
 *  H1 — MNIST regression in a separate small-scale probe (mnist_geometric.c).
 *
 *  L1 — Substrate-novelty Gate (3) remains structural; flagged as known
 *       limitation. Building a base-2 baseline would be its own project.
 */

#include "synth_proto.h"
#include "synth_close_proto.h"
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
#include <time.h>

#define N_TRAIN 2000
#define N_TEST  500
#define SIG_DIM 64
#define TOP_K 1
#define N_SEEDS 10
#define N_EPOCHS 50
#define N_FLIPS_PER_EPOCH 200

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

typedef struct {
    int random_pm; int error_pm; int geom_pm;
    int32_t random_margin; int32_t error_margin; int32_t geom_margin;
    double random_s, error_s, geom_s;
} seed_result_t;

/* Generic seed runner; the prototype/sample funcs are passed in. */
typedef enum { BENCH_FAR, BENCH_CLOSE } bench_kind_t;

static void run_seed(seed_result_t* r, bench_kind_t bench,
                       uint32_t train_seed, uint32_t test_seed, uint32_t r_seed) {
    int D, C;
    m4t_trit_t* protos;
    m4t_trit_t* train; int* train_lbl;
    m4t_trit_t* test; int* test_lbl;

    if (bench == BENCH_FAR) {
        synth_proto_config_t cfg = synth_proto_default();
        D = cfg.input_dim; C = cfg.n_classes;
        protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
        synth_proto_generate_prototypes(protos, &cfg);
        train = malloc((size_t)N_TRAIN * D * sizeof(m4t_trit_t));
        train_lbl = malloc((size_t)N_TRAIN * sizeof(int));
        synth_proto_generate_samples(train, train_lbl, N_TRAIN, protos, &cfg, train_seed);
        test = malloc((size_t)N_TEST * D * sizeof(m4t_trit_t));
        test_lbl = malloc((size_t)N_TEST * sizeof(int));
        synth_proto_generate_samples(test, test_lbl, N_TEST, protos, &cfg, test_seed);
    } else {
        synth_close_proto_config_t cfg = synth_close_proto_default();
        D = cfg.input_dim; C = cfg.n_classes;
        protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
        synth_close_proto_generate_prototypes(protos, &cfg);
        train = malloc((size_t)N_TRAIN * D * sizeof(m4t_trit_t));
        train_lbl = malloc((size_t)N_TRAIN * sizeof(int));
        synth_close_proto_generate_samples(train, train_lbl, N_TRAIN, protos, &cfg, train_seed);
        test = malloc((size_t)N_TEST * D * sizeof(m4t_trit_t));
        test_lbl = malloc((size_t)N_TEST * sizeof(int));
        synth_close_proto_generate_samples(test, test_lbl, N_TEST, protos, &cfg, test_seed);
    }

    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);
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

    /* Random R baseline timing. */
    clock_t t0 = clock();
    m4t_trit_t* train_proj = malloc((size_t)N_TRAIN * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, N_TRAIN, R_random, SIG_DIM, D);
    gesh_bank_build_class_mean(&b_rand, train_proj, train_lbl, N_TRAIN, C);
    r->random_pm = eval_pm(R_random, &b_rand, test, test_lbl, N_TEST, SIG_DIM, D);
    r->random_margin = pairwise_sum(&b_rand);
    r->random_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Error-trained timing. */
    t0 = clock();
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
    r->error_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Geometric-trained timing. */
    t0 = clock();
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
    r->geom_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    free(protos); free(train); free(train_lbl); free(test); free(test_lbl);
    free(R_random); free(R_error); free(R_geom);
    free(b_rand.tiles_packed); free(b_rand.labels);
    free(b_err.tiles_packed);  free(b_err.labels);
    free(b_geom.tiles_packed); free(b_geom.labels);
    free(train_proj);
}

static const char* verdict_label(double mean, double ci) {
    /* PASS:       lower CI > 0 AND mean ≥ +2pp.
     * WEAK PASS:  lower CI > 0 AND mean < +2pp.
     * TIE:        CI straddles zero AND |mean| < 2pp.
     * WEAK FAIL:  upper CI < 0 AND mean > -2pp.
     * FAIL:       upper CI < 0 AND mean ≤ -2pp. */
    double lo = mean - ci, hi = mean + ci;
    if (lo > 0  && mean >= 2.0) return "**PASS**";
    if (lo > 0)                  return "WEAK PASS";
    if (hi < 0  && mean <= -2.0) return "**FAIL**";
    if (hi < 0)                  return "WEAK FAIL";
    return "TIE";
}

static void aggregate_and_report(seed_result_t* results, int n,
                                    const char* bench_name) {
    int rs[64], es[64], gs[64];
    double rsum = 0, esum = 0, gsum = 0, rt = 0, et = 0, gt = 0;
    for (int s = 0; s < n; s++) {
        rs[s] = results[s].random_pm;
        es[s] = results[s].error_pm;
        gs[s] = results[s].geom_pm;
        rsum += rs[s]; esum += es[s]; gsum += gs[s];
        rt += results[s].random_s; et += results[s].error_s; gt += results[s].geom_s;
    }
    double rmean = rsum / n, emean = esum / n, gmean = gsum / n;

    /* Two paired comparisons: geom vs error, geom vs random. */
    double ge_sum = 0, ge_sq = 0;
    double gr_sum = 0, gr_sq = 0;
    for (int s = 0; s < n; s++) {
        ge_sum += (gs[s] - es[s]);
        gr_sum += (gs[s] - rs[s]);
    }
    double ge_mean = ge_sum / n, gr_mean = gr_sum / n;
    for (int s = 0; s < n; s++) {
        double d_ge = (gs[s] - es[s]) - ge_mean; ge_sq += d_ge * d_ge;
        double d_gr = (gs[s] - rs[s]) - gr_mean; gr_sq += d_gr * d_gr;
    }
    double ge_sd = sqrt(ge_sq / (n - 1)), gr_sd = sqrt(gr_sq / (n - 1));
    double ge_ci = 1.96 * ge_sd / sqrt((double)n);
    double gr_ci = 1.96 * gr_sd / sqrt((double)n);

    printf("\n## %s — %d seeds\n", bench_name, n);
    printf("  random R:       %.1f%% (mean), runtime %.2fs total\n", rmean / 10.0, rt);
    printf("  error-trained:  %.1f%% (mean), runtime %.2fs total\n", emean / 10.0, et);
    printf("  geometric:      %.1f%% (mean), runtime %.2fs total\n", gmean / 10.0, gt);
    printf("\n  geom vs random:        %+.2fpp (CI [%+.2f, %+.2f]) → %s\n",
           gr_mean / 10.0, (gr_mean - gr_ci) / 10.0, (gr_mean + gr_ci) / 10.0,
           verdict_label(gr_mean / 10.0, gr_ci / 10.0));
    printf("  geom vs error-trained: %+.2fpp (CI [%+.2f, %+.2f]) → %s\n",
           ge_mean / 10.0, (ge_mean - ge_ci) / 10.0, (ge_mean + ge_ci) / 10.0,
           verdict_label(ge_mean / 10.0, ge_ci / 10.0));
}

int main(void) {
    printf("# P0-3 geometric training verification (remediated)\n");
    printf("# %d seeds × 2 benchmarks × 3 training variants\n", N_SEEDS);
    printf("# n_train=%d, n_test=%d, sig_dim=%d, %d flip budget per training\n",
           N_TRAIN, N_TEST, SIG_DIM, N_EPOCHS * N_FLIPS_PER_EPOCH);

    /* Distinct seeds. */
    uint32_t train_seeds[N_SEEDS] = {
        0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u, 0x55555555u,
        0x66666666u, 0x77777777u, 0x88888888u, 0x99999999u, 0xaaaaaaaau };
    uint32_t test_seeds[N_SEEDS] = {
        0xb1b1b1b1u, 0xb2b2b2b2u, 0xb3b3b3b3u, 0xb4b4b4b4u, 0xb5b5b5b5u,
        0xb6b6b6b6u, 0xb7b7b7b7u, 0xb8b8b8b8u, 0xb9b9b9b9u, 0xbabababau };
    uint32_t r_seeds[N_SEEDS] = {
        0xc7c7c7c7u, 0xb22bd00du, 0xdeadc0deu, 0xfeedfaceu, 0xa5a5a5a5u,
        0xc0ffeebbu, 0x13579bdfu, 0xb16b00b5u, 0xfacefeedu, 0x12345678u };

    /* Far benchmark (synth_proto: random ±1 per class — already well-spread). */
    seed_result_t far_results[N_SEEDS];
    clock_t t0 = clock();
    for (int s = 0; s < N_SEEDS; s++) {
        run_seed(&far_results[s], BENCH_FAR,
                    train_seeds[s], test_seeds[s], r_seeds[s]);
    }
    double far_total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Close benchmark (synth_close_proto: classes differ by 1..C-1 trit flips). */
    seed_result_t close_results[N_SEEDS];
    t0 = clock();
    for (int s = 0; s < N_SEEDS; s++) {
        run_seed(&close_results[s], BENCH_CLOSE,
                    train_seeds[s], test_seeds[s], r_seeds[s]);
    }
    double close_total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    aggregate_and_report(far_results, N_SEEDS, "synth_proto (far prototypes)");
    aggregate_and_report(close_results, N_SEEDS, "synth_close_proto (close prototypes)");

    printf("\n## Total runtime\n");
    printf("  far benchmark:    %.1fs\n", far_total_s);
    printf("  close benchmark:  %.1fs\n", close_total_s);
    printf("  total:            %.1fs\n", far_total_s + close_total_s);

    printf("\n## Substrate-novelty Gate (was Gate 2)\n");
    printf("  REPLACED: tautological 'geom margin > error margin'.\n");
    printf("  NEW: 'geometric outperforms error-trained on close-prototype\n");
    printf("       benchmark' — substantive test of geometric training's\n");
    printf("       claim to capture inter-class margin structure.\n");
    printf("  See close-prototype results above for verdict.\n");

    return 0;
}
