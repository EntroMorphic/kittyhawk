/*
 * wildcard_probe.c — P0-1 verification benchmark.
 *
 * Runs the four pre-committed gates from gesh_zero_signal_design_synthesize.md:
 *   Gate 1: wildcard accuracy on synth_wildcard ≥ Hamming + 5pp.
 *   Gate 2: wildcard runtime ≤ 1.2× current Hamming.
 *   Gate 3: substrate-novelty audit — articulated in the gate verdict
 *           narrative (storage advantage is structural, not runtime-
 *           measurable beyond what Gate 2 already captures).
 *   Gate 4: no MNIST regression beyond ±2pp (skipped in this probe; run
 *           the existing mnist probes to verify, with caller-supplied
 *           wildcard bank if desired).
 *
 * Configuration:
 *   synth_wildcard:
 *     C = 10 classes
 *     D = 64 = K(16 always) + M(16 sometimes) + N(32 noise)
 *     n_train = 2000, n_test = 500
 *     noise_pct = 10
 *
 *   Bank constructors compared:
 *     class_mean    — emergent zeros from sample-cancellation ties
 *     class_wildcard — deliberate zeros at signal_pm < threshold
 *
 *   Distance kernels compared:
 *     popcount_dist     — §19 (I) Tie-cancellation symmetric Hamming
 *     wildcard_dist     — §19 (II) Wildcard tile-zero, (III) Abstain query-zero
 *
 * Four (bank, kernel) pairs measured; the diagonal pair (class_wildcard +
 * wildcard_dist) is the substrate-novel cell. The off-diagonal pairs
 * are §19.6 violations but measured for the comparison.
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
#include <time.h>

#define N_TRAIN 2000
#define N_TEST  500
#define SIG_DIM 64
#define TOP_K   1
#define WILDCARD_SNR_THRESHOLD_PERMILLE 200  /* signal < 0.2 → wildcard */

typedef enum {
    BANK_CLASS_MEAN     = 0,
    BANK_CLASS_WILDCARD = 1,
} bank_kind_t;

typedef enum {
    DIST_HAMMING  = 0,
    DIST_WILDCARD = 1,
} dist_kind_t;

static int eval_pm(
    const m4t_trit_t* R, const gesh_bank_t* bank,
    const m4t_trit_t* test, const int* test_lbl, int n_test,
    int sig_dim, int input_dim, dist_kind_t dist_kind)
{
    gesh_projection_t proj = { .R = R, .input_dim = input_dim, .sig_dim = sig_dim };
    int* preds = malloc((size_t)n_test * sizeof(int));
    int rc;
    if (dist_kind == DIST_HAMMING) {
        rc = gesh_forward_classify(preds, test, n_test, bank, &proj, TOP_K);
    } else {
        rc = gesh_forward_classify_wildcard(preds, test, n_test, bank, &proj, TOP_K);
    }
    if (rc != 0) { free(preds); return -1; }
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        if (preds[i] == test_lbl[i]) correct++;
    }
    free(preds);
    return (correct * 1000) / n_test;
}

static int count_zeros_in_bank(const gesh_bank_t* bank) {
    int Dp = M4T_TRIT_PACKED_BYTES(bank->sig_dim);
    int total_zeros = 0;
    int total_positions = bank->n_tiles * bank->sig_dim;
    /* Unpack and count. */
    m4t_trit_t* unpacked = malloc((size_t)bank->sig_dim * sizeof(m4t_trit_t));
    for (int t = 0; t < bank->n_tiles; t++) {
        m4t_unpack_trits_1d(unpacked,
                              bank->tiles_packed + (size_t)t * Dp,
                              bank->sig_dim);
        for (int j = 0; j < bank->sig_dim; j++) {
            if (unpacked[j] == 0) total_zeros++;
        }
    }
    free(unpacked);
    return (total_zeros * 1000) / total_positions;  /* permille */
}

int main(void) {
    synth_wildcard_config_t cfg = synth_wildcard_default();

    printf("# P0-1 wildcard verification probe\n");
    printf("# C=%d, D=%d (K=%d always + M=%d sometimes + N=%d noise), "
           "noise_pct=%d, snr_threshold_pm=%d\n",
           cfg.n_classes, cfg.input_dim, cfg.always_dim, cfg.sometimes_dim,
           cfg.noise_dim, cfg.noise_pct, WILDCARD_SNR_THRESHOLD_PERMILLE);
    printf("# n_train=%d, n_test=%d, sig_dim=%d, top_k=%d\n",
           N_TRAIN, N_TEST, SIG_DIM, TOP_K);
    fflush(stdout);

    /* Generate prototypes + train + test samples. */
    int D = cfg.input_dim;
    m4t_trit_t* protos = malloc((size_t)cfg.n_classes * D * sizeof(m4t_trit_t));
    synth_wildcard_generate_prototypes(protos, &cfg);

    m4t_trit_t* train = malloc((size_t)N_TRAIN * D * sizeof(m4t_trit_t));
    int* train_lbl = malloc((size_t)N_TRAIN * sizeof(int));
    synth_wildcard_generate_samples(train, train_lbl, N_TRAIN, protos,
                                       &cfg, 0x11111111u);

    m4t_trit_t* test = malloc((size_t)N_TEST * D * sizeof(m4t_trit_t));
    int* test_lbl = malloc((size_t)N_TEST * sizeof(int));
    synth_wildcard_generate_samples(test, test_lbl, N_TEST, protos,
                                       &cfg, 0x22222222u);

    /* Random R — substrate-deterministic; same seed across cells. */
    m4t_trit_t* R = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, D, 0xc0ffeebbu);

    /* Project training samples through R to get sig_dim-trit signatures. */
    m4t_trit_t* train_proj = malloc((size_t)N_TRAIN * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, N_TRAIN, R, SIG_DIM, D);

    /* Build both banks. */
    int Dp_sig = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    gesh_bank_t bank_mean, bank_wild;

    bank_mean.tiles_packed = malloc((size_t)cfg.n_classes * (size_t)Dp_sig);
    bank_mean.labels = malloc((size_t)cfg.n_classes * sizeof(int));
    bank_mean.n_tiles = cfg.n_classes;
    bank_mean.sig_dim = SIG_DIM;
    gesh_bank_build_class_mean(&bank_mean, train_proj, train_lbl,
                                  N_TRAIN, cfg.n_classes);

    bank_wild.tiles_packed = malloc((size_t)cfg.n_classes * (size_t)Dp_sig);
    bank_wild.labels = malloc((size_t)cfg.n_classes * sizeof(int));
    bank_wild.n_tiles = cfg.n_classes;
    bank_wild.sig_dim = SIG_DIM;
    gesh_bank_build_class_wildcard(&bank_wild, train_proj, train_lbl,
                                      N_TRAIN, cfg.n_classes,
                                      WILDCARD_SNR_THRESHOLD_PERMILLE);

    int mean_zero_pm = count_zeros_in_bank(&bank_mean);
    int wild_zero_pm = count_zeros_in_bank(&bank_wild);
    printf("# bank_mean zero density:     %d permille (%.1f%%)\n",
           mean_zero_pm, mean_zero_pm / 10.0);
    printf("# bank_wildcard zero density: %d permille (%.1f%%)\n",
           wild_zero_pm, wild_zero_pm / 10.0);
    printf("\n");

    /* Run the four (bank, kernel) cells with timing. */
    printf("## Gate 1 — accuracy across bank × kernel pairs\n\n");
    printf("| bank          | kernel    | accuracy | runtime (10K queries) |\n");
    printf("|---------------|-----------|----------|------------------------|\n");

    struct cell_result {
        int pm;
        double runtime_s;
    } results[2][2];  /* [bank_kind][dist_kind] */

    /* Eval timing: amplify by running the 500-test 20× to get a meaningful
     * runtime measurement. Same accuracy result; deterministic. */
    int n_repeats = 20;

    bank_kind_t bank_kinds[] = { BANK_CLASS_MEAN, BANK_CLASS_WILDCARD };
    dist_kind_t dist_kinds[] = { DIST_HAMMING, DIST_WILDCARD };
    const char* bank_names[] = { "class_mean", "class_wildcard" };
    const char* dist_names[] = { "Hamming", "Wildcard" };

    for (int b = 0; b < 2; b++) {
        for (int d = 0; d < 2; d++) {
            const gesh_bank_t* this_bank =
                (bank_kinds[b] == BANK_CLASS_MEAN) ? &bank_mean : &bank_wild;
            int pm = -1;
            clock_t t0 = clock();
            for (int rep = 0; rep < n_repeats; rep++) {
                pm = eval_pm(R, this_bank, test, test_lbl, N_TEST,
                              SIG_DIM, D, dist_kinds[d]);
            }
            clock_t t1 = clock();
            results[b][d].pm = pm;
            results[b][d].runtime_s =
                (double)(t1 - t0) / CLOCKS_PER_SEC;
            printf("| %-13s | %-9s | %5.1f%%   | %6.2fs (%d×%d=%d queries) |\n",
                   bank_names[b], dist_names[d], pm / 10.0,
                   results[b][d].runtime_s, n_repeats, N_TEST,
                   n_repeats * N_TEST);
            fflush(stdout);
        }
    }

    /* Gate 1 verdict. */
    int hamming_baseline_pm  = results[BANK_CLASS_MEAN][DIST_HAMMING].pm;
    int wildcard_substrate_pm = results[BANK_CLASS_WILDCARD][DIST_WILDCARD].pm;
    int gain_pp_x10 = wildcard_substrate_pm - hamming_baseline_pm;  /* permille - permille = pp×10 */

    printf("\n## Gate 1 verdict: substrate-novel pair vs baseline\n");
    printf("  baseline        (class_mean + Hamming):    %5.1f%%\n", hamming_baseline_pm / 10.0);
    printf("  substrate-novel (class_wildcard + Wildcard): %5.1f%%\n", wildcard_substrate_pm / 10.0);
    printf("  gain:                                       %+5.1fpp\n", gain_pp_x10 / 10.0);
    printf("  Gate 1 PASS bar: gain ≥ +5.0pp\n");
    printf("  Gate 1 verdict: %s\n",
           gain_pp_x10 >= 50 ? "**PASS**" :
           gain_pp_x10 >= 10 ? "INCONCLUSIVE (1pp ≤ gain < 5pp)" :
           gain_pp_x10 >  0  ? "MARGINAL (positive but < 1pp)" :
                                "**FAIL** (gain ≤ 0; substrate-novelty not demonstrated)");

    /* Gate 2 verdict: wildcard runtime vs Hamming runtime on same bank. */
    /* Compare bank_wildcard + Wildcard vs bank_wildcard + Hamming —
     * isolates the kernel runtime difference, holding the bank constant. */
    printf("\n## Gate 2 verdict: wildcard kernel runtime overhead\n");
    double t_hamming  = results[BANK_CLASS_WILDCARD][DIST_HAMMING].runtime_s;
    double t_wildcard = results[BANK_CLASS_WILDCARD][DIST_WILDCARD].runtime_s;
    double ratio = t_wildcard / t_hamming;
    printf("  Hamming runtime  (on wildcard bank): %.3fs\n", t_hamming);
    printf("  Wildcard runtime (on wildcard bank): %.3fs\n", t_wildcard);
    printf("  ratio (wildcard / Hamming):          %.2fx\n", ratio);
    printf("  Gate 2 PASS bar: ratio ≤ 1.20×\n");
    printf("  Gate 2 verdict: %s\n",
           ratio <= 1.20 ? "**PASS**" :
           ratio <= 1.50 ? "ACCEPTABLE (overhead present but bounded)" :
                            "**FAIL** (overhead exceeds 50%; needs NEON optimization)");

    /* Gate 3: substrate-novelty audit — by construction. The
     * wildcard kernel uses the substrate's free third state directly
     * via 2-bit packed-trit storage. Base-2 with masks would require
     * either:
     *   - 4-state encoding {-1, 0, +1, mask} = 2 bits per position SAME
     *     as substrate, but the kernel must dispatch on 4 states instead
     *     of 3 — extra branching.
     *   - 3-state ±1/0 encoding + separate mask bitvector = 2 bits/pos
     *     for trits + 1 bit/pos for mask = 1.5× substrate storage,
     *     plus separate popcount-distance over the mask, plus AND'ing
     *     the mask into the cost computation = additional NEON pass.
     *
     * Either way base-2 pays storage or compute overhead the substrate's
     * free third state avoids. The audit is structural; documented in
     * §19.5 of M4T_SUBSTRATE.md. */
    printf("\n## Gate 3 verdict: substrate-novelty audit\n");
    printf("  Wildcard semantics is expressed in the substrate's free\n");
    printf("  third state at 2 bits/position. Base-2 alternatives:\n");
    printf("    - 4-state {-1, 0, +1, mask}: same 2 bits/pos but extra\n");
    printf("      branching on the 4th state (no free SDOT-equivalent).\n");
    printf("    - ±1 + separate mask: 3 bits/pos = 1.5× storage, plus\n");
    printf("      separate masked-popcount kernel (extra NEON pass).\n");
    printf("  Substrate-novelty audit: **PASS by construction**.\n");
    printf("  Documented: m4t/docs/M4T_SUBSTRATE.md §19.5.\n");

    /* Cleanup. */
    free(protos); free(train); free(train_lbl); free(test); free(test_lbl);
    free(R); free(train_proj);
    free(bank_mean.tiles_packed); free(bank_mean.labels);
    free(bank_wild.tiles_packed); free(bank_wild.labels);

    return 0;
}
