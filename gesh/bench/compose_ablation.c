/*
 * compose_ablation.c — P0-4 RED-TEAM REMEDIATION.
 *
 * Tests P0-4 compositional routing on a hierarchical benchmark
 * (synth_compose_hier) — the FAIR benchmark for the design — across
 * multiple stage-1 / stage-2 / mask configurations:
 *
 *   Variants (all classify into N_CLASSES = N_SUPER × K sub-classes):
 *     V0  classmean single-stage       (Hamming)              baseline
 *     V1  wildcard  single-stage       (P0-1)                 baseline
 *     V2  hier: stage-1=class-mean-wildcard-PER-SUB, full mask        (P0-4 default, full mask)
 *     V3  hier: stage-1=class-mean-wildcard-PER-SUB, wildcard mask    (P0-4 as-shipped)
 *     V4  hier: stage-1=class-mean-PER-SUPER (coarse), full mask      (deliberately coarse stage-1)
 *     V5  hier: stage-1=class-mean-WILDCARD-PER-SUPER, full mask      (coarse + wildcard for stage-1)
 *     V6  hier: stage-1=class-mean-WILDCARD-PER-SUPER, wildcard mask  (coarse + wildcard binding)
 *     V7  hier: stage-1=class-mean-WILDCARD-PER-SUPER, COMPLEMENT mask (stage-2 refines on confident dims)
 *
 * V4-V7 directly test the red-team's C2 finding: with deliberately
 * coarse stage-1 (one tile per super-class, multi-class buckets),
 * does compositional routing earn its keep?
 *
 * 10 seeds. Paired t-CI (df=9, t* ≈ 2.262).
 */

#include "synth_compose_hier.h"
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

#define N_TRAIN 4000
#define N_TEST  1000
#define SIG_DIM 64
#define N_SEEDS 10
#define WILDCARD_SNR_PERMILLE 200
#define T_STAR_DF9 2.262   /* 95% two-sided t-critical, df=9 */

typedef enum { MASK_FULL = 0, MASK_WILDCARD = 1, MASK_COMPLEMENT = 2 } mask_mode_t;

/* ── Helpers ───────────────────────────────────────────────────────── */

static int correct_pm(const int* preds, const int* labels, int n) {
    int c = 0; for (int i = 0; i < n; i++) if (preds[i] == labels[i]) c++;
    return (c * 1000) / n;
}

/* Build a stage-1 wildcard bank with per-SUPER tiles (coarse). */
static void build_super_wildcard_bank(
    gesh_bank_t* bank,
    const m4t_trit_t* train_proj, const int* super_lbl,
    int n_train, int n_super)
{
    gesh_bank_build_class_wildcard(bank, train_proj, super_lbl, n_train,
                                      n_super, WILDCARD_SNR_PERMILLE);
}

static void build_super_classmean_bank(
    gesh_bank_t* bank,
    const m4t_trit_t* train_proj, const int* super_lbl,
    int n_train, int n_super)
{
    gesh_bank_build_class_mean(bank, train_proj, super_lbl, n_train, n_super);
}

/* Manual 2-stage forward: stage-1 over coarse super-bank (wildcard or
 * Hamming), stage-2 over per-bucket sub-bank (class-mean over sub-class
 * labels) with caller-provided mask mode. Returns predicted SUB-class
 * label per query. */
static void forward_coarse_compose(
    int* preds, int n_test,
    const m4t_trit_t* test, const int* test_proj_pad,  /* for projection */
    const m4t_trit_t* R, int input_dim,
    const gesh_bank_t* super_bank,
    int super_use_wildcard,
    const gesh_bank_t* sub_banks_per_super,
    const uint8_t* sub_masks_per_super,
    int n_super,
    mask_mode_t mode)
{
    (void)test_proj_pad;
    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);
    uint8_t* full_mask = malloc((size_t)Dp);
    memset(full_mask, 0xFF, (size_t)Dp);
    int tail = SIG_DIM & 3;
    if (tail > 0) full_mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);

    uint8_t* packed_sig = malloc((size_t)Dp);
    uint8_t* compl_mask = malloc((size_t)Dp);

    for (int q = 0; q < n_test; q++) {
        const m4t_trit_t* x = test + (size_t)q * input_dim;
        gesh_project_one_packed(packed_sig, x, R, SIG_DIM, input_dim);

        /* Stage-1: nearest super-tile. */
        int best_g = 0;
        int32_t best_d = INT32_MAX;
        for (int g = 0; g < n_super; g++) {
            int32_t d = super_use_wildcard
                ? m4t_route_wildcard_dist(packed_sig,
                    super_bank->tiles_packed + (size_t)g * Dp,
                    full_mask, SIG_DIM)
                : m4t_popcount_dist(packed_sig,
                    super_bank->tiles_packed + (size_t)g * Dp,
                    full_mask, Dp);
            if (d < best_d) { best_d = d; best_g = g; }
        }

        /* Build stage-2 mask per mode. */
        const uint8_t* m2 = full_mask;
        if (mode == MASK_WILDCARD) {
            m2 = sub_masks_per_super + (size_t)best_g * Dp;
        } else if (mode == MASK_COMPLEMENT) {
            const uint8_t* wc = sub_masks_per_super + (size_t)best_g * Dp;
            for (int i = 0; i < Dp; i++) compl_mask[i] = (uint8_t)(~wc[i] & full_mask[i]);
            m2 = compl_mask;
        }

        /* Empty-mask fallback: use full mask. */
        int has_active = 0;
        for (int i = 0; i < Dp; i++) if (m2[i]) { has_active = 1; break; }
        if (!has_active) m2 = full_mask;

        const gesh_bank_t* sub = &sub_banks_per_super[best_g];
        int best_t = 0; int32_t best_d2 = INT32_MAX;
        for (int t = 0; t < sub->n_tiles; t++) {
            int32_t d = m4t_popcount_dist(packed_sig,
                sub->tiles_packed + (size_t)t * Dp, m2, Dp);
            if (d < best_d2) { best_d2 = d; best_t = t; }
        }
        preds[q] = sub->labels[best_t];
    }

    free(packed_sig); free(full_mask); free(compl_mask);
}

/* ── One seed: run all variants ────────────────────────────────────── */

typedef struct { int v[8]; } seed_pm_t;

static void run_one_seed(seed_pm_t* out, uint32_t seed_base) {
    synth_compose_hier_config_t cfg = synth_compose_hier_default();
    cfg.seed = seed_base;
    int D = synth_compose_hier_input_dim(&cfg);
    int C = synth_compose_hier_n_classes(&cfg);
    int G = cfg.n_super;
    int Dp = M4T_TRIT_PACKED_BYTES(SIG_DIM);

    m4t_trit_t* protos = malloc((size_t)C * D * sizeof(m4t_trit_t));
    synth_compose_hier_generate_prototypes(protos, &cfg);

    m4t_trit_t* train = malloc((size_t)N_TRAIN * D * sizeof(m4t_trit_t));
    int* y_sub        = malloc((size_t)N_TRAIN * sizeof(int));
    int* y_sup        = malloc((size_t)N_TRAIN * sizeof(int));
    synth_compose_hier_generate_samples(train, y_sub, y_sup, N_TRAIN, protos, &cfg, seed_base ^ 0xA1u);

    m4t_trit_t* test = malloc((size_t)N_TEST * D * sizeof(m4t_trit_t));
    int* yt_sub      = malloc((size_t)N_TEST * sizeof(int));
    int* yt_sup      = malloc((size_t)N_TEST * sizeof(int));
    synth_compose_hier_generate_samples(test, yt_sub, yt_sup, N_TEST, protos, &cfg, seed_base ^ 0xB2u);

    m4t_trit_t* R = malloc((size_t)SIG_DIM * D * sizeof(m4t_trit_t));
    gesh_init_random_projection(R, SIG_DIM, D, seed_base ^ 0xC3u);
    gesh_projection_t proj = { .R = R, .input_dim = D, .sig_dim = SIG_DIM };

    /* Pre-project training samples once. */
    m4t_trit_t* train_proj = malloc((size_t)N_TRAIN * SIG_DIM * sizeof(m4t_trit_t));
    gesh_project_batch_unpacked(train_proj, train, N_TRAIN, R, SIG_DIM, D);

    int* preds = malloc((size_t)N_TEST * sizeof(int));

    /* V0 classmean Hamming single-stage. */
    {
        gesh_bank_t b = { .tiles_packed = malloc((size_t)C * Dp),
                            .labels = malloc((size_t)C * sizeof(int)),
                            .n_tiles = C, .sig_dim = SIG_DIM };
        gesh_bank_build_class_mean(&b, train_proj, y_sub, N_TRAIN, C);
        gesh_forward_classify(preds, test, N_TEST, &b, &proj, 1);
        out->v[0] = correct_pm(preds, yt_sub, N_TEST);
        free(b.tiles_packed); free(b.labels);
    }
    /* V1 wildcard single-stage. */
    {
        gesh_bank_t b = { .tiles_packed = malloc((size_t)C * Dp),
                            .labels = malloc((size_t)C * sizeof(int)),
                            .n_tiles = C, .sig_dim = SIG_DIM };
        gesh_bank_build_class_wildcard(&b, train_proj, y_sub, N_TRAIN, C, WILDCARD_SNR_PERMILLE);
        gesh_forward_classify_wildcard(preds, test, N_TEST, &b, &proj, 1);
        out->v[1] = correct_pm(preds, yt_sub, N_TEST);
        free(b.tiles_packed); free(b.labels);
    }
    /* V2/V3 (per-sub-class stage-1 hierarchical). */
    {
        gesh_bank_hier_t hb;
        gesh_bank_hier_alloc(&hb, C, SIG_DIM);
        gesh_bank_build_hierarchical(&hb, train_proj, y_sub, N_TRAIN, C, WILDCARD_SNR_PERMILLE);
        /* V2: full mask (re-implement: bypass sel mask). */
        {
            uint8_t fm[1 << 8]; (void)fm;  /* unused */
            /* Use forward_coarse_compose-like inlined: temporarily replace
             * stage2_masks with full mask via a copy. */
            uint8_t* saved = malloc((size_t)C * Dp);
            memcpy(saved, hb.stage2_masks, (size_t)C * Dp);
            uint8_t* full_mask = malloc((size_t)Dp);
            memset(full_mask, 0xFF, (size_t)Dp);
            int tail = SIG_DIM & 3;
            if (tail > 0) full_mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
            for (int g = 0; g < C; g++)
                memcpy(hb.stage2_masks + (size_t)g * Dp, full_mask, (size_t)Dp);
            gesh_forward_classify_hierarchical(preds, NULL, test, N_TEST, &hb, &proj);
            out->v[2] = correct_pm(preds, yt_sub, N_TEST);
            memcpy(hb.stage2_masks, saved, (size_t)C * Dp);
            free(saved); free(full_mask);
        }
        /* V3: P0-4 as-shipped (wildcard mask). */
        gesh_forward_classify_hierarchical(preds, NULL, test, N_TEST, &hb, &proj);
        out->v[3] = correct_pm(preds, yt_sub, N_TEST);
        gesh_bank_hier_free(&hb);
    }
    /* V4-V7: coarse stage-1 (per-super), stage-2 sub-bank built from sub-class labels of bucket members. */
    {
        gesh_bank_t super_cm = { .tiles_packed = malloc((size_t)G * Dp),
                                    .labels = malloc((size_t)G * sizeof(int)),
                                    .n_tiles = G, .sig_dim = SIG_DIM };
        gesh_bank_t super_wc = { .tiles_packed = malloc((size_t)G * Dp),
                                    .labels = malloc((size_t)G * sizeof(int)),
                                    .n_tiles = G, .sig_dim = SIG_DIM };
        build_super_classmean_bank(&super_cm, train_proj, y_sup, N_TRAIN, G);
        build_super_wildcard_bank (&super_wc, train_proj, y_sup, N_TRAIN, G);

        /* Precompute wildcard-select masks for the wildcard super-bank. */
        uint8_t* super_masks = calloc((size_t)G, (size_t)Dp);
        for (int g = 0; g < G; g++)
            m4t_route_wildcard_select_mask(super_masks + (size_t)g * Dp,
                super_wc.tiles_packed + (size_t)g * Dp, SIG_DIM);

        /* Build per-super sub-banks (over sub-class labels of bucket members).
         * Stage-1 routing: assign each sample to its nearest super-tile (by
         * wildcard-dist for wildcard variants, hamming for V4). We build
         * TWO assignment sets: wc_assign and cm_assign. */
        int* wc_assign = malloc((size_t)N_TRAIN * sizeof(int));
        int* cm_assign = malloc((size_t)N_TRAIN * sizeof(int));
        uint8_t* full_mask = malloc((size_t)Dp);
        memset(full_mask, 0xFF, (size_t)Dp);
        int tail = SIG_DIM & 3;
        if (tail > 0) full_mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
        uint8_t* sample_packed = malloc((size_t)Dp);
        m4t_trit_t* sample_buf = malloc((size_t)SIG_DIM * sizeof(m4t_trit_t));

        for (int i = 0; i < N_TRAIN; i++) {
            for (int j = 0; j < SIG_DIM; j++) sample_buf[j] = train_proj[(size_t)i * SIG_DIM + j];
            m4t_pack_trits_1d(sample_packed, sample_buf, SIG_DIM);
            int bg_w = 0, bg_c = 0; int32_t bd_w = INT32_MAX, bd_c = INT32_MAX;
            for (int g = 0; g < G; g++) {
                int32_t dw = m4t_route_wildcard_dist(sample_packed,
                    super_wc.tiles_packed + (size_t)g * Dp, full_mask, SIG_DIM);
                int32_t dc = m4t_popcount_dist(sample_packed,
                    super_cm.tiles_packed + (size_t)g * Dp, full_mask, Dp);
                if (dw < bd_w) { bd_w = dw; bg_w = g; }
                if (dc < bd_c) { bd_c = dc; bg_c = g; }
            }
            wc_assign[i] = bg_w;
            cm_assign[i] = bg_c;
        }

        /* For each variant V4..V7 we need sub_banks_per_super[G] containing
         * one tile per SUB-CLASS, label = sub-class id. */
        gesh_bank_t* sub_cm = calloc((size_t)G, sizeof(gesh_bank_t));
        gesh_bank_t* sub_wc = calloc((size_t)G, sizeof(gesh_bank_t));
        for (int g = 0; g < G; g++) {
            sub_cm[g].n_tiles = C; sub_cm[g].sig_dim = SIG_DIM;
            sub_cm[g].tiles_packed = calloc((size_t)C, (size_t)Dp);
            sub_cm[g].labels = calloc((size_t)C, sizeof(int));
            for (int k = 0; k < C; k++) sub_cm[g].labels[k] = k;
            sub_wc[g] = sub_cm[g];
            sub_wc[g].tiles_packed = calloc((size_t)C, (size_t)Dp);
            sub_wc[g].labels = calloc((size_t)C, sizeof(int));
            for (int k = 0; k < C; k++) sub_wc[g].labels[k] = k;
        }

        m4t_trit_t* bucket_samples = NULL; int bcap = 0;
        int* bucket_labels = malloc((size_t)N_TRAIN * sizeof(int));
        for (int g = 0; g < G; g++) {
            int n_in_cm = 0, n_in_wc = 0;
            for (int i = 0; i < N_TRAIN; i++) {
                if (cm_assign[i] == g) n_in_cm++;
                if (wc_assign[i] == g) n_in_wc++;
            }
            int n_max = n_in_cm > n_in_wc ? n_in_cm : n_in_wc;
            if (n_max > bcap) {
                free(bucket_samples);
                bucket_samples = malloc((size_t)n_max * (size_t)SIG_DIM * sizeof(m4t_trit_t));
                bcap = n_max;
            }
            /* sub_cm[g]: built from cm_assign bucket. */
            int idx = 0;
            for (int i = 0; i < N_TRAIN; i++) if (cm_assign[i] == g) {
                memcpy(bucket_samples + (size_t)idx * SIG_DIM,
                       train_proj + (size_t)i * SIG_DIM,
                       (size_t)SIG_DIM * sizeof(m4t_trit_t));
                bucket_labels[idx] = y_sub[i]; idx++;
            }
            if (idx > 0) gesh_bank_build_class_mean(&sub_cm[g], bucket_samples, bucket_labels, idx, C);
            /* sub_wc[g]: built from wc_assign bucket. */
            idx = 0;
            for (int i = 0; i < N_TRAIN; i++) if (wc_assign[i] == g) {
                memcpy(bucket_samples + (size_t)idx * SIG_DIM,
                       train_proj + (size_t)i * SIG_DIM,
                       (size_t)SIG_DIM * sizeof(m4t_trit_t));
                bucket_labels[idx] = y_sub[i]; idx++;
            }
            if (idx > 0) gesh_bank_build_class_mean(&sub_wc[g], bucket_samples, bucket_labels, idx, C);
        }
        free(bucket_samples); free(bucket_labels);

        /* Run forwards. */
        forward_coarse_compose(preds, N_TEST, test, NULL, R, D,
                                  &super_cm, /*super_use_wildcard=*/0,
                                  sub_cm, super_masks, G, MASK_FULL);
        out->v[4] = correct_pm(preds, yt_sub, N_TEST);

        forward_coarse_compose(preds, N_TEST, test, NULL, R, D,
                                  &super_wc, /*super_use_wildcard=*/1,
                                  sub_wc, super_masks, G, MASK_FULL);
        out->v[5] = correct_pm(preds, yt_sub, N_TEST);

        forward_coarse_compose(preds, N_TEST, test, NULL, R, D,
                                  &super_wc, /*super_use_wildcard=*/1,
                                  sub_wc, super_masks, G, MASK_WILDCARD);
        out->v[6] = correct_pm(preds, yt_sub, N_TEST);

        forward_coarse_compose(preds, N_TEST, test, NULL, R, D,
                                  &super_wc, /*super_use_wildcard=*/1,
                                  sub_wc, super_masks, G, MASK_COMPLEMENT);
        out->v[7] = correct_pm(preds, yt_sub, N_TEST);

        for (int g = 0; g < G; g++) {
            free(sub_cm[g].tiles_packed); free(sub_cm[g].labels);
            free(sub_wc[g].tiles_packed); free(sub_wc[g].labels);
        }
        free(sub_cm); free(sub_wc);
        free(super_cm.tiles_packed); free(super_cm.labels);
        free(super_wc.tiles_packed); free(super_wc.labels);
        free(super_masks);
        free(wc_assign); free(cm_assign);
        free(full_mask); free(sample_packed); free(sample_buf);
    }

    free(preds);
    free(train_proj); free(R);
    free(test); free(yt_sub); free(yt_sup);
    free(train); free(y_sub); free(y_sup);
    free(protos);
}

/* ── Aggregate + report ─────────────────────────────────────────────── */

static void mean_t_ci(const double* x, int n, double* mean, double* ci) {
    double s = 0; for (int i = 0; i < n; i++) s += x[i]; *mean = s / n;
    double v = 0;
    for (int i = 0; i < n; i++) { double d = x[i] - *mean; v += d * d; }
    v /= (n > 1 ? n - 1 : 1);
    double se = sqrt(v / n);
    *ci = T_STAR_DF9 * se;
}

static const char* verdict(double mean, double ci) {
    double lo = mean - ci, hi = mean + ci;
    if (lo > 0 && mean >= 1.0) return "**PASS**";
    if (lo > 0)                 return "WEAK PASS";
    if (hi < 0 && mean <= -1.0) return "**FAIL**";
    if (hi < 0)                 return "WEAK FAIL";
    return "TIE";
}

int main(void) {
    static const char* names[8] = {
        "V0 classmean H single-stage",
        "V1 wildcard  W single-stage",
        "V2 hier per-sub stage1, full mask  ",
        "V3 hier per-sub stage1, WILDCARD m ",
        "V4 hier per-SUPER classmean, full m",
        "V5 hier per-SUPER wildcard,  full m",
        "V6 hier per-SUPER wildcard,  WC m  ",
        "V7 hier per-SUPER wildcard,  COMPL m"
    };

    printf("# P0-4 RED-TEAM REMEDIATION: compositional routing ablation\n");
    printf("# benchmark: synth_compose_hier (n_super=4, k=5 → 20 sub-classes)\n");
    printf("# %d seeds, n_train=%d n_test=%d sig_dim=%d, t* (df=9) = %.3f\n\n",
           N_SEEDS, N_TRAIN, N_TEST, SIG_DIM, T_STAR_DF9);

    seed_pm_t results[N_SEEDS];
    clock_t t0 = clock();
    for (int s = 0; s < N_SEEDS; s++) {
        uint32_t base = 0xDADAFEEDu + (uint32_t)s * 31u;
        run_one_seed(&results[s], base);
        printf("seed %2d:", s);
        for (int v = 0; v < 8; v++) printf("  V%d=%.1f", v, results[s].v[v]/10.0);
        printf("\n");
    }
    double total_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Means. */
    double means[8] = {0};
    for (int v = 0; v < 8; v++) {
        for (int s = 0; s < N_SEEDS; s++) means[v] += results[s].v[v]/10.0;
        means[v] /= N_SEEDS;
    }

    printf("\n## Means across %d seeds\n", N_SEEDS);
    for (int v = 0; v < 8; v++) printf("  %s : %.2f%%\n", names[v], means[v]);

    /* Pairwise vs V1 (the strongest single-stage baseline). */
    printf("\n## Paired t-CI: variant vs V1 (wildcard single-stage)\n");
    for (int v = 2; v < 8; v++) {
        double d[N_SEEDS];
        for (int s = 0; s < N_SEEDS; s++)
            d[s] = (results[s].v[v] - results[s].v[1]) / 10.0;
        double m, ci; mean_t_ci(d, N_SEEDS, &m, &ci);
        printf("  %s vs V1 : %+.2fpp  CI [%+.2f, %+.2f]  → %s\n",
               names[v], m, m - ci, m + ci, verdict(m, ci));
    }

    /* Pairwise vs V0. */
    printf("\n## Paired t-CI: variant vs V0 (classmean single-stage)\n");
    for (int v = 2; v < 8; v++) {
        double d[N_SEEDS];
        for (int s = 0; s < N_SEEDS; s++)
            d[s] = (results[s].v[v] - results[s].v[0]) / 10.0;
        double m, ci; mean_t_ci(d, N_SEEDS, &m, &ci);
        printf("  %s vs V0 : %+.2fpp  CI [%+.2f, %+.2f]  → %s\n",
               names[v], m, m - ci, m + ci, verdict(m, ci));
    }

    printf("\n## Total runtime: %.2fs\n", total_s);
    return 0;
}
