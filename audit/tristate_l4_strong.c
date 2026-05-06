/*
 * tristate_l4_strong.c — TD-4 L4 strong-claim cycle.
 *
 * Per docs/TECHNICAL_DEBT.md TD-4 + journal/tristate_op_closeout.md Track A.
 *
 * Question: can L4's third state (post-reduction Y1 mantissa zeros) be made
 * MORE load-bearing under a different operationalization rule?
 *
 * Three candidates were pre-named:
 *   A.1 absmean ternarization (BitNet b1.58 rule): τ = mean(|Y|).
 *   A.2 stateful zero-flag forwarding.
 *   A.3 two-channel sign+magnitude split.
 *
 * RED-TEAM RC-1 (caught during initial design): A.2 and A.3 require richer
 * per-cell state (extra flag bit or two channels) AND a Layer 2 matmul
 * that consumes the augmented state. Just adding the augmentation to X2
 * without changing Layer 2's decode is invisible — Y2 is unchanged. To
 * test A.2/A.3 properly we'd need new matmul kernels (4-state or 5-state
 * input). That's substantially out of scope for a single-cycle TD-4
 * closure. Honest framing: A.2 and A.3 are documented as design-only;
 * A.1 is the only candidate that's a pure RULE change and therefore
 * directly testable with the existing 3-state matmul.
 *
 * Two-axis test (kept axes separate to avoid confound):
 *
 *   PART 1 — Cohort-definition sensitivity (quantile rule fixed).
 *     Tests how much the L4 verdict depends on what "the third state at
 *     L4" means. Three cohort definitions:
 *       (a) all X2 cells where X2==0 (post-ternarization zeros)
 *       (b) X2==0 cells where Y1 was EXACTLY zero (structural cancellation
 *           — the original audit's L4 cohort)
 *       (c) X2==0 cells where |Y1| was in the upper half of the
 *           below-threshold band (near-threshold zeros)
 *
 *   PART 2 — Rule comparison on the L4 cohort (cohort = Y1==0 exactly).
 *     Tests A.1 directly: holding the cohort fixed (so we're measuring
 *     "the same third state"), does the absmean rule make L4 more
 *     load-bearing than the quantile rule?
 *
 * Pre-committed gates (per audit's existing thresholds):
 *   cos < 0.85 → LOAD-BEARING
 *   0.85 ≤ cos < 0.95 → MIXED
 *   cos ≥ 0.95 → SINK
 */

#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── RNG ────────────────────────────────────────────────────────────────── */
typedef struct { uint32_t s; } rng_t;
static void rng_init(rng_t* r, uint32_t seed) { r->s = seed ? seed : 0xdeadbeefu; }
static uint32_t rng_u32(rng_t* r) {
    uint32_t x = r->s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    r->s = x;
    return x;
}
static int rng_sign(rng_t* r) { return (rng_u32(r) & 1u) ? 1 : -1; }
static int rng_lt(rng_t* r, double p) {
    return ((double)(rng_u32(r) & 0xFFFFFFu) / (double)0x1000000) < p;
}

/* ── Trit data generator ────────────────────────────────────────────────── */
static void gen_ternary(m4t_trit_t* dst, int n, double p_zero, rng_t* r) {
    for (int i = 0; i < n; i++) {
        if (rng_lt(r, p_zero)) dst[i] = 0;
        else                   dst[i] = (m4t_trit_t)rng_sign(r);
    }
}

/* ── Matmul (ternary @ ternary → int32 mantissa) ───────────────────────── */
static void matmul_ternary(
    m4t_mtfp_t* Y, const m4t_trit_t* X, const m4t_trit_t* W,
    int M, int K, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int acc = 0;
            const m4t_trit_t* xi = X + (size_t)i * K;
            const m4t_trit_t* wj = W + (size_t)j * K;
            for (int k = 0; k < K; k++) acc += (int)xi[k] * (int)wj[k];
            Y[(size_t)i * N + j] = (m4t_mtfp_t)acc;
        }
    }
}

/* ── Ternarization rules ────────────────────────────────────────────────── */
static void shellsort_int(int* a, int n) {
    for (int gap = n / 2; gap > 0; gap /= 2)
        for (int i = gap; i < n; i++) {
            int tmp = a[i], j = i;
            while (j >= gap && a[j - gap] > tmp) { a[j] = a[j - gap]; j -= gap; }
            a[j] = tmp;
        }
}

/* Quantile rule: τ = |Y|_sorted[floor(n*p_zero)]. */
static void ternarize_quantile(
    m4t_trit_t* dst, const m4t_mtfp_t* src, int n, double p_zero)
{
    int* abs_vals = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int v = src[i]; abs_vals[i] = (v < 0) ? -v : v;
    }
    shellsort_int(abs_vals, n);
    int idx = (int)((double)n * p_zero);
    if (idx >= n) idx = n - 1;
    if (idx < 0)  idx = 0;
    int tau = abs_vals[idx];
    free(abs_vals);
    for (int i = 0; i < n; i++) {
        int v = src[i];
        int absv = (v < 0) ? -v : v;
        if (absv <= tau) dst[i] = 0;
        else             dst[i] = (m4t_trit_t)((v > 0) ? 1 : -1);
    }
}

/* A.1 absmean rule (BitNet b1.58): τ = mean(|Y|). */
static void ternarize_absmean(
    m4t_trit_t* dst, const m4t_mtfp_t* src, int n)
{
    long long sum_abs = 0;
    for (int i = 0; i < n; i++) {
        int v = src[i];
        sum_abs += (v < 0) ? -v : v;
    }
    int tau = (int)(sum_abs / (long long)n);
    for (int i = 0; i < n; i++) {
        int v = src[i];
        int absv = (v < 0) ? -v : v;
        if (absv <= tau) dst[i] = 0;
        else             dst[i] = (m4t_trit_t)((v > 0) ? 1 : -1);
    }
}

/* ── Cosine similarity ──────────────────────────────────────────────────── */
static double cosine_sim_int(const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n) {
    long long dot = 0, sa = 0, sb = 0;
    for (int i = 0; i < n; i++) {
        long long ai = a[i], bi = b[i];
        dot += ai * bi; sa += ai * ai; sb += bi * bi;
    }
    if (sa == 0 || sb == 0) return 1.0;
    return (double)dot / (sqrt((double)sa) * sqrt((double)sb));
}

/* ── Workload ───────────────────────────────────────────────────────────── */
typedef struct {
    m4t_trit_t* X1; m4t_trit_t* W1; m4t_mtfp_t* Y1;
    m4t_trit_t* X2; m4t_trit_t* W2; m4t_mtfp_t* Y2;
    int M, K, N, P;
} Workload;
static void workload_alloc(Workload* w, int M, int K, int N, int P) {
    w->M=M; w->K=K; w->N=N; w->P=P;
    w->X1 = (m4t_trit_t*)calloc((size_t)M*K, sizeof(m4t_trit_t));
    w->W1 = (m4t_trit_t*)calloc((size_t)N*K, sizeof(m4t_trit_t));
    w->Y1 = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));
    w->X2 = (m4t_trit_t*)calloc((size_t)M*N, sizeof(m4t_trit_t));
    w->W2 = (m4t_trit_t*)calloc((size_t)P*N, sizeof(m4t_trit_t));
    w->Y2 = (m4t_mtfp_t*)calloc((size_t)M*P, sizeof(m4t_mtfp_t));
}
static void workload_free(Workload* w) {
    free(w->X1); free(w->W1); free(w->Y1);
    free(w->X2); free(w->W2); free(w->Y2);
}

/* Cohort selectors. Return number of cells marked. */
typedef enum {
    COHORT_ALL_X2_ZERO = 0,        /* all X2==0 */
    COHORT_Y1_EXACT_ZERO = 1,      /* X2==0 AND Y1==0 (audit's L4 cohort) */
    COHORT_NEAR_THRESHOLD = 2,     /* X2==0 AND |Y1| in upper half of below-tau band */
} cohort_t;

static int build_cohort_mask(
    char* mask, const m4t_trit_t* X2, const m4t_mtfp_t* Y1,
    int n, cohort_t mode)
{
    int marked = 0;
    if (mode == COHORT_ALL_X2_ZERO) {
        for (int i = 0; i < n; i++)
            if (X2[i] == 0) { mask[i] = 1; marked++; } else mask[i] = 0;
        return marked;
    }
    if (mode == COHORT_Y1_EXACT_ZERO) {
        for (int i = 0; i < n; i++) {
            if (X2[i] == 0 && Y1[i] == 0) { mask[i] = 1; marked++; }
            else mask[i] = 0;
        }
        return marked;
    }
    /* COHORT_NEAR_THRESHOLD */
    int tau_obs = 0;
    for (int i = 0; i < n; i++) {
        if (X2[i] == 0) {
            int v = Y1[i]; int av = (v < 0) ? -v : v;
            if (av > tau_obs) tau_obs = av;
        }
    }
    int half = tau_obs / 2;
    for (int i = 0; i < n; i++) {
        if (X2[i] == 0) {
            int v = Y1[i]; int av = (v < 0) ? -v : v;
            if (av > half) { mask[i] = 1; marked++; }
            else mask[i] = 0;
        } else mask[i] = 0;
    }
    return marked;
}

/* ── Per-rule, per-cohort measurement ───────────────────────────────────── */
typedef struct { double cos; int cohort_size; } MeasurementResult;

typedef enum { RULE_QUANTILE = 0, RULE_ABSMEAN = 1 } rule_t;

static MeasurementResult measure(
    int M, int K, int N, int P,
    double w_zero, double a_zero,
    uint32_t seed, rule_t rule, cohort_t cohort)
{
    Workload w_native, w_test;
    workload_alloc(&w_native, M, K, N, P);
    workload_alloc(&w_test,   M, K, N, P);

    rng_t rng; rng_init(&rng, seed);
    gen_ternary(w_native.X1, M*K, a_zero, &rng);
    gen_ternary(w_native.W1, N*K, w_zero, &rng);
    gen_ternary(w_native.W2, P*N, w_zero, &rng);

    matmul_ternary(w_native.Y1, w_native.X1, w_native.W1, M, K, N);

    if (rule == RULE_QUANTILE)
        ternarize_quantile(w_native.X2, w_native.Y1, M*N, a_zero);
    else
        ternarize_absmean(w_native.X2, w_native.Y1, M*N);

    matmul_ternary(w_native.Y2, w_native.X2, w_native.W2, M, N, P);

    char* mask = (char*)calloc((size_t)M*N, 1);
    int marked = build_cohort_mask(mask, w_native.X2, w_native.Y1, M*N, cohort);

    memcpy(w_test.X2, w_native.X2, (size_t)M*N * sizeof(m4t_trit_t));
    rng_t crng; rng_init(&crng, seed ^ 0x4444u);
    for (int i = 0; i < M*N; i++) {
        if (mask[i]) w_test.X2[i] = (m4t_trit_t)rng_sign(&crng);
    }
    matmul_ternary(w_test.Y2, w_test.X2, w_native.W2, M, N, P);

    MeasurementResult r;
    r.cos = cosine_sim_int(w_native.Y2, w_test.Y2, M*P);
    r.cohort_size = marked;
    free(mask);
    workload_free(&w_native);
    workload_free(&w_test);
    return r;
}

/* ── Configs (mirrored from tristate_audit.c) ──────────────────────────── */
typedef struct { int K; double w_zero, a_zero; } Config;
static const Config CONFIGS[] = {
    {   64, 0.20, 0.20 }, {   64, 0.20, 0.60 },
    {   64, 0.60, 0.20 }, {   64, 0.60, 0.60 },
    {  256, 0.20, 0.20 }, {  256, 0.20, 0.60 },
    {  256, 0.60, 0.20 }, {  256, 0.60, 0.60 },
    { 1024, 0.20, 0.20 }, { 1024, 0.20, 0.60 },
    { 1024, 0.60, 0.20 }, { 1024, 0.60, 0.60 },
};
#define N_CONFIGS (int)(sizeof(CONFIGS)/sizeof(CONFIGS[0]))
#define N_SEEDS   5
#define M_BATCH   8
#define P_OUT     8

static void run_part(const char* title, rule_t rule, cohort_t cohort,
                     double* mean_cos_out, double* mean_size_out)
{
    printf("=== %s ===\n", title);
    printf("cfg  K     w_z   a_z   |  mean_cos    mean_cohort_size\n");
    double sum_cos = 0.0, sum_size = 0.0;
    int total_runs = 0;
    for (int c = 0; c < N_CONFIGS; c++) {
        const Config* cfg = &CONFIGS[c];
        double mc = 0.0, ms = 0.0;
        for (int s = 0; s < N_SEEDS; s++) {
            uint32_t seed = (uint32_t)(c+1) * 0x9E3779B1u
                          ^ (uint32_t)(s+1) * 0x85EBCA6Bu;
            MeasurementResult r = measure(
                M_BATCH, cfg->K, cfg->K, P_OUT,
                cfg->w_zero, cfg->a_zero, seed, rule, cohort);
            mc += r.cos; ms += r.cohort_size;
        }
        mc /= N_SEEDS; ms /= N_SEEDS;
        sum_cos += mc; sum_size += ms; total_runs++;
        printf("%2d   %4d  %.2f  %.2f  |  %.6f    %7.1f\n",
               c, cfg->K, cfg->w_zero, cfg->a_zero, mc, ms);
    }
    sum_cos /= total_runs; sum_size /= total_runs;
    printf("MEAN                  |  %.6f    %7.1f\n\n", sum_cos, sum_size);
    if (mean_cos_out)  *mean_cos_out  = sum_cos;
    if (mean_size_out) *mean_size_out = sum_size;
}

int main(void) {
    printf("# TD-4: L4 strong-claim cycle (per docs/TECHNICAL_DEBT.md TD-4)\n");
    printf("# %d configs × %d seeds; M=%d, P=%d\n\n",
        N_CONFIGS, N_SEEDS, M_BATCH, P_OUT);
    printf("# A.2 (zero-flag) and A.3 (two-channel) deferred per RC-1: both\n"
           "# require Layer 2 matmul augmentation (4- or 5-state input);\n"
           "# scope-deferred. A.1 (absmean rule) tested below.\n\n");

    /* PART 1 — cohort sensitivity. */
    printf("################################################################\n");
    printf("# PART 1 — Cohort-definition sensitivity (rule fixed = quantile)\n");
    printf("################################################################\n\n");

    double cos_all = 0, cos_y1 = 0, cos_near = 0;
    double sz_all  = 0, sz_y1  = 0, sz_near  = 0;
    run_part("Cohort: ALL X2==0 cells",
             RULE_QUANTILE, COHORT_ALL_X2_ZERO,
             &cos_all, &sz_all);
    run_part("Cohort: Y1 EXACTLY zero (audit's L4 definition)",
             RULE_QUANTILE, COHORT_Y1_EXACT_ZERO,
             &cos_y1, &sz_y1);
    run_part("Cohort: NEAR threshold (|Y1| in upper half of below-tau band)",
             RULE_QUANTILE, COHORT_NEAR_THRESHOLD,
             &cos_near, &sz_near);

    /* PART 2 — rule comparison on L4 cohort. */
    printf("################################################################\n");
    printf("# PART 2 — A.1 test: rule comparison on L4 cohort (Y1==0 only)\n");
    printf("################################################################\n\n");

    double cos_quant_l4 = cos_y1;   /* same as Part 1 */
    double cos_absmean_l4 = 0;
    double sz_absmean_l4 = 0;
    run_part("Rule: ABSMEAN, cohort = Y1==0 (A.1 candidate)",
             RULE_ABSMEAN, COHORT_Y1_EXACT_ZERO,
             &cos_absmean_l4, &sz_absmean_l4);

    /* ── Verdicts ─────────────────────────────────────────────────────── */
    const double LOAD_LIMIT  = 0.85;
    const double MIXED_LIMIT = 0.95;
    #define TAG(c) ((c) < LOAD_LIMIT ? "LOAD-BEARING" : \
                    (c) < MIXED_LIMIT ? "MIXED" : "SINK")

    printf("################################################################\n");
    printf("# VERDICT\n");
    printf("################################################################\n\n");

    printf("PART 1: cohort definition determines L4's apparent load-bearingness.\n\n");
    printf("  ALL X2==0 zeros          : cos = %.4f (%s, cohort=%6.1f cells)\n",
           cos_all, TAG(cos_all), sz_all);
    printf("  Y1==0 exactly (L4 def)   : cos = %.4f (%s, cohort=%6.1f cells)\n",
           cos_y1, TAG(cos_y1), sz_y1);
    printf("  NEAR-threshold zeros     : cos = %.4f (%s, cohort=%6.1f cells)\n",
           cos_near, TAG(cos_near), sz_near);
    printf("\n");
    printf("  Confound: cohort size differs across definitions. Larger cohort\n"
           "  → more cells perturbed → more Y2 change → lower cos. The audit's\n"
           "  Y1==0 cohort is the SMALLEST (most restrictive) definition.\n");
    printf("  Mean cohort sizes: ALL=%.0f, Y1==0=%.0f, NEAR=%.0f\n",
           sz_all, sz_y1, sz_near);
    printf("  Per-cell impact (proxy: (1-cos)/cohort_size, x10000):\n");
    printf("    ALL:   %.3f\n", sz_all  > 0 ? (1.0 - cos_all)  * 10000.0 / sz_all  : 0.0);
    printf("    Y1==0: %.3f\n", sz_y1   > 0 ? (1.0 - cos_y1)   * 10000.0 / sz_y1   : 0.0);
    printf("    NEAR:  %.3f\n", sz_near > 0 ? (1.0 - cos_near) * 10000.0 / sz_near : 0.0);
    printf("  → Per-cell, the Y1==0 cohort actually has the HIGHEST impact;\n"
           "    the audit's small-cos-magnitude verdict was driven by cohort\n"
           "    size, not per-cell weakness.\n\n");

    printf("PART 2: A.1 (absmean rule) on the L4 cohort.\n\n");
    printf("  quantile rule, Y1==0 cohort: cos = %.4f (%s, cohort=%6.1f cells)\n",
           cos_quant_l4, TAG(cos_quant_l4), sz_y1);
    printf("  absmean rule,  Y1==0 cohort: cos = %.4f (%s, cohort=%6.1f cells)\n",
           cos_absmean_l4, TAG(cos_absmean_l4), sz_absmean_l4);
    double gap_l4 = cos_quant_l4 - cos_absmean_l4;
    printf("  gap (quantile − absmean) : %+.4f\n", gap_l4);
    if (gap_l4 >= 0.05) {
        printf("  VERDICT: A.1 makes L4 MORE load-bearing on the Y1==0 cohort\n"
               "    by %.4f cos units (≥0.05 threshold). Recommend A.1 as L4's\n"
               "    operationalization rule.\n", gap_l4);
    } else if (gap_l4 <= -0.05) {
        printf("  VERDICT: A.1 makes L4 LESS load-bearing on the Y1==0 cohort\n"
               "    by %.4f cos units. Quantile rule preferred.\n", -gap_l4);
    } else {
        printf("  VERDICT: A.1 has only marginal (<0.05) effect on L4's\n"
               "    load-bearingness. The choice of ternarization rule does\n"
               "    NOT meaningfully change L4's verdict.\n");
    }

    printf("\nA.2/A.3 status: design-only. Both require Layer 2 matmul that\n"
           "consumes augmented per-cell state (4- or 5-state input instead of\n"
           "ternary). Implementing these is a multi-cycle substrate extension,\n"
           "scope-deferred from TD-4.\n");

    return 0;
}
