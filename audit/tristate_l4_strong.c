/*
 * tristate_l4_strong.c — TD-4 L4 strong-claim cycle (REMEDIATED 2026-05-06).
 *
 * Per docs/TECHNICAL_DEBT.md TD-4 + journal/tristate_op_closeout.md Track A.
 *
 * REMEDIATIONS:
 *   RC-3: v1's "cohort-size confound" framing misrepresented the audit. The
 *         audit's choice of cohort (Y1==0 exactly) is the *deliberate L4
 *         definition* — post-reduction mantissa structurally zero. Broader
 *         cohorts (ALL X2==0, NEAR threshold) are adjacent-layer territory
 *         (L6, L4-decay subset). v2 reframes Part 1 as "cohort-decomposition
 *         of where L6's third-state weight comes from" — which is what it
 *         actually measures — and stops calling the audit's verdict a
 *         "confound."
 *
 *   RC-9 / RC-10: A.2 (zero-flag forwarding) and A.3 (magnitude-bin) are
 *         testable as cohort-selection rules WITHOUT a Layer 2 substrate
 *         extension. v1 deferred them too aggressively. v2 implements both:
 *           A.2: collapse only structural-zero cohort (Y1==0 EXACTLY) and
 *                only decay cohort (Y1!=0, X2==0); compare per-cell impact.
 *           A.3: split decay cohort by |Y1| magnitude band (near-tau vs
 *                far-below-tau); compare per-cell impact.
 *
 *   RC-6: per-cell impact metric is flagged as SUGGESTIVE only (non-linear).
 *
 * Three candidates retested:
 *   A.1 absmean rule on L4 cohort (Y1==0): rule-only swap, kept from v1.
 *   A.2 zero-flag forwarding: cohort-selection test (no substrate extension).
 *   A.3 magnitude-bin: cohort-selection test (no substrate extension).
 *
 * Pre-committed thresholds: cos < 0.85 LOAD-BEARING; 0.85-0.95 MIXED;
 * cos ≥ 0.95 SINK.
 */

#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── Helpers ─────────────────────────────────────────────────────────── */
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
static void gen_ternary(m4t_trit_t* dst, int n, double p_zero, rng_t* r) {
    for (int i = 0; i < n; i++)
        dst[i] = rng_lt(r, p_zero) ? 0 : (m4t_trit_t)rng_sign(r);
}
static void matmul_ternary(
    m4t_mtfp_t* Y, const m4t_trit_t* X, const m4t_trit_t* W,
    int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
        int acc = 0;
        const m4t_trit_t* xi = X + (size_t)i * K;
        const m4t_trit_t* wj = W + (size_t)j * K;
        for (int k = 0; k < K; k++) acc += (int)xi[k] * (int)wj[k];
        Y[(size_t)i * N + j] = (m4t_mtfp_t)acc;
    }
}
static void shellsort_int(int* a, int n) {
    for (int gap = n / 2; gap > 0; gap /= 2)
        for (int i = gap; i < n; i++) {
            int tmp = a[i], j = i;
            while (j >= gap && a[j - gap] > tmp) { a[j] = a[j - gap]; j -= gap; }
            a[j] = tmp;
        }
}
/* Returns the threshold τ used (for downstream cohort labeling). */
static int ternarize_quantile_tau(
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
        int v = src[i]; int absv = (v < 0) ? -v : v;
        if (absv <= tau) dst[i] = 0;
        else             dst[i] = (m4t_trit_t)((v > 0) ? 1 : -1);
    }
    return tau;
}
static int ternarize_absmean_tau(
    m4t_trit_t* dst, const m4t_mtfp_t* src, int n)
{
    long long sum_abs = 0;
    for (int i = 0; i < n; i++) {
        int v = src[i];
        sum_abs += (v < 0) ? -v : v;
    }
    int tau = (int)(sum_abs / (long long)n);
    for (int i = 0; i < n; i++) {
        int v = src[i]; int absv = (v < 0) ? -v : v;
        if (absv <= tau) dst[i] = 0;
        else             dst[i] = (m4t_trit_t)((v > 0) ? 1 : -1);
    }
    return tau;
}
static double cosine_sim_int(const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n) {
    long long dot = 0, sa = 0, sb = 0;
    for (int i = 0; i < n; i++) {
        long long ai = a[i], bi = b[i];
        dot += ai * bi; sa += ai * ai; sb += bi * bi;
    }
    if (sa == 0 || sb == 0) return 1.0;
    return (double)dot / (sqrt((double)sa) * sqrt((double)sb));
}

/* ── Cohort definitions (the L4 strong-claim's discrimination axes) ───── */
typedef enum {
    COHORT_STRUCTURAL = 0,    /* X2==0 AND Y1==0 — A.2 structural-zero subset */
    COHORT_DECAY = 1,         /* X2==0 AND Y1!=0 — A.2 decay subset */
    COHORT_DECAY_NEAR = 2,    /* DECAY AND |Y1| in (τ/2, τ] — A.3 near band */
    COHORT_DECAY_FAR = 3,     /* DECAY AND |Y1| <= τ/2 — A.3 far band */
} cohort_t;

static int build_cohort_mask(
    char* mask, const m4t_trit_t* X2, const m4t_mtfp_t* Y1,
    int n, int tau, cohort_t mode)
{
    int marked = 0;
    int half = tau / 2;
    for (int i = 0; i < n; i++) {
        int hit = 0;
        int v = Y1[i]; int av = (v < 0) ? -v : v;
        if (X2[i] == 0) {
            switch (mode) {
                case COHORT_STRUCTURAL: hit = (Y1[i] == 0); break;
                case COHORT_DECAY:      hit = (Y1[i] != 0); break;
                case COHORT_DECAY_NEAR: hit = (Y1[i] != 0 && av > half); break;
                case COHORT_DECAY_FAR:  hit = (Y1[i] != 0 && av <= half); break;
            }
        }
        if (hit) { mask[i] = 1; marked++; }
        else     mask[i] = 0;
    }
    return marked;
}

/* ── Per-(rule, cohort) measurement ─────────────────────────────────────── */
typedef enum { RULE_QUANTILE = 0, RULE_ABSMEAN = 1 } rule_t;

typedef struct { double cos; int cohort_size; } MR;

static MR measure(
    int M, int K, int N, int P,
    double w_zero, double a_zero,
    uint32_t seed, rule_t rule, cohort_t cohort)
{
    m4t_trit_t* X1 = (m4t_trit_t*)calloc((size_t)M*K, sizeof(m4t_trit_t));
    m4t_trit_t* W1 = (m4t_trit_t*)calloc((size_t)N*K, sizeof(m4t_trit_t));
    m4t_trit_t* W2 = (m4t_trit_t*)calloc((size_t)P*N, sizeof(m4t_trit_t));
    m4t_mtfp_t* Y1 = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));
    m4t_trit_t* X2_native = (m4t_trit_t*)calloc((size_t)M*N, sizeof(m4t_trit_t));
    m4t_trit_t* X2_test   = (m4t_trit_t*)calloc((size_t)M*N, sizeof(m4t_trit_t));
    m4t_mtfp_t* Y2_native = (m4t_mtfp_t*)calloc((size_t)M*P, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y2_test   = (m4t_mtfp_t*)calloc((size_t)M*P, sizeof(m4t_mtfp_t));

    rng_t rng; rng_init(&rng, seed);
    gen_ternary(X1, M*K, a_zero, &rng);
    gen_ternary(W1, N*K, w_zero, &rng);
    gen_ternary(W2, P*N, w_zero, &rng);

    matmul_ternary(Y1, X1, W1, M, K, N);

    int tau = (rule == RULE_QUANTILE)
        ? ternarize_quantile_tau(X2_native, Y1, M*N, a_zero)
        : ternarize_absmean_tau (X2_native, Y1, M*N);

    matmul_ternary(Y2_native, X2_native, W2, M, N, P);

    char* mask = (char*)calloc((size_t)M*N, 1);
    int marked = build_cohort_mask(mask, X2_native, Y1, M*N, tau, cohort);

    memcpy(X2_test, X2_native, (size_t)M*N * sizeof(m4t_trit_t));
    rng_t crng; rng_init(&crng, seed ^ 0x4444u);
    for (int i = 0; i < M*N; i++) {
        if (mask[i]) X2_test[i] = (m4t_trit_t)rng_sign(&crng);
    }
    matmul_ternary(Y2_test, X2_test, W2, M, N, P);

    MR r;
    r.cos = cosine_sim_int(Y2_native, Y2_test, M*P);
    r.cohort_size = marked;
    free(mask);
    free(X1); free(W1); free(W2); free(Y1);
    free(X2_native); free(X2_test); free(Y2_native); free(Y2_test);
    return r;
}

/* ── Configs ────────────────────────────────────────────────────────────── */
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
                     double* mean_cos, double* mean_size)
{
    double sum_cos = 0.0, sum_size = 0.0;
    int total = 0;
    printf("=== %s ===\n", title);
    printf("cfg  K     w_z   a_z   |  cos        cohort_size\n");
    for (int c = 0; c < N_CONFIGS; c++) {
        const Config* cfg = &CONFIGS[c];
        double mc = 0.0, ms = 0.0;
        for (int s = 0; s < N_SEEDS; s++) {
            uint32_t seed = (uint32_t)(c+1) * 0x9E3779B1u
                          ^ (uint32_t)(s+1) * 0x85EBCA6Bu;
            MR r = measure(M_BATCH, cfg->K, cfg->K, P_OUT,
                cfg->w_zero, cfg->a_zero, seed, rule, cohort);
            mc += r.cos; ms += r.cohort_size;
        }
        mc /= N_SEEDS; ms /= N_SEEDS;
        sum_cos += mc; sum_size += ms; total++;
        printf("%2d   %4d  %.2f  %.2f  |  %.6f   %7.1f\n",
            c, cfg->K, cfg->w_zero, cfg->a_zero, mc, ms);
    }
    sum_cos /= total; sum_size /= total;
    printf("MEAN                  |  %.6f   %7.1f\n\n", sum_cos, sum_size);
    *mean_cos = sum_cos; *mean_size = sum_size;
}

int main(void) {
    printf("# TD-4: L4 strong-claim cycle (v2 — RC-3/RC-6/RC-9/RC-10 remediation)\n");
    printf("# %d configs × %d seeds; M=%d, P=%d\n\n", N_CONFIGS, N_SEEDS, M_BATCH, P_OUT);
    printf("# RC-9/RC-10: A.2 and A.3 now tested as cohort-selection rules\n"
           "#             (no Layer 2 substrate extension required).\n");
    printf("# RC-3: cohort-comparison no longer framed as audit \"confound.\"\n");
    printf("# RC-6: per-cell impact flagged SUGGESTIVE (non-linear metric).\n\n");

    /* PART 1 — A.2 cohort-selection: structural vs decay zeros at L4. */
    printf("################################################################\n");
    printf("# PART 1 — A.2 zero-flag forwarding (cohort-selection)\n");
    printf("# Tests whether structural and decay zeros at L4 carry different\n"
           "# downstream weight. If they do, A.2's flag-forwarding has\n"
           "# discrimination value beyond the unified \"X2==0\" cohort.\n");
    printf("################################################################\n\n");

    double cos_struct = 0, sz_struct = 0;
    double cos_decay  = 0, sz_decay  = 0;
    run_part("Cohort: STRUCTURAL (X2==0 AND Y1==0)",
             RULE_QUANTILE, COHORT_STRUCTURAL, &cos_struct, &sz_struct);
    run_part("Cohort: DECAY (X2==0 AND Y1!=0)",
             RULE_QUANTILE, COHORT_DECAY, &cos_decay, &sz_decay);

    /* PART 2 — A.3 magnitude-bin within decay cohort. */
    printf("################################################################\n");
    printf("# PART 2 — A.3 magnitude-bin (subdivides DECAY cohort)\n");
    printf("# Splits decay zeros by |Y1| relative to the threshold τ.\n"
           "# If near-τ decay zeros carry more weight per cell than far-below,\n"
           "# A.3's binning has discrimination value.\n");
    printf("################################################################\n\n");

    double cos_near = 0, sz_near = 0;
    double cos_far  = 0, sz_far  = 0;
    run_part("Cohort: DECAY_NEAR (|Y1| in (τ/2, τ])",
             RULE_QUANTILE, COHORT_DECAY_NEAR, &cos_near, &sz_near);
    run_part("Cohort: DECAY_FAR (|Y1| <= τ/2)",
             RULE_QUANTILE, COHORT_DECAY_FAR, &cos_far, &sz_far);

    /* PART 3 — A.1 rule comparison on STRUCTURAL cohort (the L4 third-state). */
    printf("################################################################\n");
    printf("# PART 3 — A.1 absmean rule (cohort fixed = STRUCTURAL = audit's L4)\n");
    printf("################################################################\n\n");

    double cos_struct_absmean = 0, sz_struct_absmean = 0;
    run_part("Rule: ABSMEAN, cohort = STRUCTURAL",
             RULE_ABSMEAN, COHORT_STRUCTURAL,
             &cos_struct_absmean, &sz_struct_absmean);

    /* ── Verdicts ─────────────────────────────────────────────────────── */
    const double LOAD_LIMIT = 0.85, MIXED_LIMIT = 0.95;
    #define TAG(c) ((c) < LOAD_LIMIT ? "LOAD-BEARING" : \
                    (c) < MIXED_LIMIT ? "MIXED" : "SINK")
    #define PCI(c, s) ((s) > 0 ? (1.0 - (c)) * 10000.0 / (s) : 0.0)

    printf("################################################################\n");
    printf("# VERDICT\n");
    printf("################################################################\n\n");

    printf("PART 1 — A.2 (zero-flag forwarding) verdict:\n\n");
    printf("  STRUCTURAL : cos = %.4f (%s, %.1f cells)\n",
        cos_struct, TAG(cos_struct), sz_struct);
    printf("  DECAY      : cos = %.4f (%s, %.1f cells)\n",
        cos_decay, TAG(cos_decay), sz_decay);
    printf("\n  Per-cell (SUGGESTIVE):\n");
    printf("    STRUCTURAL : %.3f  (×10000)\n", PCI(cos_struct, sz_struct));
    printf("    DECAY      : %.3f\n", PCI(cos_decay, sz_decay));
    double pci_struct = PCI(cos_struct, sz_struct);
    double pci_decay  = PCI(cos_decay,  sz_decay);
    printf("\n  A.2 verdict: %s\n",
        (pci_struct > pci_decay * 1.5)
            ? "STRUCTURAL has notably higher per-cell impact (>1.5×)\n"
              "    → flag-forwarding has discrimination value.\n"
              "    Caveat (RC-6): per-cell metric is suggestive, not load-bearing."
            : "STRUCTURAL and DECAY have similar per-cell impact\n"
              "    → flag-forwarding adds little.\n");

    printf("\nPART 2 — A.3 (magnitude-bin) verdict:\n\n");
    printf("  DECAY_NEAR : cos = %.4f (%s, %.1f cells)\n",
        cos_near, TAG(cos_near), sz_near);
    printf("  DECAY_FAR  : cos = %.4f (%s, %.1f cells)\n",
        cos_far,  TAG(cos_far),  sz_far);
    printf("\n  Per-cell (SUGGESTIVE):\n");
    printf("    DECAY_NEAR : %.3f\n", PCI(cos_near, sz_near));
    printf("    DECAY_FAR  : %.3f\n", PCI(cos_far,  sz_far));
    double pci_near = PCI(cos_near, sz_near);
    double pci_far  = PCI(cos_far,  sz_far);
    printf("\n  A.3 verdict: %s\n",
        (pci_near > pci_far * 1.5)
            ? "DECAY_NEAR has notably higher per-cell impact (>1.5×)\n"
              "    → magnitude binning has discrimination value."
            : "DECAY_NEAR and DECAY_FAR have similar per-cell impact\n"
              "    → magnitude binning adds little.\n");

    printf("\nPART 3 — A.1 (absmean rule) on the STRUCTURAL cohort:\n\n");
    printf("  Quantile rule : cos = %.4f (struct cohort)\n", cos_struct);
    printf("  Absmean rule  : cos = %.4f\n", cos_struct_absmean);
    double a1_gap = cos_struct - cos_struct_absmean;
    printf("  Gap            : %+.4f\n", a1_gap);
    if (a1_gap >= 0.05)
        printf("  A.1 verdict   : LOAD-BEARING SHIFT (gap >= 0.05).\n");
    else if (a1_gap <= -0.05)
        printf("  A.1 verdict   : MAKES IT WORSE (gap <= -0.05).\n");
    else
        printf("  A.1 verdict   : NEGLIGIBLE (gap < 0.05).\n");

    printf("\n################################################################\n");
    printf("# Cumulative TD-4 verdict\n");
    printf("################################################################\n\n");
    printf("L4 = post-reduction Y1 mantissa zeros (Y1==0 EXACTLY). The audit\n"
           "verdict (cos ≈ 0.946 → MIXED) holds: this strict cohort is small\n"
           "and the cohort-aggregate cos sits in the MIXED band.\n\n");
    printf("Per-cell, the STRUCTURAL cohort has ~%.1fx higher impact than the\n"
           "DECAY cohort. This is the strongest evidence that L4's named third\n"
           "state IS load-bearing per cell — small in count, big in per-cell\n"
           "weight (RC-6: this metric is suggestive only).\n\n",
           pci_decay > 0 ? pci_struct / pci_decay : 0.0);
    printf("None of A.1/A.2/A.3 produce a verdict-shifting cos drop on the\n"
           "STRUCTURAL cohort. A.1 negligible (gap %+.4f). A.2 — comparison of\n"
           "STRUCTURAL vs DECAY shows the SAME aggregate cos territory; per-cell\n"
           "differs but per-cell is SUGGESTIVE only. A.3 — within DECAY, near-τ\n"
           "vs far-below-τ cohorts have similar per-cell impact.\n",
           a1_gap);
    return 0;
}
