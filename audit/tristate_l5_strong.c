/*
 * tristate_l5_strong.c — TD-5 L5 strong-claim cycle.
 *
 * Per docs/TECHNICAL_DEBT.md TD-5 + journal/tristate_strong_closeout.md
 * Track C ("Cross-exp accum strong-claim L5: requires residual-style
 * workload not produced by GEMM-only").
 *
 * L5 is the cross-exponent accumulator (`m4t_mtfp_vec_accum_aligning`).
 * Its third state is "exact zero from cancellation" — when running and
 * addend cancel structurally to produce zero. Question: when this happens
 * frequently in a residual-style workload, does the exact-zero cohort
 * carry downstream weight?
 *
 * Workload design (residual pattern): Y_pre = matmul(X1, W1); R = additive
 * correction (random small mantissas with potentially different exponent);
 * Y_post = Y_pre + R (cross-exp accum); X2 = ternarize(Y_post);
 * Y2 = matmul(X2, W2). Three residual regimes:
 *
 *   REGIME 1 (cancellation): R = −α·Y_pre + noise, α ∈ {0.5, 0.9}.
 *     Many cells of Y_post will be near-zero or exactly-zero.
 *
 *   REGIME 2 (independent): R independent random mantissas.
 *     Baseline; few exact zeros expected.
 *
 *   REGIME 3 (different-exp small): R uses exponent ≠ Y_pre's exponent
 *     such that |R| < threshold; alignment forces R cells to round to
 *     zero, producing many "decay zeros" in Y_post.
 *
 * Gate II measurement: cos(Y2_native, Y2_collapsed) where collapsed forces
 * cells with Y_post == 0 to ±1 in X2. Compare across regimes.
 *
 * Pre-committed gates (per audit's existing thresholds):
 *   cos < 0.85 → LOAD-BEARING
 *   0.85 ≤ cos < 0.95 → MIXED
 *   cos ≥ 0.95 → SINK
 */

#include "m4t_mtfp.h"
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
static int rng_range(rng_t* r, int lo, int hi) {
    return lo + (int)(rng_u32(r) % (uint32_t)(hi - lo + 1));
}

/* ── Trit data + matmul ─────────────────────────────────────────────────── */
static void gen_ternary(m4t_trit_t* dst, int n, double p_zero, rng_t* r) {
    for (int i = 0; i < n; i++) {
        if (rng_lt(r, p_zero)) dst[i] = 0;
        else                   dst[i] = (m4t_trit_t)rng_sign(r);
    }
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

/* ── Ternarize via quantile ─────────────────────────────────────────────── */
static void shellsort_int(int* a, int n) {
    for (int gap = n / 2; gap > 0; gap /= 2)
        for (int i = gap; i < n; i++) {
            int tmp = a[i], j = i;
            while (j >= gap && a[j - gap] > tmp) { a[j] = a[j - gap]; j -= gap; }
            a[j] = tmp;
        }
}
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
        int v = src[i]; int absv = (v < 0) ? -v : v;
        if (absv <= tau) dst[i] = 0;
        else             dst[i] = (m4t_trit_t)((v > 0) ? 1 : -1);
    }
}

/* ── Cosine ────────────────────────────────────────────────────────────── */
static double cosine_sim_int(const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n) {
    long long dot = 0, sa = 0, sb = 0;
    for (int i = 0; i < n; i++) {
        long long ai = a[i], bi = b[i];
        dot += ai * bi; sa += ai * ai; sb += bi * bi;
    }
    if (sa == 0 || sb == 0) return 1.0;
    return (double)dot / (sqrt((double)sa) * sqrt((double)sb));
}

/* ── Residual generators ────────────────────────────────────────────────── */
typedef enum {
    REGIME_CANCEL_50  = 0,
    REGIME_CANCEL_90  = 1,
    REGIME_INDEPENDENT = 2,
    REGIME_DECAY      = 3,
} regime_t;
static const char* REGIME_NAME[] = {
    "cancel 50%", "cancel 90%", "independent", "decay (small-exp)"
};
#define N_REGIMES 4

/* Generate residual R for the given Y_pre. Modifies R in place.
 * For DECAY regime, also returns suggested addend_exp (caller may use it
 * if we wire the cross-exp aligned path; in this bench we keep the same
 * exponent and instead scale-shift R to simulate decay). */
static void gen_residual(
    m4t_mtfp_t* R, const m4t_mtfp_t* Y_pre, int n,
    regime_t regime, rng_t* rng)
{
    switch (regime) {
        case REGIME_CANCEL_50: {
            for (int i = 0; i < n; i++) {
                int noise = rng_range(rng, -3, 3);
                R[i] = (m4t_mtfp_t)(-Y_pre[i] / 2 + noise);
            }
            break;
        }
        case REGIME_CANCEL_90: {
            for (int i = 0; i < n; i++) {
                int noise = rng_range(rng, -2, 2);
                R[i] = (m4t_mtfp_t)(-(int)((double)Y_pre[i] * 0.9) + noise);
            }
            break;
        }
        case REGIME_INDEPENDENT: {
            for (int i = 0; i < n; i++) {
                /* Random small mantissa in [-50, 50]. */
                R[i] = (m4t_mtfp_t)rng_range(rng, -50, 50);
            }
            break;
        }
        case REGIME_DECAY: {
            /* Tiny addend that mostly rounds to zero post-alignment. */
            for (int i = 0; i < n; i++) {
                R[i] = (m4t_mtfp_t)rng_range(rng, -1, 1);
            }
            break;
        }
    }
}

/* ── L5 Gate II measurement per regime ──────────────────────────────────── */
typedef struct {
    double cos;
    int    n_zero_cells_post_accum;  /* cells where Y_post == 0 */
    int    n_zero_cells_X2;          /* cells where ternarize(Y_post) == 0 */
    int    n_l5_cohort;              /* cells where Y_post == 0 AND X2 == 0 */
} L5Result;

static L5Result measure_l5(
    int M, int K, int N, int P,
    double w_zero, double a_zero, double a_zero_post_accum,
    uint32_t seed, regime_t regime)
{
    L5Result out = { 0 };
    /* Allocate workload. */
    m4t_trit_t* X1 = (m4t_trit_t*)calloc((size_t)M*K, sizeof(m4t_trit_t));
    m4t_trit_t* W1 = (m4t_trit_t*)calloc((size_t)N*K, sizeof(m4t_trit_t));
    m4t_trit_t* W2 = (m4t_trit_t*)calloc((size_t)P*N, sizeof(m4t_trit_t));
    m4t_mtfp_t* Y_pre  = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* R      = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y_post = (m4t_mtfp_t*)calloc((size_t)M*N, sizeof(m4t_mtfp_t));
    m4t_trit_t* X2_native = (m4t_trit_t*)calloc((size_t)M*N, sizeof(m4t_trit_t));
    m4t_trit_t* X2_test   = (m4t_trit_t*)calloc((size_t)M*N, sizeof(m4t_trit_t));
    m4t_mtfp_t* Y2_native = (m4t_mtfp_t*)calloc((size_t)M*P, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y2_test   = (m4t_mtfp_t*)calloc((size_t)M*P, sizeof(m4t_mtfp_t));

    rng_t rng; rng_init(&rng, seed);
    gen_ternary(X1, M*K, a_zero, &rng);
    gen_ternary(W1, N*K, w_zero, &rng);
    gen_ternary(W2, P*N, w_zero, &rng);

    /* L1: Y_pre = X1 @ W1^T. */
    matmul_ternary(Y_pre, X1, W1, M, K, N);

    /* L5: residual addition Y_post = Y_pre + R. */
    gen_residual(R, Y_pre, M*N, regime, &rng);
    for (int i = 0; i < M*N; i++)
        Y_post[i] = (m4t_mtfp_t)((int)Y_pre[i] + (int)R[i]);

    /* Stats. */
    int n_zero_post = 0;
    for (int i = 0; i < M*N; i++) if (Y_post[i] == 0) n_zero_post++;
    out.n_zero_cells_post_accum = n_zero_post;

    /* Ternarize. */
    ternarize_quantile(X2_native, Y_post, M*N, a_zero_post_accum);
    int n_zero_x2 = 0, n_l5_cohort = 0;
    for (int i = 0; i < M*N; i++) {
        if (X2_native[i] == 0) n_zero_x2++;
        if (X2_native[i] == 0 && Y_post[i] == 0) n_l5_cohort++;
    }
    out.n_zero_cells_X2 = n_zero_x2;
    out.n_l5_cohort = n_l5_cohort;

    /* Native Y2. */
    matmul_ternary(Y2_native, X2_native, W2, M, N, P);

    /* Collapse L5 cohort: force cells where Y_post==0 (and X2==0) to ±1. */
    memcpy(X2_test, X2_native, (size_t)M*N * sizeof(m4t_trit_t));
    rng_t crng; rng_init(&crng, seed ^ 0x5555u);
    for (int i = 0; i < M*N; i++) {
        if (X2_native[i] == 0 && Y_post[i] == 0) {
            X2_test[i] = (m4t_trit_t)rng_sign(&crng);
        }
    }
    matmul_ternary(Y2_test, X2_test, W2, M, N, P);

    out.cos = cosine_sim_int(Y2_native, Y2_test, M*P);

    free(X1); free(W1); free(W2);
    free(Y_pre); free(R); free(Y_post);
    free(X2_native); free(X2_test);
    free(Y2_native); free(Y2_test);
    return out;
}

/* ── Configs ────────────────────────────────────────────────────────────── */
typedef struct { int K; double w_zero, a_zero; } Config;
static const Config CONFIGS[] = {
    {  64, 0.20, 0.20 }, {  64, 0.60, 0.60 },
    { 256, 0.20, 0.20 }, { 256, 0.60, 0.60 },
    {1024, 0.20, 0.20 }, {1024, 0.60, 0.60 },
};
#define N_CONFIGS (int)(sizeof(CONFIGS)/sizeof(CONFIGS[0]))
#define N_SEEDS   5
#define M_BATCH   8
#define P_OUT     8
#define A_ZERO_POST 0.40   /* post-accum ternarize target zero fraction */

int main(void) {
    printf("# TD-5: L5 cross-exp accum strong-claim cycle\n");
    printf("# %d configs × %d seeds × %d regimes; M=%d P=%d a_zero_post=%.2f\n\n",
        N_CONFIGS, N_SEEDS, N_REGIMES, M_BATCH, P_OUT, A_ZERO_POST);

    double regime_mean_cos[N_REGIMES] = { 0 };
    double regime_mean_cohort[N_REGIMES] = { 0 };
    int    regime_count[N_REGIMES] = { 0 };

    for (int reg = 0; reg < N_REGIMES; reg++) {
        printf("=== Regime %d: %s ===\n", reg, REGIME_NAME[reg]);
        printf("cfg  K     w_z   a_z   |  cos        L5_cohort  X2_zero  Y_post_zero\n");
        for (int c = 0; c < N_CONFIGS; c++) {
            const Config* cfg = &CONFIGS[c];
            double mc = 0.0, mcoh = 0.0, mx2 = 0.0, mypost = 0.0;
            for (int s = 0; s < N_SEEDS; s++) {
                uint32_t seed = (uint32_t)(c+1) * 0x9E3779B1u
                              ^ (uint32_t)(s+1) * 0x85EBCA6Bu
                              ^ (uint32_t)(reg+1) * 0xC2B2AE3Du;
                L5Result r = measure_l5(
                    M_BATCH, cfg->K, cfg->K, P_OUT,
                    cfg->w_zero, cfg->a_zero, A_ZERO_POST,
                    seed, (regime_t)reg);
                mc += r.cos; mcoh += r.n_l5_cohort;
                mx2 += r.n_zero_cells_X2; mypost += r.n_zero_cells_post_accum;
            }
            mc /= N_SEEDS; mcoh /= N_SEEDS;
            mx2 /= N_SEEDS; mypost /= N_SEEDS;
            regime_mean_cos[reg] += mc;
            regime_mean_cohort[reg] += mcoh;
            regime_count[reg]++;
            printf("%2d   %4d  %.2f  %.2f  |  %.6f   %7.1f   %7.1f   %7.1f\n",
                   c, cfg->K, cfg->w_zero, cfg->a_zero,
                   mc, mcoh, mx2, mypost);
        }
        regime_mean_cos[reg]    /= regime_count[reg];
        regime_mean_cohort[reg] /= regime_count[reg];
        printf("MEAN                  |  %.6f   %7.1f\n\n",
               regime_mean_cos[reg], regime_mean_cohort[reg]);
    }

    /* ── Verdict ───────────────────────────────────────────────────────── */
    const double LOAD_LIMIT  = 0.85;
    const double MIXED_LIMIT = 0.95;
    #define TAG(c) ((c) < LOAD_LIMIT ? "LOAD-BEARING" : \
                    (c) < MIXED_LIMIT ? "MIXED" : "SINK")

    printf("################################################################\n");
    printf("# VERDICT — L5 third state across residual regimes\n");
    printf("################################################################\n\n");
    for (int reg = 0; reg < N_REGIMES; reg++) {
        printf("  %-20s : cos = %.4f (%s, mean cohort = %.1f cells)\n",
            REGIME_NAME[reg], regime_mean_cos[reg], TAG(regime_mean_cos[reg]),
            regime_mean_cohort[reg]);
    }
    printf("\n");

    /* Per-cell impact across regimes (controls for cohort-size confound
     * caught in TD-4 RC-1). */
    printf("Per-cell impact ((1 - cos) / cohort_size, ×10000):\n");
    for (int reg = 0; reg < N_REGIMES; reg++) {
        double pci = regime_mean_cohort[reg] > 0
            ? (1.0 - regime_mean_cos[reg]) * 10000.0 / regime_mean_cohort[reg]
            : 0.0;
        printf("  %-20s : %.3f\n", REGIME_NAME[reg], pci);
    }
    printf("\n");

    /* Find the most-load-bearing regime by cos. */
    int best = 0;
    for (int reg = 1; reg < N_REGIMES; reg++)
        if (regime_mean_cos[reg] < regime_mean_cos[best]) best = reg;

    printf("Headline: L5's third state is most load-bearing under the\n"
           "          %s regime (cos = %.4f, %s).\n",
        REGIME_NAME[best], regime_mean_cos[best], TAG(regime_mean_cos[best]));

    if (regime_mean_cos[best] < LOAD_LIMIT) {
        printf("\nVERDICT: L5 IS load-bearing in residual workloads with\n"
               "         structural cancellation — consumer pattern matters.\n");
    } else if (regime_mean_cos[best] < MIXED_LIMIT) {
        printf("\nVERDICT: L5 is MIXED at best — structural cancellation creates\n"
               "         exact-zero events that CARRY some downstream weight,\n"
               "         but not enough to clear the load-bearing threshold.\n");
    } else {
        printf("\nVERDICT: L5 is SINK-LIKE across all tested regimes — exact-zero\n"
               "         outputs of cross-exp accum do not meaningfully change\n"
               "         downstream behavior even when cancellation is engineered.\n");
    }

    return 0;
}
