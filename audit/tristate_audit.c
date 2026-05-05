/*
 * audit/tristate_audit.c — third-state utilization audit
 *
 * Per journal/tristate_op_synthesize.md. Measures third-state utilization
 * across substrate layers on a 2-layer ternary GEMM workload that mirrors
 * a 1.58-bit LLM forward pass.
 *
 * Layers measured:
 *   L1: weight third-state distribution
 *   L2: activation third-state distribution (input to layer 1)
 *   L3: per-MAC product third-state distribution (within layer 1)
 *   L4: post-reduction mantissa zero-fraction (Y1)
 *   L5: cross-exp accumulator third-state — DEFERRED (not naturally
 *       exercised by a GEMM-only workload; requires residual-style
 *       follow-on cycle)
 *   L6: post-ternarization third-state (X2 = ternarize(Y1))
 *
 * Two gates per layer:
 *   Gate I  (info-theoretic): Shannon entropy of the third-state
 *           distribution at this layer.
 *   Gate II (algorithmic): cosine similarity between native Y2 and
 *           collapsed Y2, where "collapsed" means this layer's third
 *           state has been forcibly mapped to a random non-zero state.
 *
 * Workload: ternary X1[M,K] @ ternary W1[N,K]^T = Y1[M,N] (MTFP19);
 *           X2 = ternarize(Y1); ternary X2 @ ternary W2[P,N]^T = Y2[M,P].
 *
 * 12 configs × 5 seeds. Realism gate validates achieved zero-fractions
 * are within ±5pp of target.
 *
 * Output: CSV table to stdout for downstream tabulation.
 *
 * Build: links against libm4t. Audit is a measurement tool; uses
 * <math.h> for log2/sqrt — sanctioned outside the substrate.
 */

#include "m4t_types.h"
#include "m4t_ternary_matmul.h"
#include "m4t_mtfp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>

/* ── Deterministic PRNG (xorshift32) ────────────────────────────────────── */

typedef struct { uint32_t state; } rng_t;

static uint32_t rng_next(rng_t* r) {
    uint32_t x = r->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->state = x;
    return x;
}

static void rng_init(rng_t* r, uint32_t seed) {
    r->state = seed ? seed : 0xdeadbeefu;
    /* warm up */
    for (int i = 0; i < 8; i++) (void)rng_next(r);
}

/* Uniform [0, 1) double. */
static double rng_uniform(rng_t* r) {
    return (double)(rng_next(r) >> 8) / (double)(1u << 24);
}

/* Random sign: ±1 with equal probability. */
static int rng_sign(rng_t* r) {
    return (rng_next(r) & 1u) ? 1 : -1;
}

/* ── Trit generation ─────────────────────────────────────────────────────── */

/* Generate n trits with target zero-fraction p_zero and balanced ±1.
 * Each trit independent. */
static void gen_ternary(m4t_trit_t* dst, int n, double p_zero, rng_t* r) {
    for (int i = 0; i < n; i++) {
        double u = rng_uniform(r);
        if (u < p_zero) dst[i] = 0;
        else            dst[i] = (m4t_trit_t)rng_sign(r);
    }
}

/* Replace every 0 in src with a random ±1; pass through ±1 unchanged.
 * Used by Gate II to forcibly collapse the third state. */
static void binary_collapse(m4t_trit_t* dst, const m4t_trit_t* src, int n, rng_t* r) {
    for (int i = 0; i < n; i++) {
        if (src[i] == 0) dst[i] = (m4t_trit_t)rng_sign(r);
        else             dst[i] = src[i];
    }
}

/* ── Ternarization (MTFP19 → ternary) ───────────────────────────────────── */

static int int_cmp_asc(const void* a, const void* b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}

/* Ternarize Y to dst with target zero-fraction. Uses an absolute-value
 * quantile threshold: τ = |Y|_sorted[floor(n*p_zero)], then dst[i] = 0 if
 * |Y[i]| <= τ else sign(Y[i]). Achieves p_zero exactly modulo ties. */
static void ternarize_quantile(
    m4t_trit_t* dst,
    const m4t_mtfp_t* src,
    int n,
    double p_zero)
{
    int* abs_vals = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int v = src[i];
        abs_vals[i] = (v < 0) ? -v : v;
    }
    qsort(abs_vals, (size_t)n, sizeof(int), int_cmp_asc);
    int idx = (int)((double)n * p_zero);
    if (idx >= n) idx = n - 1;
    if (idx < 0) idx = 0;
    int tau = abs_vals[idx];
    free(abs_vals);

    for (int i = 0; i < n; i++) {
        int v = src[i];
        int absv = (v < 0) ? -v : v;
        if (absv <= tau) dst[i] = 0;
        else             dst[i] = (m4t_trit_t)((v > 0) ? 1 : -1);
    }
}

/* ── Distribution + entropy measurement ─────────────────────────────────── */

typedef struct {
    double frac_neg;
    double frac_zero;
    double frac_pos;
    double entropy_bits;   /* Shannon entropy in bits, max log2(3) ≈ 1.585 */
} TritStats;

static TritStats measure_trit_stats(const m4t_trit_t* arr, int n) {
    int neg = 0, zer = 0, pos = 0;
    for (int i = 0; i < n; i++) {
        if      (arr[i] < 0) neg++;
        else if (arr[i] > 0) pos++;
        else                  zer++;
    }
    TritStats s;
    s.frac_neg  = (double)neg / (double)n;
    s.frac_zero = (double)zer / (double)n;
    s.frac_pos  = (double)pos / (double)n;
    s.entropy_bits = 0.0;
    double ps[3] = { s.frac_neg, s.frac_zero, s.frac_pos };
    for (int j = 0; j < 3; j++) {
        if (ps[j] > 0.0) s.entropy_bits -= ps[j] * log2(ps[j]);
    }
    return s;
}

/* L4-style measurement: distribution of int32 mantissas, treating "exactly
 * zero" as the third state and bucketing non-zeros by sign. Reports the
 * three-way distribution with the same TritStats shape. */
static TritStats measure_int_stats_zerocenter(const m4t_mtfp_t* arr, int n) {
    int neg = 0, zer = 0, pos = 0;
    for (int i = 0; i < n; i++) {
        if      (arr[i] <  0) neg++;
        else if (arr[i] >  0) pos++;
        else                   zer++;
    }
    TritStats s;
    s.frac_neg  = (double)neg / (double)n;
    s.frac_zero = (double)zer / (double)n;
    s.frac_pos  = (double)pos / (double)n;
    s.entropy_bits = 0.0;
    double ps[3] = { s.frac_neg, s.frac_zero, s.frac_pos };
    for (int j = 0; j < 3; j++) {
        if (ps[j] > 0.0) s.entropy_bits -= ps[j] * log2(ps[j]);
    }
    return s;
}

/* ── Cosine similarity ──────────────────────────────────────────────────── */

static double cosine_sim_int(const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        double ai = (double)a[i], bi = (double)b[i];
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    if (na == 0.0 || nb == 0.0) return 1.0;  /* degenerate; treat as identical */
    return dot / (sqrt(na) * sqrt(nb));
}

/* ── Custom matmul that lets us inject Gate II's L3 collapse ────────────── */

/* Y[M,N] = X[M,K] @ W^T[N,K], all ternary, no flags, MTFP19 output.
 *
 * If l3_collapse_rng != NULL: when the per-MAC product X[i,k]*W[j,k] is
 * zero, replace the contribution with a random ±1. This is the Gate II
 * collapse for L3. */
static void matmul_ternary_with_optional_l3_collapse(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_trit_t* W,   /* shape [N, K] */
    int M, int K, int N,
    rng_t* l3_collapse_rng)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int64_t acc = 0;
            const m4t_trit_t* xi = X + (size_t)i * K;
            const m4t_trit_t* wj = W + (size_t)j * K;
            for (int k = 0; k < K; k++) {
                int prod = (int)xi[k] * (int)wj[k];
                if (prod == 0 && l3_collapse_rng) {
                    prod = rng_sign(l3_collapse_rng);
                }
                acc += (int64_t)prod;
            }
            Y[(size_t)i * N + j] = m4t_mtfp_clamp64(acc);
        }
    }
}

/* L4 collapse: replace each Y[i] == 0 with a random ±median-magnitude value.
 * The magnitude substituted is the median of |Y[Y!=0]|. */
static void l4_collapse(m4t_mtfp_t* dst, const m4t_mtfp_t* src, int n, rng_t* r) {
    /* Compute median of non-zero |Y| */
    int* nonzero_abs = (int*)malloc((size_t)n * sizeof(int));
    int nz = 0;
    for (int i = 0; i < n; i++) {
        if (src[i] != 0) {
            int v = src[i];
            nonzero_abs[nz++] = (v < 0) ? -v : v;
        }
    }
    int sub = 1;
    if (nz > 0) {
        qsort(nonzero_abs, (size_t)nz, sizeof(int), int_cmp_asc);
        sub = nonzero_abs[nz / 2];
        if (sub == 0) sub = 1;
    }
    free(nonzero_abs);
    for (int i = 0; i < n; i++) {
        if (src[i] == 0) dst[i] = (m4t_mtfp_t)(rng_sign(r) * sub);
        else             dst[i] = src[i];
    }
}

/* ── 2-layer forward pass ───────────────────────────────────────────────── */

typedef struct {
    /* Layer 1 inputs */
    m4t_trit_t* X1;          /* [M, K] */
    m4t_trit_t* W1;          /* [N, K] */
    /* Layer 1 output / Layer 2 input */
    m4t_mtfp_t* Y1;          /* [M, N] */
    m4t_trit_t* X2;          /* [M, N] (post-ternarization) */
    /* Layer 2 weights / output */
    m4t_trit_t* W2;          /* [P, N] */
    m4t_mtfp_t* Y2;          /* [M, P] */
    int M, K, N, P;
} Workload;

static void workload_alloc(Workload* w, int M, int K, int N, int P) {
    w->M = M; w->K = K; w->N = N; w->P = P;
    w->X1 = (m4t_trit_t*)calloc((size_t)M * K, sizeof(m4t_trit_t));
    w->W1 = (m4t_trit_t*)calloc((size_t)N * K, sizeof(m4t_trit_t));
    w->Y1 = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    w->X2 = (m4t_trit_t*)calloc((size_t)M * N, sizeof(m4t_trit_t));
    w->W2 = (m4t_trit_t*)calloc((size_t)P * N, sizeof(m4t_trit_t));
    w->Y2 = (m4t_mtfp_t*)calloc((size_t)M * P, sizeof(m4t_mtfp_t));
}

static void workload_free(Workload* w) {
    free(w->X1); free(w->W1); free(w->Y1);
    free(w->X2); free(w->W2); free(w->Y2);
}

static void forward_pass(
    Workload* w,
    double act_zero_frac_layer2,
    rng_t* l3_rng /* nullable for native */)
{
    /* Layer 1: Y1 = X1 @ W1^T */
    matmul_ternary_with_optional_l3_collapse(
        w->Y1, w->X1, w->W1, w->M, w->K, w->N, l3_rng);
    /* Ternarize Y1 → X2 */
    ternarize_quantile(w->X2, w->Y1, w->M * w->N, act_zero_frac_layer2);
    /* Layer 2: Y2 = X2 @ W2^T (no L3 collapse on layer 2) */
    matmul_ternary_with_optional_l3_collapse(
        w->Y2, w->X2, w->W2, w->M, w->N, w->P, NULL);
}

/* ── Configs ────────────────────────────────────────────────────────────── */

typedef struct {
    int K;                      /* hidden size; N = K, P = 8, M = 8 */
    double weight_zero_frac;
    double act_zero_frac;
} Config;

static const Config CONFIGS[] = {
    /* K     w_zero  a_zero */
    {   64,  0.20,   0.20 },
    {   64,  0.20,   0.60 },
    {   64,  0.60,   0.20 },
    {   64,  0.60,   0.60 },
    {  256,  0.20,   0.20 },
    {  256,  0.20,   0.60 },
    {  256,  0.60,   0.20 },
    {  256,  0.60,   0.60 },
    { 1024,  0.20,   0.20 },
    { 1024,  0.20,   0.60 },
    { 1024,  0.60,   0.20 },
    { 1024,  0.60,   0.60 },
};
#define N_CONFIGS (int)(sizeof(CONFIGS)/sizeof(CONFIGS[0]))
#define N_SEEDS   5
#define M_BATCH   8
#define P_OUT     8

/* Realism gate tolerance (±5pp on observed zero-fraction). */
#define REALISM_TOL 0.05

/* ── Per-run measurements ───────────────────────────────────────────────── */

typedef struct {
    /* Gate I per layer */
    TritStats l1, l2, l3, l4_int, l6;
    /* Gate II per layer (cosine similarity to native Y2) */
    double cos_l1, cos_l2, cos_l3, cos_l4, cos_l6;
    /* Realism */
    double observed_w_zero_frac;
    double observed_a_zero_frac;
    int realism_pass;
} RunResult;

static void measure_per_mac_distribution(
    Workload* w, TritStats* out)
{
    long long M = w->M, K = w->K, N = w->N;
    long long total = M * K * N;
    long long neg = 0, zer = 0, pos = 0;
    for (int i = 0; i < w->M; i++) {
        for (int j = 0; j < w->N; j++) {
            const m4t_trit_t* xi = w->X1 + (size_t)i * K;
            const m4t_trit_t* wj = w->W1 + (size_t)j * K;
            for (int k = 0; k < w->K; k++) {
                int prod = (int)xi[k] * (int)wj[k];
                if      (prod < 0) neg++;
                else if (prod > 0) pos++;
                else                zer++;
            }
        }
    }
    out->frac_neg  = (double)neg / (double)total;
    out->frac_zero = (double)zer / (double)total;
    out->frac_pos  = (double)pos / (double)total;
    out->entropy_bits = 0.0;
    double ps[3] = { out->frac_neg, out->frac_zero, out->frac_pos };
    for (int j = 0; j < 3; j++) {
        if (ps[j] > 0.0) out->entropy_bits -= ps[j] * log2(ps[j]);
    }
}

static RunResult run_one(const Config* cfg, uint32_t seed) {
    rng_t rng;
    rng_init(&rng, seed);

    int M = M_BATCH, K = cfg->K, N = cfg->K, P = P_OUT;

    Workload w_native;
    workload_alloc(&w_native, M, K, N, P);
    /* Generate native inputs */
    gen_ternary(w_native.X1, M * K, cfg->act_zero_frac, &rng);
    gen_ternary(w_native.W1, N * K, cfg->weight_zero_frac, &rng);
    gen_ternary(w_native.W2, P * N, cfg->weight_zero_frac, &rng);

    /* Realism gate: observed vs target */
    TritStats w1_stats = measure_trit_stats(w_native.W1, N * K);
    TritStats x1_stats = measure_trit_stats(w_native.X1, M * K);
    double w_obs = w1_stats.frac_zero;
    double x_obs = x1_stats.frac_zero;
    int realism = (fabs(w_obs - cfg->weight_zero_frac) <= REALISM_TOL)
               && (fabs(x_obs - cfg->act_zero_frac) <= REALISM_TOL);

    /* Native forward pass */
    forward_pass(&w_native, cfg->act_zero_frac, NULL);

    /* Gate I measurements */
    RunResult r;
    memset(&r, 0, sizeof(r));
    r.l1 = w1_stats;
    r.l2 = x1_stats;
    measure_per_mac_distribution(&w_native, &r.l3);
    r.l4_int = measure_int_stats_zerocenter(w_native.Y1, M * N);
    r.l6 = measure_trit_stats(w_native.X2, M * N);
    r.observed_w_zero_frac = w_obs;
    r.observed_a_zero_frac = x_obs;
    r.realism_pass = realism;

    /* Gate II per layer: collapse, rerun, compare Y2 */
    Workload w_test;
    workload_alloc(&w_test, M, K, N, P);

    /* L1 collapse: replace zeros in W1 (and W2) with random ±1 */
    binary_collapse(w_test.X1, w_native.X1, M * K, &rng);
    memcpy(w_test.X1, w_native.X1, (size_t)M * K * sizeof(m4t_trit_t));
    binary_collapse(w_test.W1, w_native.W1, N * K, &rng);
    binary_collapse(w_test.W2, w_native.W2, P * N, &rng);
    forward_pass(&w_test, cfg->act_zero_frac, NULL);
    r.cos_l1 = cosine_sim_int(w_native.Y2, w_test.Y2, M * P);

    /* L2 collapse: replace zeros in X1 with random ±1 */
    binary_collapse(w_test.X1, w_native.X1, M * K, &rng);
    memcpy(w_test.W1, w_native.W1, (size_t)N * K * sizeof(m4t_trit_t));
    memcpy(w_test.W2, w_native.W2, (size_t)P * N * sizeof(m4t_trit_t));
    forward_pass(&w_test, cfg->act_zero_frac, NULL);
    r.cos_l2 = cosine_sim_int(w_native.Y2, w_test.Y2, M * P);

    /* L3 collapse: keep all inputs native, but replace zero MAC products
     * with random ±1 inside layer 1. (Layer 2 stays native.) */
    memcpy(w_test.X1, w_native.X1, (size_t)M * K * sizeof(m4t_trit_t));
    memcpy(w_test.W1, w_native.W1, (size_t)N * K * sizeof(m4t_trit_t));
    memcpy(w_test.W2, w_native.W2, (size_t)P * N * sizeof(m4t_trit_t));
    {
        rng_t l3_rng;
        rng_init(&l3_rng, seed ^ 0x33333333u);
        forward_pass(&w_test, cfg->act_zero_frac, &l3_rng);
    }
    r.cos_l3 = cosine_sim_int(w_native.Y2, w_test.Y2, M * P);

    /* L4 collapse: native layer 1, replace Y1 zeros with random ±median |Y1|,
     * then continue with the patched Y1 through ternarize and layer 2. */
    matmul_ternary_with_optional_l3_collapse(
        w_test.Y1, w_native.X1, w_native.W1, M, K, N, NULL);
    {
        m4t_mtfp_t* y1_collapsed = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
        l4_collapse(y1_collapsed, w_test.Y1, M * N, &rng);
        ternarize_quantile(w_test.X2, y1_collapsed, M * N, cfg->act_zero_frac);
        free(y1_collapsed);
    }
    matmul_ternary_with_optional_l3_collapse(
        w_test.Y2, w_test.X2, w_native.W2, M, N, P, NULL);
    r.cos_l4 = cosine_sim_int(w_native.Y2, w_test.Y2, M * P);

    /* L6 collapse: native through ternarize, then collapse zeros in X2,
     * then layer 2 with collapsed X2. */
    matmul_ternary_with_optional_l3_collapse(
        w_test.Y1, w_native.X1, w_native.W1, M, K, N, NULL);
    ternarize_quantile(w_test.X2, w_test.Y1, M * N, cfg->act_zero_frac);
    {
        m4t_trit_t* x2_collapsed = (m4t_trit_t*)calloc((size_t)M * N, sizeof(m4t_trit_t));
        binary_collapse(x2_collapsed, w_test.X2, M * N, &rng);
        matmul_ternary_with_optional_l3_collapse(
            w_test.Y2, x2_collapsed, w_native.W2, M, N, P, NULL);
        free(x2_collapsed);
    }
    r.cos_l6 = cosine_sim_int(w_native.Y2, w_test.Y2, M * P);

    workload_free(&w_test);
    workload_free(&w_native);
    return r;
}

/* ── Aggregation + output ───────────────────────────────────────────────── */

typedef struct {
    double mean, sd;
} Stat;

static Stat reduce(const double* xs, int n) {
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < n; i++) { s += xs[i]; s2 += xs[i]*xs[i]; }
    Stat r;
    r.mean = s / n;
    double var = s2 / n - r.mean * r.mean;
    if (var < 0.0) var = 0.0;
    r.sd = sqrt(var);
    return r;
}

int main(void) {
    /* CSV header */
    printf("config_idx,K,w_zero,a_zero,seed,realism_pass,"
           "L1_H,L2_H,L3_H,L4_H,L6_H,"
           "L4_zero_frac,"
           "cos_L1,cos_L2,cos_L3,cos_L4,cos_L6\n");

    /* Per-config aggregation */
    double l1_H[N_SEEDS], l2_H[N_SEEDS], l3_H[N_SEEDS], l4_H[N_SEEDS], l6_H[N_SEEDS];
    double l4_zero[N_SEEDS];
    double cos1[N_SEEDS], cos2[N_SEEDS], cos3[N_SEEDS], cos4[N_SEEDS], cos6[N_SEEDS];

    int realism_fail_total = 0;
    int total_runs = 0;

    for (int c = 0; c < N_CONFIGS; c++) {
        const Config* cfg = &CONFIGS[c];
        for (int s = 0; s < N_SEEDS; s++) {
            uint32_t seed = (uint32_t)(c * 1000 + s + 1);
            RunResult r = run_one(cfg, seed);
            total_runs++;
            if (!r.realism_pass) realism_fail_total++;
            printf("%d,%d,%.2f,%.2f,%u,%d,"
                   "%.4f,%.4f,%.4f,%.4f,%.4f,"
                   "%.4f,"
                   "%.6f,%.6f,%.6f,%.6f,%.6f\n",
                c, cfg->K, cfg->weight_zero_frac, cfg->act_zero_frac, seed, r.realism_pass,
                r.l1.entropy_bits, r.l2.entropy_bits, r.l3.entropy_bits,
                r.l4_int.entropy_bits, r.l6.entropy_bits,
                r.l4_int.frac_zero,
                r.cos_l1, r.cos_l2, r.cos_l3, r.cos_l4, r.cos_l6);

            l1_H[s] = r.l1.entropy_bits;
            l2_H[s] = r.l2.entropy_bits;
            l3_H[s] = r.l3.entropy_bits;
            l4_H[s] = r.l4_int.entropy_bits;
            l6_H[s] = r.l6.entropy_bits;
            l4_zero[s] = r.l4_int.frac_zero;
            cos1[s] = r.cos_l1; cos2[s] = r.cos_l2; cos3[s] = r.cos_l3;
            cos4[s] = r.cos_l4; cos6[s] = r.cos_l6;
        }
        /* Per-config summary */
        Stat s_l1H = reduce(l1_H, N_SEEDS);
        Stat s_l2H = reduce(l2_H, N_SEEDS);
        Stat s_l3H = reduce(l3_H, N_SEEDS);
        Stat s_l4H = reduce(l4_H, N_SEEDS);
        Stat s_l6H = reduce(l6_H, N_SEEDS);
        Stat s_l4z = reduce(l4_zero, N_SEEDS);
        Stat s_c1  = reduce(cos1, N_SEEDS);
        Stat s_c2  = reduce(cos2, N_SEEDS);
        Stat s_c3  = reduce(cos3, N_SEEDS);
        Stat s_c4  = reduce(cos4, N_SEEDS);
        Stat s_c6  = reduce(cos6, N_SEEDS);
        fprintf(stderr,
            "[summary] cfg %d K=%d w_zero=%.2f a_zero=%.2f | "
            "H: L1=%.3f L2=%.3f L3=%.3f L4=%.3f L6=%.3f | "
            "L4_zero=%.3f | "
            "cos: L1=%.4f L2=%.4f L3=%.4f L4=%.4f L6=%.4f\n",
            c, cfg->K, cfg->weight_zero_frac, cfg->act_zero_frac,
            s_l1H.mean, s_l2H.mean, s_l3H.mean, s_l4H.mean, s_l6H.mean,
            s_l4z.mean,
            s_c1.mean, s_c2.mean, s_c3.mean, s_c4.mean, s_c6.mean);
    }

    fprintf(stderr,
        "[overall] %d runs total; %d realism failures (%.1f%%)\n",
        total_runs, realism_fail_total, 100.0 * realism_fail_total / total_runs);

    return 0;
}
