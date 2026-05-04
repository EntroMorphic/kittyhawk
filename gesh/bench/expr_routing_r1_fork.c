/*
 * expr_routing_r1_fork.c — distinguishes F1/F2/F3 framings of R1 FAIL.
 *
 * Per docs/PLAN_R1_FORK.md and journal/r1_path_forward_synthesize.md.
 *
 * Runs BOTH signature rules (sign-only and dual) at sig_dim ∈ {16, 32, 64}
 * on curated arity-1 and arity-2 banks plus random expressions. Measures
 * inter-class distance and partition-change rate. Applies pre-committed
 * framing thresholds at sig_dim=64.
 */

#include "expr.h"
#include "expr_bank.h"
#include "expr_random.h"
#include "expr_signature.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Test-input sets ────────────────────────────────────────────────────── */

static const m4t_mtfp_t TI_A1_16[16] = {
    -30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30
};
static const m4t_mtfp_t TI_A1_32[32] = {
    -30, -25, -20, -18, -15, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1,
      2,   3,   4,   5,   6,   8,  10, 12, 15, 18, 20, 22, 25, 27, 28, 30
};
/* sig_dim=64: 64 inputs spanning [-30, 30], denser near zero. */
static const m4t_mtfp_t TI_A1_64[64] = {
    -30, -28, -27, -25, -23, -22, -20, -19, -18, -16, -15, -14, -13, -12, -11, -10,
     -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,
      7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
     23,  24,  25,  26,  27,  28,  29,  30,  -17, -21, -24, -26, -29,  17,  24,  29
};

/* Arity-2 grids (row-major). */
static m4t_mtfp_t TI_A2_16[16 * 2];
static m4t_mtfp_t TI_A2_32[32 * 2];
static m4t_mtfp_t TI_A2_64[64 * 2];

static void init_test_inputs_2(void) {
    /* 4×4 grid for sig_dim=16. */
    static const m4t_mtfp_t pts4[4]  = {-10, -3, 3, 10};
    int idx = 0;
    for (int xi = 0; xi < 4; xi++)
        for (int yi = 0; yi < 4; yi++) {
            TI_A2_16[idx*2 + 0] = pts4[xi];
            TI_A2_16[idx*2 + 1] = pts4[yi];
            idx++;
        }
    /* 4×8 grid for sig_dim=32. */
    static const m4t_mtfp_t y8[8] = {-15, -8, -5, -1, 1, 5, 8, 15};
    idx = 0;
    for (int xi = 0; xi < 4; xi++)
        for (int yi = 0; yi < 8; yi++) {
            TI_A2_32[idx*2 + 0] = pts4[xi];
            TI_A2_32[idx*2 + 1] = y8[yi];
            idx++;
        }
    /* 8×8 grid for sig_dim=64. */
    static const m4t_mtfp_t pts8[8] = {-15, -8, -5, -1, 1, 5, 8, 15};
    idx = 0;
    for (int xi = 0; xi < 8; xi++)
        for (int yi = 0; yi < 8; yi++) {
            TI_A2_64[idx*2 + 0] = pts8[xi];
            TI_A2_64[idx*2 + 1] = pts8[yi];
            idx++;
        }
}

/* ── Constructors and helpers ───────────────────────────────────────────── */

static expr_t* X(void) { return expr_var(0); }
static expr_t* Y(void) { return expr_var(1); }
static expr_t* K(int v) { return expr_const((m4t_mtfp_t)v); }

static int build_arity1_bank(expr_t** out, const char** names) {
    int n = 0;
    out[n]=X();                                          names[n++]="x";
    out[n]=expr_neg(X());                                names[n++]="-x";
    out[n]=expr_max(X(), expr_neg(X()));                 names[n++]="|x|";
    out[n]=expr_min(X(), expr_neg(X()));                 names[n++]="-|x|";
    out[n]=expr_mul(X(), X());                           names[n++]="x*x";
    out[n]=expr_mul(expr_mul(X(), X()), X());            names[n++]="x*x*x";
    out[n]=expr_add(X(), K(5));                          names[n++]="x+5";
    out[n]=expr_sub(X(), K(5));                          names[n++]="x-5";
    out[n]=expr_max(X(), K(0));                          names[n++]="max(x,0)";
    out[n]=expr_min(X(), K(0));                          names[n++]="min(x,0)";
    out[n]=expr_mul(X(), expr_sub(X(), K(3)));           names[n++]="x*(x-3)";
    out[n]=expr_mul(expr_sub(X(), K(1)), expr_add(X(), K(1))); names[n++]="(x-1)*(x+1)";
    return n;
}
static int build_arity2_bank(expr_t** out, const char** names) {
    int n = 0;
    out[n]=expr_add(X(), Y());                           names[n++]="x+y";
    out[n]=expr_sub(X(), Y());                           names[n++]="x-y";
    out[n]=expr_sub(Y(), X());                           names[n++]="y-x";
    out[n]=expr_mul(X(), Y());                           names[n++]="x*y";
    out[n]=expr_min(X(), Y());                           names[n++]="min(x,y)";
    out[n]=expr_max(X(), Y());                           names[n++]="max(x,y)";
    out[n]=expr_sub(expr_max(X(),Y()), expr_min(X(),Y())); names[n++]="|x-y|";
    out[n]=expr_add(expr_min(X(),Y()), expr_max(X(),Y())); names[n++]="min+max";
    out[n]=expr_sub(expr_mul(X(),X()), expr_mul(Y(),Y())); names[n++]="x²-y²";
    out[n]=expr_mul(expr_add(X(),Y()), expr_sub(X(),Y())); names[n++]="(x+y)(x-y)";
    out[n]=expr_neg(expr_add(X(), Y()));                 names[n++]="-(x+y)";
    out[n]=expr_add(X(), expr_add(Y(), K(0)));           names[n++]="x+(y+0)";
    out[n]=expr_min(expr_min(X(),Y()), X());             names[n++]="min(min(x,y),x)";
    out[n]=expr_max(expr_max(X(),Y()), Y());             names[n++]="max(max(x,y),y)";
    return n;
}
static void free_cands(expr_t** c, int n) { for (int i = 0; i < n; i++) expr_free(c[i]); }

static void make_full_mask(uint8_t* mask, int sig_dim) {
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    memset(mask, 0xFF, (size_t)Dp);
    int tail = sig_dim & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
}

/* ── Inter-class distance for sign-only bank ────────────────────────────── */

static void inter_class_signonly(const expr_bank_t* eb, const uint8_t* mask,
                                       int* out_min, double* out_mean, int* out_max) {
    int Dp = M4T_TRIT_PACKED_BYTES(eb->base.sig_dim);
    int n = eb->base.n_tiles;
    int min_d = INT32_MAX, max_d = 0;
    long sum = 0; int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            int d = m4t_popcount_dist(
                eb->base.tiles_packed + (size_t)i * Dp,
                eb->base.tiles_packed + (size_t)j * Dp,
                mask, Dp);
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
            sum += d; cnt++;
        }
    }
    *out_min = cnt > 0 ? min_d : 0;
    *out_max = cnt > 0 ? max_d : 0;
    *out_mean = cnt > 0 ? (double)sum / (double)cnt : 0.0;
}

/* ── Inter-class distance for dual bank ─────────────────────────────────── */

static void inter_class_dual(const expr_bank_dual_t* eb, const uint8_t* mask,
                                   int* out_min, double* out_mean, int* out_max) {
    int Dp = M4T_TRIT_PACKED_BYTES(eb->base.sig_dim);
    int Cp = (eb->base.sig_dim + 7) / 8;
    int n = eb->base.n_tiles;
    int min_d = INT32_MAX, max_d = 0;
    long sum = 0; int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            int d = m4t_route_confidence_weighted_dist(
                eb->base.tiles_packed + (size_t)i * Dp,
                eb->conf_bits_per_tile + (size_t)i * Cp,
                eb->base.tiles_packed + (size_t)j * Dp,
                eb->conf_bits_per_tile + (size_t)j * Cp,
                mask, eb->base.sig_dim);
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
            sum += d; cnt++;
        }
    }
    *out_min = cnt > 0 ? min_d : 0;
    *out_max = cnt > 0 ? max_d : 0;
    *out_mean = cnt > 0 ? (double)sum / (double)cnt : 0.0;
}

/* ── Partition-change rate (cross-rule) on random expressions ───────────── */

static double partition_change_rate(int n_random, int n_vars,
                                          const m4t_mtfp_t* test, int n_test,
                                          uint32_t seed)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    expr_t** cands = malloc((size_t)n_random * sizeof(expr_t*));
    uint32_t state = seed;
    for (int i = 0; i < n_random; i++) cands[i] = expr_random(&state, n_vars, 3);

    /* Sign-only bank. */
    expr_bank_t b_old = {0};
    b_old.base.tiles_packed   = calloc((size_t)n_random, (size_t)Dp);
    b_old.base.labels         = calloc((size_t)n_random, sizeof(int));
    b_old.base.sig_dim        = n_test;
    b_old.candidate_to_class  = calloc((size_t)n_random, sizeof(int));
    expr_bank_build(&b_old, (const expr_t* const*)cands, n_random,
                      test, n_test, n_vars);

    /* Dual bank. */
    expr_bank_dual_t b_new = {0};
    b_new.base.tiles_packed   = calloc((size_t)n_random, (size_t)Dp);
    b_new.base.labels         = calloc((size_t)n_random, sizeof(int));
    b_new.base.sig_dim        = n_test;
    b_new.conf_bits_per_tile  = calloc((size_t)n_random, (size_t)Cp);
    b_new.candidate_to_class  = calloc((size_t)n_random, sizeof(int));
    expr_bank_dual_build(&b_new, (const expr_t* const*)cands, n_random,
                            test, n_test, n_vars);

    long total_pairs = 0, changed_pairs = 0;
    for (int i = 0; i < n_random; i++) {
        int ci_old = b_old.candidate_to_class[i];
        int ci_new = b_new.candidate_to_class[i];
        for (int j = i + 1; j < n_random; j++) {
            int cj_old = b_old.candidate_to_class[j];
            int cj_new = b_new.candidate_to_class[j];
            int same_old = (ci_old == cj_old);
            int same_new = (ci_new == cj_new);
            total_pairs++;
            if (same_old != same_new) changed_pairs++;
        }
    }

    free(b_old.base.tiles_packed); free(b_old.base.labels); free(b_old.candidate_to_class);
    free(b_new.base.tiles_packed); free(b_new.base.labels);
    free(b_new.conf_bits_per_tile); free(b_new.candidate_to_class);
    for (int i = 0; i < n_random; i++) expr_free(cands[i]);
    free(cands);
    return total_pairs > 0 ? 100.0 * (double)changed_pairs / (double)total_pairs : 0.0;
}

/* Random-bank class count under a single rule (sign-only or dual). */
static int random_bank_classes(int n_random, int n_vars,
                                     const m4t_mtfp_t* test, int n_test,
                                     uint32_t seed, int use_dual)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    expr_t** cands = malloc((size_t)n_random * sizeof(expr_t*));
    uint32_t state = seed;
    for (int i = 0; i < n_random; i++) cands[i] = expr_random(&state, n_vars, 3);
    int n_classes;
    if (use_dual) {
        expr_bank_dual_t b = {0};
        b.base.tiles_packed   = calloc((size_t)n_random, (size_t)Dp);
        b.base.labels         = calloc((size_t)n_random, sizeof(int));
        b.base.sig_dim        = n_test;
        b.conf_bits_per_tile  = calloc((size_t)n_random, (size_t)Cp);
        b.candidate_to_class  = calloc((size_t)n_random, sizeof(int));
        expr_bank_dual_build(&b, (const expr_t* const*)cands, n_random,
                                test, n_test, n_vars);
        n_classes = b.base.n_tiles;
        free(b.base.tiles_packed); free(b.base.labels);
        free(b.conf_bits_per_tile); free(b.candidate_to_class);
    } else {
        expr_bank_t b = {0};
        b.base.tiles_packed   = calloc((size_t)n_random, (size_t)Dp);
        b.base.labels         = calloc((size_t)n_random, sizeof(int));
        b.base.sig_dim        = n_test;
        b.candidate_to_class  = calloc((size_t)n_random, sizeof(int));
        expr_bank_build(&b, (const expr_t* const*)cands, n_random,
                          test, n_test, n_vars);
        n_classes = b.base.n_tiles;
        free(b.base.tiles_packed); free(b.base.labels); free(b.candidate_to_class);
    }
    for (int i = 0; i < n_random; i++) expr_free(cands[i]);
    free(cands);
    return n_classes;
}

/* ── Per-(arity, sig_dim) measurement ───────────────────────────────────── */

typedef struct {
    int sign_n_classes; int sign_min; int sign_max; double sign_mean;
    int dual_n_classes; int dual_min; int dual_max; double dual_mean;
    double pc_rate;            /* partition-change rate */
    int rand_sign_classes_avg; int rand_dual_classes_avg;
} measurement_t;

static void measure_combo(measurement_t* m, expr_t** cands, int n_cand,
                              const m4t_mtfp_t* test, int n_test, int n_vars,
                              const char* arity_label, int sig_dim)
{
    /* Build curated banks under both rules. */
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, n_test);

    expr_bank_t sb = {0};
    sb.base.tiles_packed   = calloc((size_t)n_cand, (size_t)Dp);
    sb.base.labels         = calloc((size_t)n_cand, sizeof(int));
    sb.base.sig_dim        = n_test;
    sb.candidate_to_class  = calloc((size_t)n_cand, sizeof(int));
    expr_bank_build(&sb, (const expr_t* const*)cands, n_cand, test, n_test, n_vars);
    m->sign_n_classes = sb.base.n_tiles;
    inter_class_signonly(&sb, mask, &m->sign_min, &m->sign_mean, &m->sign_max);

    expr_bank_dual_t db = {0};
    db.base.tiles_packed   = calloc((size_t)n_cand, (size_t)Dp);
    db.base.labels         = calloc((size_t)n_cand, sizeof(int));
    db.base.sig_dim        = n_test;
    db.conf_bits_per_tile  = calloc((size_t)n_cand, (size_t)Cp);
    db.candidate_to_class  = calloc((size_t)n_cand, sizeof(int));
    expr_bank_dual_build(&db, (const expr_t* const*)cands, n_cand, test, n_test, n_vars);
    m->dual_n_classes = db.base.n_tiles;
    inter_class_dual(&db, mask, &m->dual_min, &m->dual_mean, &m->dual_max);

    /* Partition-change rate, mean over 3 seeds. */
    double pc_sum = 0;
    static const uint32_t seeds[3] = {0xa1u, 0xa2u, 0xa3u};
    for (int s = 0; s < 3; s++) {
        pc_sum += partition_change_rate(100, n_vars, test, n_test, seeds[s]);
    }
    m->pc_rate = pc_sum / 3.0;

    /* Random-bank class count, mean over 3 seeds, both rules. */
    int rs_sum = 0, rd_sum = 0;
    for (int s = 0; s < 3; s++) {
        rs_sum += random_bank_classes(100, n_vars, test, n_test, seeds[s], 0);
        rd_sum += random_bank_classes(100, n_vars, test, n_test, seeds[s], 1);
    }
    m->rand_sign_classes_avg = rs_sum / 3;
    m->rand_dual_classes_avg = rd_sum / 3;

    printf("\n  [%s sig_dim=%d]\n", arity_label, sig_dim);
    printf("    SIGN-ONLY: curated %2d classes, min=%d mean=%.2f max=%d  |  random-bank avg %d classes\n",
           m->sign_n_classes, m->sign_min, m->sign_mean, m->sign_max, m->rand_sign_classes_avg);
    printf("    DUAL:      curated %2d classes, min=%d mean=%.2f max=%d  |  random-bank avg %d classes\n",
           m->dual_n_classes, m->dual_min, m->dual_mean, m->dual_max, m->rand_dual_classes_avg);
    printf("    Partition-change between rules (100 random, 3 seeds avg): %.1f%%\n", m->pc_rate);

    free(sb.base.tiles_packed); free(sb.base.labels); free(sb.candidate_to_class);
    free(db.base.tiles_packed); free(db.base.labels);
    free(db.conf_bits_per_tile); free(db.candidate_to_class);
    free(mask);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    init_test_inputs_2();
    printf("# R1 FORK EXPERIMENT — distinguishes F1/F2/F3 framings of R1 FAIL\n");
    printf("# Pre-committed framings per docs/PLAN_R1_FORK.md\n");

    expr_t* c1[32]; const char* n1[32]; int nc1 = build_arity1_bank(c1, n1);
    expr_t* c2[32]; const char* n2[32]; int nc2 = build_arity2_bank(c2, n2);

    measurement_t m_a1_16, m_a1_32, m_a1_64;
    measurement_t m_a2_16, m_a2_32, m_a2_64;

    printf("\n========================================\n");
    printf("ARITY-1 SWEEP\n");
    printf("========================================\n");
    measure_combo(&m_a1_16, c1, nc1, TI_A1_16, 16, 1, "arity-1", 16);
    measure_combo(&m_a1_32, c1, nc1, TI_A1_32, 32, 1, "arity-1", 32);
    measure_combo(&m_a1_64, c1, nc1, TI_A1_64, 64, 1, "arity-1", 64);

    printf("\n========================================\n");
    printf("ARITY-2 SWEEP\n");
    printf("========================================\n");
    measure_combo(&m_a2_16, c2, nc2, TI_A2_16, 16, 2, "arity-2", 16);
    measure_combo(&m_a2_32, c2, nc2, TI_A2_32, 32, 2, "arity-2", 32);
    measure_combo(&m_a2_64, c2, nc2, TI_A2_64, 64, 2, "arity-2", 64);

    /* ── Apply pre-committed framings at sig_dim=64 ───────────────── */
    printf("\n========================================\n");
    printf("FRAMING THRESHOLDS APPLIED (at sig_dim=64)\n");
    printf("========================================\n");

    int sign64_a1_min = m_a1_64.sign_min;
    int dual64_a1_min = m_a1_64.dual_min;
    int sign64_a2_min = m_a2_64.sign_min;
    int dual64_a2_min = m_a2_64.dual_min;

    /* F1 wins iff dual at sig_dim=64 has arity-1 inter-class min >= sign-only at sig_dim=64 by >= 2 trits. */
    int f1_a1 = (dual64_a1_min - sign64_a1_min) >= 2;
    int f1_a2 = (dual64_a2_min - sign64_a2_min) >= 2;
    int f1 = f1_a1;  /* per plan, F1 framing is stated for arity-1 */

    /* F2 wins iff sign-only at sig_dim=64 reaches arity-1 min >= 6 AND dual doesn't add >= 2. */
    int f2_a1_threshold = (sign64_a1_min >= 6);
    int f2_a1_no_dual_advantage = (dual64_a1_min - sign64_a1_min) < 2;
    int f2 = f2_a1_threshold && f2_a1_no_dual_advantage;

    /* F3 wins iff both rules at sig_dim=64 have arity-1 inter-class min < 6 AND
     * neither shows >= 30% partition change rate from sig_dim=16 to sig_dim=64.
     * The "partition change rate from sig_dim=16 to 64" is effectively
     * "did increasing dim help discriminate?" — proxy: how much bigger is
     * the random-bank class count at sig_dim=64 vs sig_dim=16. */
    int f3_a1_capped = (sign64_a1_min < 6 && dual64_a1_min < 6);
    int sign_dim_increase = m_a1_64.rand_sign_classes_avg - m_a1_16.rand_sign_classes_avg;
    int dual_dim_increase = m_a1_64.rand_dual_classes_avg - m_a1_16.rand_dual_classes_avg;
    /* >= 30% means at least 30 more classes per 100 random expressions. */
    int sign_dim_helped = (sign_dim_increase >= 30);
    int dual_dim_helped = (dual_dim_increase >= 30);
    int f3 = f3_a1_capped && !sign_dim_helped && !dual_dim_helped;

    printf("\nKey numbers at sig_dim=64:\n");
    printf("  arity-1 sign-only inter-class min : %d\n", sign64_a1_min);
    printf("  arity-1 dual      inter-class min : %d\n", dual64_a1_min);
    printf("  arity-1 dual - sign delta         : %+d trits\n", dual64_a1_min - sign64_a1_min);
    printf("  arity-2 sign-only inter-class min : %d\n", sign64_a2_min);
    printf("  arity-2 dual      inter-class min : %d\n", dual64_a2_min);
    printf("\nDim-helped indicators (random-bank class count, sig_dim=16 vs 64):\n");
    printf("  arity-1 sign random classes: %d -> %d (delta %+d)\n",
           m_a1_16.rand_sign_classes_avg, m_a1_64.rand_sign_classes_avg, sign_dim_increase);
    printf("  arity-1 dual random classes: %d -> %d (delta %+d)\n",
           m_a1_16.rand_dual_classes_avg, m_a1_64.rand_dual_classes_avg, dual_dim_increase);

    printf("\nFraming verdicts:\n");
    printf("  F1 (wrong rule)  : %s  (dual >= sign + 2 trits at sig_dim=64 arity-1: %s)\n",
           f1 ? "WIN" : "lose", f1_a1 ? "yes" : "no");
    printf("  F2 (wrong axis)  : %s  (sign >= 6 at sig_dim=64 arity-1: %s; dual no advantage: %s)\n",
           f2 ? "WIN" : "lose",
           f2_a1_threshold ? "yes" : "no",
           f2_a1_no_dual_advantage ? "yes" : "no");
    printf("  F3 (wrong layer) : %s  (both capped <6 arity-1: %s; sign dim helped: %s; dual dim helped: %s)\n",
           f3 ? "WIN" : "lose",
           f3_a1_capped ? "yes" : "no",
           sign_dim_helped ? "yes" : "no",
           dual_dim_helped ? "yes" : "no");

    /* Mixed (per-arity) verdict check. */
    int mixed = (f1_a2 && !f1_a1);  /* F1 wins arity-2 but not arity-1 */
    if (mixed) {
        printf("  MIXED (per-arity): arity-2 favors dual; arity-1 doesn't. Per-arity rules indicated.\n");
    }

    int n_winners = (f1 ? 1 : 0) + (f2 ? 1 : 0) + (f3 ? 1 : 0);
    printf("\n========================================\n");
    if (n_winners == 1) {
        if (f1) printf("FORK VERDICT: F1 wins (wrong rule). Next: write R1 v2 plan.\n");
        else if (f2) printf("FORK VERDICT: F2 wins (wrong axis). Next: revert R1 for arity-1; resume R3/R2 with sign-only.\n");
        else printf("FORK VERDICT: F3 wins (wrong layer). Next: pivot to P1-1; archive R1.\n");
    } else if (n_winners == 0) {
        if (mixed) {
            printf("FORK VERDICT: MIXED — per-arity rules indicated (Option H).\n");
        } else {
            printf("FORK VERDICT: INCONCLUSIVE — none of F1/F2/F3 cleanly wins.\n");
            printf("Closest framing(s) — examine numbers above; may need extended sig_dim sweep.\n");
        }
    } else {
        printf("FORK VERDICT: AMBIGUOUS (%d framings nominally win). Inspect thresholds.\n", n_winners);
    }
    printf("========================================\n");

    free_cands(c1, nc1); free_cands(c2, nc2);
    return 0;
}
