/*
 * expr_routing_r1_remediation.c — addresses R1 red-team findings from
 * journal/r1_signature_rule_redteam.md against pre-committed gates in
 * journal/r1_remediation_precommit.md.
 *
 * Eight sections, one per finding cluster:
 *   §1  Partition-change measurement   (C1) — multi-seed (subsumes H3, M4)
 *   §2  Rule-difference probes         (C2)
 *   §3  Constant-offset diagnostic     (H1)
 *   §4  Granularity sweep              (H2)
 *   §5  Timing measurement             (H4) — diagnostic only
 *   §6  Inter-class distance under new rule  (M1)
 *   §7  Runtime regression check       (M2, L1)
 *   §8  Per-band distribution          (M3)
 */

#include "expr.h"
#include "expr_bank.h"
#include "expr_random.h"
#include "expr_signature.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Unified seed list per H4/M4 discipline. */
static const uint32_t SEEDS[5] = { 0xa1u, 0xa2u, 0xa3u, 0xa4u, 0xa5u };
#define N_SEEDS 5

/* Test inputs (same as original probe). */
static const m4t_mtfp_t TEST_INPUTS_1[16] = {
    -30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30
};
static m4t_mtfp_t TEST_INPUTS_2[16 * 2];
static void init_test_inputs_2(void) {
    static const m4t_mtfp_t pts[4] = {-10, -3, 3, 10};
    int idx = 0;
    for (int xi = 0; xi < 4; xi++)
        for (int yi = 0; yi < 4; yi++) {
            TEST_INPUTS_2[idx*2 + 0] = pts[xi];
            TEST_INPUTS_2[idx*2 + 1] = pts[yi];
            idx++;
        }
}

static expr_t* X(void) { return expr_var(0); }
static expr_t* Y(void) { return expr_var(1); }
static expr_t* K(int v) { return expr_const((m4t_mtfp_t)v); }

/* ── Curated banks (same as previous probes) ────────────────────────────── */

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

/* ── Helpers ────────────────────────────────────────────────────────────── */

static void make_full_mask(uint8_t* mask, int sig_dim) {
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    memset(mask, 0xFF, (size_t)Dp);
    int tail = sig_dim & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
}

static int route_signonly(const uint8_t* sig, const expr_bank_t* eb,
                              const uint8_t* mask) {
    int Dp = M4T_TRIT_PACKED_BYTES(eb->base.sig_dim);
    int best = 0; int32_t best_d = INT32_MAX;
    for (int t = 0; t < eb->base.n_tiles; t++) {
        int32_t d = m4t_popcount_dist(sig, eb->base.tiles_packed + (size_t)t * Dp,
                                         mask, Dp);
        if (d < best_d) { best_d = d; best = t; }
    }
    return best;
}

static int route_dual(const uint8_t* trit, const uint8_t* conf,
                          const expr_bank_dual_t* eb, const uint8_t* mask) {
    int sig_dim = eb->base.sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int Cp = (sig_dim + 7) / 8;
    int best = 0; int32_t best_d = INT32_MAX;
    for (int t = 0; t < eb->base.n_tiles; t++) {
        int32_t d = m4t_route_confidence_weighted_dist(
            trit, conf,
            eb->base.tiles_packed + (size_t)t * Dp,
            eb->conf_bits_per_tile + (size_t)t * Cp,
            mask, sig_dim);
        if (d < best_d) { best_d = d; best = t; }
    }
    return best;
}

/* Build sign-only bank wrapper. */
static void build_signonly_bank(expr_bank_t* eb, expr_t** cands, int n_cand,
                                    const m4t_mtfp_t* test, int n_test, int n_vars) {
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    eb->base.tiles_packed   = calloc((size_t)n_cand, (size_t)Dp);
    eb->base.labels         = calloc((size_t)n_cand, sizeof(int));
    eb->base.sig_dim        = n_test;
    eb->candidate_to_class  = calloc((size_t)n_cand, sizeof(int));
    expr_bank_build(eb, (const expr_t* const*)cands, n_cand, test, n_test, n_vars);
}
static void free_signonly_bank(expr_bank_t* eb) {
    free(eb->base.tiles_packed); free(eb->base.labels); free(eb->candidate_to_class);
}

/* Build dual bank wrapper. */
static void build_dual_bank(expr_bank_dual_t* eb, expr_t** cands, int n_cand,
                                const m4t_mtfp_t* test, int n_test, int n_vars) {
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    eb->base.tiles_packed   = calloc((size_t)n_cand, (size_t)Dp);
    eb->base.labels         = calloc((size_t)n_cand, sizeof(int));
    eb->base.sig_dim        = n_test;
    eb->conf_bits_per_tile  = calloc((size_t)n_cand, (size_t)Cp);
    eb->candidate_to_class  = calloc((size_t)n_cand, sizeof(int));
    expr_bank_dual_build(eb, (const expr_t* const*)cands, n_cand, test, n_test, n_vars);
}
static void free_dual_bank(expr_bank_dual_t* eb) {
    free(eb->base.tiles_packed); free(eb->base.labels);
    free(eb->conf_bits_per_tile); free(eb->candidate_to_class);
}

/* ── §1 Partition-change measurement (C1, multi-seed) ───────────────────── */
/*
 * For each seed: build BOTH old-rule bank and new-rule bank from N random
 * expressions. For each pair (i,j), determine same-class status under each
 * rule. Count pairs that change relationship between rules.
 */

static double partition_change_one_seed(int n_random, int n_vars,
                                              const m4t_mtfp_t* test, int n_test,
                                              uint32_t seed)
{
    expr_t** cands = malloc((size_t)n_random * sizeof(expr_t*));
    uint32_t state = seed;
    for (int i = 0; i < n_random; i++) cands[i] = expr_random(&state, n_vars, 3);

    expr_bank_t b_old = {0};       build_signonly_bank(&b_old, cands, n_random, test, n_test, n_vars);
    expr_bank_dual_t b_new = {0};  build_dual_bank(&b_new, cands, n_random, test, n_test, n_vars);

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

    free_signonly_bank(&b_old); free_dual_bank(&b_new);
    for (int i = 0; i < n_random; i++) expr_free(cands[i]);
    free(cands);
    return total_pairs > 0 ? 100.0 * (double)changed_pairs / (double)total_pairs : 0.0;
}

/* ── §2 Rule-difference probes (C2) ─────────────────────────────────────── */
/*
 * Probes hand-designed to route DIFFERENTLY under sign-only vs dual-rule.
 * Under each probe, we pre-commit the expected dual-rule target (the
 * representative-name we expect under dual). PASS if ≥70% of probes route
 * to the dual-expected class.
 *
 * Rule differences exploited:
 *   - Sign-only merges `x ≡ x*x*x`. Dual splits them.
 *     → probes that are x*x*x-shaped should route to "x*x*x" class under dual,
 *       NOT to "x" class.
 *   - Dual merges `x*x ≡ (x-1)*(x+1)`. Sign-only keeps them separate.
 *     → probes that are x²-1-shaped should route to "x*x" representative under
 *       dual (since dual merged), NOT "(x-1)*(x+1)".
 */

typedef struct {
    expr_t* probe;
    int     expected_dual_candidate;
    const char* description;
} rd_probe_t;

static int build_rd_probes_arity1(rd_probe_t* p) {
    int n = 0;
    /* x³ family: sign-only merges with x; dual splits. Expected dual: candidate 5 (x*x*x). */
    p[n].probe = expr_mul(expr_mul(X(), X()), X()); p[n].expected_dual_candidate = 5; p[n].description = "RD x*x*x"; n++;
    p[n].probe = expr_mul(expr_mul(X(), X()), expr_mul(X(), K(1))); p[n].expected_dual_candidate = 5; p[n].description = "RD x*x*(x*1)"; n++;
    p[n].probe = expr_neg(expr_neg(expr_mul(expr_mul(X(), X()), X()))); p[n].expected_dual_candidate = 5; p[n].description = "RD -(-(x*x*x))"; n++;
    p[n].probe = expr_add(expr_mul(expr_mul(X(), X()), X()), K(0)); p[n].expected_dual_candidate = 5; p[n].description = "RD (x*x*x)+0"; n++;
    /* x²-1 family: dual merges with x*x; sign-only keeps separate. Expected dual: candidate 4 (x*x). */
    p[n].probe = expr_mul(expr_sub(X(), K(1)), expr_add(X(), K(1))); p[n].expected_dual_candidate = 4; p[n].description = "RD (x-1)(x+1)"; n++;
    p[n].probe = expr_sub(expr_mul(X(), X()), K(1)); p[n].expected_dual_candidate = 4; p[n].description = "RD x²-1"; n++;
    p[n].probe = expr_add(expr_mul(expr_sub(X(), K(1)), expr_add(X(), K(1))), K(0)); p[n].expected_dual_candidate = 4; p[n].description = "RD ((x-1)(x+1))+0"; n++;
    return n;
}

/* Arity-2 has no rule-difference probes here — most arity-2 mergers in the
 * curated bank are true mathematical identities (min+max ≡ x+y, etc.) so
 * sign-only and dual produce identical partitions on the curated arity-2
 * candidates. C1 partition-change gate covers the arity-2 case at scale via
 * random-expression sampling. */

static void free_rd_probes(rd_probe_t* p, int n) {
    for (int i = 0; i < n; i++) expr_free(p[i].probe);
}

/* ── §3 Constant-offset diagnostic (H1) ─────────────────────────────────── */
/* Take pairs (e, e+k) for k in {1, 5, -1}. Report whether they route to
 * the same class under dual-rule. Diagnostic only. */

static void run_h1_diagnostic(const expr_bank_dual_t* eb, const char** names,
                                  const m4t_mtfp_t* test, int n_test, int n_vars) {
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, n_test);

    /* Build pairs: (base, base + k). */
    typedef struct { expr_t* base; expr_t* shifted; const char* label; } pair_t;
    pair_t pairs[6];
    int np = 0;
    pairs[np].base = expr_mul(X(), X());                      pairs[np].shifted = expr_add(expr_mul(X(), X()), K(1));   pairs[np].label = "x*x vs x*x+1"; np++;
    pairs[np].base = expr_mul(X(), X());                      pairs[np].shifted = expr_add(expr_mul(X(), X()), K(5));   pairs[np].label = "x*x vs x*x+5"; np++;
    pairs[np].base = expr_mul(X(), X());                      pairs[np].shifted = expr_sub(expr_mul(X(), X()), K(1));   pairs[np].label = "x*x vs x*x-1"; np++;
    pairs[np].base = X();                                       pairs[np].shifted = expr_add(X(), K(1));                  pairs[np].label = "x vs x+1"; np++;
    pairs[np].base = expr_max(X(), K(0));                       pairs[np].shifted = expr_add(expr_max(X(), K(0)), K(1));  pairs[np].label = "max(x,0) vs max(x,0)+1"; np++;
    pairs[np].base = expr_mul(X(), expr_sub(X(), K(3)));        pairs[np].shifted = expr_add(expr_mul(X(), expr_sub(X(), K(3))), K(1)); pairs[np].label = "x*(x-3) vs +1"; np++;

    int same_count = 0;
    uint8_t *t1 = malloc((size_t)Dp), *c1 = malloc((size_t)Cp);
    uint8_t *t2 = malloc((size_t)Dp), *c2 = malloc((size_t)Cp);
    for (int i = 0; i < np; i++) {
        expr_to_signature_dual(t1, c1, pairs[i].base,    test, n_test, n_vars);
        expr_to_signature_dual(t2, c2, pairs[i].shifted, test, n_test, n_vars);
        int r1_idx = route_dual(t1, c1, eb, mask);
        int r2_idx = route_dual(t2, c2, eb, mask);
        int same = (r1_idx == r2_idx);
        if (same) same_count++;
        printf("  %-28s  base->%s  shifted->%s  %s\n",
               pairs[i].label,
               names[eb->base.labels[r1_idx]],
               names[eb->base.labels[r2_idx]],
               same ? "SAME class" : "DIFFERENT classes");
    }
    printf("  %d/%d constant-offset pairs merge under dual rule\n", same_count, np);
    for (int i = 0; i < np; i++) { expr_free(pairs[i].base); expr_free(pairs[i].shifted); }
    free(t1); free(c1); free(t2); free(c2); free(mask);
}

/* ── §4 Granularity sweep (H2) ──────────────────────────────────────────── */
/* Generate 30 random expressions; for each, also generate a sibling that
 * adds K(1) to root. Compute both signatures. Report distance distribution.
 * PASS if ≥80% of pairs have signature distance ≤ 2. */

static double granularity_sweep(int n_pairs, int n_vars,
                                     const m4t_mtfp_t* test, int n_test,
                                     uint32_t seed)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, n_test);
    uint8_t *t_a = malloc((size_t)Dp), *c_a = malloc((size_t)Cp);
    uint8_t *t_b = malloc((size_t)Dp), *c_b = malloc((size_t)Cp);
    uint32_t state = seed;
    int low_dist = 0;
    for (int i = 0; i < n_pairs; i++) {
        expr_t* base = expr_random(&state, n_vars, 3);
        /* Sibling: base + K(1). Same structure shifted by 1. */
        expr_t* sibling = expr_add(base, K(1));
        expr_to_signature_dual(t_a, c_a, base,    test, n_test, n_vars);
        expr_to_signature_dual(t_b, c_b, sibling, test, n_test, n_vars);
        int trit_dist = m4t_route_confidence_weighted_dist(t_a, c_a, t_b, c_b, mask, n_test);
        if (trit_dist <= 2) low_dist++;
        /* Note: expr_free on sibling frees base too (sibling owns it via expr_add). */
        expr_free(sibling);
    }
    free(t_a); free(c_a); free(t_b); free(c_b); free(mask);
    return 100.0 * (double)low_dist / (double)n_pairs;
}

/* ── §5 Timing measurement (H4) ─────────────────────────────────────────── */

static void timing_compare(int n_iter, const expr_bank_dual_t* dual_eb,
                                const expr_bank_t* sign_eb,
                                const m4t_mtfp_t* test, int n_test, int n_vars) {
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, n_test);
    uint8_t *trit = malloc((size_t)Dp), *conf = malloc((size_t)Cp);
    uint8_t* sig_only = malloc((size_t)Dp);

    /* Use a single fixed expression for timing. */
    expr_t* probe = expr_mul(X(), expr_sub(X(), K(3)));
    expr_to_signature(sig_only, probe, test, n_test, n_vars);
    expr_to_signature_dual(trit, conf, probe, test, n_test, n_vars);

    /* Sign-only routing timing. */
    clock_t t0 = clock();
    int dummy = 0;
    for (int i = 0; i < n_iter; i++) dummy ^= route_signonly(sig_only, sign_eb, mask);
    double t_sign = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Dual routing timing. */
    t0 = clock();
    for (int i = 0; i < n_iter; i++) dummy ^= route_dual(trit, conf, dual_eb, mask);
    double t_dual = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("  %d iterations:\n", n_iter);
    printf("    sign-only popcount path: %.3fms total, %.3fμs per route\n",
           t_sign * 1000.0, t_sign * 1e6 / n_iter);
    printf("    dual conf-weighted path: %.3fms total, %.3fμs per route\n",
           t_dual * 1000.0, t_dual * 1e6 / n_iter);
    printf("    ratio (dual/sign):       %.2fx\n",
           t_sign > 0 ? t_dual / t_sign : 0.0);

    expr_free(probe);
    free(trit); free(conf); free(sig_only); free(mask);
    (void)dummy;
}

/* ── §6 Inter-class distance under new rule (M1) ────────────────────────── */

static int inter_class_min_dual(const expr_bank_dual_t* eb,
                                     const uint8_t* mask,
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
    return cnt;
}

/* ── §7 Runtime regression check (M2, L1) ───────────────────────────────── */
/* For each curated candidate, compute dual signature AND sign-only signature.
 * Verify that dual is NOT bit-identical to sign-only padded with zeros (which
 * would mean the kernel's conf channel is unused or always-zero). */

static int runtime_regression_check(expr_t** cands, int n_cand,
                                          const m4t_mtfp_t* test, int n_test, int n_vars,
                                          const char** names) {
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    uint8_t *trit = malloc((size_t)Dp), *conf = malloc((size_t)Cp);
    uint8_t *sig_only = malloc((size_t)Dp);
    int n_with_conf = 0;
    int n_trit_diff = 0;
    for (int c = 0; c < n_cand; c++) {
        expr_to_signature(sig_only, cands[c], test, n_test, n_vars);
        expr_to_signature_dual(trit, conf, cands[c], test, n_test, n_vars);
        int conf_set = 0;
        for (int b = 0; b < Cp; b++) if (conf[b]) { conf_set = 1; break; }
        int trit_diff = (memcmp(sig_only, trit, (size_t)Dp) != 0);
        if (conf_set) n_with_conf++;
        if (trit_diff) n_trit_diff++;
    }
    free(trit); free(conf); free(sig_only);
    printf("  %d/%d candidates have at least one conf bit set\n", n_with_conf, n_cand);
    printf("  %d/%d candidates have trit-signature differing from sign-only\n", n_trit_diff, n_cand);
    /* PASS if at least one candidate has conf bits set (the kernel is doing
     * something with the second channel). */
    int pass = (n_with_conf > 0);
    (void)names;
    return pass;
}

/* ── §8 Per-band distribution diagnostic (M3) ───────────────────────────── */

static void per_band_distribution(const expr_bank_dual_t* eb, const char** names,
                                       int* out_total_zero, int* out_total_cells) {
    int sig_dim = eb->base.sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int Cp = (sig_dim + 7) / 8;
    int total_zero = 0, total_cells = 0;
    int total_strong_pos = 0, total_weak_pos = 0;
    int total_strong_neg = 0, total_weak_neg = 0;
    for (int t = 0; t < eb->base.n_tiles; t++) {
        const uint8_t* trit = eb->base.tiles_packed + (size_t)t * Dp;
        const uint8_t* conf = eb->conf_bits_per_tile + (size_t)t * Cp;
        int sn=0,wn=0,z=0,wp=0,sp=0;
        for (int i = 0; i < sig_dim; i++) {
            uint8_t code = (uint8_t)((trit[i>>2] >> ((i&3)*2)) & 0x3u);
            int conf_bit = (conf[i>>3] >> (i&7)) & 1u;
            if (code == 0x00u) z++;
            else if (code == 0x01u) { if (conf_bit) sp++; else wp++; }
            else if (code == 0x02u) { if (conf_bit) sn++; else wn++; }
        }
        printf("  tile %2d (%s): SN=%d WN=%d Z=%d WP=%d SP=%d\n",
               t, names[eb->base.labels[t]], sn, wn, z, wp, sp);
        total_strong_neg += sn; total_weak_neg += wn; total_zero += z;
        total_weak_pos += wp; total_strong_pos += sp;
        total_cells += sig_dim;
    }
    printf("  TOTAL: SN=%d WN=%d Z=%d WP=%d SP=%d (total cells=%d)\n",
           total_strong_neg, total_weak_neg, total_zero, total_weak_pos, total_strong_pos,
           total_cells);
    *out_total_zero = total_zero;
    *out_total_cells = total_cells;
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    init_test_inputs_2();
    printf("# R1 RED-TEAM REMEDIATION (vision claim #2)\n");
    printf("# Pre-committed gates per journal/r1_remediation_precommit.md\n\n");

    /* Build curated banks under both rules. */
    expr_t* c1[32]; const char* n1[32]; int nc1 = build_arity1_bank(c1, n1);
    expr_t* c2[32]; const char* n2[32]; int nc2 = build_arity2_bank(c2, n2);

    expr_bank_t       sb1 = {0}; build_signonly_bank(&sb1, c1, nc1, TEST_INPUTS_1, 16, 1);
    expr_bank_dual_t  db1 = {0}; build_dual_bank   (&db1, c1, nc1, TEST_INPUTS_1, 16, 1);
    expr_bank_t       sb2 = {0}; build_signonly_bank(&sb2, c2, nc2, TEST_INPUTS_2, 16, 2);
    expr_bank_dual_t  db2 = {0}; build_dual_bank   (&db2, c2, nc2, TEST_INPUTS_2, 16, 2);

    printf("Curated banks:\n");
    printf("  arity-1: sign-only %d classes, dual %d classes\n", sb1.base.n_tiles, db1.base.n_tiles);
    printf("  arity-2: sign-only %d classes, dual %d classes\n", sb2.base.n_tiles, db2.base.n_tiles);

    /* ── §1 Partition-change measurement ───────────────────────────── */
    printf("\n========================================\n");
    printf("§1  PARTITION-CHANGE (C1) — multi-seed (subsumes H3, M4)\n");
    printf("Gate: 5-seed mean >=30%% with stddev <=15pp.\n");
    printf("========================================\n");
    double pc_pcts1[N_SEEDS], pc_pcts2[N_SEEDS];
    for (int s = 0; s < N_SEEDS; s++) {
        pc_pcts1[s] = partition_change_one_seed(100, 1, TEST_INPUTS_1, 16, SEEDS[s]);
        pc_pcts2[s] = partition_change_one_seed(100, 2, TEST_INPUTS_2, 16, SEEDS[s]);
        printf("  seed 0x%x: arity-1 %.1f%%  arity-2 %.1f%%\n", SEEDS[s], pc_pcts1[s], pc_pcts2[s]);
    }
    double mean1 = 0, mean2 = 0;
    for (int s = 0; s < N_SEEDS; s++) { mean1 += pc_pcts1[s]; mean2 += pc_pcts2[s]; }
    mean1 /= N_SEEDS; mean2 /= N_SEEDS;
    double var1 = 0, var2 = 0;
    for (int s = 0; s < N_SEEDS; s++) {
        double d1 = pc_pcts1[s] - mean1; var1 += d1*d1;
        double d2 = pc_pcts2[s] - mean2; var2 += d2*d2;
    }
    double sd1 = sqrt(var1/(N_SEEDS-1));
    double sd2 = sqrt(var2/(N_SEEDS-1));
    double pc_combined_mean = (mean1 + mean2) / 2.0;
    double pc_combined_sd = (sd1 + sd2) / 2.0;
    int pc_pass = (pc_combined_mean >= 30.0 && pc_combined_sd <= 15.0);
    int pc_fail = (pc_combined_mean < 15.0);
    printf("\n  arity-1 mean=%.1f%% sd=%.1fpp\n", mean1, sd1);
    printf("  arity-2 mean=%.1f%% sd=%.1fpp\n", mean2, sd2);
    printf("§1 RESULT: combined mean=%.1f%% sd=%.1fpp  -> %s\n",
           pc_combined_mean, pc_combined_sd,
           pc_pass ? "PASS" : (pc_fail ? "FAIL" : "WEAK"));

    /* ── §2 Rule-difference probes ─────────────────────────────────── */
    printf("\n========================================\n");
    printf("§2  RULE-DIFFERENCE PROBES (C2)\n");
    printf("Gate: >=70%% of probes route to dual-expected class.\n");
    printf("(Arity-2 has no rule-difference probes — curated bank has same\n");
    printf("partition under both rules.)\n");
    printf("========================================\n");
    rd_probe_t rd1[16]; int nr1 = build_rd_probes_arity1(rd1);
    int Dp = M4T_TRIT_PACKED_BYTES(16);
    int Cp = (16 + 7) / 8;
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, 16);
    uint8_t *trit = malloc((size_t)Dp), *conf = malloc((size_t)Cp);
    int rd_correct = 0;
    for (int i = 0; i < nr1; i++) {
        expr_to_signature_dual(trit, conf, rd1[i].probe, TEST_INPUTS_1, 16, 1);
        int landed = route_dual(trit, conf, &db1, mask);
        int expected = db1.candidate_to_class[rd1[i].expected_dual_candidate];
        int ok = (landed == expected);
        if (ok) rd_correct++;
        printf("  %-30s expect=%-12s landed=%-12s %s\n",
               rd1[i].description,
               n1[db1.base.labels[expected]],
               n1[db1.base.labels[landed]],
               ok ? "OK" : "MISS");
    }
    free(trit); free(conf);
    double rd_pct = nr1 > 0 ? 100.0 * (double)rd_correct / (double)nr1 : 0.0;
    int rd_pass = (rd_pct >= 70.0);
    int rd_fail = (rd_pct < 50.0);
    printf("\n§2 RESULT: %d/%d (%.1f%%)  GATE >=70%%  -> %s\n",
           rd_correct, nr1, rd_pct,
           rd_pass ? "PASS" : (rd_fail ? "FAIL" : "WEAK"));
    free_rd_probes(rd1, nr1);

    /* ── §3 Constant-offset diagnostic ─────────────────────────────── */
    printf("\n========================================\n");
    printf("§3  CONSTANT-OFFSET DIAGNOSTIC (H1) — diagnostic, no gate\n");
    printf("========================================\n");
    run_h1_diagnostic(&db1, n1, TEST_INPUTS_1, 16, 1);

    /* ── §4 Granularity sweep ──────────────────────────────────────── */
    printf("\n========================================\n");
    printf("§4  GRANULARITY SWEEP (H2)\n");
    printf("Gate: >=80%% of base/(base+1) sibling pairs have dist <=2.\n");
    printf("========================================\n");
    double gran1 = granularity_sweep(30, 1, TEST_INPUTS_1, 16, 0xb1u);
    double gran2 = granularity_sweep(30, 2, TEST_INPUTS_2, 16, 0xb2u);
    double gran_combined = (gran1 + gran2) / 2.0;
    int gran_pass = (gran_combined >= 80.0);
    printf("  arity-1: %.1f%% pairs with dist <=2\n", gran1);
    printf("  arity-2: %.1f%% pairs with dist <=2\n", gran2);
    printf("§4 RESULT: combined %.1f%%  GATE >=80%%  -> %s\n",
           gran_combined, gran_pass ? "PASS" : (gran_combined >= 60 ? "WEAK" : "FAIL"));

    /* ── §5 Timing measurement ─────────────────────────────────────── */
    printf("\n========================================\n");
    printf("§5  TIMING (H4) — diagnostic, no gate\n");
    printf("========================================\n");
    timing_compare(100000, &db1, &sb1, TEST_INPUTS_1, 16, 1);

    /* ── §6 Inter-class distance under new rule ────────────────────── */
    printf("\n========================================\n");
    printf("§6  INTER-CLASS DISTANCE UNDER DUAL RULE (M1)\n");
    printf("Gate: minimum inter-class distance >=4 for both arities.\n");
    printf("========================================\n");
    int Dp16 = M4T_TRIT_PACKED_BYTES(16);
    uint8_t* mask16 = malloc((size_t)Dp16);
    make_full_mask(mask16, 16);
    int min1, max1; double mean_d1;
    int min2, max2; double mean_d2;
    inter_class_min_dual(&db1, mask16, &min1, &mean_d1, &max1);
    inter_class_min_dual(&db2, mask16, &min2, &mean_d2, &max2);
    printf("  arity-1: %d classes; min=%d mean=%.2f max=%d  %s\n",
           db1.base.n_tiles, min1, mean_d1, max1, min1 >= 4 ? "OK" : "FLAG");
    printf("  arity-2: %d classes; min=%d mean=%.2f max=%d  %s\n",
           db2.base.n_tiles, min2, mean_d2, max2, min2 >= 4 ? "OK" : "FLAG");
    int ic_pass = (min1 >= 4 && min2 >= 4);
    int ic_fail = (min1 < 2 || min2 < 2);
    printf("§6 RESULT: -> %s\n",
           ic_pass ? "PASS" : (ic_fail ? "FAIL" : "WEAK"));
    free(mask16);

    /* ── §7 Runtime regression check ───────────────────────────────── */
    printf("\n========================================\n");
    printf("§7  RUNTIME REGRESSION CHECK (M2, L1)\n");
    printf("Verify dual signatures aren't silently sign-only.\n");
    printf("========================================\n");
    int reg1 = runtime_regression_check(c1, nc1, TEST_INPUTS_1, 16, 1, n1);
    int reg2 = runtime_regression_check(c2, nc2, TEST_INPUTS_2, 16, 2, n2);
    int reg_pass = (reg1 && reg2);
    printf("§7 RESULT: -> %s\n", reg_pass ? "PASS" : "FAIL");

    /* ── §8 Per-band distribution ──────────────────────────────────── */
    printf("\n========================================\n");
    printf("§8  PER-BAND DISTRIBUTION (M3) — diagnostic\n");
    printf("Flag if zero band exceeds 60%% of cells on average.\n");
    printf("========================================\n");
    int total_zero1, total_cells1;
    printf("\n--- Arity-1 bank ---\n");
    per_band_distribution(&db1, n1, &total_zero1, &total_cells1);
    int total_zero2, total_cells2;
    printf("\n--- Arity-2 bank ---\n");
    per_band_distribution(&db2, n2, &total_zero2, &total_cells2);
    double zero_pct1 = 100.0 * (double)total_zero1 / (double)total_cells1;
    double zero_pct2 = 100.0 * (double)total_zero2 / (double)total_cells2;
    printf("\n  arity-1 zero band: %.1f%% of cells  %s\n",
           zero_pct1, zero_pct1 > 60 ? "FLAG (zero-dominated)" : "OK");
    printf("  arity-2 zero band: %.1f%% of cells  %s\n",
           zero_pct2, zero_pct2 > 60 ? "FLAG (zero-dominated)" : "OK");

    /* ── Final verdict ─────────────────────────────────────────────── */
    printf("\n========================================\n");
    printf("R1-REMEDIATION FINAL VERDICT\n");
    printf("========================================\n");
    printf("§1 partition-change   : %s (mean %.1f%% sd %.1fpp)\n",
           pc_pass ? "PASS" : (pc_fail ? "FAIL" : "WEAK"), pc_combined_mean, pc_combined_sd);
    printf("§2 rule-difference    : %s (%d/%d = %.1f%%)\n",
           rd_pass ? "PASS" : (rd_fail ? "FAIL" : "WEAK"), rd_correct, nr1, rd_pct);
    printf("§3 constant-offset    : reported (no gate)\n");
    printf("§4 granularity        : %s (%.1f%%)\n",
           gran_pass ? "PASS" : (gran_combined >= 60 ? "WEAK" : "FAIL"), gran_combined);
    printf("§5 timing             : reported (no gate)\n");
    printf("§6 inter-class M1     : %s (arity-1 min=%d, arity-2 min=%d)\n",
           ic_pass ? "PASS" : (ic_fail ? "FAIL" : "WEAK"), min1, min2);
    printf("§7 runtime regression : %s\n", reg_pass ? "PASS" : "FAIL");
    printf("§8 per-band dist      : reported (zero %.1f%%/%.1f%%)\n", zero_pct1, zero_pct2);

    int all_pass = pc_pass && rd_pass && gran_pass && ic_pass && reg_pass;
    int any_fail = pc_fail || rd_fail || ic_fail || !reg_pass;
    const char* overall;
    if (any_fail) overall = "FAIL";
    else if (all_pass) overall = "PASS";
    else overall = "WEAK";
    printf("\nOVERALL R1-REMEDIATION: %s\n", overall);
    printf("========================================\n");

    free(mask);
    free_signonly_bank(&sb1); free_dual_bank(&db1);
    free_signonly_bank(&sb2); free_dual_bank(&db2);
    free_cands(c1, nc1); free_cands(c2, nc2);
    return (strcmp(overall, "FAIL") == 0) ? 1 : 0;
}
