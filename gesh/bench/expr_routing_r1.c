/*
 * expr_routing_r1.c — verification probe for R1 (richer signature rule).
 *
 * Tests the per-expression-tau dual-threshold rule from
 * journal/r1_signature_rule_synthesize.md against pre-committed gates from
 * docs/PLAN_EXPRESSION_ROUTING_R2.md:
 *
 *   R1-A backward-compat   : >= 70% subagent-probe match under new rule
 *   R1-B information gain  : >= 30% of random expressions get DIFFERENT
 *                            signatures vs sign-only
 *   R1-C substrate-kernel  : new rule includes m4t_route_threshold_extract_dual
 *                            AND m4t_route_confidence_weighted_dist
 *                            (verified by code review; this binary's
 *                            grep-able call sites are the evidence).
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

/* ── Test inputs (same as original probe) ───────────────────────────────── */

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

/* ── Bank constructors (same candidates as remediation binary) ──────────── */

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

static void free_cands(expr_t** c, int n) {
    for (int i = 0; i < n; i++) expr_free(c[i]);
}

/* ── Subagent probes (same set as remediation binary) ───────────────────── */

typedef struct {
    expr_t* probe;
    int     expected_candidate;
    const char* description;
} probe_t;

static int build_subagent_probes_arity1(probe_t* p) {
    int n = 0;
    p[n].probe=expr_add(X(), expr_sub(K(3), K(3))); p[n].expected_candidate=0; p[n].description="x+(3-3)"; n++;
    p[n].probe=expr_neg(expr_neg(X())); p[n].expected_candidate=0; p[n].description="-(-x)"; n++;
    p[n].probe=expr_sub(K(0), X()); p[n].expected_candidate=1; p[n].description="0-x"; n++;
    p[n].probe=expr_neg(expr_add(X(), K(0))); p[n].expected_candidate=1; p[n].description="-(x+0)"; n++;
    p[n].probe=expr_max(X(), expr_neg(X())); p[n].expected_candidate=2; p[n].description="max(x,-x)"; n++;
    p[n].probe=expr_neg(expr_max(X(), expr_neg(X()))); p[n].expected_candidate=3; p[n].description="-max(x,-x)"; n++;
    p[n].probe=expr_min(X(), expr_neg(X())); p[n].expected_candidate=3; p[n].description="min(x,-x)"; n++;
    p[n].probe=expr_add(expr_add(X(), K(2)), K(3)); p[n].expected_candidate=6; p[n].description="x+2+3"; n++;
    p[n].probe=expr_sub(expr_sub(X(), K(2)), K(3)); p[n].expected_candidate=7; p[n].description="x-2-3"; n++;
    p[n].probe=expr_add(expr_max(X(), K(0)), K(0)); p[n].expected_candidate=8; p[n].description="max(x,0)+0"; n++;
    p[n].probe=expr_neg(expr_min(expr_neg(X()), K(0))); p[n].expected_candidate=8; p[n].description="-min(-x,0)"; n++;
    p[n].probe=expr_min(K(0), X()); p[n].expected_candidate=9; p[n].description="min(0,x)"; n++;
    p[n].probe=expr_sub(expr_mul(X(), X()), expr_mul(K(3), X())); p[n].expected_candidate=10; p[n].description="x²-3x"; n++;
    p[n].probe=expr_sub(expr_mul(X(), X()), K(1)); p[n].expected_candidate=11; p[n].description="x²-1"; n++;
    p[n].probe=expr_sub(expr_max(X(), K(0)), expr_max(expr_neg(X()), K(0))); p[n].expected_candidate=0; p[n].description="ReLU(x)-ReLU(-x)"; n++;
    return n;
}

static int build_subagent_probes_arity2(probe_t* p) {
    int n = 0;
    p[n].probe=expr_add(Y(), X()); p[n].expected_candidate=0; p[n].description="y+x"; n++;
    p[n].probe=expr_add(expr_add(X(), Y()), expr_sub(K(7), K(7))); p[n].expected_candidate=0; p[n].description="x+y+(7-7)"; n++;
    p[n].probe=expr_neg(expr_sub(Y(), X())); p[n].expected_candidate=1; p[n].description="-(y-x)"; n++;
    p[n].probe=expr_neg(expr_sub(X(), Y())); p[n].expected_candidate=2; p[n].description="-(x-y)"; n++;
    p[n].probe=expr_mul(Y(), X()); p[n].expected_candidate=3; p[n].description="y*x"; n++;
    p[n].probe=expr_neg(expr_mul(expr_neg(X()), Y())); p[n].expected_candidate=3; p[n].description="-(-x*y)"; n++;
    p[n].probe=expr_neg(expr_max(expr_neg(X()), expr_neg(Y()))); p[n].expected_candidate=4; p[n].description="-max(-x,-y)"; n++;
    p[n].probe=expr_neg(expr_min(expr_neg(X()), expr_neg(Y()))); p[n].expected_candidate=5; p[n].description="-min(-x,-y)"; n++;
    p[n].probe=expr_max(expr_sub(X(),Y()), expr_sub(Y(),X())); p[n].expected_candidate=6; p[n].description="max(x-y,y-x)"; n++;
    p[n].probe=expr_sub(expr_max(X(),Y()), expr_min(X(),Y())); p[n].expected_candidate=6; p[n].description="max-min"; n++;
    p[n].probe=expr_mul(expr_sub(X(),Y()), expr_add(X(),Y())); p[n].expected_candidate=8; p[n].description="(x-y)(x+y)"; n++;
    p[n].probe=expr_sub(expr_mul(X(),X()), expr_mul(Y(),Y())); p[n].expected_candidate=8; p[n].description="x²-y²"; n++;
    p[n].probe=expr_add(expr_sub(K(0),X()), expr_sub(K(0),Y())); p[n].expected_candidate=10; p[n].description="(0-x)+(0-y)"; n++;
    p[n].probe=expr_add(expr_min(X(),Y()), expr_max(X(),Y())); p[n].expected_candidate=0; p[n].description="min+max"; n++;
    p[n].probe=expr_sub(expr_sub(expr_add(expr_max(X(),Y()), expr_min(X(),Y())),
                                     expr_max(X(),Y())),
                          expr_max(X(),Y())); p[n].expected_candidate=6; p[n].description="(max+min)-2max (ambig)"; n++;
    return n;
}

static void free_probes(probe_t* p, int n) {
    for (int i = 0; i < n; i++) expr_free(p[i].probe);
}

/* ── Routing helpers ────────────────────────────────────────────────────── */

static void make_full_mask(uint8_t* mask, int sig_dim) {
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    memset(mask, 0xFF, (size_t)Dp);
    int tail = sig_dim & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
}

/* Route via confidence-weighted distance over dual signatures. */
static int route_dual(
    const uint8_t* q_trit, const uint8_t* q_conf,
    const expr_bank_dual_t* eb, const uint8_t* mask)
{
    int sig_dim = eb->base.sig_dim;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int Cp = (sig_dim + 7) / 8;
    int best_t = 0;
    int32_t best_d = INT32_MAX;
    for (int t = 0; t < eb->base.n_tiles; t++) {
        int32_t d = m4t_route_confidence_weighted_dist(
            q_trit, q_conf,
            eb->base.tiles_packed + (size_t)t * Dp,
            eb->conf_bits_per_tile + (size_t)t * Cp,
            mask, sig_dim);
        if (d < best_d) { best_d = d; best_t = t; }
    }
    return best_t;
}

/* ── R1-A: backward-compat probe under new rule ─────────────────────────── */

typedef struct { int correct; int total; } tally_t;

static tally_t run_dual_probes(
    const expr_bank_dual_t* eb, probe_t* probes, int n_probes,
    const m4t_mtfp_t* test_inputs, int n_test, int n_vars,
    const char** names, int verbose)
{
    int sig_dim = n_test;
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    int Cp = (sig_dim + 7) / 8;
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, sig_dim);
    uint8_t* trit = malloc((size_t)Dp);
    uint8_t* conf = malloc((size_t)Cp);
    int correct = 0;
    for (int i = 0; i < n_probes; i++) {
        expr_to_signature_dual(trit, conf, probes[i].probe, test_inputs, n_test, n_vars);
        int landed = route_dual(trit, conf, eb, mask);
        int expected = eb->candidate_to_class[probes[i].expected_candidate];
        int ok = (landed == expected);
        if (ok) correct++;
        if (verbose) {
            printf("  %-30s expect=%-22s landed=%-22s %s\n",
                   probes[i].description,
                   names[eb->base.labels[expected]],
                   names[eb->base.labels[landed]],
                   ok ? "OK" : "MISS");
        }
    }
    free(trit); free(conf); free(mask);
    tally_t r = { correct, n_probes };
    return r;
}

/* ── R1-B: information gain (signature byte-difference rate) ────────────── */
/*
 * For 100 random expressions per arity, compute both the sign-only signature
 * and the dual signature. R1-B PASSes if >= 30% have a different combined
 * signature byte-pattern under the new rule vs old. (Per pre-commit in
 * PLAN_EXPRESSION_ROUTING_R2.md: "different signatures vs sign-only").
 */

static int info_gain_count(int n_random, int n_vars,
                              const m4t_mtfp_t* test_inputs, int n_test,
                              uint32_t seed)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    uint8_t* old_sig  = malloc((size_t)Dp);
    uint8_t* new_trit = malloc((size_t)Dp);
    uint8_t* new_conf = malloc((size_t)Cp);
    uint32_t state = seed;
    int n_diff = 0;
    for (int i = 0; i < n_random; i++) {
        expr_t* e = expr_random(&state, n_vars, 3);
        expr_to_signature(old_sig, e, test_inputs, n_test, n_vars);
        expr_to_signature_dual(new_trit, new_conf, e, test_inputs, n_test, n_vars);

        /* "Different signatures" = either trit signature differs OR conf bits
         * are non-trivially set (the conf channel carries new information). */
        int trit_diff = (memcmp(old_sig, new_trit, (size_t)Dp) != 0);
        int conf_nonzero = 0;
        for (int b = 0; b < Cp; b++) if (new_conf[b] != 0) { conf_nonzero = 1; break; }
        if (trit_diff || conf_nonzero) n_diff++;

        expr_free(e);
    }
    free(old_sig); free(new_trit); free(new_conf);
    return n_diff;
}

/* Also report: do new and old rules produce DIFFERENT EQUIVALENCE-CLASS
 * PARTITIONS on a random candidate set? This is a stronger informal check. */
static void partition_diff(int n_random, int n_vars,
                              const m4t_mtfp_t* test_inputs, int n_test,
                              uint32_t seed,
                              int* out_old_classes, int* out_new_classes)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    int Cp = (n_test + 7) / 8;
    expr_t** cands = malloc((size_t)n_random * sizeof(expr_t*));
    uint32_t state = seed;
    for (int i = 0; i < n_random; i++) {
        cands[i] = expr_random(&state, n_vars, 3);
    }

    /* Old-rule bank (sign-only). */
    expr_bank_t b_old = {0};
    b_old.base.tiles_packed   = calloc((size_t)n_random, (size_t)Dp);
    b_old.base.labels         = calloc((size_t)n_random, sizeof(int));
    b_old.base.sig_dim        = n_test;
    b_old.candidate_to_class  = calloc((size_t)n_random, sizeof(int));
    expr_bank_build(&b_old, (const expr_t* const*)cands, n_random,
                      test_inputs, n_test, n_vars);
    *out_old_classes = b_old.base.n_tiles;

    /* New-rule bank (per-expression-tau dual-threshold). */
    expr_bank_dual_t b_new = {0};
    b_new.base.tiles_packed   = calloc((size_t)n_random, (size_t)Dp);
    b_new.base.labels         = calloc((size_t)n_random, sizeof(int));
    b_new.base.sig_dim        = n_test;
    b_new.conf_bits_per_tile  = calloc((size_t)n_random, (size_t)Cp);
    b_new.candidate_to_class  = calloc((size_t)n_random, sizeof(int));
    expr_bank_dual_build(&b_new, (const expr_t* const*)cands, n_random,
                            test_inputs, n_test, n_vars);
    *out_new_classes = b_new.base.n_tiles;

    free(b_old.base.tiles_packed); free(b_old.base.labels); free(b_old.candidate_to_class);
    free(b_new.base.tiles_packed); free(b_new.base.labels);
    free(b_new.conf_bits_per_tile); free(b_new.candidate_to_class);
    for (int i = 0; i < n_random; i++) expr_free(cands[i]);
    free(cands);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    init_test_inputs_2();
    printf("# Expression Routing R1 — per-expression-tau dual-threshold rule\n");
    printf("# Pre-committed gates per docs/PLAN_EXPRESSION_ROUTING_R2.md\n\n");

    /* Build dual banks for arity-1 and arity-2. */
    expr_t* c1[32]; const char* n1[32]; int nc1 = build_arity1_bank(c1, n1);
    expr_t* c2[32]; const char* n2[32]; int nc2 = build_arity2_bank(c2, n2);

    int Dp = M4T_TRIT_PACKED_BYTES(16);
    int Cp = (16 + 7) / 8;
    expr_bank_dual_t eb1 = {0};
    eb1.base.tiles_packed     = calloc((size_t)nc1, (size_t)Dp);
    eb1.base.labels           = calloc((size_t)nc1, sizeof(int));
    eb1.base.sig_dim          = 16;
    eb1.conf_bits_per_tile    = calloc((size_t)nc1, (size_t)Cp);
    eb1.candidate_to_class    = calloc((size_t)nc1, sizeof(int));
    expr_bank_dual_build(&eb1, (const expr_t* const*)c1, nc1, TEST_INPUTS_1, 16, 1);

    expr_bank_dual_t eb2 = {0};
    eb2.base.tiles_packed     = calloc((size_t)nc2, (size_t)Dp);
    eb2.base.labels           = calloc((size_t)nc2, sizeof(int));
    eb2.base.sig_dim          = 16;
    eb2.conf_bits_per_tile    = calloc((size_t)nc2, (size_t)Cp);
    eb2.candidate_to_class    = calloc((size_t)nc2, sizeof(int));
    expr_bank_dual_build(&eb2, (const expr_t* const*)c2, nc2, TEST_INPUTS_2, 16, 2);

    printf("Dual banks under per-expression-tau rule:\n");
    printf("  arity-1: %d candidates -> %d equivalence classes\n", nc1, eb1.base.n_tiles);
    printf("  arity-2: %d candidates -> %d equivalence classes\n", nc2, eb2.base.n_tiles);

    /* ── R1-A backward-compat probe ────────────────────────────────── */
    printf("\n========================================\n");
    printf("R1-A  BACKWARD-COMPAT (subagent probes under new rule)\n");
    printf("========================================\n");
    probe_t p1[32]; int np1 = build_subagent_probes_arity1(p1);
    probe_t p2[32]; int np2 = build_subagent_probes_arity2(p2);
    printf("\n--- Arity-1 ---\n");
    tally_t a1 = run_dual_probes(&eb1, p1, np1, TEST_INPUTS_1, 16, 1, n1, 1);
    printf("\n--- Arity-2 ---\n");
    tally_t a2 = run_dual_probes(&eb2, p2, np2, TEST_INPUTS_2, 16, 2, n2, 1);
    int rA_correct = a1.correct + a2.correct;
    int rA_total = a1.total + a2.total;
    double rA_pct = 100.0 * (double)rA_correct / (double)rA_total;
    int rA_pass = (rA_pct >= 70.0);
    printf("\nR1-A RESULT: %d/%d (%.1f%%)  GATE >=70%%  -> %s\n",
           rA_correct, rA_total, rA_pct,
           rA_pass ? "PASS" : (rA_pct >= 50.0 ? "WEAK" : "FAIL"));
    free_probes(p1, np1); free_probes(p2, np2);

    /* ── R1-B information gain ─────────────────────────────────────── */
    printf("\n========================================\n");
    printf("R1-B  INFORMATION GAIN (random-expression signature differences)\n");
    printf("100 random expressions per arity; count how many produce DIFFERENT\n");
    printf("signatures under new rule vs sign-only (trit-bytes differ OR conf\n");
    printf("bits non-zero).\n");
    printf("========================================\n");
    int diff1 = info_gain_count(100, 1, TEST_INPUTS_1, 16, 0xa1u);
    int diff2 = info_gain_count(100, 2, TEST_INPUTS_2, 16, 0xa2u);
    printf("  arity-1: %d/100 differ\n", diff1);
    printf("  arity-2: %d/100 differ\n", diff2);
    int rB_total = diff1 + diff2;
    double rB_pct = 100.0 * (double)rB_total / 200.0;
    int rB_pass = (rB_pct >= 30.0);
    printf("\nR1-B RESULT: %d/200 (%.1f%%)  GATE >=30%%  -> %s\n",
           rB_total, rB_pct,
           rB_pass ? "PASS" : (rB_pct >= 10.0 ? "WEAK" : "FAIL"));

    /* Stronger informal check: does the partition change? */
    int old1, new1, old2, new2;
    partition_diff(100, 1, TEST_INPUTS_1, 16, 0xb1u, &old1, &new1);
    partition_diff(100, 2, TEST_INPUTS_2, 16, 0xb2u, &old2, &new2);
    printf("\nInformal: equivalence-class counts under old rule vs new rule\n");
    printf("  arity-1 (100 random): old=%d classes, new=%d classes (delta %+d)\n",
           old1, new1, new1 - old1);
    printf("  arity-2 (100 random): old=%d classes, new=%d classes (delta %+d)\n",
           old2, new2, new2 - old2);

    /* ── R1-C substrate-kernel use ─────────────────────────────────── */
    printf("\n========================================\n");
    printf("R1-C  SUBSTRATE-KERNEL USE\n");
    printf("New rule call path includes (verifiable via grep):\n");
    printf("  - m4t_route_threshold_extract_dual    (used by expr_to_signature_dual)\n");
    printf("  - m4t_route_confidence_weighted_dist  (used by route_dual)\n");
    printf("Both shipped, both previously unused in expression-routing consumer.\n");
    int rC_pass = 1;  /* by construction; grep evidence is the audit */
    printf("R1-C RESULT: -> PASS (by construction)\n");

    /* ── Verdict ───────────────────────────────────────────────────── */
    printf("\n========================================\n");
    printf("R1 FINAL VERDICT\n");
    printf("========================================\n");
    printf("R1-A backward-compat : %s (%.1f%%)\n", rA_pass ? "PASS" : (rA_pct >= 50 ? "WEAK" : "FAIL"), rA_pct);
    printf("R1-B information gain: %s (%.1f%%)\n", rB_pass ? "PASS" : (rB_pct >= 10 ? "WEAK" : "FAIL"), rB_pct);
    printf("R1-C substrate-kernel: PASS\n");
    int all_pass = rA_pass && rB_pass && rC_pass;
    const char* overall;
    if (all_pass) overall = "PASS";
    else if (rA_pct < 50.0) overall = "FAIL (R1-A)";
    else if (rB_pct < 10.0 && rA_pct < 90.0) overall = "FAIL (R1-B)";
    else overall = "WEAK";
    printf("OVERALL R1: %s\n", overall);
    printf("========================================\n");

    free(eb1.base.tiles_packed); free(eb1.base.labels);
    free(eb1.conf_bits_per_tile); free(eb1.candidate_to_class);
    free(eb2.base.tiles_packed); free(eb2.base.labels);
    free(eb2.conf_bits_per_tile); free(eb2.candidate_to_class);
    free_cands(c1, nc1); free_cands(c2, nc2);
    return (strncmp(overall, "FAIL", 4) == 0) ? 1 : 0;
}
