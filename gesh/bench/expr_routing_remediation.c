/*
 * expr_routing_remediation.c — addresses red-team findings from
 * journal/expression_routing_redteam.md against the pre-committed gates in
 * journal/expression_routing_remediation_precommit.md.
 *
 * Five sections, one per remediated finding cluster:
 *   §1  Subagent blind probes        (C1, C2)
 *   §2  Scale-collapse probes        (H3)
 *   §3  Multi-input-set sweep        (H1)
 *   §4  Random-bank multi-seed       (H2, M2)
 *   §5  Inter-class distance diag    (M1)
 *
 * Each section reports its own pass/fail against the pre-committed gates.
 * Final verdict: PASS only if all gated sections meet their bars.
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

/* ── Shared helpers ─────────────────────────────────────────────────────── */

static expr_t* X(void) { return expr_var(0); }
static expr_t* Y(void) { return expr_var(1); }
static expr_t* K(int v) { return expr_const((m4t_mtfp_t)v); }

static void make_full_mask(uint8_t* mask, int sig_dim) {
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    memset(mask, 0xFF, (size_t)Dp);
    int tail = sig_dim & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
}

static int route_signature(
    const uint8_t* sig_packed, const gesh_bank_t* bank, const uint8_t* mask)
{
    int Dp = M4T_TRIT_PACKED_BYTES(bank->sig_dim);
    int best_t = 0;
    int32_t best_d = INT32_MAX;
    for (int t = 0; t < bank->n_tiles; t++) {
        int32_t d = m4t_popcount_dist(
            sig_packed,
            bank->tiles_packed + (size_t)t * Dp,
            mask, Dp);
        if (d < best_d) { best_d = d; best_t = t; }
    }
    return best_t;
}

/* ── Curated banks (same as original probe; we reuse for consistency) ───── */

static const m4t_mtfp_t TEST_INPUTS_1_DEFAULT[16] = {
    -30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30
};

static int build_arity1_bank(expr_t** out_cand, const char** out_names) {
    int n = 0;
    out_cand[n]=X();                                        out_names[n++]="x";
    out_cand[n]=expr_neg(X());                              out_names[n++]="-x";
    out_cand[n]=expr_max(X(), expr_neg(X()));               out_names[n++]="|x|";
    out_cand[n]=expr_min(X(), expr_neg(X()));               out_names[n++]="-|x|";
    out_cand[n]=expr_mul(X(), X());                         out_names[n++]="x*x";
    out_cand[n]=expr_mul(expr_mul(X(), X()), X());          out_names[n++]="x*x*x";
    out_cand[n]=expr_add(X(), K(5));                        out_names[n++]="x+5";
    out_cand[n]=expr_sub(X(), K(5));                        out_names[n++]="x-5";
    out_cand[n]=expr_max(X(), K(0));                        out_names[n++]="max(x,0)";
    out_cand[n]=expr_min(X(), K(0));                        out_names[n++]="min(x,0)";
    out_cand[n]=expr_mul(X(), expr_sub(X(), K(3)));         out_names[n++]="x*(x-3)";
    out_cand[n]=expr_mul(expr_sub(X(), K(1)), expr_add(X(), K(1)));
                                                            out_names[n++]="(x-1)*(x+1)";
    return n;
}

static int build_arity2_bank(expr_t** out_cand, const char** out_names) {
    int n = 0;
    out_cand[n]=expr_add(X(), Y());                          out_names[n++]="x+y";
    out_cand[n]=expr_sub(X(), Y());                          out_names[n++]="x-y";
    out_cand[n]=expr_sub(Y(), X());                          out_names[n++]="y-x";
    out_cand[n]=expr_mul(X(), Y());                          out_names[n++]="x*y";
    out_cand[n]=expr_min(X(), Y());                          out_names[n++]="min(x,y)";
    out_cand[n]=expr_max(X(), Y());                          out_names[n++]="max(x,y)";
    out_cand[n]=expr_sub(expr_max(X(),Y()), expr_min(X(),Y())); out_names[n++]="|x-y|";
    out_cand[n]=expr_add(expr_min(X(),Y()), expr_max(X(),Y())); out_names[n++]="min+max";
    out_cand[n]=expr_sub(expr_mul(X(),X()), expr_mul(Y(),Y())); out_names[n++]="x²-y²";
    out_cand[n]=expr_mul(expr_add(X(),Y()), expr_sub(X(),Y())); out_names[n++]="(x+y)(x-y)";
    out_cand[n]=expr_neg(expr_add(X(), Y()));                out_names[n++]="-(x+y)";
    out_cand[n]=expr_add(X(), expr_add(Y(), K(0)));          out_names[n++]="x+(y+0)";
    out_cand[n]=expr_min(expr_min(X(),Y()), X());            out_names[n++]="min(min(x,y),x)";
    out_cand[n]=expr_max(expr_max(X(),Y()), Y());            out_names[n++]="max(max(x,y),y)";
    return n;
}

/* Bank wrapper: build, returns expr_bank_t with tiles_packed/labels/c2c
 * allocated. Caller frees via free_bank. */
typedef struct {
    expr_bank_t bank;
    int n_cand;
    const char** names;  /* borrowed from caller */
} eb_t;

static void build_eb(eb_t* eb, expr_t** cands, const char** names, int n_cand,
                       const m4t_mtfp_t* test_inputs, int n_test, int n_vars) {
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    eb->bank.base.tiles_packed = calloc((size_t)n_cand, (size_t)Dp);
    eb->bank.base.labels       = calloc((size_t)n_cand, sizeof(int));
    eb->bank.base.sig_dim      = n_test;
    eb->bank.base.n_tiles      = 0;
    eb->bank.candidate_to_class = calloc((size_t)n_cand, sizeof(int));
    expr_bank_build(&eb->bank, (const expr_t* const*)cands, n_cand,
                      test_inputs, n_test, n_vars);
    eb->n_cand = n_cand;
    eb->names = names;
}

static void free_eb(eb_t* eb) {
    free(eb->bank.base.tiles_packed);
    free(eb->bank.base.labels);
    free(eb->bank.candidate_to_class);
}

/* ── §1 Subagent blind probes (C1, C2) ──────────────────────────────────── */
/*
 * These probes were authored by an independent subagent with NO access to
 * the signature derivation rule or test-input set. They saw only the bank
 * representative names ("x", "-x", "|x|", ...) and were told to predict
 * routings via mathematical intuition.
 *
 * If routing matches predictions ≥70%, mathematical intuition aligns with
 * the routing rule — non-tautological evidence for vision claim #2.
 */

typedef struct {
    expr_t* probe;
    int     expected_candidate;
    const char* description;
} probe_t;

static int build_subagent_probes_arity1(probe_t* p) {
    int n = 0;
    /* 1: x + (3 - 3) -> x */
    p[n].probe=expr_add(X(), expr_sub(K(3), K(3))); p[n].expected_candidate=0; p[n].description="SUB x+(3-3) -> x"; n++;
    /* 2: -(-x) -> x */
    p[n].probe=expr_neg(expr_neg(X())); p[n].expected_candidate=0; p[n].description="SUB -(-x) -> x"; n++;
    /* 3: 0 - x -> -x */
    p[n].probe=expr_sub(K(0), X()); p[n].expected_candidate=1; p[n].description="SUB 0-x -> -x"; n++;
    /* 4: -(x + 0) -> -x */
    p[n].probe=expr_neg(expr_add(X(), K(0))); p[n].expected_candidate=1; p[n].description="SUB -(x+0) -> -x"; n++;
    /* 5: max(x, -x) -> |x| */
    p[n].probe=expr_max(X(), expr_neg(X())); p[n].expected_candidate=2; p[n].description="SUB max(x,-x) -> |x|"; n++;
    /* 6: -max(x, -x) -> -|x| */
    p[n].probe=expr_neg(expr_max(X(), expr_neg(X()))); p[n].expected_candidate=3; p[n].description="SUB -max(x,-x) -> -|x|"; n++;
    /* 7: min(x, -x) -> -|x| */
    p[n].probe=expr_min(X(), expr_neg(X())); p[n].expected_candidate=3; p[n].description="SUB min(x,-x) -> -|x|"; n++;
    /* 8: x + 2 + 3 -> x+5 */
    p[n].probe=expr_add(expr_add(X(), K(2)), K(3)); p[n].expected_candidate=6; p[n].description="SUB x+2+3 -> x+5"; n++;
    /* 9: x - 2 - 3 -> x-5 */
    p[n].probe=expr_sub(expr_sub(X(), K(2)), K(3)); p[n].expected_candidate=7; p[n].description="SUB x-2-3 -> x-5"; n++;
    /* 10: max(x, 0) + 0 -> max(x, 0) */
    p[n].probe=expr_add(expr_max(X(), K(0)), K(0)); p[n].expected_candidate=8; p[n].description="SUB max(x,0)+0 -> max(x,0)"; n++;
    /* 11: -min(-x, 0) -> max(x, 0) */
    p[n].probe=expr_neg(expr_min(expr_neg(X()), K(0))); p[n].expected_candidate=8; p[n].description="SUB -min(-x,0) -> max(x,0)"; n++;
    /* 12: min(0, x) -> min(x, 0) */
    p[n].probe=expr_min(K(0), X()); p[n].expected_candidate=9; p[n].description="SUB min(0,x) -> min(x,0)"; n++;
    /* 13: x*x - 3*x -> x*(x-3) */
    p[n].probe=expr_sub(expr_mul(X(), X()), expr_mul(K(3), X())); p[n].expected_candidate=10; p[n].description="SUB x²-3x -> x*(x-3)"; n++;
    /* 14: x*x - 1 -> (x-1)*(x+1) */
    p[n].probe=expr_sub(expr_mul(X(), X()), K(1)); p[n].expected_candidate=11; p[n].description="SUB x²-1 -> (x-1)(x+1)"; n++;
    /* 15: max(x,0) - max(-x,0) -> x */
    p[n].probe=expr_sub(expr_max(X(), K(0)), expr_max(expr_neg(X()), K(0))); p[n].expected_candidate=0; p[n].description="SUB ReLU(x)-ReLU(-x) -> x"; n++;
    return n;
}

static int build_subagent_probes_arity2(probe_t* p) {
    int n = 0;
    /* 1: y + x -> x+y */
    p[n].probe=expr_add(Y(), X()); p[n].expected_candidate=0; p[n].description="SUB y+x -> x+y"; n++;
    /* 2: x + y + (7 - 7) -> x+y */
    p[n].probe=expr_add(expr_add(X(), Y()), expr_sub(K(7), K(7))); p[n].expected_candidate=0; p[n].description="SUB x+y+(7-7) -> x+y"; n++;
    /* 3: -(y - x) -> x-y */
    p[n].probe=expr_neg(expr_sub(Y(), X())); p[n].expected_candidate=1; p[n].description="SUB -(y-x) -> x-y"; n++;
    /* 4: -(x - y) -> y-x */
    p[n].probe=expr_neg(expr_sub(X(), Y())); p[n].expected_candidate=2; p[n].description="SUB -(x-y) -> y-x"; n++;
    /* 5: y * x -> x*y */
    p[n].probe=expr_mul(Y(), X()); p[n].expected_candidate=3; p[n].description="SUB y*x -> x*y"; n++;
    /* 6: -(-x * y) -> x*y */
    p[n].probe=expr_neg(expr_mul(expr_neg(X()), Y())); p[n].expected_candidate=3; p[n].description="SUB -(-x*y) -> x*y"; n++;
    /* 7: -max(-x, -y) -> min(x, y) */
    p[n].probe=expr_neg(expr_max(expr_neg(X()), expr_neg(Y()))); p[n].expected_candidate=4; p[n].description="SUB -max(-x,-y) -> min(x,y)"; n++;
    /* 8: -min(-x, -y) -> max(x, y) */
    p[n].probe=expr_neg(expr_min(expr_neg(X()), expr_neg(Y()))); p[n].expected_candidate=5; p[n].description="SUB -min(-x,-y) -> max(x,y)"; n++;
    /* 9: max(x-y, y-x) -> |x-y| */
    p[n].probe=expr_max(expr_sub(X(),Y()), expr_sub(Y(),X())); p[n].expected_candidate=6; p[n].description="SUB max(x-y,y-x) -> |x-y|"; n++;
    /* 10: max(x,y) - min(x,y) -> |x-y| */
    p[n].probe=expr_sub(expr_max(X(),Y()), expr_min(X(),Y())); p[n].expected_candidate=6; p[n].description="SUB max-min -> |x-y|"; n++;
    /* 11: (x-y)(x+y) -> x²-y² */
    p[n].probe=expr_mul(expr_sub(X(),Y()), expr_add(X(),Y())); p[n].expected_candidate=8; p[n].description="SUB (x-y)(x+y) -> x²-y²"; n++;
    /* 12: x*x - y*y -> x²-y² */
    p[n].probe=expr_sub(expr_mul(X(),X()), expr_mul(Y(),Y())); p[n].expected_candidate=8; p[n].description="SUB x²-y² -> x²-y²"; n++;
    /* 13: (0 - x) + (0 - y) -> -(x+y) */
    p[n].probe=expr_add(expr_sub(K(0),X()), expr_sub(K(0),Y())); p[n].expected_candidate=10; p[n].description="SUB (0-x)+(0-y) -> -(x+y)"; n++;
    /* 14: min(x,y) + max(x,y) -> x+y (HARD - tempting to think max-related) */
    p[n].probe=expr_add(expr_min(X(),Y()), expr_max(X(),Y())); p[n].expected_candidate=0; p[n].description="SUB min+max -> x+y"; n++;
    /* 15: max(x,y) + min(x,y) - max(x,y) - max(x,y) -> |x-y|
     *     (subagent self-corrected: simplifies to -|x-y|; closest available
     *     is |x-y|. Honest hard probe with ambiguous expected answer.) */
    p[n].probe=expr_sub(expr_sub(expr_add(expr_max(X(),Y()), expr_min(X(),Y())),
                                    expr_max(X(),Y())),
                          expr_max(X(),Y())); p[n].expected_candidate=6; p[n].description="SUB (max+min)-2max -> |x-y| (ambig)"; n++;
    return n;
}

/* Run a probe set against a bank, return correct count and total. */
typedef struct { int correct; int total; } probe_tally_t;

static probe_tally_t run_probes(
    const char* label, const eb_t* eb, probe_t* probes, int n_probes,
    const m4t_mtfp_t* test_inputs, int n_test, int n_vars, int verbose)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, n_test);
    uint8_t* sig = malloc((size_t)Dp);
    int correct = 0;
    if (verbose) printf("\n--- %s ---\n", label);
    for (int i = 0; i < n_probes; i++) {
        expr_to_signature(sig, probes[i].probe, test_inputs, n_test, n_vars);
        int landed = route_signature(sig, &eb->bank.base, mask);
        int expected = eb->bank.candidate_to_class[probes[i].expected_candidate];
        int ok = (landed == expected);
        if (ok) correct++;
        if (verbose) {
            printf("  %-45s expect=%-22s landed=%-22s %s\n",
                   probes[i].description,
                   eb->names[eb->bank.base.labels[expected]],
                   eb->names[eb->bank.base.labels[landed]],
                   ok ? "OK" : "MISS");
        }
    }
    free(sig); free(mask);
    probe_tally_t r = { correct, n_probes };
    return r;
}

static void free_probes(probe_t* p, int n) {
    for (int i = 0; i < n; i++) expr_free(p[i].probe);
}
static void free_cands(expr_t** c, int n) {
    for (int i = 0; i < n; i++) expr_free(c[i]);
}

/* ── §2 Scale-collapse probes (H3) ──────────────────────────────────────── */
/*
 * Sign-only signatures must collapse magnitude-scaled equivalents. These
 * probes confirm the collapse is the rule's behavior. EXPECTED 100% PASS
 * by construction; reporting honestly that this is a known limit, not a
 * surprise.
 */

static int build_scale_collapse_arity1(probe_t* p) {
    int n = 0;
    /* 2*x has same sign as x at every input → should route to x. */
    p[n].probe=expr_mul(K(2), X()); p[n].expected_candidate=0; p[n].description="SCALE 2x -> x"; n++;
    /* 5*x same. */
    p[n].probe=expr_mul(K(5), X()); p[n].expected_candidate=0; p[n].description="SCALE 5x -> x"; n++;
    /* 2*|x|: same sign as |x|. */
    p[n].probe=expr_mul(K(2), expr_max(X(), expr_neg(X()))); p[n].expected_candidate=2; p[n].description="SCALE 2|x| -> |x|"; n++;
    /* -2*x: same sign as -x. */
    p[n].probe=expr_mul(K(-2), X()); p[n].expected_candidate=1; p[n].description="SCALE -2x -> -x"; n++;
    /* 3*max(x,0) same as max(x,0). */
    p[n].probe=expr_mul(K(3), expr_max(X(), K(0))); p[n].expected_candidate=8; p[n].description="SCALE 3*max(x,0) -> max(x,0)"; n++;
    return n;
}

static int build_scale_collapse_arity2(probe_t* p) {
    int n = 0;
    /* 2*(x+y) same sign as x+y. */
    p[n].probe=expr_mul(K(2), expr_add(X(),Y())); p[n].expected_candidate=0; p[n].description="SCALE 2(x+y) -> x+y"; n++;
    /* 5*(x*y) same sign as x*y. */
    p[n].probe=expr_mul(K(5), expr_mul(X(),Y())); p[n].expected_candidate=3; p[n].description="SCALE 5*x*y -> x*y"; n++;
    /* -3*(x-y) same sign as -(x-y) = y-x. */
    p[n].probe=expr_mul(K(-3), expr_sub(X(),Y())); p[n].expected_candidate=2; p[n].description="SCALE -3(x-y) -> y-x"; n++;
    /* 2*min(x,y) same sign as min(x,y). */
    p[n].probe=expr_mul(K(2), expr_min(X(),Y())); p[n].expected_candidate=4; p[n].description="SCALE 2*min(x,y) -> min(x,y)"; n++;
    /* 5*max(x,y) same sign as max(x,y). */
    p[n].probe=expr_mul(K(5), expr_max(X(),Y())); p[n].expected_candidate=5; p[n].description="SCALE 5*max(x,y) -> max(x,y)"; n++;
    return n;
}

/* ── §3 Multi-input-set sweep (H1) ──────────────────────────────────────── */
/*
 * Run the subagent probe set under 4 different test-input sets. Whether
 * routing PASSes under each lens is the data; the gate is ≥3/4 PASS.
 */

static const char* BAND_NAMES[4] = {
    "tight {-3..3}", "mid {-30..30}", "wide-positive {1..1000}", "powers-of-10"
};

/* ── §4 Random-bank multi-seed (H2, M2) ─────────────────────────────────── */
/*
 * Build a bank from random expression trees. Report merger rate. Run
 * the subagent probe set through this random bank — expected to fail
 * because subagent probes target specific named classes that may not
 * exist in a random bank. The interesting metric is: are random-bank
 * routings INTERNALLY consistent (the same equivalent probes route to
 * the same random-bank class)?
 *
 * Internal-consistency check: take pairs of subagent probes that target
 * the same class (e.g., both target `x`), compute their signatures
 * against the random bank, check if they land in the same random-bank
 * class. Expected ≥ chance.
 */

static int internal_consistency_check(
    const eb_t* eb, probe_t* probes, int n_probes,
    const m4t_mtfp_t* test_inputs, int n_test, int n_vars,
    int* out_total_pairs)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test);
    uint8_t* mask = malloc((size_t)Dp);
    make_full_mask(mask, n_test);

    /* Compute landing class for each probe. */
    int* landings = malloc((size_t)n_probes * sizeof(int));
    uint8_t* sig = malloc((size_t)Dp);
    for (int i = 0; i < n_probes; i++) {
        expr_to_signature(sig, probes[i].probe, test_inputs, n_test, n_vars);
        landings[i] = route_signature(sig, &eb->bank.base, mask);
    }
    free(sig); free(mask);

    /* Count pairs (i,j) where probes[i].expected_candidate ==
     * probes[j].expected_candidate, and check if landings agree. */
    int agree = 0, total = 0;
    for (int i = 0; i < n_probes; i++) {
        for (int j = i+1; j < n_probes; j++) {
            if (probes[i].expected_candidate == probes[j].expected_candidate) {
                total++;
                if (landings[i] == landings[j]) agree++;
            }
        }
    }
    free(landings);
    *out_total_pairs = total;
    return agree;
}

/* ── §5 Inter-class distance diagnostic (M1) ────────────────────────────── */

static void inter_class_distances(
    const gesh_bank_t* bank, const uint8_t* mask,
    int* out_min, double* out_mean, int* out_max)
{
    int Dp = M4T_TRIT_PACKED_BYTES(bank->sig_dim);
    int n = bank->n_tiles;
    int min_d = INT32_MAX, max_d = 0;
    long sum = 0; int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            int d = m4t_popcount_dist(
                bank->tiles_packed + (size_t)i * Dp,
                bank->tiles_packed + (size_t)j * Dp,
                mask, Dp);
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
            sum += d;
            cnt++;
        }
    }
    *out_min = (cnt > 0) ? min_d : 0;
    *out_max = (cnt > 0) ? max_d : 0;
    *out_mean = (cnt > 0) ? (double)sum / (double)cnt : 0.0;
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("# Expression Routing REMEDIATION (vision claim #2 P0 red-team)\n");
    printf("# Pre-committed gates per journal/expression_routing_remediation_precommit.md\n\n");

    /* Build curated banks (arity-1, arity-2) on default test inputs. */
    expr_t* cand1[32]; const char* names1[32];
    int n_cand1 = build_arity1_bank(cand1, names1);
    eb_t eb1;
    build_eb(&eb1, cand1, names1, n_cand1, TEST_INPUTS_1_DEFAULT, 16, 1);

    /* Arity-2 4x4 grid. */
    static const m4t_mtfp_t pts[4] = {-10, -3, 3, 10};
    m4t_mtfp_t test2[16 * 2];
    {
        int idx = 0;
        for (int xi = 0; xi < 4; xi++)
            for (int yi = 0; yi < 4; yi++) {
                test2[idx*2 + 0] = pts[xi];
                test2[idx*2 + 1] = pts[yi];
                idx++;
            }
    }
    expr_t* cand2[32]; const char* names2[32];
    int n_cand2 = build_arity2_bank(cand2, names2);
    eb_t eb2;
    build_eb(&eb2, cand2, names2, n_cand2, test2, 16, 2);

    printf("Curated banks: arity-1 has %d classes (%d cand, %d mergers); "
           "arity-2 has %d classes (%d cand, %d mergers)\n",
           eb1.bank.base.n_tiles, n_cand1, n_cand1 - eb1.bank.base.n_tiles,
           eb2.bank.base.n_tiles, n_cand2, n_cand2 - eb2.bank.base.n_tiles);

    /* ── §1 Subagent blind probes ─────────────────────────────────────── */
    printf("\n========================================\n");
    printf("§1  SUBAGENT BLIND PROBES (C1, C2)\n");
    printf("========================================\n");
    probe_t sub1[32]; int n_sub1 = build_subagent_probes_arity1(sub1);
    probe_t sub2[32]; int n_sub2 = build_subagent_probes_arity2(sub2);
    probe_tally_t s1 = run_probes("Arity-1 subagent probes", &eb1, sub1, n_sub1, TEST_INPUTS_1_DEFAULT, 16, 1, 1);
    probe_tally_t s2 = run_probes("Arity-2 subagent probes", &eb2, sub2, n_sub2, test2, 16, 2, 1);
    int sub_correct = s1.correct + s2.correct;
    int sub_total   = s1.total   + s2.total;
    double sub_pct = 100.0 * (double)sub_correct / (double)sub_total;
    int sub_pass = (sub_pct >= 70.0);
    printf("\n§1 RESULT: %d/%d (%.1f%%)  GATE >=70%%  -> %s\n",
           sub_correct, sub_total, sub_pct, sub_pass ? "PASS" : (sub_pct >= 50.0 ? "WEAK" : "FAIL"));
    free_probes(sub1, n_sub1); free_probes(sub2, n_sub2);

    /* ── §2 Scale-collapse probes ─────────────────────────────────────── */
    printf("\n========================================\n");
    printf("§2  SCALE-COLLAPSE PROBES (H3)\n");
    printf("(Expected 100%% routing to unscaled class — confirms documented limit)\n");
    printf("========================================\n");
    probe_t sc1[16]; int n_sc1 = build_scale_collapse_arity1(sc1);
    probe_t sc2[16]; int n_sc2 = build_scale_collapse_arity2(sc2);
    probe_tally_t sc_a1 = run_probes("Arity-1 scale-collapse", &eb1, sc1, n_sc1, TEST_INPUTS_1_DEFAULT, 16, 1, 1);
    probe_tally_t sc_a2 = run_probes("Arity-2 scale-collapse", &eb2, sc2, n_sc2, test2, 16, 2, 1);
    int sc_correct = sc_a1.correct + sc_a2.correct;
    int sc_total = sc_a1.total + sc_a2.total;
    int sc_pass = (sc_correct == sc_total);
    printf("\n§2 RESULT: %d/%d (%.1f%%)  EXPECTED 100%%  -> %s\n",
           sc_correct, sc_total, 100.0*(double)sc_correct/(double)sc_total,
           sc_pass ? "as-expected PASS (sign-only collapses scale)" : "ANOMALY");
    free_probes(sc1, n_sc1); free_probes(sc2, n_sc2);

    /* ── §3 Multi-input-set sweep ─────────────────────────────────────── */
    printf("\n========================================\n");
    printf("§3  MULTI-INPUT-SET SWEEP (H1)\n");
    printf("Run subagent probes under 4 different test-input bands.\n");
    printf("Gate: PASS criteria met under >=3/4 input sets.\n");
    printf("========================================\n");
    int sets_passed = 0;
    for (int band = 0; band < 4; band++) {
        m4t_mtfp_t band_inputs1[16];
        m4t_mtfp_t band_inputs2[16 * 2];
        uint32_t s = 0xb1u + (uint32_t)band;
        inputs_band(band_inputs1, 16, 1, band, &s);
        s = 0xb2u + (uint32_t)band;
        inputs_band(band_inputs2, 16, 2, band, &s);
        /* Rebuild banks under this band. */
        expr_t* c1[32]; const char* n1[32]; int nc1 = build_arity1_bank(c1, n1);
        expr_t* c2[32]; const char* n2[32]; int nc2 = build_arity2_bank(c2, n2);
        eb_t b1, b2;
        build_eb(&b1, c1, n1, nc1, band_inputs1, 16, 1);
        build_eb(&b2, c2, n2, nc2, band_inputs2, 16, 2);
        /* Re-build subagent probes (they were freed above). */
        probe_t p1[32]; int np1 = build_subagent_probes_arity1(p1);
        probe_t p2[32]; int np2 = build_subagent_probes_arity2(p2);
        probe_tally_t r1 = run_probes("(silent)", &b1, p1, np1, band_inputs1, 16, 1, 0);
        probe_tally_t r2 = run_probes("(silent)", &b2, p2, np2, band_inputs2, 16, 2, 0);
        int correct = r1.correct + r2.correct;
        int total = r1.total + r2.total;
        double pct = 100.0 * (double)correct / (double)total;
        int band_pass = (pct >= 70.0);
        printf("  band %d (%-30s): bank1=%d cls bank2=%d cls   probes %d/%d (%.1f%%)  %s\n",
               band, BAND_NAMES[band], b1.bank.base.n_tiles, b2.bank.base.n_tiles,
               correct, total, pct, band_pass ? "PASS" : (pct >= 50 ? "WEAK" : "FAIL"));
        if (band_pass) sets_passed++;
        free_probes(p1, np1); free_probes(p2, np2);
        free_eb(&b1); free_eb(&b2); free_cands(c1, nc1); free_cands(c2, nc2);
    }
    int multi_pass = (sets_passed >= 3);
    printf("\n§3 RESULT: %d/4 input bands PASS  GATE >=3/4  -> %s\n",
           sets_passed, multi_pass ? "PASS" : (sets_passed >= 2 ? "WEAK" : "FAIL"));

    /* ── §4 Random-bank multi-seed ────────────────────────────────────── */
    printf("\n========================================\n");
    printf("§4  RANDOM-BANK MULTI-SEED (H2, M2)\n");
    printf("Build banks from random expression trees. Report merger rate.\n");
    printf("Internal-consistency: subagent probes targeting same class should\n");
    printf("land in the same RANDOM-bank class. Gate: 5-seed mean >=70%% with stddev <=15pp.\n");
    printf("========================================\n");
    double seed_pcts[5];
    int n_random_cand = 20;
    for (int seed = 0; seed < 5; seed++) {
        uint32_t state = 0xdada0001u + (uint32_t)seed * 31u;
        expr_t* rcand[32]; const char* rnames[32];
        for (int i = 0; i < n_random_cand; i++) {
            rcand[i] = expr_random(&state, 1, 3);
            rnames[i] = "random";
        }
        eb_t reb;
        build_eb(&reb, rcand, rnames, n_random_cand, TEST_INPUTS_1_DEFAULT, 16, 1);
        int n_classes = reb.bank.base.n_tiles;
        int mergers = n_random_cand - n_classes;
        /* Build subagent arity-1 probes again. */
        probe_t p[32]; int np = build_subagent_probes_arity1(p);
        int total_pairs = 0;
        int agree = internal_consistency_check(&reb, p, np, TEST_INPUTS_1_DEFAULT, 16, 1, &total_pairs);
        double pct = (total_pairs > 0) ? (100.0 * (double)agree / (double)total_pairs) : 0.0;
        seed_pcts[seed] = pct;
        printf("  seed %u: random bank n_cand=%d -> %d classes (%d mergers); "
               "consistency %d/%d pairs agree (%.1f%%)\n",
               (unsigned)seed, n_random_cand, n_classes, mergers, agree, total_pairs, pct);
        free_probes(p, np);
        free_eb(&reb); free_cands(rcand, n_random_cand);
    }
    double mean_pct = 0.0;
    for (int i = 0; i < 5; i++) mean_pct += seed_pcts[i];
    mean_pct /= 5.0;
    double var = 0.0;
    for (int i = 0; i < 5; i++) {
        double d = seed_pcts[i] - mean_pct;
        var += d * d;
    }
    double sd = sqrt(var / 4.0);  /* sample stddev, n-1 */
    int rb_pass = (mean_pct >= 70.0 && sd <= 15.0);
    printf("\n§4 RESULT: 5-seed mean = %.1f%% +/- %.1fpp  GATE mean>=70%% AND sd<=15pp  -> %s\n",
           mean_pct, sd, rb_pass ? "PASS" : (mean_pct >= 50.0 ? "WEAK" : "FAIL"));

    /* ── §5 Inter-class distance diagnostic ───────────────────────────── */
    printf("\n========================================\n");
    printf("§5  INTER-CLASS DISTANCE DIAGNOSTIC (M1)\n");
    printf("Min, mean, max pairwise Hamming distance between bank classes.\n");
    printf("Flag if min < 4 (one quarter of sig_dim=16).\n");
    printf("========================================\n");
    {
        int Dp = M4T_TRIT_PACKED_BYTES(16);
        uint8_t* mask = malloc((size_t)Dp);
        make_full_mask(mask, 16);
        int min_d, max_d; double mean_d;
        inter_class_distances(&eb1.bank.base, mask, &min_d, &mean_d, &max_d);
        printf("  arity-1: %d classes; pairwise dist min=%d mean=%.2f max=%d  %s\n",
               eb1.bank.base.n_tiles, min_d, mean_d, max_d,
               (min_d < 4) ? "FLAG (low headroom)" : "OK");
        inter_class_distances(&eb2.bank.base, mask, &min_d, &mean_d, &max_d);
        printf("  arity-2: %d classes; pairwise dist min=%d mean=%.2f max=%d  %s\n",
               eb2.bank.base.n_tiles, min_d, mean_d, max_d,
               (min_d < 4) ? "FLAG (low headroom)" : "OK");
        free(mask);
    }

    /* ── Final verdict ────────────────────────────────────────────────── */
    printf("\n========================================\n");
    printf("REMEDIATION FINAL VERDICT\n");
    printf("========================================\n");
    printf("§1 subagent blind  : %s (%.1f%%)\n", sub_pass ? "PASS" : (sub_pct>=50?"WEAK":"FAIL"), sub_pct);
    printf("§2 scale-collapse  : %s (%.1f%%, expected 100%%)\n", sc_pass ? "PASS" : "ANOMALY",
           100.0*(double)sc_correct/(double)sc_total);
    printf("§3 multi-input-set : %s (%d/4 bands)\n", multi_pass ? "PASS" : (sets_passed>=2?"WEAK":"FAIL"), sets_passed);
    printf("§4 random-bank x5  : %s (mean %.1f%% sd %.1fpp)\n", rb_pass ? "PASS" : (mean_pct>=50?"WEAK":"FAIL"), mean_pct, sd);
    printf("§5 inter-class diag: reported (no gate)\n");

    int all_gated_pass = sub_pass && multi_pass && rb_pass && sc_pass;
    const char* overall;
    if (all_gated_pass) overall = "PASS";
    else if (!sub_pass && sub_pct < 50.0) overall = "FAIL";
    else if (!multi_pass && sets_passed <= 1) overall = "FAIL";
    else if (!rb_pass && mean_pct <= 50.0) overall = "FAIL";
    else overall = "WEAK";
    printf("\nOVERALL REMEDIATION: %s\n", overall);
    printf("========================================\n");

    free_eb(&eb1); free_eb(&eb2);
    free_cands(cand1, n_cand1); free_cands(cand2, n_cand2);
    return (strcmp(overall, "FAIL") == 0) ? 1 : 0;
}
