/*
 * expr_routing_probe.c — P0-4 probe for vision claim #2.
 *
 * Builds two banks (arity-1, arity-2) of candidate expressions, lets the
 * equivalence-class machinery merge them, then routes ~30 syntactically
 * distinct equivalents per bank to test whether routing-as-equivalence
 * recognition works.
 *
 * Pre-committed verdict gate (per docs/PLAN_EXPRESSION_ROUTING.md and
 * journal/expression_routing_synthesize.md):
 *   PASS  : ≥51/60 (≥85%) AND every class with ≥1 probe gets ≥1/3 correct
 *   WEAK  : 36-50/60
 *   FAIL  : ≤35/60
 */

#include "expr.h"
#include "expr_bank.h"
#include "expr_signature.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Test-input sets ────────────────────────────────────────────────────── */

static const m4t_mtfp_t TEST_INPUTS_1[16] = {
    -30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30
};

/* 4×4 grid over {-10, -3, 3, 10}, row-major (x, y) pairs. */
static m4t_mtfp_t TEST_INPUTS_2[16 * 2];
static void init_test_inputs_2(void) {
    static const m4t_mtfp_t pts[4] = {-10, -3, 3, 10};
    int idx = 0;
    for (int xi = 0; xi < 4; xi++) {
        for (int yi = 0; yi < 4; yi++) {
            TEST_INPUTS_2[idx*2 + 0] = pts[xi];
            TEST_INPUTS_2[idx*2 + 1] = pts[yi];
            idx++;
        }
    }
}

/* Convenience constructors. x = var 0, y = var 1. */
static expr_t* X(void) { return expr_var(0); }
static expr_t* Y(void) { return expr_var(1); }
static expr_t* K(int v) { return expr_const((m4t_mtfp_t)v); }

/* ── Routing: nearest tile via Hamming, return class index ──────────────── */

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

static void make_full_mask(uint8_t* mask, int sig_dim) {
    int Dp = M4T_TRIT_PACKED_BYTES(sig_dim);
    memset(mask, 0xFF, (size_t)Dp);
    int tail = sig_dim & 3;
    if (tail > 0) mask[Dp - 1] = (uint8_t)((1u << (tail * 2)) - 1u);
}

/* ── Probe descriptor ───────────────────────────────────────────────────── */

typedef struct {
    expr_t* probe;
    int     expected_candidate;   /* index into the candidate list */
    const char* description;
} probe_t;

/* Per-class hit tracking. */
typedef struct {
    int n_total;
    int n_correct;
} class_stats_t;

/* ── Run one bank: build, probe, tally, verdict-feed ────────────────────── */

typedef struct {
    int n_classes;
    int n_probes;
    int n_correct;
    int n_classes_meeting_floor;   /* classes where >=1/3 of their probes correct */
    int n_classes_with_probes;
} bank_result_t;

static bank_result_t run_bank(
    const char* bank_name,
    const expr_t* const* candidates, const char* const* candidate_names,
    int n_candidates,
    probe_t* probes, int n_probes,
    const m4t_mtfp_t* test_inputs, int n_test_inputs, int n_vars)
{
    int Dp = M4T_TRIT_PACKED_BYTES(n_test_inputs);

    /* Allocate bank storage. */
    expr_bank_t bank = {0};
    bank.base.tiles_packed = (uint8_t*)calloc((size_t)n_candidates, (size_t)Dp);
    bank.base.labels       = (int*)calloc((size_t)n_candidates, sizeof(int));
    bank.base.sig_dim      = n_test_inputs;
    bank.base.n_tiles      = 0;
    bank.candidate_to_class = (int*)calloc((size_t)n_candidates, sizeof(int));

    expr_bank_build(&bank, candidates, n_candidates,
                      test_inputs, n_test_inputs, n_vars);

    printf("\n=== %s ===\n", bank_name);
    printf("Candidates: %d → Equivalence classes: %d (mergers: %d)\n",
           n_candidates, bank.base.n_tiles, n_candidates - bank.base.n_tiles);

    /* Report each class with its representative + members. */
    for (int k = 0; k < bank.base.n_tiles; k++) {
        printf("  class %d: rep = %s   |  members:", k,
               candidate_names[bank.base.labels[k]]);
        for (int c = 0; c < n_candidates; c++) {
            if (bank.candidate_to_class[c] == k) printf(" [%d]%s", c, candidate_names[c]);
        }
        printf("\n");
    }

    /* Run probes. */
    uint8_t* mask = (uint8_t*)malloc((size_t)Dp);
    make_full_mask(mask, n_test_inputs);
    uint8_t* probe_sig = (uint8_t*)malloc((size_t)Dp);

    class_stats_t* stats = (class_stats_t*)calloc((size_t)bank.base.n_tiles,
                                                       sizeof(class_stats_t));
    int n_correct_total = 0;

    printf("\nProbes:\n");
    for (int i = 0; i < n_probes; i++) {
        expr_to_signature(probe_sig, probes[i].probe,
                            test_inputs, n_test_inputs, n_vars);
        int landed_class = route_signature(probe_sig, &bank.base, mask);
        int expected_class = bank.candidate_to_class[probes[i].expected_candidate];
        int correct = (landed_class == expected_class);

        stats[expected_class].n_total++;
        if (correct) {
            stats[expected_class].n_correct++;
            n_correct_total++;
        }

        printf("  %-36s expect=class%d (%s)  landed=class%d (%s)  %s\n",
               probes[i].description,
               expected_class, candidate_names[bank.base.labels[expected_class]],
               landed_class,   candidate_names[bank.base.labels[landed_class]],
               correct ? "OK" : "MISS");
    }

    /* Per-class floor check. */
    int n_meeting_floor = 0;
    int n_with_probes   = 0;
    for (int k = 0; k < bank.base.n_tiles; k++) {
        if (stats[k].n_total == 0) continue;
        n_with_probes++;
        if (stats[k].n_correct * 3 >= stats[k].n_total) n_meeting_floor++;
    }

    printf("\n%s tally: %d/%d correct (%.1f%%)\n",
           bank_name, n_correct_total, n_probes,
           100.0 * (double)n_correct_total / (double)n_probes);
    printf("Per-class floor (>=1/3 correct): %d/%d classes meet it\n",
           n_meeting_floor, n_with_probes);

    /* Free per-probe expressions and bank storage. */
    free(stats);
    free(probe_sig);
    free(mask);
    free(bank.base.tiles_packed);
    free(bank.base.labels);
    free(bank.candidate_to_class);

    bank_result_t r;
    r.n_classes = bank.base.n_tiles;
    r.n_probes  = n_probes;
    r.n_correct = n_correct_total;
    r.n_classes_meeting_floor = n_meeting_floor;
    r.n_classes_with_probes   = n_with_probes;
    return r;
}

/* ── Arity-1 candidate set ──────────────────────────────────────────────── */

static int build_arity1_bank_args(
    expr_t** out_cand, const char** out_names)
{
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

/* ── Arity-1 probes ─────────────────────────────────────────────────────── */
/* Index references the arity1 candidate list above (0 = x, 1 = -x, ...). */

static int build_arity1_probes(probe_t* p) {
    int n = 0;
    /* Class containing x (cand 0) */
    p[n].probe=expr_neg(expr_neg(X())); p[n].expected_candidate=0; p[n].description="-(-x) -> x"; n++;
    p[n].probe=expr_add(X(), K(0));     p[n].expected_candidate=0; p[n].description="x+0 -> x"; n++;
    p[n].probe=expr_sub(X(), K(0));     p[n].expected_candidate=0; p[n].description="x-0 -> x"; n++;
    p[n].probe=expr_mul(X(), K(1));     p[n].expected_candidate=0; p[n].description="x*1 -> x"; n++;
    p[n].probe=expr_add(expr_sub(X(), X()), X()); p[n].expected_candidate=0; p[n].description="(x-x)+x -> x"; n++;

    /* Class containing -x (cand 1) */
    p[n].probe=expr_sub(K(0), X());     p[n].expected_candidate=1; p[n].description="0-x -> -x"; n++;
    p[n].probe=expr_mul(K(-1), X());    p[n].expected_candidate=1; p[n].description="-1*x -> -x"; n++;
    p[n].probe=expr_neg(X());           p[n].expected_candidate=1; p[n].description="-(x) -> -x"; n++;

    /* Class containing |x| (cand 2; merges with x*x cand 4) */
    p[n].probe=expr_mul(X(), X());                p[n].expected_candidate=2; p[n].description="x*x -> |x|"; n++;
    p[n].probe=expr_max(expr_neg(X()), X());      p[n].expected_candidate=2; p[n].description="max(-x,x) -> |x|"; n++;
    p[n].probe=expr_mul(expr_neg(X()), expr_neg(X())); p[n].expected_candidate=2; p[n].description="(-x)*(-x) -> |x|"; n++;
    p[n].probe=expr_max(X(), expr_neg(X()));      p[n].expected_candidate=2; p[n].description="max(x,-x) -> |x|"; n++;
    p[n].probe=expr_neg(expr_neg(expr_mul(X(), X()))); p[n].expected_candidate=2; p[n].description="-(-(x*x)) -> |x|"; n++;

    /* Class containing -|x| (cand 3) */
    p[n].probe=expr_min(X(), expr_neg(X()));      p[n].expected_candidate=3; p[n].description="min(x,-x) -> -|x|"; n++;
    p[n].probe=expr_neg(expr_max(X(), expr_neg(X()))); p[n].expected_candidate=3; p[n].description="-max(x,-x) -> -|x|"; n++;
    p[n].probe=expr_neg(expr_mul(X(), X()));      p[n].expected_candidate=3; p[n].description="-(x*x) -> -|x|"; n++;

    /* Class containing x+5 (cand 6) */
    p[n].probe=expr_add(expr_add(X(), K(2)), K(3));   p[n].expected_candidate=6; p[n].description="(x+2)+3 -> x+5"; n++;
    p[n].probe=expr_add(K(5), X());                   p[n].expected_candidate=6; p[n].description="5+x -> x+5"; n++;
    p[n].probe=expr_sub(X(), K(-5));                  p[n].expected_candidate=6; p[n].description="x-(-5) -> x+5"; n++;

    /* Class containing x-5 (cand 7) */
    p[n].probe=expr_add(X(), K(-5));                  p[n].expected_candidate=7; p[n].description="x+(-5) -> x-5"; n++;
    p[n].probe=expr_sub(X(), K(5));                   p[n].expected_candidate=7; p[n].description="x-5 (literal)"; n++;
    p[n].probe=expr_add(K(-5), X());                  p[n].expected_candidate=7; p[n].description="-5+x -> x-5"; n++;

    /* Class containing max(x,0) (cand 8) */
    p[n].probe=expr_max(K(0), X());                   p[n].expected_candidate=8; p[n].description="max(0,x) -> max(x,0)"; n++;
    p[n].probe=expr_max(X(), K(0));                   p[n].expected_candidate=8; p[n].description="max(x,0) literal"; n++;

    /* Class containing min(x,0) (cand 9) */
    p[n].probe=expr_min(K(0), X());                   p[n].expected_candidate=9; p[n].description="min(0,x) -> min(x,0)"; n++;
    p[n].probe=expr_min(X(), K(0));                   p[n].expected_candidate=9; p[n].description="min(x,0) literal"; n++;

    /* Class containing x*(x-3) (cand 10) */
    p[n].probe=expr_mul(expr_sub(X(), K(3)), X());    p[n].expected_candidate=10; p[n].description="(x-3)*x -> x*(x-3)"; n++;
    p[n].probe=expr_sub(expr_mul(X(), X()), expr_mul(K(3), X())); p[n].expected_candidate=10; p[n].description="x*x-3x -> x*(x-3)"; n++;

    /* Class containing (x-1)*(x+1) (cand 11) */
    p[n].probe=expr_mul(expr_add(X(), K(1)), expr_sub(X(), K(1))); p[n].expected_candidate=11; p[n].description="(x+1)*(x-1) -> x²-1"; n++;
    p[n].probe=expr_sub(expr_mul(X(), X()), K(1));    p[n].expected_candidate=11; p[n].description="x*x-1 -> x²-1"; n++;

    return n;
}

/* ── Arity-2 candidate set ──────────────────────────────────────────────── */

static int build_arity2_bank_args(
    expr_t** out_cand, const char** out_names)
{
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

/* ── Arity-2 probes ─────────────────────────────────────────────────────── */

static int build_arity2_probes(probe_t* p) {
    int n = 0;
    /* x+y class (cand 0; merges with min+max=7, x+(y+0)=11) */
    p[n].probe=expr_add(Y(), X());                              p[n].expected_candidate=0; p[n].description="y+x -> x+y"; n++;
    p[n].probe=expr_add(expr_add(X(), K(0)), Y());              p[n].expected_candidate=0; p[n].description="(x+0)+y -> x+y"; n++;
    p[n].probe=expr_add(expr_min(X(),Y()), expr_max(X(),Y()));  p[n].expected_candidate=0; p[n].description="min+max -> x+y"; n++;
    p[n].probe=expr_add(expr_add(X(), Y()), K(0));              p[n].expected_candidate=0; p[n].description="(x+y)+0 -> x+y"; n++;
    p[n].probe=expr_add(expr_add(X(), K(1)), expr_sub(Y(), K(1))); p[n].expected_candidate=0; p[n].description="(x+1)+(y-1) -> x+y"; n++;

    /* x-y class (cand 1) */
    p[n].probe=expr_add(X(), expr_neg(Y()));                    p[n].expected_candidate=1; p[n].description="x+(-y) -> x-y"; n++;
    p[n].probe=expr_add(expr_neg(Y()), X());                    p[n].expected_candidate=1; p[n].description="-y+x -> x-y"; n++;
    p[n].probe=expr_sub(expr_sub(X(), Y()), K(0));              p[n].expected_candidate=1; p[n].description="(x-y)-0 -> x-y"; n++;

    /* y-x class (cand 2) */
    p[n].probe=expr_neg(expr_sub(X(), Y()));                    p[n].expected_candidate=2; p[n].description="-(x-y) -> y-x"; n++;
    p[n].probe=expr_add(Y(), expr_neg(X()));                    p[n].expected_candidate=2; p[n].description="y+(-x) -> y-x"; n++;
    p[n].probe=expr_add(expr_sub(Y(), X()), K(0));              p[n].expected_candidate=2; p[n].description="(y-x)+0 -> y-x"; n++;

    /* x*y class (cand 3) */
    p[n].probe=expr_mul(Y(), X());                              p[n].expected_candidate=3; p[n].description="y*x -> x*y"; n++;
    p[n].probe=expr_mul(expr_mul(X(), K(1)), Y());              p[n].expected_candidate=3; p[n].description="(x*1)*y -> x*y"; n++;
    p[n].probe=expr_mul(X(), expr_add(Y(), K(0)));              p[n].expected_candidate=3; p[n].description="x*(y+0) -> x*y"; n++;
    p[n].probe=expr_add(expr_mul(X(), Y()), K(0));              p[n].expected_candidate=3; p[n].description="(x*y)+0 -> x*y"; n++;

    /* min(x,y) class (cand 4; merges with min(min(x,y),x)=12) */
    p[n].probe=expr_min(Y(), X());                              p[n].expected_candidate=4; p[n].description="min(y,x) -> min(x,y)"; n++;
    p[n].probe=expr_min(expr_min(X(), Y()), X());               p[n].expected_candidate=4; p[n].description="min(min(x,y),x) -> min(x,y)"; n++;
    p[n].probe=expr_min(X(), expr_min(X(), Y()));               p[n].expected_candidate=4; p[n].description="min(x,min(x,y)) -> min(x,y)"; n++;

    /* max(x,y) class (cand 5; merges with max(max(x,y),y)=13) */
    p[n].probe=expr_max(Y(), X());                              p[n].expected_candidate=5; p[n].description="max(y,x) -> max(x,y)"; n++;
    p[n].probe=expr_max(expr_max(X(), Y()), Y());               p[n].expected_candidate=5; p[n].description="max(max(x,y),y) -> max(x,y)"; n++;
    p[n].probe=expr_max(Y(), expr_max(X(), Y()));               p[n].expected_candidate=5; p[n].description="max(y,max(x,y)) -> max(x,y)"; n++;

    /* |x-y| class (cand 6) */
    p[n].probe=expr_sub(expr_max(X(),Y()), expr_min(X(),Y()));  p[n].expected_candidate=6; p[n].description="max-min -> |x-y|"; n++;
    p[n].probe=expr_sub(expr_max(Y(),X()), expr_min(Y(),X()));  p[n].expected_candidate=6; p[n].description="max(y,x)-min(y,x) -> |x-y|"; n++;

    /* x²-y² class (cand 8; merges with (x+y)(x-y)=9) */
    p[n].probe=expr_mul(expr_add(X(),Y()), expr_sub(X(),Y()));  p[n].expected_candidate=8; p[n].description="(x+y)(x-y) -> x²-y²"; n++;
    p[n].probe=expr_mul(expr_sub(X(),Y()), expr_add(X(),Y()));  p[n].expected_candidate=8; p[n].description="(x-y)(x+y) -> x²-y²"; n++;
    p[n].probe=expr_sub(expr_mul(X(),X()), expr_mul(Y(),Y()));  p[n].expected_candidate=8; p[n].description="x*x-y*y -> x²-y²"; n++;

    /* -(x+y) class (cand 10) */
    p[n].probe=expr_sub(expr_neg(X()), Y());                    p[n].expected_candidate=10; p[n].description="-x-y -> -(x+y)"; n++;
    p[n].probe=expr_add(expr_neg(X()), expr_neg(Y()));          p[n].expected_candidate=10; p[n].description="(-x)+(-y) -> -(x+y)"; n++;
    p[n].probe=expr_sub(K(0), expr_add(X(), Y()));              p[n].expected_candidate=10; p[n].description="0-(x+y) -> -(x+y)"; n++;
    p[n].probe=expr_neg(expr_add(Y(), X()));                    p[n].expected_candidate=10; p[n].description="-(y+x) -> -(x+y)"; n++;

    return n;
}

/* ── Tightened (HARD) probes ──────────────────────────────────────────────
 *
 * Probes that are NOT byte-identical to any bank candidate, but are signature-
 * close to a specific class. Tests "did we build a routing system" vs "did we
 * build a hash table". Expected classes pre-computed by hand from signature
 * analysis (see journal/expression_routing_synthesize.md follow-up).
 */

static int build_arity1_probes_hard(probe_t* p) {
    int n = 0;
    /* Each probe shifts/perturbs a candidate; target is the CLOSEST
     * candidate by sign-Hamming over the 16 test inputs. */
    p[n].probe=expr_add(X(), K(1));                                p[n].expected_candidate=0; p[n].description="HARD x+1 -> x"; n++;
    p[n].probe=expr_sub(X(), K(1));                                p[n].expected_candidate=0; p[n].description="HARD x-1 -> x"; n++;
    p[n].probe=expr_add(X(), K(10));                               p[n].expected_candidate=6; p[n].description="HARD x+10 -> x+5"; n++;
    p[n].probe=expr_sub(X(), K(10));                               p[n].expected_candidate=7; p[n].description="HARD x-10 -> x-5"; n++;
    p[n].probe=expr_mul(expr_sub(X(),K(2)), expr_add(X(),K(2)));   p[n].expected_candidate=11; p[n].description="HARD (x-2)(x+2) -> x²-1"; n++;
    p[n].probe=expr_mul(X(), expr_sub(X(), K(1)));                 p[n].expected_candidate=2; p[n].description="HARD x*(x-1) -> |x|"; n++;
    p[n].probe=expr_mul(expr_sub(X(),K(3)), expr_sub(X(),K(5)));   p[n].expected_candidate=2; p[n].description="HARD (x-3)(x-5) -> |x|"; n++;
    p[n].probe=expr_neg(expr_mul(X(), expr_sub(X(), K(3))));       p[n].expected_candidate=3; p[n].description="HARD -(x*(x-3)) -> -|x|"; n++;
    p[n].probe=expr_mul(expr_add(X(),K(1)), expr_sub(X(),K(2)));   p[n].expected_candidate=11; p[n].description="HARD (x+1)(x-2) -> x²-1"; n++;
    p[n].probe=expr_sub(K(1), X());                                p[n].expected_candidate=1; p[n].description="HARD 1-x -> -x"; n++;
    return n;
}

static int build_arity2_probes_hard(probe_t* p) {
    int n = 0;
    p[n].probe=expr_add(expr_add(X(),Y()), K(1));                  p[n].expected_candidate=0; p[n].description="HARD x+y+1 -> x+y"; n++;
    p[n].probe=expr_sub(expr_add(X(),Y()), K(1));                  p[n].expected_candidate=0; p[n].description="HARD x+y-1 -> x+y"; n++;
    p[n].probe=expr_add(X(), expr_mul(K(2), Y()));                 p[n].expected_candidate=0; p[n].description="HARD x+2y -> x+y"; n++;
    p[n].probe=expr_add(expr_mul(K(2), X()), Y());                 p[n].expected_candidate=0; p[n].description="HARD 2x+y -> x+y"; n++;
    p[n].probe=expr_mul(expr_add(X(),Y()), expr_add(X(),Y()));     p[n].expected_candidate=6; p[n].description="HARD (x+y)² -> |x-y|"; n++;
    p[n].probe=expr_add(expr_mul(X(),X()), expr_mul(Y(),Y()));     p[n].expected_candidate=6; p[n].description="HARD x²+y² -> |x-y|"; n++;
    p[n].probe=expr_sub(expr_min(X(),Y()), K(5));                  p[n].expected_candidate=4; p[n].description="HARD min(x,y)-5 -> min(x,y)"; n++;
    p[n].probe=expr_add(expr_max(X(),Y()), K(5));                  p[n].expected_candidate=6; p[n].description="HARD max(x,y)+5 -> |x-y|"; n++;
    return n;
}

/* ── Cleanup ────────────────────────────────────────────────────────────── */

static void free_probes(probe_t* p, int n) {
    for (int i = 0; i < n; i++) expr_free(p[i].probe);
}
static void free_candidates(expr_t** c, int n) {
    for (int i = 0; i < n; i++) expr_free(c[i]);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    init_test_inputs_2();

    printf("# Expression Routing Probe (P0-4, vision claim #2)\n");
    printf("# arity-1: 16 test inputs; arity-2: 16 input pairs over 4x4 grid\n");
    printf("# sig_dim = 16 (one trit per test input)\n");

    /* Arity-1 bank — easy probes (algebraic equivalents). */
    expr_t* cand1[32]; const char* names1[32];
    int n_cand1 = build_arity1_bank_args(cand1, names1);
    probe_t probes1[64];
    int n_probes1 = build_arity1_probes(probes1);

    bank_result_t r1_easy = run_bank("ARITY-1 (easy: algebraic equivalents)",
        (const expr_t* const*)cand1, names1, n_cand1,
        probes1, n_probes1, TEST_INPUTS_1, 16, 1);

    free_probes(probes1, n_probes1);

    /* Arity-1 bank — HARD probes (near-equivalent, tests routing-not-hashing). */
    probe_t probes1h[32];
    int n_probes1h = build_arity1_probes_hard(probes1h);
    bank_result_t r1_hard = run_bank("ARITY-1 (HARD: near-equivalent)",
        (const expr_t* const*)cand1, names1, n_cand1,
        probes1h, n_probes1h, TEST_INPUTS_1, 16, 1);
    free_probes(probes1h, n_probes1h);
    free_candidates(cand1, n_cand1);

    /* Arity-2 bank — easy probes. */
    expr_t* cand2[32]; const char* names2[32];
    int n_cand2 = build_arity2_bank_args(cand2, names2);
    probe_t probes2[64];
    int n_probes2 = build_arity2_probes(probes2);

    bank_result_t r2_easy = run_bank("ARITY-2 (easy: algebraic equivalents)",
        (const expr_t* const*)cand2, names2, n_cand2,
        probes2, n_probes2, TEST_INPUTS_2, 16, 2);

    free_probes(probes2, n_probes2);

    /* Arity-2 bank — HARD probes. */
    probe_t probes2h[32];
    int n_probes2h = build_arity2_probes_hard(probes2h);
    bank_result_t r2_hard = run_bank("ARITY-2 (HARD: near-equivalent)",
        (const expr_t* const*)cand2, names2, n_cand2,
        probes2h, n_probes2h, TEST_INPUTS_2, 16, 2);
    free_probes(probes2h, n_probes2h);
    free_candidates(cand2, n_cand2);

    /* ── Verdicts (BOTH gates pre-committed before running) ──────────────
     *
     * EASY gate (original, sanity-check):
     *   PASS  : >= 51/60 AND every class with probes >= 1/3 correct
     *   WEAK  : 36-50/60
     *   FAIL  : <= 35/60
     *
     * HARD gate (tightened, the real test of routing-as-meaning):
     *   HARD_PASS : >= 14/18 AND no class with all probes wrong
     *   HARD_WEAK : 9-13/18
     *   HARD_FAIL : <= 8/18
     *
     * OVERALL = PASS iff both EASY and HARD pass; FAIL iff either FAILs;
     *           WEAK otherwise. */

    int easy_correct = r1_easy.n_correct + r2_easy.n_correct;
    int easy_probes  = r1_easy.n_probes  + r2_easy.n_probes;
    int easy_floor_ok = (r1_easy.n_classes_meeting_floor == r1_easy.n_classes_with_probes) &&
                          (r2_easy.n_classes_meeting_floor == r2_easy.n_classes_with_probes);

    int hard_correct = r1_hard.n_correct + r2_hard.n_correct;
    int hard_probes  = r1_hard.n_probes  + r2_hard.n_probes;
    /* For HARD, "no class with all probes wrong" = every class with probes
     * has at least one correct. The class-stats inside run_bank tracks
     * meeting-1/3-floor; for HARD we need ≥1/N where N is per-class probe
     * count, which is approximately ≥1 of any. The 1/3 floor is strictly
     * tighter than "≥1 correct" for classes with ≥4 probes; equal for
     * classes with 1-3 probes. Re-use it for the HARD floor; if it fails,
     * we report which class failed. */
    int hard_floor_ok = (r1_hard.n_classes_meeting_floor == r1_hard.n_classes_with_probes) &&
                          (r2_hard.n_classes_meeting_floor == r2_hard.n_classes_with_probes);

    const char* easy_verdict;
    if (easy_correct >= 51 && easy_floor_ok)        easy_verdict = "PASS";
    else if (easy_correct >= 36)                     easy_verdict = "WEAK";
    else                                              easy_verdict = "FAIL";

    const char* hard_verdict;
    if (hard_correct >= 14 && hard_floor_ok)        hard_verdict = "PASS";
    else if (hard_correct >= 9)                      hard_verdict = "WEAK";
    else                                              hard_verdict = "FAIL";

    const char* overall;
    if (strcmp(easy_verdict, "FAIL") == 0 || strcmp(hard_verdict, "FAIL") == 0) overall = "FAIL";
    else if (strcmp(easy_verdict, "PASS") == 0 && strcmp(hard_verdict, "PASS") == 0) overall = "PASS";
    else overall = "WEAK";

    printf("\n=================================================\n");
    printf("EASY  : %d/%d correct (%.1f%%)  floor=%s  -> %s\n",
           easy_correct, easy_probes, 100.0*(double)easy_correct/(double)easy_probes,
           easy_floor_ok ? "OK" : "FAIL", easy_verdict);
    printf("HARD  : %d/%d correct (%.1f%%)  floor=%s  -> %s\n",
           hard_correct, hard_probes, 100.0*(double)hard_correct/(double)hard_probes,
           hard_floor_ok ? "OK" : "FAIL", hard_verdict);
    printf("OVERALL VERDICT: %s\n", overall);
    printf("=================================================\n");

    return (strcmp(overall, "FAIL") == 0) ? 1 : 0;
}
