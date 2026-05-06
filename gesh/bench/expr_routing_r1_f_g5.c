/*
 * expr_routing_r1_f_g5.c — F-G5 (held-out routing accuracy) for R1.
 *
 * Per docs/TECHNICAL_DEBT.md TD-8 + journal/r1_falsify_closeout.md F-G5.
 * Closes the 5th axis of the R1 falsification matrix that was deferred in
 * the original closeout for "external equivalence ground truth requires
 * substantial engineering."
 *
 * Approach:
 *   1. Generate K_TOTAL random arity-1 expressions.
 *   2. For each, compute a "behavioral fingerprint": int64 evaluation on
 *      N_FP fixed test inputs (independent from the signature inputs).
 *      Two expressions with byte-identical fingerprints are declared
 *      *equivalent* — strong proxy for algebraic equivalence at this
 *      input scale (collisions rare given int64 outputs and N_FP=32).
 *   3. Group expressions into equivalence classes by fingerprint.
 *   4. Filter to classes with >= 4 members (need anchor + held-out test).
 *   5. Per class: first member = bank anchor; rest = held-out test set.
 *   6. Build TWO banks from anchors: sign-only (expr_to_signature) and
 *      dual (expr_to_signature_dual). Each bank tile is the anchor's
 *      signature; tile labels are the equivalence-class IDs.
 *   7. Route each held-out test expr → predicted class = nearest bank
 *      tile (Hamming for sign-only, confidence-weighted for dual).
 *   8. Routing accuracy = (predicted_class == true_class) / total_test.
 *
 * R1's verdict shifts ONLY if dual >> sign-only here. Otherwise R1
 * remains methodically falsified (now across 5 axes instead of 4).
 *
 * Pre-committed gate: dual must beat sign-only by ≥ 2 pp absolute to
 * count as a verdict shift. Anything below that is "R1 still falsified
 * on F-G5; dual no better than sign-only at routing under behavioral-
 * equivalence ground truth."
 */

#include "expr.h"
#include "expr_random.h"
#include "expr_signature.h"
#include "m4t_route.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K_TOTAL          8000   /* expressions to generate */
#define N_FP             32     /* fingerprint inputs (int64 ground truth) */
#define N_SIG            16     /* signature test inputs (independent) */
#define MIN_PER_CLASS    4      /* need 1 anchor + 3 held-out minimum */
#define MAX_DEPTH        4
#define VERDICT_GAP_PP   2      /* dual beats sign-only by this for verdict shift */

/* Per RED-TEAM RC-1: int64 overflow on wide-range inputs (depth-4 mul-heavy
 * expressions on [-30, +30] inputs can hit |x|^16 ≈ 10^23, blowing past
 * int64). Overflow fragments otherwise-equivalent expressions into distinct
 * fingerprints, biasing the equivalence-class set toward "trivial" classes
 * (constants, all-zero outputs) where dual's magnitude info happens to help.
 *
 * Run both bands and report both. Tight is the canonical verdict (clean
 * ground truth, no overflow artifacts); wide is reported as a sanity-check
 * showing how much the verdict depends on input range. */
typedef struct { int band; const char* name; const char* note; } band_t;
static const band_t BANDS[] = {
    { 0, "tight  {-3..3}",  "canonical (no int64 overflow)" },
    { 1, "wide   {-30..30}", "sanity-check (some overflow expected at depth 4)" },
};
#define N_BANDS (int)(sizeof(BANDS) / sizeof(BANDS[0]))

typedef struct {
    int    expr_idx;            /* index into expressions[] */
    int    class_id;            /* equivalence-class index (post-filter) */
} labeled_t;

static int compare_fp_then_idx(const void* a, const void* b) {
    /* Used to group expressions sharing fingerprints together for the
     * equivalence-class scan. */
    return memcmp(a, b, sizeof(int64_t) * N_FP);
}

static int run_one_band(int band_idx);

int main(void) {
    printf("# F-G5: held-out routing accuracy for R1 (TD-8)\n");
    printf("# K_TOTAL=%d N_FP=%d N_SIG=%d MIN_PER_CLASS=%d MAX_DEPTH=%d\n",
        K_TOTAL, N_FP, N_SIG, MIN_PER_CLASS, MAX_DEPTH);
    printf("# Pre-committed gate: |dual − sign-only| ≥ %d pp triggers verdict.\n\n",
        VERDICT_GAP_PP);

    int rc = 0;
    for (int b = 0; b < N_BANDS; b++) {
        printf("########################################################\n");
        printf("# Input band: %s — %s\n", BANDS[b].name, BANDS[b].note);
        printf("########################################################\n");
        rc |= run_one_band(b);
        printf("\n");
    }
    return rc;
}

static int run_one_band(int band_idx) {
    int band = BANDS[band_idx].band;
    /* 1. Generate fingerprint inputs and signature inputs (deterministic). */
    m4t_mtfp_t fp_inputs[N_FP];
    m4t_mtfp_t sig_inputs[N_SIG];
    {
        uint32_t s_fp = 0x1F4B5C7Du;
        inputs_band(fp_inputs, N_FP, /*n_vars=*/1, band, &s_fp);
        uint32_t s_sig = 0x9E3779B1u;
        inputs_band(sig_inputs, N_SIG, /*n_vars=*/1, band, &s_sig);
    }

    /* 2. Generate K_TOTAL random expressions. */
    expr_t** exprs = (expr_t**)malloc((size_t)K_TOTAL * sizeof(expr_t*));
    if (!exprs) { fprintf(stderr, "OOM exprs\n"); return 1; }
    {
        uint32_t s_gen = 0xDEADBEEFu;
        for (int i = 0; i < K_TOTAL; i++) {
            exprs[i] = expr_random(&s_gen, /*n_vars=*/1, MAX_DEPTH);
            if (!exprs[i]) { fprintf(stderr, "OOM expr %d\n", i); return 1; }
        }
    }

    /* 3. Compute fingerprints. fp_table[i] = N_FP int64 values for expr i. */
    int64_t* fp_table = (int64_t*)malloc((size_t)K_TOTAL * N_FP * sizeof(int64_t));
    if (!fp_table) { fprintf(stderr, "OOM fp\n"); return 1; }
    for (int i = 0; i < K_TOTAL; i++) {
        for (int t = 0; t < N_FP; t++) {
            fp_table[(size_t)i * N_FP + t] =
                expr_eval(exprs[i], &fp_inputs[t], 1);
        }
    }

    /* 4. Group by fingerprint into equivalence classes. */
    /* Build (fingerprint || expr_idx) records, sort by fingerprint. */
    typedef struct {
        int64_t fp[N_FP];
        int     expr_idx;
    } fp_record_t;
    fp_record_t* recs = (fp_record_t*)malloc((size_t)K_TOTAL * sizeof(fp_record_t));
    if (!recs) { fprintf(stderr, "OOM recs\n"); return 1; }
    for (int i = 0; i < K_TOTAL; i++) {
        memcpy(recs[i].fp, fp_table + (size_t)i * N_FP,
               sizeof(int64_t) * N_FP);
        recs[i].expr_idx = i;
    }
    qsort(recs, K_TOTAL, sizeof(fp_record_t), compare_fp_then_idx);

    /* Walk sorted list, identify runs (= equivalence classes). */
    int total_classes_seen = 0;
    int n_kept_classes = 0;
    int n_anchors = 0;
    int n_test = 0;
    /* Worst case: every expr is its own class. Bound storage by K_TOTAL. */
    labeled_t* anchors = (labeled_t*)malloc((size_t)K_TOTAL * sizeof(labeled_t));
    labeled_t* tests   = (labeled_t*)malloc((size_t)K_TOTAL * sizeof(labeled_t));
    if (!anchors || !tests) { fprintf(stderr, "OOM labels\n"); return 1; }

    int run_start = 0;
    for (int i = 1; i <= K_TOTAL; i++) {
        int end_of_run = (i == K_TOTAL) ||
            (memcmp(recs[i].fp, recs[run_start].fp,
                    sizeof(int64_t) * N_FP) != 0);
        if (end_of_run) {
            int run_len = i - run_start;
            total_classes_seen++;
            if (run_len >= MIN_PER_CLASS) {
                /* First member = anchor; rest = test. Stable order from qsort. */
                int class_id = n_kept_classes;
                anchors[n_anchors].expr_idx = recs[run_start].expr_idx;
                anchors[n_anchors].class_id = class_id;
                n_anchors++;
                for (int k = run_start + 1; k < i; k++) {
                    tests[n_test].expr_idx = recs[k].expr_idx;
                    tests[n_test].class_id = class_id;
                    n_test++;
                }
                n_kept_classes++;
            }
            run_start = i;
        }
    }

    printf("Generated %d expressions → %d distinct fingerprints\n",
           K_TOTAL, total_classes_seen);
    printf("Kept %d classes (≥ %d members each) → %d anchors + %d held-out tests\n\n",
           n_kept_classes, MIN_PER_CLASS, n_anchors, n_test);

    if (n_test == 0) {
        printf("ABORT: no held-out test expressions. Increase K_TOTAL or "
               "decrease MIN_PER_CLASS.\n");
        return 2;
    }

    /* 5. Build banks from anchors. One tile per equivalence class.
     *    Tiles laid out row-major: tile[c] = anchor[c]'s signature. */
    int sig_packed_bytes  = M4T_TRIT_PACKED_BYTES(N_SIG);
    int conf_bytes        = (N_SIG + 7) / 8;

    uint8_t* bank_sign  = (uint8_t*)calloc((size_t)n_anchors * sig_packed_bytes, 1);
    uint8_t* bank_dual_trit = (uint8_t*)calloc((size_t)n_anchors * sig_packed_bytes, 1);
    uint8_t* bank_dual_conf = (uint8_t*)calloc((size_t)n_anchors * conf_bytes, 1);
    if (!bank_sign || !bank_dual_trit || !bank_dual_conf) {
        fprintf(stderr, "OOM banks\n"); return 1;
    }

    for (int c = 0; c < n_anchors; c++) {
        const expr_t* e = exprs[anchors[c].expr_idx];
        expr_to_signature(
            bank_sign + (size_t)c * sig_packed_bytes,
            e, sig_inputs, N_SIG, /*n_vars=*/1);
        expr_to_signature_dual(
            bank_dual_trit + (size_t)c * sig_packed_bytes,
            bank_dual_conf + (size_t)c * conf_bytes,
            e, sig_inputs, N_SIG, /*n_vars=*/1);
    }

    /* 6. Route held-out test exprs and tally accuracy. */
    int correct_sign  = 0;
    int correct_dual  = 0;
    /* Per-class breakdown. */
    int* per_class_total   = (int*)calloc((size_t)n_anchors, sizeof(int));
    int* per_class_sign_ok = (int*)calloc((size_t)n_anchors, sizeof(int));
    int* per_class_dual_ok = (int*)calloc((size_t)n_anchors, sizeof(int));
    if (!per_class_total || !per_class_sign_ok || !per_class_dual_ok) {
        fprintf(stderr, "OOM per-class\n"); return 1;
    }

    /* All-ones mask for sign-only popcount distance. */
    uint8_t* mask_all = (uint8_t*)malloc((size_t)sig_packed_bytes);
    if (!mask_all) { fprintf(stderr, "OOM mask\n"); return 1; }
    memset(mask_all, 0xFF, (size_t)sig_packed_bytes);
    /* Mask only valid bits in last byte (each cell = 2 bits). */
    int valid_bits_last = (N_SIG % 4) * 2;
    if (valid_bits_last != 0) {
        mask_all[sig_packed_bytes - 1] = (uint8_t)((1u << valid_bits_last) - 1u);
    }

    uint8_t test_sign[M4T_TRIT_PACKED_BYTES(N_SIG)];
    uint8_t test_dual_trit[M4T_TRIT_PACKED_BYTES(N_SIG)];
    uint8_t test_dual_conf[(N_SIG + 7) / 8];

    for (int t = 0; t < n_test; t++) {
        const expr_t* e = exprs[tests[t].expr_idx];
        int true_class = tests[t].class_id;
        per_class_total[true_class]++;

        expr_to_signature(test_sign, e, sig_inputs, N_SIG, /*n_vars=*/1);
        expr_to_signature_dual(test_dual_trit, test_dual_conf,
                               e, sig_inputs, N_SIG, /*n_vars=*/1);

        /* Sign-only: argmin Hamming distance over bank tiles.
         * Tie-break: lowest class index wins (deterministic). */
        int best_sign = 0;
        int32_t best_d_sign = INT32_MAX;
        for (int c = 0; c < n_anchors; c++) {
            int32_t d = m4t_popcount_dist(
                test_sign,
                bank_sign + (size_t)c * sig_packed_bytes,
                mask_all, sig_packed_bytes);
            if (d < best_d_sign) { best_d_sign = d; best_sign = c; }
        }
        if (best_sign == true_class) {
            correct_sign++;
            per_class_sign_ok[true_class]++;
        }

        /* Dual: confidence-weighted distance. */
        int best_dual = 0;
        int32_t best_d_dual = INT32_MAX;
        for (int c = 0; c < n_anchors; c++) {
            int32_t d = m4t_route_confidence_weighted_dist(
                test_dual_trit,
                test_dual_conf,
                bank_dual_trit + (size_t)c * sig_packed_bytes,
                bank_dual_conf + (size_t)c * conf_bytes,
                mask_all,
                N_SIG);
            if (d < best_d_dual) { best_d_dual = d; best_dual = c; }
        }
        if (best_dual == true_class) {
            correct_dual++;
            per_class_dual_ok[true_class]++;
        }
    }

    double acc_sign = (double)correct_sign / (double)n_test * 100.0;
    double acc_dual = (double)correct_dual / (double)n_test * 100.0;
    double gap_pp   = acc_dual - acc_sign;

    printf("=== Routing accuracy on held-out test set (n=%d) ===\n", n_test);
    printf("  sign-only : %d / %d  =  %5.2f%%\n", correct_sign, n_test, acc_sign);
    printf("  R1 dual   : %d / %d  =  %5.2f%%\n", correct_dual, n_test, acc_dual);
    printf("  gap (dual − sign-only) : %+5.2f pp\n\n", gap_pp);

    /* Honest-baseline: random class assignment expected accuracy. */
    double expected_random = 100.0 / (double)n_anchors;
    printf("  baseline (random pick from %d classes) ≈ %5.3f%%\n",
        n_anchors, expected_random);

    /* Per-class accuracy distribution: count classes where each rule is
     * better, equal, or worse. */
    int n_classes_with_test = 0;
    int dual_better = 0, dual_equal = 0, dual_worse = 0;
    for (int c = 0; c < n_anchors; c++) {
        if (per_class_total[c] == 0) continue;
        n_classes_with_test++;
        if (per_class_dual_ok[c] > per_class_sign_ok[c]) dual_better++;
        else if (per_class_dual_ok[c] == per_class_sign_ok[c]) dual_equal++;
        else dual_worse++;
    }
    printf("\n=== Per-class breakdown over %d classes (with held-out exprs) ===\n",
        n_classes_with_test);
    printf("  classes where dual > sign-only : %d\n", dual_better);
    printf("  classes where dual = sign-only : %d\n", dual_equal);
    printf("  classes where dual < sign-only : %d\n", dual_worse);

    printf("\n=== Pre-committed verdict gate (gap >= +%d pp for dual to win) ===\n",
        VERDICT_GAP_PP);
    if (gap_pp >= (double)VERDICT_GAP_PP) {
        printf("  VERDICT SHIFT: R1 dual rule beats sign-only on F-G5.\n");
        printf("  Refer this back to journal/r1_falsify_closeout.md status.\n");
    } else if (gap_pp <= -(double)VERDICT_GAP_PP) {
        printf("  R1 STILL FALSIFIED (now 5-axis): dual UNDERPERFORMS sign-only by >=%d pp.\n",
            VERDICT_GAP_PP);
    } else {
        printf("  R1 STILL FALSIFIED (now 5-axis): dual within ±%d pp of sign-only.\n",
            VERDICT_GAP_PP);
        printf("  Routing accuracy axis confirms: dual rule provides no measurable\n"
               "  routing advantage over sign-only under behavioral-equivalence\n"
               "  ground truth.\n");
    }

    /* Cleanup. */
    free(mask_all);
    free(per_class_total); free(per_class_sign_ok); free(per_class_dual_ok);
    free(bank_sign); free(bank_dual_trit); free(bank_dual_conf);
    free(anchors); free(tests);
    free(recs); free(fp_table);
    for (int i = 0; i < K_TOTAL; i++) expr_free(exprs[i]);
    free(exprs);

    return 0;
}
