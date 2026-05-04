/*
 * expr_bank.h — equivalence-class bank constructor for expressions.
 *
 * Closes vision-claim-#2 P0-3.
 *
 * The bank holds one tile per equivalence class of expressions. Each tile's
 * label is the index of that class's canonical representative in the input
 * candidate list. The "vote" of the data-side forward path collapses here
 * to 1-NN nearest-tile lookup — equivalence-class lookup, not class-vote.
 *
 * The constructor:
 *   1. Computes the signature of every candidate via expr_to_signature.
 *   2. Detects equivalence classes (signatures byte-equal).
 *   3. Picks the FIRST candidate seen for each class as the representative
 *      (input order = simplicity ordering by convention; the user supplies
 *      simplest-first).
 *   4. Builds the bank tiles from representatives only (post-merger).
 *   5. Records the candidate→class map so the probe can compare against
 *      "the equivalence class of candidate X" rather than "candidate X
 *      specifically."
 *
 * Substrate-discipline: bank tiles use the same packing as gesh_bank_t.
 * No open-coded sign step (the sign step happens inside expr_to_signature
 * via m4t_route_threshold_extract).
 *
 * This bank type is single-arity: every candidate evaluates against the
 * same test-input set, with the same n_vars. Cross-arity routing is P1.
 */

#ifndef GESH_EXPR_BANK_H
#define GESH_EXPR_BANK_H

#include "expr.h"
#include "gesh_bank.h"
#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Standard tile-and-label bank, post-merger.
     *   base.n_tiles  = number of equivalence classes after merging.
     *   base.labels[k] = the candidate index of the representative for class k
     *                    (so the probe can name what class k "is").
     *   base.sig_dim  = the test-input set size used. */
    gesh_bank_t base;

    /* Per-candidate metadata. */
    int n_candidates;
    int* candidate_to_class;    /* [n_candidates] → class index in [0, base.n_tiles) */

    /* Arity metadata. */
    int n_vars;
    int n_test_inputs;
} expr_bank_t;

/* Build an equivalence-class bank from `n_candidates` candidate expressions.
 *
 * Caller responsibilities:
 *   - bank->base.tiles_packed must be allocated for at least n_candidates
 *     tiles' worth of storage (we may use fewer post-merger; the unused
 *     suffix is left untouched but accessible).
 *   - bank->base.labels must be allocated for at least n_candidates ints.
 *   - bank->base.sig_dim must equal n_test_inputs.
 *   - bank->candidate_to_class must be allocated for n_candidates ints.
 *
 * The constructor:
 *   - Sets bank->n_candidates, bank->n_vars, bank->n_test_inputs.
 *   - Sets bank->base.n_tiles to the post-merger count.
 *   - Fills bank->base.tiles_packed and bank->base.labels (first n_tiles
 *     entries valid).
 *   - Fills bank->candidate_to_class with class indices.
 *
 * Stable: same candidates in same order produce same bank.
 *
 * Preconditions:
 *   bank, candidates, test_inputs non-NULL
 *   n_candidates > 0, n_test_inputs > 0, n_vars >= 1
 *   bank->base.sig_dim == n_test_inputs
 */
void expr_bank_build(
    expr_bank_t* bank,
    const expr_t* const* candidates,
    int n_candidates,
    const m4t_mtfp_t* test_inputs,    /* [n_test_inputs × n_vars] */
    int n_test_inputs,
    int n_vars);

/* R1 — dual-signature bank using per-expression-tau dual-threshold rule.
 *
 * Per journal/r1_signature_rule_synthesize.md. Uses expr_to_signature_dual
 * to produce (trit, conf-bits) signature pairs and routes via
 * m4t_route_confidence_weighted_dist. Equivalence-class detection is
 * memcmp on the concatenation (trit_sig || conf_bits).
 *
 * Storage: per tile, both packed-trit signature AND parallel conf bitmap.
 */
typedef struct {
    gesh_bank_t base;                       /* tiles_packed = trit signatures */
    uint8_t* conf_bits_per_tile;            /* [n_tiles × ceil(sig_dim/8)] */
    int n_candidates;
    int* candidate_to_class;
    int n_vars;
    int n_test_inputs;
} expr_bank_dual_t;

/* Build a dual-signature equivalence-class bank.
 *
 * Caller responsibilities (same shape as expr_bank_build, plus):
 *   - bank->conf_bits_per_tile must be allocated for at least
 *     n_candidates × ceil(sig_dim/8) bytes.
 *
 * The constructor merges candidates with byte-equal (trit_sig + conf_bits)
 * pairs into the same class.
 */
void expr_bank_dual_build(
    expr_bank_dual_t* bank,
    const expr_t* const* candidates,
    int n_candidates,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars);

#ifdef __cplusplus
}
#endif

#endif /* GESH_EXPR_BANK_H */
