/*
 * expr_signature.h — turn an expression into a packed-trit signature by
 * evaluating it on a fixed test-input set and ternarizing the outputs.
 *
 * Closes vision-claim-#2 P0-1. The rule:
 *
 *   For each test_input[i]:
 *     value[i] = expr_eval(expr, test_input[i], n_vars)
 *   Then ternarize via the substrate kernel:
 *     signature = m4t_route_threshold_extract(values, tau=0)
 *
 * Two expressions that behave identically on the test-input set produce
 * byte-equal signatures. Routing convergence on the same address is the
 * operational definition of "these expressions implement the same function."
 *
 * Substrate-discipline: all ternarization through m4t_route_threshold_extract.
 * No open-coded sign step.
 *
 * Test-input layout:
 *   inputs is [n_test_inputs × n_vars] in row-major order.
 *   test_input[i] is the i-th row, holding n_vars values for that probe point.
 *   For arity 1, n_vars=1 and each row is one scalar input.
 *   For arity 2, n_vars=2 and each row is one (x, y) pair.
 *
 * Output: out_packed_sig is M4T_TRIT_PACKED_BYTES(n_test_inputs) bytes;
 * sig_dim equals n_test_inputs by construction (one trit per test input).
 */

#ifndef GESH_EXPR_SIGNATURE_H
#define GESH_EXPR_SIGNATURE_H

#include "expr.h"
#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the signature of expr by evaluating on the test-input set.
 *
 * Preconditions:
 *   out_packed_sig: at least M4T_TRIT_PACKED_BYTES(n_test_inputs) bytes
 *   expr non-NULL
 *   test_inputs non-NULL when n_test_inputs > 0
 *   n_test_inputs > 0
 *   n_vars matches the expression's arity
 */
void expr_to_signature(
    uint8_t* out_packed_sig,
    const expr_t* expr,
    const m4t_mtfp_t* test_inputs,    /* [n_test_inputs × n_vars] */
    int n_test_inputs,
    int n_vars);

/* R1 — per-expression-tau dual-threshold signature.
 *
 * Closes red-team concerns 2, 3, 7, partial 8 from
 * journal/expression_routing_redteam.md per the LMM cycle in
 * journal/r1_signature_rule_synthesize.md.
 *
 * The rule:
 *   1. Evaluate expr at all n_test_inputs → int64 values[n_test].
 *   2. Compute max_abs = max over i of |values[i]|.
 *   3. Set tau_weak = max_abs / 4, tau_strong = max_abs / 2.
 *   4. Call m4t_route_threshold_extract_dual to produce the (trit, conf-bit)
 *      pair, where each cell is in one of 5 states:
 *      strong-neg, weak-neg, zero, weak-pos, strong-pos.
 *
 * The signature reflects the expression's *internal* magnitude structure,
 * not its absolute scale. `x` and `2*x` produce identical signatures (same
 * shape, different scale); `x` and `x²` produce different signatures
 * (different shapes).
 *
 * Substrate-discipline: ternarization + band classification entirely inside
 * m4t_route_threshold_extract_dual. No open-coded sign or band step.
 *
 * Output: out_trit_packed is M4T_TRIT_PACKED_BYTES(n_test_inputs) bytes;
 * out_conf_bits is (n_test_inputs + 7) / 8 bytes.
 *
 * Edge case: max_abs == 0 (e.g., `x - x`). tau_weak = tau_strong = 0;
 * dual-threshold collapses to sign-only behavior; all-zero values produce
 * all-zero trit signature with no confidence bits set. Mathematically
 * correct.
 */
void expr_to_signature_dual(
    uint8_t* out_trit_packed,         /* [M4T_TRIT_PACKED_BYTES(n_test)] */
    uint8_t* out_conf_bits,           /* [(n_test + 7) / 8] */
    const expr_t* expr,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars);

#ifdef __cplusplus
}
#endif

#endif /* GESH_EXPR_SIGNATURE_H */
