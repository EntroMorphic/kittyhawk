/*
 * expr_signature.c — implementation of expr_to_signature.
 *
 * Substrate-discipline: ternarization via m4t_route_threshold_extract;
 * no open-coded sign step.
 */

#include "expr_signature.h"
#include "m4t_route.h"

#include <assert.h>
#include <stdlib.h>

void expr_to_signature(
    uint8_t* out_packed_sig,
    const expr_t* expr,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars)
{
    assert(out_packed_sig && expr);
    assert(n_test_inputs > 0);
    assert(test_inputs);

    /* Evaluate at each test input. Collect into an int64 buffer. */
    int64_t* values = (int64_t*)malloc((size_t)n_test_inputs * sizeof(int64_t));
    assert(values);

    for (int i = 0; i < n_test_inputs; i++) {
        const m4t_mtfp_t* row = test_inputs + (size_t)i * (size_t)n_vars;
        values[i] = expr_eval(expr, row, n_vars);
    }

    /* Ternarize through the substrate kernel at tau=0 (sign-extract).
     * Sanctioned input class: integer-valued inputs with non-trivial
     * probability of exact zero — exactly our case. */
    m4t_route_threshold_extract(out_packed_sig, values, /*tau=*/0,
                                  n_test_inputs);

    free(values);
}

void expr_to_signature_dual(
    uint8_t* out_trit_packed,
    uint8_t* out_conf_bits,
    const expr_t* expr,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars)
{
    assert(out_trit_packed && out_conf_bits && expr);
    assert(n_test_inputs > 0);
    assert(test_inputs);

    /* Evaluate at each test input. */
    int64_t* values = (int64_t*)malloc((size_t)n_test_inputs * sizeof(int64_t));
    assert(values);

    for (int i = 0; i < n_test_inputs; i++) {
        const m4t_mtfp_t* row = test_inputs + (size_t)i * (size_t)n_vars;
        values[i] = expr_eval(expr, row, n_vars);
    }

    /* Per-expression tau: scaled to this expression's max absolute output.
     * Edge case: max_abs == 0 → tau values = 0; kernel handles tau=0 as
     * sign-only (all zero values → all zero trits, no conf bits set). */
    int64_t max_abs = 0;
    for (int i = 0; i < n_test_inputs; i++) {
        int64_t a = values[i] < 0 ? -values[i] : values[i];
        if (a > max_abs) max_abs = a;
    }
    int64_t tau_weak   = max_abs / 4;
    int64_t tau_strong = max_abs / 2;

    /* Substrate kernel does both the trit-state classification and the
     * confidence-bit extraction in one pass. */
    m4t_route_threshold_extract_dual(
        out_trit_packed, out_conf_bits,
        values, tau_weak, tau_strong,
        n_test_inputs);

    free(values);
}
