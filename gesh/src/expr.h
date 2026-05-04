/*
 * expr.h — small expression tree over the substrate's primitive ops.
 *
 * Closes vision-claim-#2 P0-2. The representation is a tree of nodes, where
 * each node is either a variable, a constant, or one of the substrate-native
 * operations (neg/add/sub/mul/max/min). exp/log are deliberately absent;
 * they belong to P1-1 (substrate primitives floor).
 *
 * Evaluator returns int64. The substrate's MTFP19 mantissa range is large
 * (|x| ≤ 581_130_733), but nested products on bounded test inputs can climb
 * fast; int64 has the headroom to evaluate any reasonable expression depth
 * on the P0 test-input range without overflow concerns. The signature step
 * (expr_signature.h) ternarizes by sign at tau=0, so absolute magnitudes
 * don't matter to the routing decision — only signs do, and signs are
 * preserved by int64 throughout.
 *
 * No float anywhere. No malloc-free scaffolding beyond explicit ownership:
 * constructors allocate; expr_free recursively frees a tree.
 */

#ifndef GESH_EXPR_H
#define GESH_EXPR_H

#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EXPR_VAR,    /* variables: x = var_idx 0, y = var_idx 1, ... */
    EXPR_CONST,  /* constant MTFP value */
    EXPR_NEG,    /* unary: -a */
    EXPR_ADD,    /* binary: a + b */
    EXPR_SUB,    /* binary: a - b */
    EXPR_MUL,    /* binary: a * b */
    EXPR_MAX,    /* binary: max(a, b) */
    EXPR_MIN,    /* binary: min(a, b) */
} expr_op_t;

typedef struct expr {
    expr_op_t       op;
    int             var_idx;     /* EXPR_VAR */
    m4t_mtfp_t      const_val;   /* EXPR_CONST */
    struct expr*    a;           /* unary/binary first operand */
    struct expr*    b;           /* binary second operand */
} expr_t;

/* Constructors — each allocates and returns ownership.
 * Inputs to binary/unary constructors are CONSUMED (the returned tree owns
 * them). expr_free on the root frees the entire subtree. */
expr_t* expr_var(int idx);
expr_t* expr_const(m4t_mtfp_t val);
expr_t* expr_neg(expr_t* a);
expr_t* expr_add(expr_t* a, expr_t* b);
expr_t* expr_sub(expr_t* a, expr_t* b);
expr_t* expr_mul(expr_t* a, expr_t* b);
expr_t* expr_max(expr_t* a, expr_t* b);
expr_t* expr_min(expr_t* a, expr_t* b);

/* Recursive free. NULL-safe. */
void expr_free(expr_t* e);

/* Evaluate the expression at the given input vector. inputs is indexed by
 * var_idx, has at least n_vars entries. Returns int64 — sign is what the
 * downstream signature step consumes; magnitude is informational.
 *
 * Preconditions:
 *   e non-NULL
 *   inputs non-NULL when expression references any EXPR_VAR
 *   var_idx of every EXPR_VAR in e is < n_vars
 *
 * Aborts via assert on out-of-range var_idx. */
int64_t expr_eval(const expr_t* e, const m4t_mtfp_t* inputs, int n_vars);

#ifdef __cplusplus
}
#endif

#endif /* GESH_EXPR_H */
