/*
 * expr.c — implementation of gesh/src/expr.h.
 */

#include "expr.h"

#include <assert.h>
#include <stdlib.h>

static expr_t* alloc_node(expr_op_t op) {
    expr_t* e = (expr_t*)calloc(1, sizeof(expr_t));
    assert(e);
    e->op = op;
    return e;
}

expr_t* expr_var(int idx) {
    assert(idx >= 0);
    expr_t* e = alloc_node(EXPR_VAR);
    e->var_idx = idx;
    return e;
}

expr_t* expr_const(m4t_mtfp_t val) {
    expr_t* e = alloc_node(EXPR_CONST);
    e->const_val = val;
    return e;
}

expr_t* expr_neg(expr_t* a) {
    assert(a);
    expr_t* e = alloc_node(EXPR_NEG);
    e->a = a;
    return e;
}

static expr_t* binary(expr_op_t op, expr_t* a, expr_t* b) {
    assert(a && b);
    expr_t* e = alloc_node(op);
    e->a = a;
    e->b = b;
    return e;
}

expr_t* expr_add(expr_t* a, expr_t* b) { return binary(EXPR_ADD, a, b); }
expr_t* expr_sub(expr_t* a, expr_t* b) { return binary(EXPR_SUB, a, b); }
expr_t* expr_mul(expr_t* a, expr_t* b) { return binary(EXPR_MUL, a, b); }
expr_t* expr_max(expr_t* a, expr_t* b) { return binary(EXPR_MAX, a, b); }
expr_t* expr_min(expr_t* a, expr_t* b) { return binary(EXPR_MIN, a, b); }

void expr_free(expr_t* e) {
    if (!e) return;
    expr_free(e->a);
    expr_free(e->b);
    free(e);
}

int64_t expr_eval(const expr_t* e, const m4t_mtfp_t* inputs, int n_vars) {
    assert(e);
    switch (e->op) {
    case EXPR_VAR: {
        assert(e->var_idx >= 0 && e->var_idx < n_vars);
        assert(inputs);
        return (int64_t)inputs[e->var_idx];
    }
    case EXPR_CONST:
        return (int64_t)e->const_val;
    case EXPR_NEG:
        return -expr_eval(e->a, inputs, n_vars);
    case EXPR_ADD:
        return expr_eval(e->a, inputs, n_vars) + expr_eval(e->b, inputs, n_vars);
    case EXPR_SUB:
        return expr_eval(e->a, inputs, n_vars) - expr_eval(e->b, inputs, n_vars);
    case EXPR_MUL: {
        int64_t va = expr_eval(e->a, inputs, n_vars);
        int64_t vb = expr_eval(e->b, inputs, n_vars);
        return va * vb;
    }
    case EXPR_MAX: {
        int64_t va = expr_eval(e->a, inputs, n_vars);
        int64_t vb = expr_eval(e->b, inputs, n_vars);
        return (va > vb) ? va : vb;
    }
    case EXPR_MIN: {
        int64_t va = expr_eval(e->a, inputs, n_vars);
        int64_t vb = expr_eval(e->b, inputs, n_vars);
        return (va < vb) ? va : vb;
    }
    }
    /* Unreachable; the enum is closed. Silence compiler warnings. */
    assert(0 && "unknown expr_op_t");
    return 0;
}
