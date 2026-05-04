/*
 * expr_random.c — random expression and input generators.
 */

#include "expr_random.h"

#include <assert.h>
#include <stdlib.h>

/* Local xorshift32 (same shape as gesh_train.c's xs32). */
static uint32_t xs32(uint32_t* state) {
    uint32_t x = *state;
    if (x == 0) x = 0x12345678u;  /* L4-style scale-coupling note: 0 seed
                                   * mapped to a fixed nonzero so callers
                                   * can use 0 without hitting the all-zero
                                   * degenerate case. */
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

expr_t* expr_random(uint32_t* state, int n_vars, int max_depth) {
    assert(n_vars >= 1);
    assert(max_depth >= 0);

    if (max_depth == 0) {
        /* Leaf: 70% var, 30% const in {-3,-2,-1,0,1,2,3,5}. */
        if ((xs32(state) % 10u) < 7u) {
            int v = (int)(xs32(state) % (uint32_t)n_vars);
            return expr_var(v);
        } else {
            static const m4t_mtfp_t consts[] = {-5, -3, -2, -1, 0, 1, 2, 3, 5};
            int idx = (int)(xs32(state) % (uint32_t)(sizeof(consts)/sizeof(consts[0])));
            return expr_const(consts[idx]);
        }
    }

    /* Operator. 6 ops: neg(1) + 5 binary (add/sub/mul/max/min). */
    int op_choice = (int)(xs32(state) % 6u);
    int sub_depth = (int)(xs32(state) % (uint32_t)max_depth);  /* 0..max_depth-1 */
    if (op_choice == 0) {
        return expr_neg(expr_random(state, n_vars, sub_depth));
    }
    expr_t* a = expr_random(state, n_vars, sub_depth);
    expr_t* b = expr_random(state, n_vars, sub_depth);
    switch (op_choice) {
    case 1: return expr_add(a, b);
    case 2: return expr_sub(a, b);
    case 3: return expr_mul(a, b);
    case 4: return expr_max(a, b);
    case 5: return expr_min(a, b);
    }
    return expr_add(a, b);  /* unreachable */
}

void inputs_random_arity1(m4t_mtfp_t* out, int n, uint32_t* state) {
    assert(out && n > 0);
    /* Sample uniformly from [-30, +30] inclusive. Range matches the curated
     * test-input set's span. L4 note: callers using a different bank-scale
     * regime should use inputs_band with an explicit band. */
    for (int i = 0; i < n; i++) {
        uint32_t r = xs32(state);
        out[i] = (m4t_mtfp_t)((int32_t)(r % 61u) - 30);
    }
}

void inputs_random_arity2(m4t_mtfp_t* out, int n_pairs, uint32_t* state) {
    assert(out && n_pairs > 0);
    for (int i = 0; i < n_pairs; i++) {
        uint32_t rx = xs32(state);
        uint32_t ry = xs32(state);
        out[i*2 + 0] = (m4t_mtfp_t)((int32_t)(rx % 61u) - 30);
        out[i*2 + 1] = (m4t_mtfp_t)((int32_t)(ry % 61u) - 30);
    }
}

void inputs_band(m4t_mtfp_t* out, int n, int n_vars, int band, uint32_t* state) {
    assert(out && n > 0 && n_vars >= 1);
    int total = n * n_vars;
    switch (band) {
    case 0: {
        /* Tight: {-3..3}. */
        for (int i = 0; i < total; i++) {
            uint32_t r = xs32(state);
            out[i] = (m4t_mtfp_t)((int32_t)(r % 7u) - 3);
        }
        break;
    }
    case 1: {
        /* Mid: {-30..30} (matches curated set scale). */
        for (int i = 0; i < total; i++) {
            uint32_t r = xs32(state);
            out[i] = (m4t_mtfp_t)((int32_t)(r % 61u) - 30);
        }
        break;
    }
    case 2: {
        /* Wide-positive: {1..1000}. */
        for (int i = 0; i < total; i++) {
            uint32_t r = xs32(state);
            out[i] = (m4t_mtfp_t)((int32_t)(r % 1000u) + 1);
        }
        break;
    }
    case 3: {
        /* Powers of 10 spanning ±. Fixed pattern, state ignored. */
        static const m4t_mtfp_t pow10[] = {
            -1000, -100, -10, -1, 0, 1, 10, 100, 1000
        };
        int len = (int)(sizeof(pow10) / sizeof(pow10[0]));
        for (int i = 0; i < total; i++) {
            out[i] = pow10[i % len];
        }
        (void)state;
        break;
    }
    default:
        assert(0 && "unknown input band");
    }
}
