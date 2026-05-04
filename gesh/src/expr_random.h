/*
 * expr_random.h — random expression-tree and random test-input generators.
 *
 * For the expression-routing remediation cycle (vision claim #2 P0 red-team
 * remediation). Used to address H1 (test-input set choice) and H2 (bank
 * candidate curation bias) from journal/expression_routing_redteam.md.
 *
 * Deterministic given seed (xorshift32). Same seed = same output, so
 * multi-seed runs are reproducible.
 */

#ifndef GESH_EXPR_RANDOM_H
#define GESH_EXPR_RANDOM_H

#include "expr.h"
#include "m4t_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generate a random expression tree of bounded depth.
 *
 * n_vars: 1 for arity-1, 2 for arity-2. Leaf-vars sampled uniformly from
 *         {0, ..., n_vars-1}.
 * max_depth: 0 → returns a leaf (var or small const). >0 → returns an
 *            operator (neg, add, sub, mul, max, min) with random subtrees
 *            of depth <= max_depth-1.
 * state: xorshift32 state, advanced. Seed it with any uint32_t.
 *
 * Caller owns the returned tree; expr_free it.
 */
expr_t* expr_random(uint32_t* state, int n_vars, int max_depth);

/* Generate n random scalar inputs in the range [-30, +30] (matching the
 * scale of the curated test-input set in expr_routing_probe.c).
 *
 * out: caller-allocated, [n] elements.
 */
void inputs_random_arity1(m4t_mtfp_t* out, int n, uint32_t* state);

/* Generate n_pairs random (x, y) pairs in the range [-30, +30] each.
 * out: caller-allocated, [n_pairs * 2] elements in row-major order
 * (out[2i] = x_i, out[2i+1] = y_i).
 */
void inputs_random_arity2(m4t_mtfp_t* out, int n_pairs, uint32_t* state);

/* Generate inputs spanning a specified scale band. Used for the multi-input
 * set sweep (H1). band ∈ {0=tight {-3..3}, 1=mid {-30..30}, 2=wide-positive
 * {1..1000}, 3=powers-of-10 {-1000,-100,-10,-1,0,1,10,100,1000,...}}.
 *
 * For arity-1: produces n inputs.
 * For arity-2: produces n input pairs (out is [n*2]).
 * For powers-of-10 (band 3) the output is fixed-pattern and ignores state. */
void inputs_band(m4t_mtfp_t* out, int n, int n_vars, int band, uint32_t* state);

#ifdef __cplusplus
}
#endif

#endif /* GESH_EXPR_RANDOM_H */
