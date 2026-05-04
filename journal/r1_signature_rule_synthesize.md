# Synthesis: R1 Signature Rule (per-expression-tau dual-threshold)

## Architecture

Replace the sign-only ternarization (current rule) with a 5-state encoding that uses the substrate's `m4t_route_threshold_extract_dual` kernel paired with `m4t_route_confidence_weighted_dist`. Tau values are computed per-expression as fractions of the expression's own maximum absolute output on the test inputs. Bank type extends to carry both the trit signature and the parallel confidence bitmap per tile. Equivalence-class machinery unchanged.

## Key Decisions

**D1: Per-expression tau scaled to expression's max absolute output.** [from REFLECT core insight]
For each expression: compute max_abs across test-input evaluations. Set tau_weak = max_abs / 4, tau_strong = max_abs / 2. The signature reflects the expression's internal magnitude structure, not its absolute scale.

**D2: 5-state encoding via `m4t_route_threshold_extract_dual`.** [Path A from REFLECT]
States: strong-neg, weak-neg, zero, weak-pos, strong-pos. Stored as (packed-trit, conf-bits) pair per signature. Distance via `m4t_route_confidence_weighted_dist`.

**D3: Bank type extends, doesn't replace.** [from REFLECT, A4 challenge]
Add a parallel `conf_bits` buffer per tile. The existing memcmp-based equivalence detection extends to memcmp over (trit + conf) concatenated. Bank constructor and routing function update; bank framing doesn't.

**D4: Tau-=0 edge case is well-defined.** [Node 8 in NODES]
Expressions with max_abs == 0 (e.g., `x - x`) get tau_weak = tau_strong = 0. The kernel handles tau=0 (sign-only behavior); all-zero values produce all-zero trit signature with no confidence bits set. All-zero expressions merge into one class, mathematically correct.

## Implementation Spec

### Signature derivation function

```c
void expr_to_signature_dual(
    uint8_t* out_trit_packed,        /* M4T_TRIT_PACKED_BYTES(n_test) bytes */
    uint8_t* out_conf_bits,           /* (n_test + 7) / 8 bytes */
    const expr_t* expr,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars);
```

Algorithm:
1. Evaluate expr at each of n_test inputs → int64 values[n_test].
2. Compute max_abs = max over i of |values[i]|.
3. tau_weak = max_abs / 4, tau_strong = max_abs / 2.
4. Call `m4t_route_threshold_extract_dual(out_trit_packed, out_conf_bits, values, tau_weak, tau_strong, n_test_inputs)`.

Substrate-discipline: ternarization through the kernel; no open-coded sign step or band classification.

### Bank type extension

```c
typedef struct {
    gesh_bank_t base;                 /* unchanged */
    uint8_t* conf_bits_per_tile;      /* [n_tiles × ceil(sig_dim/8)] */
    int n_candidates;
    int* candidate_to_class;
    int n_vars;
    int n_test_inputs;
} expr_bank_dual_t;
```

### Bank constructor

```c
void expr_bank_dual_build(
    expr_bank_dual_t* bank,
    const expr_t* const* candidates,
    int n_candidates,
    const m4t_mtfp_t* test_inputs,
    int n_test_inputs,
    int n_vars);
```

Algorithm: same shape as `expr_bank_build`, but:
- Computes both trit_sig and conf_bits per candidate via `expr_to_signature_dual`.
- Equivalence-class detection: memcmp on (trit_sig || conf_bits) concatenation. Two candidates with byte-identical trit AND byte-identical conf go to same class.
- Stores both buffers per representative tile.

### Routing function

```c
int route_signature_dual(
    const uint8_t* query_trit_packed,
    const uint8_t* query_conf_bits,
    const expr_bank_dual_t* bank,
    const uint8_t* mask);
```

Iterates tiles, calls `m4t_route_confidence_weighted_dist` per tile, returns nearest tile index.

### Verification probe

`gesh/bench/expr_routing_r1.c` — runs all three R1 gates against the curated banks (arity-1, arity-2):

- **R1-A (backward-compat):** route the original 30 subagent probes through the new rule. PASS if ≥70% match expected class.
- **R1-B (information gain):** generate 100 random expressions per arity. Compute both old-rule and new-rule signatures. PASS if ≥30% have *different equivalence-class assignments* under the new rule vs old.
- **R1-C (substrate-kernel use):** verifiable via grep + code review — the new rule's call path includes `m4t_route_threshold_extract_dual` AND `m4t_route_confidence_weighted_dist`.

## Pre-committed gates (carried forward from PLAN_EXPRESSION_ROUTING_R2.md)

- **R1-A** ≥70% subagent-probe match under new rule
- **R1-B** ≥30% of random expressions get different equivalence-class assignment vs sign-only
- **R1-C** new rule includes ≥1 previously-unused substrate kernel

R1 PASS = all three. WEAK = R1-A in [50%, 70%]. FAIL = R1-A < 50% OR R1-C absent OR (R1-B < 10% AND R1-A < 90%).

## Substrate-discipline notes

- Ternarization through `m4t_route_threshold_extract_dual`. No open-coded band classification.
- Distance via `m4t_route_confidence_weighted_dist`. No open-coded distance computation.
- max_abs computation in C is a simple loop over int64 values; not a substrate operation. Acceptable as it's pre-kernel scratch.
- Bank constructor's memcmp on concatenated buffers is C library, not substrate; same acceptable category as the existing memcmp on plain trit signatures.

## What this synthesis does NOT do

- Does not retire the existing sign-only `expr_to_signature` or `expr_bank_build`. Those keep working for the original probe and remediation binaries.
- Does not change the equivalence-class framing. The bank still holds one tile per class, label = representative-id.
- Does not address adversarial probes, cross-arity routing, or exp/log primitives. All P1 or later.
- Does not pre-tune tau ratios beyond max/4 and max/2. If the gates surface that these are wrong, a follow-on cycle iterates.

## Loop-back triggers

- **Back to RAW** if R1-A drops below 70% AND the failures are mathematically defensible (the new rule splits classes the old rule wrongly merged) — that's a successful split, not a regression. Reframe the gate.
- **Back to NODES** if R1-B is borderline (10-30%) AND we discover the random-expression generator's depth/op-distribution is the cause (not the rule).
- **Back to REFLECT** if R1-C is somehow not satisfied (e.g., the kernel is called but the rule's behavior is sign-only-equivalent). That would mean the rule is using the kernel but not its richer semantics.
- **Run a full new cycle** if R1-A < 50% — the new rule is fundamentally broken; revisit Path A vs Path B vs other.
