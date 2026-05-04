# Nodes of Interest: R1 Signature Rule Design

## Node 1: Tau choice is load-bearing
Both paths need tau values. Fixed tau can't fit expressions spanning 1.5+ orders of magnitude (linear ~30, quadratic ~900). Per-expression tau scales to that expression's own magnitude range.
**Why it matters:** the rule's information capture depends on tau matching expression scale. Wrong tau → either everything is "strong" (no discrimination within an expression) or everything is "weak" (no use of the strong band).

## Node 2: Per-expression tau is actually the right scale-invariance
With per-expression tau, `x` and `2*x` (different absolute scale, same internal structure) get matching signatures by construction. `x` and `x*x` (different internal structures) get distinct signatures. This is the kind of scale-invariance vision claim #2 wants.
**Why it matters:** scale-invariance was previously achieved by accident (sign-only collapsed all magnitudes). Now it's achieved by design at the right granularity.

## Node 3: Path A captures monotone-magnitude information; Path B doesn't
`x`, `x³`, `x⁵` are all sign-equivalent (sign(x) at every input). Path A's magnitude bands could distinguish them (linear vs cubic vs quintic profiles). Path B's wildcard rule can't (no near-zero cells differ between them).
**Why it matters:** concern 7 (compose-equivalence collapsing precision-distinct expressions) is partly addressed by Path A's monotone-magnitude capture. Path B doesn't address it.

## Node 4: Path B eats information at the most informative positions (T1)
Wildcard at near-zero marks small values as "uncertain." But small values include sign-flip points like `x²-1` at x=0 (=-1, small but the *only* negative value in the signature). Marking it wildcard loses the discriminating signal.
**Why it matters:** Path B's third-state semantic ("uncertain") is the wrong direction for our use. We want the third state to *carry* information.

## Node 5: Storage cost (T2)
Path A: trit + conf bit per position → doubles signature storage. At sig_dim=16 that's 4 bytes + 2 bytes = 6 bytes per tile. Tiny absolute, but compounds at scale.
Path B: single-trit per position → 4 bytes per tile.
**Why it matters:** at toy scale, irrelevant. At 1000-class banks (R2 territory), Path A's 6KB vs Path B's 4KB still tiny. Storage isn't the constraint.

## Node 6: Distance kernel cost (T3)
`m4t_popcount_dist`: pure SIMD popcount, fast.
`m4t_route_confidence_weighted_dist`: per-position scan with branches, slower.
`m4t_route_wildcard_dist`: popcount + bit-correction, fast.
**Why it matters:** Path A's confidence-weighted distance is the slowest of the three. At toy scale, irrelevant. At scale, could be 5-10x slower than popcount per probe. Worth measuring if scaling becomes an actual problem.

## Node 7: R1-B gate (information gain ≥30%) needs care
The rule must produce *different* signatures from sign-only on at least 30% of random expressions. Path A with per-expression tau will easily clear this for expressions with non-uniform magnitudes (quadratics, products). For purely sign-monotone expressions (`x`, `-x`, `x+5`), Path A's bands may or may not differ from sign-only depending on tau choice.
**Why it matters:** if tau_weak is too close to tau_strong, all cells land in the same band and Path A degenerates to sign-only. The choice tau_weak = max/4, tau_strong = max/2 leaves room for the bands to actually differ.

## Node 8: Edge case — expressions that evaluate to all-zero
`x - x`, `0 * x`, etc. max_abs = 0. tau values = 0. Dual-threshold at tau=0 collapses to sign-only behavior, which produces all-zero signature for all-zero values. All these expressions merge into one class (the "always-zero" class). Mathematically correct.
**Why it matters:** edge case is well-defined; no special handling needed.

## Node 9: Bank type extension
`expr_bank_t.base.tiles_packed` currently holds packed-trit signatures. Path A needs an additional conf_bits array. Either extend `expr_bank_t` directly or create a new `expr_bank_dual_t`. The existing equivalence-class detection (memcmp on signatures) extends naturally to memcmp on (trit_sig + conf_bits) concatenated.
**Why it matters:** P0-3's bank framing survives this change; only the constructor and routing function need updating. The equivalence-class machinery doesn't change.

## Node 10: Backward-compat gate (R1-A) tests preservation, not improvement
R1-A says ≥70% of original 30 subagent probes still match under the new rule. The gate accepts a *small drop* from the current 96.7% — that's the price of richer discrimination if the new rule splits classes the old rule merged. Below 70% means the new rule has broken what worked, not just made it richer.
**Why it matters:** the gate gives the new rule room to be different without requiring it to be uniformly better. Acceptable tradeoff: lose some easy probes if it gains substrate-distinctness and addresses concerns 2/3/7.

---

## Tension Summary

- **T1 (Node 4):** Path B's wildcard at near-zero eats information at sign-flip points. Resolved by picking Path A (uses third state for magnitude bands, not for "uncertainty").
- **T2 (Node 5):** Path A's storage cost is 1.5x Path B's. Resolved by noting the absolute cost is tiny at any scale we care about.
- **T3 (Node 6):** Path A's distance kernel is slower than popcount. Real but not relevant at toy scale; revisit if R2 scaling shows it's a bottleneck.
- **T4 (cross-Node):** Per-expression tau adds a setup-time evaluator pass. Resolved by noting setup-time cost is acceptable; queries are single-pass either way.

## Dependencies

- **D1:** tau choice (Node 1) depends on expression scale, which is per-expression. Resolved by per-expression tau (Node 2).
- **D2:** R1-B gate's bite (Node 7) depends on tau being non-trivial. Resolved by tau_weak = max/4, tau_strong = max/2 leaving real room between bands.
- **D3:** Bank type changes (Node 9) cascade to the constructor and routing function but NOT the equivalence-class machinery.
