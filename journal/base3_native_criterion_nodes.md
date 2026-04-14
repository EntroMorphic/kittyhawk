---
date: 2026-04-14
phase: NODES
topic: What is the right criterion for "base-3 native," and what replaces sign_extract?
---

# Nodes

## N1. The criterion has to be mechanical, not vibes-based
The sign_extract problem hid for weeks because my gut-level "is this ternary?" check was permeable to type-system theater (three output codes, two in practice). A mechanical criterion — one that can be applied the same way by any auditor and returns the same answer — is what's needed. C-final is mechanical: collapse a state, check if utility degrades.

## N2. C-final: "collapse a state, check material degradation"
A primitive is base-3 native iff removing any one of its three states (treating it as equivalent to another, or as measure-zero) materially degrades utility on realistic inputs. This is a counterfactual test, checkable per primitive.

## N3. C-B (base-2 impossible) is too strict
Would invalidate SDOT, TBL, popcount — primitives that are base-2-implemented but expose base-3 abstractions. NORTH_STAR explicitly frames M4T's job as "surface the hidden ternary nature of the hardware" — the silicon is binary; what matters is the abstraction. C-B fails NORTH_STAR.

## N4. "Realistic inputs" is a moving target
C-final requires naming realistic inputs. For sign_extract+MTFP projections, zero is measure-zero. For sign_extract+sparse-count-reductions, zero might be common. Different consumers produce different input distributions. The criterion has to be evaluated *per consumer* or for the *intended class of consumers*.

## N5. The three replacement options are not parallel alternatives
Options 1 and 2 (`trit_extract(k)`, `threshold_extract(τ)`) are both extraction primitives — they turn ints into trits via a parameter. Option 3 (no extraction, signature IS mantissa) is structurally different — it eliminates a primitive category. Treating all three as "alternatives" conflates replacement with removal.

## N6. sign_extract is the τ=0 degenerate case of threshold_extract
If threshold_extract uses `|v| < τ → 0, v ≥ τ → +1, v ≤ -τ → -1`, then τ=0 gives `v > 0 → +1, v < 0 → -1, v == 0 → 0` — exactly sign_extract. This is structural: sign_extract was never a separate primitive; it was a special case we mistook for a general primitive.

## N7. Option 3 (no extraction) changes the routing-pass shape
Currently: `MTFP projection → extract-to-trits → packed-trit distance → decide`. Under Option 3: `MTFP4 projection → pack-mantissas → packed-trit distance → decide`. The distance step is the same; the extraction step is absent; the projection step changes output width. This is a consumer-pipeline change, not just a primitive swap.

## N8. Options 1, 2, 3 are not mutually exclusive at substrate level
The substrate could provide Option 2 (`threshold_extract`) without precluding Option 3 (MTFP4-native consumers). Option 1 is subsumed by Option 2 (with τ = 3^k). So the real choice is "Option 2 primitive, plus consumers free to use Option 3 pipeline."

## N9. `signature_update` needs a compound fix
signature_update's final step is sign_extract. Fixing requires parameterizing the final step. If we pick threshold_extract, signature_update becomes `signature_update(weights, ..., tau)`. The tau for sign-of-(col_sum - mean) is typically zero (that's the sign-flip boundary) — so signature_update with τ=0 regresses to the current behavior. That's informative: for compound ops where the natural boundary IS zero, threshold_extract with τ=0 is structurally correct (not a base-2 failure). The difference from sign_extract-over-MTFP-projections is that here the zero state occurs meaningfully often (col_sum == mean is possible on integer inputs).

## N10. Not every use of sign_extract is binary-shape
Corollary to N9: sign_extract fails C-final for MTFP-projection inputs because zero is measure-zero. It might PASS C-final for other inputs (sparse-count differences, integer-mean subtractions) where zero occurs naturally. So the question isn't "is sign_extract always wrong" but "does the consumer's input distribution realize all three states."

## N11. C-final has to be applied per primitive per consumer
N4 + N10 imply: C-final is not a primitive-only property. It depends on the input distribution from the consumer. The substrate can't know in advance; the consumer has to assert "my inputs realize all three states at this primitive" as a precondition.

## N12. This matches the throughline
"Substrate guarantees invariants; consumer names preconditions." C-final becomes a PRECONDITION a consumer names: "under my input distribution, the three states at this primitive are all realized." The substrate provides primitives that are three-state-capable in principle; the consumer warrants that its inputs activate all three states in practice.

## N13. Criterion location: substrate spec vs meta
Should C-final be written into the substrate spec (§18 or similar), or stay as meta-criterion in NORTH_STAR / LMM journals? C-final is a property of the abstraction the substrate exposes — it belongs in the spec. Spec-level statement: "a primitive is 'three-state-capable' iff C-final holds; consumers are responsible for preconditions that realize all three states."

## N14. Option 3 is still the deepest move architecturally
Even granting that threshold_extract is structurally correct, Option 3 eliminates an entire primitive category by making the signature representation coincide with the data representation. In the long run, routing pipelines that don't need an "extract-to-trits" step are simpler to reason about.

---

## Tensions

### T1. Extraction-primitive-required vs extraction-primitive-unneeded
Options 1/2 presuppose that a signature is a DIFFERENT representation from the data (needs extraction). Option 3 says they can be the same (no extraction). Both are architecturally defensible. Which is right for M4T?

### T2. Per-primitive vs per-use evaluation of "base-3 native"
N10/N11: some primitives that LOOK binary-shape (sign_extract) are actually three-state-capable under certain input distributions. The criterion is per-primitive-per-consumer, not per-primitive. That complicates audit.

### T3. Elegance vs migration cost
Option 3 is architecturally cleanest but requires consumers to re-shape their pipelines. Option 2 is smallest-delta. If we care most about "don't do this wrong twice," Option 3. If we care most about "move fast on measurable experiments," Option 2.

### T4. Criterion rigor vs criterion usability
C-final is more rigorous than C-A but still requires judgment about "material degradation." Stricter criteria (like "zero state must be strictly more than 10% of outputs") are easier to apply but arbitrary.

### T5. Sign_extract might not be wrong everywhere
N9/N10: signature_update with col_sum-vs-mean integer subtraction genuinely realizes all three states. Deleting sign_extract would force a rewrite of signature_update even though its current behavior is structurally correct for its inputs.

---

## Dependencies

- Criterion decision → scope of audit → scope of replacement
- Replacement choice (Option 2 vs 3) → consumer-pipeline impact
- Per-use evaluation (N11) → whether sign_extract stays or goes wholesale
- Spec-level formalization (N13) → future primitive reviews
