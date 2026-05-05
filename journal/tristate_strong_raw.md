# RAW: strong-claim test setup

Stream of consciousness on what the user means by "set up the STRONG claim test" and what the cycle should look like.

## What the strong claim is, restated

Vision claim 3 (strong form): **base-3 carries information that base-2 collapses, in a way that's structurally cheaper or more accurate than base-2's workaround (e.g., binary value + separate masking machinery).**

The audit (post-fix) established the WEAK form: third state is load-bearing in our substrate's algorithms. L1, L2, L6 unambiguously load-bearing per both gates; L4 mixed; L3 sparsity-dominated.

The strong form is comparative. We need a base-2 reference implementation of one or more substrate layers, side-by-side measured against the base-3 implementation. The verdict: does base-3 outperform base-2-with-workaround on at least one axis (density, precision, kernel cost) without losing on the others?

## Which layer to compare?

The audit ranked layers by Gate II load-bearing-ness:
- L3 (mean cos 0.46) — most load-bearing. But sparsity-dominated; base-2-with-sparsity-flag is most likely to match.
- L1 (mean cos 0.49) — uniformly load-bearing. Direct comparison: ternary weights vs base-2 weights with mask.
- L2 (mean cos 0.62) — uniformly load-bearing.
- L6 (mean cos 0.74) — load-bearing in dense regime, partial in sparse.
- L4 (mean cos 0.94) — least load-bearing. Tightening opportunity.

L1 (weights) is the cleanest strong-claim test: most direct comparison, most defensible base-2 alternative, and it's where most "ternary network" papers have made claims about base-3's value.

The L1 test asks: **Is the substrate's packed-ternary weight representation + ternary matmul kernel structurally better than a base-2 alternative (signed binary + sparsity mask) on the same workload?**

## Three candidate base-2 alternatives

For weight storage with the same {-1, 0, +1} value range:

**B2-A: Signed binary only.** 1 bit per cell. No third state — collapse zeros to either +1 or -1 (or to a fixed value). Information is LOST: the algorithm can no longer distinguish "this weight is off" from "this weight is +1."
- Density: 1 bit/cell (DENSER than base-3)
- Functionality: LOSES the third state entirely; sparsity is impossible.
- Not a fair comparison if the substrate genuinely uses the third state for sparsity. Functional equivalence FAILS.

**B2-B: Sign + sparsity bit.** 2 bits per cell: 1 sign bit + 1 mask bit. {sparse=1, sign=anything} encodes 0; {sparse=0, sign=±} encodes ±1.
- Density: 2 bits/cell (EQUAL to packed ternary)
- Functionality: matches base-3 exactly.
- Kernel: load sign byte, load mask byte, conditionally apply mask. Extra ops vs raw SDOT.
- **THIS is the canonical "base-2 + workaround" alternative.**

**B2-C: Sparse-storage (CSR-like).** Store only non-zero weights with their indices. Highly variable density — depends on sparsity. Can be MORE dense than 2 bits/cell when zero-fraction is high (e.g., 60% zero → 40% × 9 bits ≈ 3.6 bits/cell at K=256, vs 2 bits/cell ternary).
- Density: variable; depends on sparsity
- Functionality: matches base-3 exactly.
- Kernel: gather operations, no SDOT-friendly layout.
- Probably WORSE than ternary on kernel cost in most regimes.

The cleanest strong-claim test uses **B2-B (sign + sparsity)** as the comparison. Same density (2 bits/cell), same functional range, but the third state is encoded as an explicit overlay rather than a native value.

## Comparison axes

Per memory `feedback_substrate_claim_scope.md`: density, precision, kernel cost. Pre-commit thresholds.

**Density:**
- Base-3: 2 bits/cell (current packing)
- B2-B: 2 bits/cell
- TIE expected. Either wins iff one packs strictly tighter than the other on the same workload.

**Precision (algorithmic equivalence):**
- Both encode {-1, 0, +1}. Same matmul produces same output.
- TIE expected by construction. Either wins iff one produces measurably better output (which would require one to be lossy, contradiction).
- Pre-commit gate: bit-exact equivalence on the same workload. PASS = both produce identical Y.

**Kernel cost:**
- Base-3 substrate (SDOT path): TBL decode + signed int8 multiply via SDOT, 16 cells per cycle.
- B2-B kernel: load sign + load mask + conditional zero (via vbslq or vbicq) + signed multiply. Extra ops per 16-cell block.
- Pre-commit gate: count NEON instructions per 16-cell output block under disassembly.
- Base-3 wins iff it uses fewer NEON ops without other regressions.

**Honest expectation:** base-3 likely wins on kernel cost in DENSE weights (no skip benefit available), and might LOSE in sparse weights (where B2-B could skip masked cells entirely if the kernel is sufficiently clever). The verdict could be regime-dependent.

## Construction concerns

**1. The B2-B kernel must be implementable.** It's not in the substrate. Need to write it. Some scope:
- A single matmul function: takes packed sign bytes + packed mask bytes + ternary activations + output buffer.
- Bit-exact reference for verification.
- NEON-only (per project rule); no scalar fallback in production paths.
- Doesn't have to live in libm4t — can be a standalone reference in `audit/` or a new dir.

**2. Measurement methodology must be fair.** Both kernels should be:
- Compiled with same flags (-O3, LTO).
- Profiled on the same workload (same K, same zero-fraction, same activations).
- Disassembled to count NEON instructions per inner block.

**3. Wall-clock benchmarking is lower priority** (per project rule "function over speed; no scalar in production"). The CYCLE COUNT or NEON-OP COUNT is the deliverable. Wall-clock is informational at best; the project gates on op count, not throughput.

**4. The measurement should match the audit's workload** so the strong-claim verdict is comparable to the weak-claim verdict. Same M, K, N; same configs; same zero-fractions.

**5. Sparsity is a confound.** In SDOT, every cell contributes whether it's zero or not (zero × X = 0 in int8 arithmetic, but the multiply still happens). In B2-B, a clever kernel could skip masked cells. To compare fairly, both kernels should EITHER process every cell (fair on dense) OR exploit sparsity (fair on sparse). Need to pre-commit which mode of comparison.

## What "wins" means for the strong claim

Per `feedback_substrate_claim_scope.md`: base-3 wins iff it outperforms B2-B on at least one axis without losing on the others.

Possible outcomes:
- **Base-3 wins density tie + precision tie + kernel cost.** Strong claim SUPPORTED.
- **Base-3 wins density and ties others.** Strong claim SUPPORTED (density advantage exists).
- **Base-3 ties on all axes.** Strong claim NOT SUPPORTED (no distinctive advantage).
- **Base-3 loses on any axis without compensating wins elsewhere.** Strong claim FALSIFIED.

The most informative outcome is probably "base-3 wins kernel cost in dense regime, loses in highly sparse regime." That gives a regime-dependent verdict and tells us where base-3's value actually lives.

## Open questions

- Should the strong-claim test include B2-A (1-bit) as a third arm to test the "matters that we DON'T collapse" question? B2-A tests "what happens if you collapse the third state entirely." Answer: lose information. Confirms the audit's L1 finding from a different angle. Worth including?
- Should the cycle build the B2-B kernel to PRODUCTION quality or PROTOTYPE quality? Production means full NEON, error checking, proper API. Prototype means "just enough to measure." For a science cycle, prototype is fine; production is overkill.
- Should the cycle use the AUDIT'S workload or a different one? Reusing the audit's workload makes the strong-claim verdict directly comparable. New workload could surface different regimes.
- How does cross-exp accum (L5) factor in? Not at all for L1 strong claim; L5 is a different layer. Strong claim for L5 would need a different reference impl, deferred.

## Where I'd land

**Cycle scope:** strong-claim test on L1 (weights). Build B2-B (sign + mask) reference matmul kernel. Same workload as audit. Measurements:
- Density: bits/cell at the storage level — both 2/cell (tie expected).
- Precision: bit-exact output equivalence on the same workload — gate pass = identical Y.
- Kernel cost: NEON instruction count per 16-cell inner block via disassembly.

Pre-commit gate values:
- Density: tie iff equal-bits-per-cell.
- Precision: pass iff bit-exact Y match.
- Kernel cost: base-3 wins iff fewer NEON ops; loses iff more.

Verdict logic:
- Win on cost + tie elsewhere → STRONG CLAIM SUPPORTED.
- Tie on cost + tie elsewhere → STRONG CLAIM NOT SUPPORTED.
- Loss on cost + tie elsewhere → STRONG CLAIM FALSIFIED on this layer.
- Mixed (regime-dependent) → REPORTED HONESTLY; partial support.

**Output:** `journal/tristate_strong_closeout.md` with verdict + per-axis evidence + regime-dependent notes. New code: `audit/b2b_matmul.c` (B2-B reference kernel) + integration into audit harness.

## Concerns

1. **Selection bias on the layer.** Picking L1 because it's "most load-bearing per audit" is defensible, but a different layer might give a different verdict. The cycle's verdict is L1-specific.
2. **B2-B kernel design choices affect the verdict.** A poorly-written B2-B will lose on cost; a cleverly-optimized one might win. The cycle should pick a "reasonably tight but not heroically optimized" B2-B that represents how a skilled implementer would write this.
3. **NEON op count is a proxy for cycles, not a perfect measure.** Some ops are cheaper than others; pipelining matters; cache effects exist. The cycle should report op count AND wall-clock for cross-check; gate on op count primarily.
4. **The "no scalar in production" rule.** B2-B kernel needs a NEON path. Scalar reference for testing is fine. Production-style B2-B should be NEON.
5. **The strong claim is asymmetric: base-2 has decades of optimization; base-3 has a few months.** A "tie" on kernel cost might still favor base-3 on a per-engineering-effort basis. Hard to quantify; mention as caveat.

## What feels right

This is a build cycle (write the B2-B kernel) + science cycle (run the comparison). Output is a verdict on the strong claim FOR L1, plus reusable B2-B infrastructure that future cycles can extend to other layers.

Estimated complexity: ~200-400 lines of NEW C for the B2-B kernel + harness, plus integration into the existing audit framework. Manageable in one cycle.
