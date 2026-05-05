# NODES: strong-claim test setup

Atomic claims extracted from `tristate_strong_raw.md`.

## Claim restatement

- **N1.** Strong vision claim 3: base-3 carries information that base-2 collapses, in a way structurally cheaper or more accurate than base-2's workaround (binary value + masking machinery).
- **N2.** Strong claim is COMPARATIVE — requires a reference base-2 implementation alongside the base-3 substrate.
- **N3.** Per `feedback_substrate_claim_scope.md`: comparison axes are density (bits/cell), precision (algorithmic equivalence), kernel cost (NEON ops).

## Layer to compare

- **N4.** Audit Gate II ranking (post-fix): L3 (cos 0.46) < L1 (0.49) < L2 (0.62) < L6 (0.74) < L4 (0.94). L3 most load-bearing but sparsity-dominated. L1 most directly comparable to a base-2 alternative.
- **N5.** L1 (weights) is the cleanest strong-claim test. Most direct comparison; most defensible base-2 alt; most prior work (BitNet, ternary networks) has made strong-claim assertions about base-3 weight value.
- **N6.** L2, L6 (also LOAD-BEARING) reserved for follow-on cycles; L3 (sparsity-dominated) is least likely to show a base-3 advantage.

## Base-2 alternative selection

- **N7.** Three candidate base-2 alternatives: B2-A (1-bit signed only), B2-B (sign + sparsity bit, 2 bits/cell), B2-C (CSR-like sparse storage).
- **N8.** B2-A is functionally inequivalent — loses the third state entirely. Not a fair comparison; would only confirm "the third state matters."
- **N9.** B2-B is the canonical "base-2 + workaround" alternative: same density (2 bits/cell), same functional range, third state encoded as explicit overlay.
- **N10.** B2-C is variable-density; better in highly sparse regimes but worse on kernel structure (gather ops, no SDOT-friendly layout). Not a clean comparison.
- **N11.** **Decision: use B2-B (sign + sparsity bit) as the comparison.**

## Comparison axes (pre-committed)

- **N12.** **Density:** bits/cell. Both 2 bits/cell. TIE expected. Either wins iff one packs strictly tighter on the same workload.
- **N13.** **Precision:** bit-exact equivalence on identical inputs. PASS = identical Y output. Both encode {-1, 0, +1}, so should match by construction.
- **N14.** **Kernel cost:** NEON instruction count per inner block (16 cells per output) via disassembly. Base-3 wins iff fewer NEON ops without other regressions.

## Verdict logic (pre-committed)

- **N15.** Win on kernel cost + tie elsewhere → STRONG CLAIM SUPPORTED.
- **N16.** Tie everywhere → STRONG CLAIM NOT SUPPORTED.
- **N17.** Loss on cost + tie elsewhere → STRONG CLAIM FALSIFIED on L1.
- **N18.** Mixed (regime-dependent) → REPORTED HONESTLY; partial support.

## Construction scope

- **N19.** Need to write a B2-B reference matmul kernel. NEON-only per project rule. Production-style not required (prototype quality acceptable for a science cycle).
- **N20.** Both kernels (base-3 substrate + B2-B reference) must be measurable with: (a) bit-exact output verification, (b) disassembly for NEON op count.
- **N21.** Compile both with same flags (-O3, LTO-on per project default).
- **N22.** Reuse the audit's workload (same M, K, N; same configs; same zero-fractions) for direct verdict comparability.

## Sparsity confound

- **N23.** SDOT in the base-3 path processes every cell uniformly; zero × X = 0 happens in int8 arithmetic but the multiply still consumes a slot. No skip benefit.
- **N24.** B2-B kernel CAN skip masked cells if the inner loop is structured to. A "skip-aware" B2-B might win on sparse weights; a "uniform" B2-B (same shape as SDOT) cannot.
- **N25.** Pre-commit which mode of B2-B comparison: uniform (fair to base-3 on dense; possibly unfair to base-2 on sparse) or skip-aware (fair to base-2 on sparse; possibly unfair to base-3 in dense).
- **N26.** **Decision:** measure BOTH modes of B2-B. Report verdict per regime. Honest framing.

## Methodology constraints

- **N27.** Per memory: function over speed; no scalar in production. B2-B reference kernel must have a NEON path; scalar reference for verification is fine.
- **N28.** Per memory: no consumer-demand framing. The strong-claim test is foundational; doesn't gate on measured demand.
- **N29.** Per CONTRIBUTING: substrate-novelty audit. The strong-claim test IS a substrate-novelty audit at the comparative level.
- **N30.** Per CONTRIBUTING: multi-config gates the story. Same 12 configs as audit (3 sizes × 2 weight zero-fracs × 2 act zero-fracs).

## Risk register

- **N31.** B2-B kernel design choices affect verdict. Pre-commit "reasonably-tight, not heroically-optimized" implementation.
- **N32.** NEON op count is a proxy for cycles; not perfect. Report op count AND wall-clock for cross-check; gate on op count.
- **N33.** Selection bias: L1-specific verdict doesn't generalize to other layers. Documented as scope.
- **N34.** "Tie on kernel cost" might still favor base-3 on engineering-effort basis (decades of base-2 vs months of base-3). Hard to quantify; mention as caveat.

## Honest expectation

- **N35.** Likely outcome: base-3 wins on dense regimes (no skip benefit), B2-B-skip-aware wins on highly sparse regimes (skip exceeds mask overhead). Verdict probably regime-dependent.
- **N36.** This is informative: surfaces where base-3 substrate's value lives, rather than producing a single decisive verdict.

## What this cycle is NOT

- **N37.** NOT a perf cycle. NEON op count is the deliverable, not throughput.
- **N38.** NOT validating strong claim broadly. Only L1; L2/L6 follow-on.
- **N39.** NOT a re-implementation of the substrate. B2-B is a separate reference, not a replacement.
- **N40.** NOT a base-2 advocacy. The cycle's purpose is to test base-3's claim against a fair alternative.

## Open questions for REFLECT

- **N41.** Should the cycle also test B2-A (1-bit) as a third arm to confirm "the third state matters at all"? Adds confirmation but increases scope.
- **N42.** What's the right disassembly target? `objdump -d` on the kernel function, count instructions in the inner loop?
- **N43.** Wall-clock measurement methodology — same workload sizes used in `bench_m4t_tier2_perf`?
- **N44.** Where does the B2-B kernel live in the source tree? `audit/`, `m4t/src/` (no — substrate is base-3 only), or new top-level `b2ref/`?
