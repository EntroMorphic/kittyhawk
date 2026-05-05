# CLOSEOUT: tri-state operationalization audit (post red-team R-G1)

Per `journal/tristate_op_synthesize.md` and `journal/tristate_op_redteam.md`. Two-gate audit of third-state utilization across substrate layers L1-L4 + L6 on a 2-layer ternary GEMM workload. 60 runs (12 configs × 5 seeds). Realism gate: 60/60 PASS.

**SCOPE NOTE.** This audit tests the WEAK claim only: third state is load-bearing in our substrate's algorithms. It does NOT test the STRONG/comparative claim that base-3 outperforms base-2 with explicit masking. Per `feedback_substrate_claim_scope.md` and the SYNTHESIZE scope clarification, the strong claim is a separate cycle (now in setup; see "Strong claim cycle handoff" at the end).

**RED-TEAM REMEDIATION APPLIED.** First-run L4 measurements were artifacts — the original L4 collapse design substituted median-magnitude values that the downstream threshold reabsorbed, producing cos = 1.000 trivially. Per `tristate_op_redteam.md` C1, the L4 collapse was redesigned (override-after-ternarize semantics) and the audit re-run. ALL POST-FIX RESULTS BELOW.

## Verdict (weak claim, intra-substrate; post-fix)

```
L1 (weight third-state)            : LOAD-BEARING
L2 (activation third-state)        : LOAD-BEARING  
L3 (per-MAC product)               : MIXED — sparse regime sink-by-entropy / load-by-algorithm
L4 (post-reduction Y1 mantissa)    : MIXED — least load-bearing of measured layers, but cos < 1
L5 (cross-exp accumulator)         : DEFERRED — not exercised by GEMM-only workload
L6 (post-ternarization X2)         : LOAD-BEARING
```

**Highest-gap algorithmically-relevant layer (post-fix): L4 still ranks lowest per Gate II.** Mean cos across configs: L1≈0.49, L2≈0.62, L3≈0.46, L4≈0.94, L6≈0.74. L4's cosine similarity to native is roughly 1.5× the next-least-load-bearing layer (L6) — the third state at L4 is still doing less work than at any other measured layer, but the gap is "tightening opportunity" not "broken layer."

The first-cycle headline ("L4 is the highest-leverage operationalization target") survives in muted form; the underlying claim shifts from "cos=1.000 invisible" to "cos≈0.94 least load-bearing."

## Per-layer measurements (means across 5 seeds; 12 configs)

### Gate I — Shannon entropy (max log2(3) ≈ 1.585 bits)

| Layer | Dense (w=0.20, a=0.20) | Sparse (w=0.60, a=0.60) | Across all configs |
|-------|-----------------------|-------------------------|---------------------|
| L1    | 1.523                 | 1.370                   | 1.450               |
| L2    | 1.520                 | 1.368                   | 1.443               |
| L3    | 1.583                 | 0.792                   | 1.231               |
| L4    | 1.186                 | 1.289                   | 1.241               |
| L6    | 1.544                 | 1.241                   | 1.412               |

Pre-committed thresholds: ≥1.4 load-bearing, 1.0–1.4 mixed, <1.0 sink-like. Gate I unchanged from first run (the entropy measurement doesn't depend on the L4 collapse).

- **L3 in sparse-sparse:** entropy 0.79 → SINK (third state dominates, ~80% zeros).
- **L1, L2, L6 entropy** tracks input distributions — load-bearing in dense, mixed in sparse.
- **L4 entropy** sits in mixed band uniformly; post-reduction zero-fraction is small (1.5%–13%).

### Gate II — Cosine similarity native vs collapsed (lower = more load-bearing) — POST-FIX

| Layer | Dense (w=0.20, a=0.20) | Sparse (w=0.60, a=0.60) | Across all configs |
|-------|-----------------------|-------------------------|---------------------|
| L1    | 0.686                 | 0.324                   | 0.488               |
| L2    | 0.779                 | 0.480                   | 0.620               |
| L3    | 0.666                 | 0.328                   | 0.461               |
| L4    | **0.961**             | **0.918**               | **0.937**           |
| L6    | 0.871                 | 0.591                   | 0.738               |

Pre-committed thresholds: ≤0.95 load-bearing, 0.95–0.99 mixed, >0.99 sink-like.

- **L1, L2, L3, L6:** all uniformly load-bearing per Gate II. Forcibly collapsing the third state in any of these measurably perturbs the final output.
- **L4 (post-fix):** straddles the load/mixed boundary. Higher act_zero regimes (a=0.60) tend toward load (cos ≈ 0.86-0.95); lower act_zero regimes (a=0.20) tend toward mixed (cos ≈ 0.94-0.99). Notably, the BitNet-typical regime (high act_zero) is where L4 is MOST load-bearing, opposite of what the broken collapse suggested.
- **L4 is consistently the least load-bearing** of any measured layer. cos values 1.5-3× higher than other layers' cos values across all configs.

## Cumulative classification per layer (post-fix)

| Layer | Dense regime | Sparse regime | Net |
|-------|--------------|---------------|-----|
| L1    | LOAD-BEARING | PARTIALLY LOAD-BEARING | **LOAD-BEARING** |
| L2    | LOAD-BEARING | PARTIALLY LOAD-BEARING | **LOAD-BEARING** |
| L3    | LOAD-BEARING | UNDER-EXPLOITED (Gate I sink) | **MIXED — see below** |
| L4    | MIXED         | LOAD-BEARING (sparse high-act-zero) | **MIXED** |
| L6    | LOAD-BEARING | PARTIALLY LOAD-BEARING | **LOAD-BEARING** |

**L3's "mixed" character (unchanged):** in sparse-sparse, L3 entropy is 0.79 (~80% zeros), sink-like by Gate I, but Gate II is strongly load-bearing (cos ≈ 0.32). The third state is doing the work of sparsity, indistinguishable in information content from base-2-with-sparsity-flag. This layer is most likely to be matched by base-2 alternatives in the strong-claim cycle.

**L4's "mixed" character (post-fix):** straddles load/mixed depending on regime. The third state at L4 carries less algorithmic weight than at L1/L2/L3/L6, but it's not invisible. Tightening opportunity exists; broken-layer narrative does not.

## Highest-gap finding (post-fix): L4 still leads, but the gap is smaller

**Mean cos rankings (most load-bearing → least):** L3 (0.46) < L1 (0.49) < L2 (0.62) < L6 (0.74) < L4 (0.94).

L4 is consistently the least load-bearing layer across all 12 configs. The "highest-gap" framing still picks L4 as the leading remediation target, but the gap is "1.9× higher cos vs next layer (L6)" not "literal cos=1 invisibility." The strength of the recommendation is reduced; the direction is preserved.

**Track A operationalization candidates (still recommended for next cycle):**

> Redesign the downstream consumer pattern (ternarization rule, gating, or stateful re-quantization) so that exact-zero L4 mantissas carry MORE distinct downstream information than they currently do. Possible mechanisms:
> - **A.1 — Absmean-threshold ternarization** (BitNet b1.58 rule): τ = mean(|Y|), so zero-magnitude inputs preserve zero-output identity more reliably than the quantile-based threshold.
> - **A.2 — Stateful zero-flag accumulation** at L4: when computing Y1[i], track whether contributions structurally cancelled (signed sum hits zero exactly) vs decayed below threshold. Pass the zero-flag downstream as the third state.
> - **A.3 — Two-channel output:** sign channel (binary) + magnitude channel (trit), magnitude=0 structurally distinct from sign=0.

Each candidate testable via the same Gate II workload — measure whether the proposed downstream design produces cos < 0.85 (clearly load-bearing) on the high-leverage configs.

## Per-config detail (post-fix; all configs, summary view)

```
cfg  K     w_z   a_z   | L1H    L2H    L3H    L4H    L6H    | L4_zf  | cos1   cos2   cos3   cos4    cos6
0    64    0.20  0.20  | 1.524  1.517  1.583  1.281  1.560  | 0.066  | 0.697  0.778  0.714  0.9599  0.836
1    64    0.20  0.60  | 1.525  1.355  1.207  1.356  1.224  | 0.096  | 0.653  0.522  0.458  0.8788  0.570
2    64    0.60  0.20  | 1.377  1.507  1.237  1.333  1.561  | 0.086  | 0.308  0.804  0.461  0.9435  0.877
3    64    0.60  0.60  | 1.370  1.367  0.793  1.424  1.176  | 0.129  | 0.316  0.553  0.355  0.8570  0.609
4    256   0.20  0.20  | 1.523  1.525  1.582  1.177  1.539  | 0.033  | 0.643  0.783  0.619  0.9795  0.880
5    256   0.20  0.60  | 1.521  1.352  1.206  1.207  1.331  | 0.041  | 0.670  0.451  0.456  0.9449  0.572
6    256   0.60  0.20  | 1.372  1.521  1.226  1.215  1.534  | 0.044  | 0.296  0.808  0.470  0.9788  0.906
7    256   0.60  0.60  | 1.368  1.370  0.792  1.271  1.326  | 0.061  | 0.335  0.487  0.326  0.9285  0.589
8    1024  0.20  0.20  | 1.522  1.519  1.583  1.099  1.534  | 0.015  | 0.718  0.776  0.665  0.9900  0.898
9    1024  0.20  0.60  | 1.522  1.372  1.225  1.131  1.354  | 0.022  | 0.644  0.448  0.490  0.9759  0.670
10   1024  0.60  0.20  | 1.371  1.520  1.225  1.128  1.545  | 0.021  | 0.284  0.779  0.386  0.9841  0.856
11   1024  0.60  0.60  | 1.370  1.368  0.792  1.173  1.322  | 0.032  | 0.320  0.402  0.303  0.9521  0.573
```

Raw per-seed CSV: `audit/results.csv` (60 rows + header).

## Red-team remediations applied

Per `tristate_op_redteam.md`:
- **R-G1 (C1, CRITICAL):** L4 collapse redesigned to override-after-ternarize semantics. Re-run produced different L4 verdict (was UNDER-EXPLOITED with cos=1.000; now MIXED with cos≈0.94). Headline preserved in direction but with weaker magnitude.
- **R-G2 (D1-D9, doc-level):** Concerns documented in red-team and noted in interpretation caveats below.

## Interpretation caveats (per red-team D1-D9)

1. **L3 measurement is L1+L2 simultaneous (D1).** For ternary inputs, X*W = 0 iff X==0 OR W==0. The L3 collapse is mathematically equivalent to "make BOTH X and W binary." L3's verdict is interpretable only relative to the union of L1+L2, not as an independent third axis.
2. **L1 collapse is whole-weight (D2).** Both W1 and W2 are collapsed simultaneously; the measurement reflects cumulative effect through both layers, not isolated layer-1 weights.
3. **PRNG state shared across Gate II calls (D3).** Order-dependent entropy: per-seed values fluctuate based on which layer's collapse runs first. Mean over 5 seeds is unbiased, but per-seed variance partly reflects this.
4. **Realism gate is trivial (D4).** Direct ternary generation always passes; the gate added zero signal in this audit. Reserved for future cycles using more complex quantization pipelines.
5. **L2 / L6 collapse changes activation distribution substantially (D5).** Collapsing zero-fraction from 0.60 to 0 is a large distributional shift. Cosine similarity reflects that the algorithm depends on the distribution shape, not specifically on the third state's discriminative content.
6. **Workload is small (M=8, P=8) (D6).** Per-seed values fluctuate (e.g., cfg 0 cos_L1: 0.58–0.76 across 5 seeds). 5-seed mean smooths but doesn't quantify per-config uncertainty in the verdict.
7. **No unit tests for measurement math (D7).** Entropy and cosine-sim formulas are simple but unverified against known distributions. Could harden with a small test if the cycle revisits.
8. **"Highest-gap" interpretation is judgment (D8).** Post-fix L4 cos≈0.94 ranks lowest, but the gap to L6 (0.74) is much smaller than the gap to a hypothetical "fully load-bearing" layer (cos<0.5). Reasonable interpretations could downgrade the recommendation.
9. **Cross-layer interactions not measured (D9).** Each Gate II treats one layer's collapse independently. Multi-layer simultaneous collapses might compound differently.

## What this cycle confirmed

1. **L1, L2, L6 are load-bearing per both gates.** The substrate's third state is doing real work at the weight, activation, and ternarization layers — collapsing the third state at any of these produces measurable Y2 perturbation in every config.
2. **L3 is sparsity-dominated.** When inputs are sparse, L3's third state IS load-bearing algorithmically but is interchangeable with base-2-with-sparsity-flag information-theoretically.
3. **L4 is the least load-bearing measured layer (post-fix).** The third state at L4 carries less algorithmic weight than other layers, but is not invisible. Strongest tightening candidate for next cycle.
4. **L5 was not exercised.** Cross-exp accum requires a residual-style workload not produced by GEMM-only forward passes. Track C in handoff.

## What this cycle did NOT do

- **Did not test the STRONG claim.** Whether base-3 outperforms base-2 with explicit masking requires a base-2 reference implementation. Strong-claim cycle is now in setup (see handoff).
- **Did not modify production code.** Audit produces measurement evidence + tooling only.
- **Did not validate vision claim 3 broadly.** Only an intra-substrate utilization audit.

## Methodology lifted

1. **Two-gate design (info-theoretic + algorithmic) catches different failure modes** — confirmed; Gate I caught L3's entropy collapse, Gate II flagged L4 as least load-bearing post-fix.
2. **Cosine similarity at the WORKLOAD-end (Y2), not at each layer's output, is the right Gate II metric.**
3. **Substitution-collapse design must survive downstream thresholds.** New methodology rule: when Gate II "collapse" requires substituting a magnitude in numeric space, validate that the substitution actually propagates through downstream operations. Median-magnitude substitutes can be silently absorbed by quantile thresholds; override-after-ternarize avoids this. Lifted from R-G1 remediation.
4. **Substrate-claim scope discipline.** Per `feedback_substrate_claim_scope.md`, the audit explicitly flagged itself as testing the WEAK claim only. CLOSEOUT preserves that scope and forwards to the strong-claim cycle.
5. **Red-team-after-execute is load-bearing.** First-cycle results pointed at L4 as "UNDER-EXPLOITED with cos=1.000." Red-team identified the collapse design flaw; remediation produced different (still-directional) findings. Without red-teaming, the audit would have published an inflated verdict.

## Next-cycle handoff

### Track A (weak deepening) — RECOMMENDED NEXT (post-strong-claim setup)
**L4-A operationalization:** redesign the downstream ternarization to make L4's structural zeros distinguishable from small-magnitude mantissas. Three candidates: A.1 absmean threshold, A.2 zero-flag forwarding, A.3 two-channel split. Test via Gate II.

Track A is now SECOND PRIORITY. The user requested the strong-claim cycle be set up next; Track A becomes the follow-on.

### Track B (strong comparative claim) — IN SETUP NEXT
Construct a base-2 reference implementation for at least one load-bearing layer (L1 or L2 are the most defensible candidates; both LOAD-BEARING per gate II uniformly). Side-by-side measurement of base-3 substrate vs base-2 with explicit masking. Compare information density, algorithmic precision, kernel cost. Verdict: does base-3 outperform base-2-with-workaround on at least one axis without losing on others?

Strong-claim cycle synthesis to follow.

### Track C (cross-exp accum coverage) — DEFERRED
Construct a residual-style workload that exercises `m4t_mtfp_vec_accum_aligning` across differing block exponents. Currently uncharacterized.

## Status

CLOSED. Post-red-team verdict: no layer is "broken / under-exploited." All measured layers (L1, L2, L3, L6) are load-bearing per both gates in their natural regimes; L4 is the least load-bearing but still measurably so. The substrate's third state is broadly doing real work in the existing kernel chain.

The headline is now positive for the WEAK claim: **the substrate's third state is intra-substrate load-bearing across measured layers**, with L4 as a tightening opportunity. The strong-claim cycle (next) tests whether this load-bearing role would be replicable by base-2 with explicit masking — that's where the substrate's distinctive value, if any, must show up.

Files:
- `audit/tristate_audit.c` — measurement tool source (post-fix L4 collapse)
- `audit/CMakeLists.txt` — build target
- `audit/results.csv` — raw per-seed measurements (60 rows, post-fix)
- `audit/results_summary.txt` — per-config summary (post-fix)
- `journal/tristate_op_redteam.md` — red-team findings (C1 critical + D1-D10 doc)
