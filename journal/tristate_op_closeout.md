# CLOSEOUT: tri-state operationalization audit

Per `journal/tristate_op_synthesize.md`. Two-gate audit of third-state utilization across substrate layers L1-L4 + L6 on a 2-layer ternary GEMM workload. 60 runs (12 configs × 5 seeds). Realism gate: 60/60 PASS.

**Verdict — VERY-IMPORTANT-AND-LOAD-BEARING NOTE FIRST.** This audit tests the WEAK claim only: third state is load-bearing in our substrate's algorithms. It does NOT test the STRONG/comparative claim that base-3 outperforms base-2 with explicit masking. Per `feedback_substrate_claim_scope.md` and the SYNTHESIZE scope clarification, the strong claim is a separate cycle.

## Verdict (weak claim, intra-substrate)

```
L1 (weight third-state)            : LOAD-BEARING        (cos uniformly low; H near max in dense regime)
L2 (activation third-state)        : LOAD-BEARING        (cos uniformly low; H near max in low-zero regime)
L3 (per-MAC product third-state)   : MIXED — sparse regime is sink-by-entropy but load-by-algorithm
L4 (post-reduction Y1 mantissa)    : UNDER-EXPLOITED — cos=1.000 in high-act-zero configs
L5 (cross-exp accumulator)         : DEFERRED — not exercised by GEMM-only workload
L6 (post-ternarization activation) : LOAD-BEARING        (cos uniformly low; H mixed)
```

**Highest-gap algorithmically-relevant layer: L4.** When downstream ternarization is aggressive (high act zero target), L4's exact-zero mantissas are INDISTINGUISHABLE from small-magnitude mantissas — replacing them with random ±median values produces cosine similarity of 1.000 to native Y2. The third state at L4 contributes literal zero information that propagates downstream. Largest opportunity for next-cycle remediation.

## Per-layer measurements (means across 5 seeds; 12 configs)

### Gate I — Shannon entropy (max log2(3) ≈ 1.585 bits)

| Layer | Dense (w=0.20, a=0.20) | Sparse (w=0.60, a=0.60) | Across all configs |
|-------|-----------------------|-------------------------|---------------------|
| L1    | 1.523                 | 1.370                   | 1.450               |
| L2    | 1.520                 | 1.368                   | 1.443               |
| L3    | 1.583                 | 0.792                   | 1.231               |
| L4    | 1.186                 | 1.289                   | 1.241               |
| L6    | 1.544                 | 1.241                   | 1.412               |

Pre-committed thresholds: ≥1.4 load-bearing, 1.0–1.4 mixed, <1.0 sink-like.

- **L3 in sparse-sparse:** entropy 0.79 → SINK (third state dominates, 66-78% zeros).
- **L1, L2, L6 entropy** tracks the input distributions — load-bearing in dense, mixed in sparse.
- **L4 entropy** sits in mixed band uniformly; the post-reduction zero-fraction is small (1.5%–13%) because most output mantissas are non-zero integers from K-trit accumulations.

### Gate II — Cosine similarity native vs collapsed (lower = more load-bearing)

| Layer | Dense (w=0.20, a=0.20) | Sparse (w=0.60, a=0.60) | Across all configs |
|-------|-----------------------|-------------------------|---------------------|
| L1    | 0.686                 | 0.324                   | 0.488               |
| L2    | 0.779                 | 0.480                   | 0.620               |
| L3    | 0.666                 | 0.328                   | 0.461               |
| L4    | 0.948                 | 1.000                   | 0.957               |
| L6    | 0.871                 | 0.591                   | 0.738               |

Pre-committed thresholds: ≤0.95 load-bearing, 0.95–0.99 mixed, >0.99 sink-like.

- **L1, L2, L3, L6:** all uniformly load-bearing per Gate II. Forcibly collapsing the third state in any of these layers measurably perturbs the final output.
- **L4 in high-act-zero configs (a=0.60):** cos = 1.000 EXACTLY. The third state at L4 has zero algorithmic effect downstream when the next ternarization aggressively zeroes out small magnitudes. This is the SINK / UNDER-EXPLOITED finding.
- **L4 in low-act-zero configs (a=0.20):** cos in [0.82, 0.98], near the load/mixed boundary. The downstream is sensitive enough that L4's third state matters somewhat.

## Cumulative classification per layer

Applying the SYNTHESIZE rule (both gates per layer; either sink-like → UNDER-EXPLOITED):

| Layer | Dense regime | Sparse regime | Net |
|-------|--------------|---------------|-----|
| L1    | LOAD-BEARING | PARTIALLY LOAD-BEARING | **LOAD-BEARING** |
| L2    | LOAD-BEARING | PARTIALLY LOAD-BEARING | **LOAD-BEARING** |
| L3    | LOAD-BEARING | UNDER-EXPLOITED (Gate I sink) | **MIXED — see below** |
| L4    | PARTIALLY LOAD-BEARING | **UNDER-EXPLOITED** (Gate II sink) | **UNDER-EXPLOITED** |
| L6    | LOAD-BEARING | PARTIALLY LOAD-BEARING | **LOAD-BEARING** |

**L3's "mixed" character explained:** in sparse-sparse, L3 entropy is 0.79 (~80% of MAC products are zero, by direct propagation from input zero-fractions). This is sink-like by Gate I (third state dominates). But Gate II is strongly load-bearing (cos ≈ 0.32) — the algorithm DEPENDS on the sparsity pattern. So L3 is "informational but sparsity-dominated" — the third state is doing the work of sparsity, not of distinct semantic content. Per the substrate-novelty framing, this is exactly the place where base-2 with explicit zero-flag could match (binary value + 1-bit sparsity flag carries the same information).

## Highest-gap finding: L4

**The biggest gap between potential third-state utilization and actual is at L4 in high-act-zero configurations.** Cos = 1.000 means the third state at L4 contributes ZERO information that affects Y2. The current downstream design (quantile-based ternarization) treats exact-zero mantissas identically to small-magnitude mantissas because the threshold τ falls above the substituted random ±median value, sending it back to zero.

This is a STRUCTURAL property of the current substrate's downstream design, not a bug. The kernel chain `matmul → ternarize_quantile → matmul` does not preserve any distinction between "Y1[i] was structurally zero" and "Y1[i] was a small accumulator value." Both go to zero in X2.

**Next cycle's target — operationalization candidate L4-A:**

> Redesign the downstream consumer pattern (ternarization rule, gating, or stateful re-quantization) so that exact-zero L4 mantissas carry information distinct from small-magnitude L4 mantissas. Possible mechanisms:
> - **Absmean-threshold ternarization** (BitNet b1.58 rule): τ = mean(|Y|), so zero-magnitude inputs preserve zero-output identity.
> - **Stateful zero-flag accumulation** at L4: when computing Y1[i], track whether contributions structurally cancelled (signed sum of trit products hits zero) vs decayed below the threshold. Pass the zero-flag downstream as the third state.
> - **Two-channel output:** sign channel (binary) + magnitude channel (trit), where magnitude=0 is structurally distinct from sign=0.

Each candidate is testable via Gate II — measure whether the proposed downstream design produces cos < 0.95 on the same workload + L4 collapse.

## Per-config detail (all configs, summary view)

```
cfg  K     w_z   a_z   | L1H    L2H    L3H    L4H    L6H    | L4_zf  | cos1   cos2   cos3   cos4    cos6
0    64    0.20  0.20  | 1.524  1.517  1.583  1.281  1.560  | 0.066  | 0.697  0.778  0.714  0.9315  0.836
1    64    0.20  0.60  | 1.525  1.355  1.207  1.356  1.224  | 0.096  | 0.653  0.522  0.458  1.0000  0.570
2    64    0.60  0.20  | 1.377  1.507  1.237  1.333  1.561  | 0.086  | 0.308  0.804  0.461  0.8241  0.877
3    64    0.60  0.60  | 1.370  1.367  0.793  1.424  1.176  | 0.129  | 0.316  0.553  0.355  1.0000  0.609
4    256   0.20  0.20  | 1.523  1.525  1.582  1.177  1.539  | 0.033  | 0.643  0.783  0.619  0.9339  0.880
5    256   0.20  0.60  | 1.521  1.352  1.206  1.207  1.331  | 0.041  | 0.670  0.451  0.456  1.0000  0.572
6    256   0.60  0.20  | 1.372  1.521  1.226  1.215  1.534  | 0.044  | 0.296  0.808  0.470  0.9348  0.906
7    256   0.60  0.60  | 1.368  1.370  0.792  1.271  1.326  | 0.061  | 0.335  0.487  0.326  1.0000  0.589
8    1024  0.20  0.20  | 1.522  1.519  1.583  1.099  1.534  | 0.015  | 0.718  0.776  0.665  0.9784  0.898
9    1024  0.20  0.60  | 1.522  1.372  1.225  1.131  1.354  | 0.022  | 0.644  0.448  0.490  1.0000  0.670
10   1024  0.60  0.20  | 1.371  1.520  1.225  1.128  1.545  | 0.021  | 0.284  0.779  0.386  0.9786  0.856
11   1024  0.60  0.60  | 1.370  1.368  0.792  1.173  1.322  | 0.032  | 0.320  0.402  0.303  1.0000  0.573
```

Raw per-seed CSV: `audit/results.csv` (60 rows + header). Reproduce: `cmake --build build --target tristate_audit && ./build/audit/tristate_audit > audit/results.csv 2> audit/results_summary.txt`.

## What this cycle confirmed

1. **L1, L2, L6 already load-bearing per both gates.** The substrate's existing kernels at the weight/activation/ternarization layers are already exploiting the third state algorithmically — collapsing the third state at any of these produces measurable Y2 perturbation in every config.
2. **L3 is sparsity-dominated.** When inputs are sparse, L3's third state is the substrate doing sparsity work — load-bearing algorithmically but indistinguishable from base-2-with-sparsity-flag in information content. Per the strong-claim scope note, this is a layer where base-2 alternatives are most likely to match.
3. **L4 has the largest under-utilization gap.** The current downstream ternarization design renders L4's third state algorithmically invisible in the most common (high-act-zero) regime. This is the highest-leverage layer for next-cycle remediation.
4. **L5 was not exercised.** Cross-exp accum requires a residual-style workload not naturally produced by GEMM-only forward passes. Deferred.

## What this cycle did NOT do

- **Did not test the STRONG/comparative claim.** Whether base-3 outperforms base-2 with explicit masking machinery requires a base-2 reference implementation and side-by-side measurement. Out of scope per SYNTHESIZE.
- **Did not modify production code.** Audit produces measurement evidence + tooling only.
- **Did not validate vision claim 3 broadly.** Only an intra-substrate utilization audit. The audit's L4 finding identifies an opportunity; the next cycle's job is to test whether a redesigned consumer pattern lifts L4 to load-bearing. Even if it does, the strong claim still requires a separate cycle.

## Methodology lifted

1. **Two-gate design (info-theoretic + algorithmic) catches different failure modes.** Gate I caught L3's entropy collapse in sparse regimes. Gate II caught L4's algorithmic invisibility under the current downstream design. Either gate alone would have missed one of these.
2. **Cosine similarity at the WORKLOAD-end (Y2), not at each layer's output, is the right Gate II metric.** Earlier I considered measuring cos sim AT each layer; rejected because the L4 case (cos = 1.000 to native) only emerges through the downstream ternarization — measuring at the layer would have shown L4 dependence. Workload-end metric exposes the consumer-pattern interaction.
3. **The realism gate (target zero-fraction within ±5pp) was a non-event** — 60/60 pass under direct independent generation. Documented as future-proof: if a future cycle generates trits via a more complex pipeline (e.g., quantizing real Gaussian samples), the realism gate becomes informative.
4. **Substrate-claim scope discipline.** Per `feedback_substrate_claim_scope.md`, the audit explicitly flagged itself as testing the WEAK (intra-substrate) claim. The CLOSEOUT preserves that scope and forwards to TWO follow-up cycles (weak deepening + strong comparative).

## Next-cycle handoff

### Track A (weak deepening) — RECOMMENDED NEXT
**L4-A operationalization:** redesign the downstream ternarization to preserve L4's structural zeros distinctly from small-magnitude mantissas. Specific candidates (one-line each):
- A.1 **Absmean threshold ternarization** (replace quantile with τ = mean(|Y|)).
- A.2 **Zero-flag forwarding** (track structural-zero events at MAC reduction; pass as a sidecar bit).
- A.3 **Two-channel split** (sign + magnitude trit, treated independently downstream).

Test each via Gate II on the same workload. R1's failure-mode lesson applies: the third state must be a deliberate output of a designed algorithm, not a residue of a quantization choice.

### Track B (strong comparative claim) — DEFERRED
For any layer that emerges as load-bearing (currently L1, L2, L6; or L4 if Track A succeeds): construct a base-2 reference implementation with explicit masking / sign-flag / zero-flag machinery. Side-by-side on the same workload. Compare information density (bits/cell), algorithmic precision (cos to ground truth), kernel cost. Verdict: does base-3 outperform base-2-with-workaround on at least one axis without losing on the others?

### Track C (cross-exp accum coverage)
Construct a residual-style workload that exercises `m4t_mtfp_vec_accum_aligning` across differing block exponents. Measure L5's third-state utilization. Currently uncharacterized.

## Honest concerns

1. **The "highest-gap" framing is interpretation-dependent.** L4's cos=1.000 is the largest single-axis gap, but L3's sink-like entropy in sparse regimes is also substantial. A different ranking criterion could prioritize L3-sparse instead. The recommendation defaults to L4 because cos=1.000 is unambiguous (literal zero algorithmic contribution); reasonable people could choose L3 instead.

2. **The L4 finding is conditional on the downstream design.** Cos=1.000 specifically reflects how `ternarize_quantile` interacts with random-substituted mantissas. A different ternarization (absmean) would produce different cos numbers. The finding is therefore: "L4's third state is invisible to this substrate's specific kernel chain in high-act-zero regimes" — not a universal property of L4.

3. **Realism gate was trivially passing.** Direct ternary generation hits target zero-fraction by construction; the gate added no signal. Future cycles using non-trivial quantization pipelines will need the gate to bite.

4. **L5 deferral is a real coverage hole.** Cross-exp accum is a substrate-level kernel and the audit doesn't speak to it. Track C above documents the deferred follow-up; the audit's conclusions don't extend to L5.

5. **Workload size is small (M=8, P=8).** Adequate for the third-state distribution measurements (per-layer stats are not size-sensitive once K is large), but a larger workload might surface size-dependent effects (e.g., saturation behavior at K=1024 was not specifically probed).

6. **Multi-config coverage was 12 configs, not exhaustive.** The audit's regimes (sparse vs dense × dense vs sparse) cover the main cases, but a continuous sweep over zero-fractions could surface non-linearities.

## Status

CLOSED. L4 is the highest-leverage operationalization target; recommended Track A (weak deepening) candidates A.1–A.3 above. L5 deferred to Track C. Strong-claim cycle (Track B) deferred until at least one Track A candidate produces a positive Gate II result, at which point the comparative test becomes meaningful.

Files:
- `audit/tristate_audit.c` — measurement tool source
- `audit/CMakeLists.txt` — build target (not in ctest)
- `audit/results.csv` — raw per-seed measurements (60 rows)
- `audit/results_summary.txt` — per-config summary
