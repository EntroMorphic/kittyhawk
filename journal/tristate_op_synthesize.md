# SYNTHESIZE: useful operationalization of the tri-state potential

Pre-committed plan + gates derived from `tristate_op_reflect.md`.

## Decision

**Run an audit-style cycle that measures third-state utilization across every layer of our existing substrate on a representative ternary-ML workload, with pre-committed information-theoretic AND algorithmic gates per layer. Output: a ranked finding identifying the layer where the third state has the largest gap between potential utilization and actual utilization.**

This is a science cycle. Output is a journal verdict + measurement evidence, not a code change to production. The next cycle will pursue the highest-gap layer with a direct operationalization test.

## The question being operationalized

**For the existing substrate (m4t_mtfp, ternary matmul, cross-exp accum, mtfp4) running a representative ternary-ML workload, where does the third state act as a load-bearing information carrier vs as a sparsity-only / sink artifact?**

If the third state is load-bearing somewhere, that layer's existing usage shows the third state does WORK in our substrate. If it's load-bearing nowhere, the third state's potential is unrealized in the existing kernels, and the next cycle pursues operationalization OUTSIDE them.

## Claim scope (important distinction)

Vision claim 3 has two interpretations, only one of which this audit tests:

**WEAK CLAIM (this audit):** the third state in our substrate carries information at some layer. Tested by Gate I (info-theoretic) + Gate II (algorithmic dependence within the layer's existing kernel structure).

**STRONG CLAIM (NOT tested here):** base-3 carries information that base-2 collapses, in a way that's structurally cheaper or more accurate than base-2's workaround (e.g., binary value + separate mask bit, or sign+magnitude with explicit zero flag). The strong claim is a COMPARATIVE assertion against an alternative substrate.

A layer that passes both audit gates demonstrates the WEAK claim — the third state does work in the current substrate. It does NOT demonstrate the strong claim — the layer might be doing something a hypothetical base-2 implementation with explicit masking machinery could match in functionality, possibly at higher cost (2 bits / cell instead of log2(3) ≈ 1.58) but with equivalent algorithmic behavior.

Demonstrating the strong claim requires:
- A base-2 reference implementation of the same layer with explicit masking / sign-flag / zero-flag machinery.
- Side-by-side measurement: information density (bits / unit work), algorithmic precision, throughput on the same workload.
- Verdict: does base-3 outperform base-2-with-workaround on at least one axis (density, precision, or kernel cost) while matching on the others?

This is a separate, harder cycle, **out of scope here.** The audit identifies WHERE the third state is load-bearing within our substrate; the strong-claim cycle would test WHETHER that load-bearing role is ACHIEVED MORE EFFICIENTLY than the base-2 alternative.

Without this scope note, a finding like "L3 is load-bearing per both gates" could be misread as "claim 3 validated at L3." The honest interpretation is "the third state does work at L3 in our substrate; whether that work would be done differently or worse in a base-2 substrate is the next question."

## Workload specification (pre-committed)

**Workload:** small ternary GEMM, modeling the inner loop of a 1.58-bit LLM forward pass.
- Weight matrix W: M × K trits, distribution matched to published 1.58-bit LLM weight statistics (zero-fraction, magnitude balance).
- Activation vector / matrix A: K × N trits, derived from realistic activation patterns (sampled from a typical pre-quantization distribution then ternarized via the same rule used in the LLM's forward pass).
- Output: ternary or quantized accumulator.

**Multi-config (per CONTRIBUTING):**
- 3 sizes: K ∈ {64, 256, 1024}.
- 2 weight distributions: (a) zero-fraction ≈ 60% (sparse, BitNet-1.58 typical); (b) zero-fraction ≈ 20% (dense, less typical).
- 2 activation distributions: (a) Gaussian-then-ternarized; (b) Real-input-stats-then-ternarized.

Total: 12 configs. Multi-seed (5 seeds per config) per CONTRIBUTING.

**Realism gate:** before running measurements, validate that the workload's weight zero-fraction matches published 1.58-bit LLM stats within ±5pp. If it doesn't, fix the workload before drawing conclusions.

## Layers to audit

Each layer is measured separately:

- **L1 — Weight storage (mtfp packed trits):** distribution of trits in W. Third state = zero trits.
- **L2 — Activation storage (same):** distribution of trits in A.
- **L3 — Per-cell MAC contribution (ternary matmul inner):** for each output cell, what's the distribution of {+1 contributions, 0 contributions, -1 contributions} from individual trits before summation?
- **L4 — Block-level reduction (within mtfp block):** what's the distribution of partial sums after a single block's MAC? This tests whether the third state is preserved after reduction or collapses.
- **L5 — Cross-exp accumulation (cross_exp_accum):** when accumulating across blocks of differing exponents, what's the distribution of accumulator state? Does the third state appear in the accumulator's outputs?
- **L6 — Output quantization (if applicable):** when the accumulator's int output is re-quantized to ternary for the next layer, what's the distribution of the resulting trits?

## Pre-committed gates per layer

For each of L1-L6, two independent gates:

### Gate I (Information-theoretic)
Measure: empirical entropy H(third-state distribution) at this layer.

- H ≈ log2(3) ≈ 1.585 bits → third state is balanced with the other two; load-bearing in the entropic sense.
- H ≈ log2(2) = 1 bit → one of the three states (likely zero) dominates; third state is sink-like.
- H < 1 bit → severe imbalance; third state mostly absent.

**Pre-committed thresholds:**
- Load-bearing: H ≥ 1.4 bits (close to log2(3))
- Mixed: 1.0 ≤ H < 1.4
- Sink-like: H < 1.0

### Gate II (Algorithmic)
Measure: does collapsing the third state to one of the other two states (forced binary projection) DEGRADE this layer's downstream contribution?

For each layer L, run two parallel measurements:
- L_native: layer operates with native third state.
- L_collapsed: layer operates with third state forcibly mapped to nearest non-zero state (random tie-break).

Measure cosine similarity (or correlation, or task-relevant metric) between L_native's output and L_collapsed's output at the SAME downstream point.

**Pre-committed thresholds:**
- Load-bearing: cosine similarity ≤ 0.95 (collapsing the third state changes the layer's output meaningfully).
- Mixed: 0.95 < cos ≤ 0.99.
- Sink-like: cos > 0.99 (collapsing the third state is nearly imperceptible — the layer doesn't depend on the third state).

### Cumulative classification per layer
- Both gates "load-bearing" → **L is LOAD-BEARING.**
- One gate "load-bearing", other "mixed" → **L is PARTIALLY LOAD-BEARING.**
- Either gate "sink-like" → **L is SINK-LIKE / UNDER-EXPLOITED.**
- Both gates "sink-like" → **L is FULLY UNDER-EXPLOITED.**

## Output: ranked finding

After measurement, rank layers L1-L6 by **gap between potential and actual utilization**:
- Gap = (theoretical max H, log2(3)) − (measured H)
- Adjusted by Gate II (algorithmic dependence) — high entropy with low algorithmic dependence is "noise," not signal.

The HIGHEST-gap layer that is also algorithmically meaningful is the next cycle's target.

## Order of execution

1. Build workload spec (synthetic data generators + realism gate).
2. Validate workload against published 1.58-bit LLM stats (realism gate).
3. Measure L1-L6 information-theoretic gates across all 12 configs × 5 seeds.
4. Measure L1-L6 algorithmic gates across all 12 configs × 5 seeds.
5. Tabulate results; classify each layer.
6. Identify the highest-gap algorithmically-meaningful layer.
7. CLOSEOUT with finding + next-cycle handoff.

## Risk register

- **R1 (workload not realistic):** if the synthetic workload doesn't match real 1.58-bit LLM stats, the audit's findings don't generalize. Mitigation: realism gate before measurement; abort if it fails.
- **R2 (information-theoretic gate alone is fakeable):** uniform-random third state passes Gate I but isn't useful. Mitigation: Gate II (algorithmic) is the bite-test.
- **R3 (algorithmic gate is task-dependent):** "downstream contribution" requires choosing a downstream metric. Mitigation: cosine similarity between native and collapsed is task-agnostic; if a more task-specific metric is needed, document it and treat as a sub-gate.
- **R4 (a layer is partially load-bearing on Gate I but uniformly load-bearing on Gate II, or vice versa):** classification rules above are pre-committed but might not match the data cleanly. Mitigation: pre-commit the rules; report the data honestly even if classification is awkward; refine in CLOSEOUT methodology lift.
- **R5 (the audit reveals NO layer is load-bearing):** would suggest claim 3 needs operationalization OUTSIDE the existing kernels. Honest finding; would inform vision claim 3's status.

## What this cycle is NOT

- **Not testing the STRONG version of claim 3** (base-3 outperforms base-2 with explicit masking machinery). That requires a base-2 reference implementation; out of scope here. See "Claim scope" above.
- Not validating claim 3 broadly. Only auditing the existing substrate's third-state utilization (the WEAK claim).
- Not modifying production code. Audit produces journal evidence + measurement scripts only.
- Not a perf cycle. No throughput measurements.
- Not a re-run of R1. R1 was about signature derivation at the consumer layer; this is about utilization at the substrate layer.

## Done when

L1-L6 measured under all 12 configs × 5 seeds, both gates per layer. Classification table produced. Highest-gap algorithmically-meaningful layer identified. CLOSEOUT records:
- Per-layer gate measurements + classification
- Ranked finding (which layer has the largest gap between potential and actual utilization, **noting this is the WEAK-claim ranking**)
- Methodology lifted (if any)
- Forward pointer to TWO follow-up cycles:
  - **WEAK-claim follow-up:** the highest-gap layer's operationalization is deepened or remediated within the existing substrate.
  - **STRONG-claim follow-up:** for any layer flagged as load-bearing, design a base-2 reference implementation with explicit masking / sign-flag / zero-flag machinery and measure comparative information density + precision + kernel cost. This is a separate cycle, scope and gates to be defined when prioritized.

## Status

Pre-committed. Awaiting user gate before execution.

The execution would be: a measurement script (likely a new file in `m4t/bench/` or `gesh/bench/` or a dedicated `audit/` directory), running the workload + measuring L1-L6 + producing tabulated output. Estimated 200-400 lines of new C / Python (depending on host). No production substrate change.
