# Phase α geometric-structure diagnostic — pre-research

**Status:** PRE-RESEARCH. This document records design output from two
research subagents launched 2026-05-12 to scope a geometric measurement
of substrate signature spaces. It is NOT a frozen pre-registration yet
— that will be a follow-on commit when implementation is undertaken.
The discipline here is: capture the research output before any
implementation drift contaminates it.

## Motivation

Persistent project concern: the empirical claim ("substrate routing is
a useful primitive") and the philosophical claim ("base-3 carries
information base-2 collapses") are diverging. The audit cycles (P0-3,
P0-4) measured scalar aggregates (utilization, entropy). Those don't
test whether the substrate's distinctive value is GEOMETRIC — local
metric structure, effective dimensionality, curvature analogues — that
scalar measures can't see.

Phase α asks: **does substrate signature space have geometric
structure that scalar/binary representations of the same data lose?**
If yes, the audits were measuring the wrong axis and the substrate has
substrate-distinctive properties not yet surfaced. If no, the third
state is decorative.

## Subagent output 1 — codebase survey

Subagent ran read-only inventory of the codebase. Findings:

### Prior geometric work in the project

- **P0-3 (closed 2026-05-10):** shipped `m4t_route_pairwise_hamming_sum`
  — ternary Hamming distance over packed-trit tile signatures. Gate 2
  PASSED (geometric loss IS optimizable on substrate; pairwise margin
  increases under flip-based coordinate descent). Gate 1 was MIXED
  → PASS only in close-prototype regime (+6.24pp). **Key
  methodological precedent:** substrate-claim earns its name only in
  the regime that would expose it (close-prototype, not far). The
  Phase α design should inherit this discipline.

- **Cycle 2 Axis 5 (substrate routing):** routed > random by +16.7pp
  at k=16; gap widens with sparsity. Mechanism = direction-awareness
  in the substrate's signature representation (sign + zero is
  natively encoded; oracle's `|score|` is direction-blind). This
  is the closest prior to a geometric finding in the empirical
  record.

- **P0-1 / P0-2 design (not shipped):** structural zero as routing
  signal (P0-1); MTFP block-exponent as routing signal (P0-2). Both
  philosophically aligned with "third state as load-bearing" but not
  yet operationalized.

### Available data

- **242 ACTV2 dump files** in `data/c_dump/` (~38 MB total), covering
  30 layers × multiple positions × 2 prompts (single-token, multi-token).
- **Schema** (per `gesh/bitnet/scripts/measure_activation_sparsity.py`):
  ACTV2 format with 12 capture sites per layer per position
  (x_norm_input, q_pre_rope, k_pre_rope, v, q_post_rope, k_post_rope,
  attn_sub_norm, x_norm, gate, up, ffn_sub_norm, block_output).
- **Cycle 2 results** (`journal/cycle2_full_battery/results.tsv`):
  475 rows × 30-token outputs across 24 prompts × 4 arms × 6 k values.
- **Routing primitives** in `m4t/src/m4t_route.h`:
  threshold_extract, distance_batch, pairwise_hamming_sum,
  wildcard_dist, confidence_weighted_dist.

### Gaps for Phase α

- K-signature dumps don't exist directly. Need to re-apply
  `m4t_route_threshold_extract` on dumped K values with a chosen tau.
- τ values weren't saved. Phase α uses fixed τ=5000 (production
  default per #1's K-sig caching).
- Cycle 2 result dumps lack per-token routing decisions; that's a
  gap for sigdist-specific measurements but not for Phase α (which
  works on K signatures themselves).

**Verdict from agent 1:** existing data is sufficient to test M1,
M2, and partially M3 on this commit's data. No new inference runs
required.

## Subagent output 2 — measure design

Subagent designed three geometric measures and three baselines with
peer-reviewed methodology.

### Measures (recommended in priority order)

**M1 — Local intrinsic dimensionality via discrete-Hamming I3D
estimator.**
- Algorithm: for each signature `x_i`, compute Hamming distance to
  all others. Take r1 ≤ r2 (two nearest neighbor distances). The
  ratio μ_i = r2/r1 is Pareto-distributed under TwoNN (Facco et al.
  2017, *Scientific Reports*) with shape d = local intrinsic
  dimension. For the DISCRETE Hamming case, use Macocco/Glielmo/Laio
  2023 (PRL: arxiv.org/abs/2207.09688) which replaces continuous
  Pareto likelihood with a binomial-on-lattice-shells likelihood
  via Ehrhart polynomials.
- **Critical:** this is the only candidate with a peer-reviewed
  estimator SPECIFICALLY for Hamming/discrete metrics. Other
  estimators assume Euclidean and would bias against substrate.
- Cost: all-pairs Hamming on 10K × 128-trit signatures (~10⁸
  popcount-XOR), well under 1 sec on M4 NEON. Estimator is O(N).
- The discrete-Hamming version for *ternary* (not just binary)
  Hamming is straightforward extension — shell volume polynomial
  changes from binomial to trinomial — but not, to agent's
  knowledge, published. **That extension is itself a claimable
  novelty for the substrate research thread.**

**M2 — k-NN graph topology divergence.**
- Mutual-kNN reciprocity (fraction of edges (i,j) where j is in
  i's kNN AND vice versa)
- Local clustering coefficient distribution (median, IQR, tail)
- Degree distribution in symmetrized k-NN graph (hub-ness test)
- Across k ∈ {5, 10, 20, 50}; substrate-vs-baseline ordering must
  be stable across all four for a credible signal.

**M3 — Persistent Betti-0 across Hamming-radius filtration.**
- For each integer Hamming radius r, form graph where edges connect
  pairs within distance r. Compute β_0(r) = connected components
  via union-find (O(N · α(N)) per scale).
- Curve r → β_0(r) is persistence-0. Substrate's claim (close-
  prototype P0-3 finding): distinct token-type clusters stay
  separate longer. Should appear as PLATEAU in β_0(r) before
  fast collapse.
- H0 only — full persistent homology (H1, H2) is O(N³) memory
  and not the right tool at this scale. β_0 captures cluster
  structure without the cost.

### Baselines

| Baseline | What it isolates | Construction |
|---|---|---|
| **B1: BitNet K-values in mtfp19** | substrate vs raw substrate input | L2 distance on int-vectors of same K. Tests: does extraction *add* structure, or just preserve what's already there? |
| **B2: Sign-only binarization** | **whether the third state matters** | sign() on the same K-values, packed to 1-bit-per-cell. Hamming distance on bits. **THE LOAD-BEARING FALSIFICATION TEST.** If substrate ≈ B2, the trit machinery is theatrical. |
| **B3: Random Gaussian projection → sign** | whether trained τ matters | Random projection to 128 dim, then sign. Same Hamming. Tests: does any 128-bit hash work, or does the substrate's threshold extraction add value? |

PCA → threshold is explicitly **not** recommended as a baseline:
different inductive bias (variance maximization) and confounds the
comparison.

### Comparison protocol (pre-registered)

For each measure M ∈ {M1, M2, M3}, compute on substrate and each
baseline B ∈ {B1, B2, B3}. Across multiple layers (0, mid, last)
and multiple sequence positions (early-context vs trained-position).

**Statistical bound:** Bootstrap over 1000 resamples. For each
measure, report point estimate + 95% bootstrap CI for the difference
(substrate − baseline) + effect size, not just p-value.

**Substrate-claim earns its name iff (FROZEN before
implementation):**
1. M1: `d̂_substrate < d̂_{B2}` with bootstrap CI excluding zero on
   the close-prototype regime.
2. M2: substrate has higher mutual-kNN reciprocity than B2 AND
   less hub-dominated degree distribution. Both must hold.
3. M3: substrate's β_0(r) shows discernible plateau (longest /
   second-longest persistence bar > 2.0) where B2's doesn't.

**Verdict rule:** at least 2 of 3 measures must show CI-excluding-
zero advantages on the close-prototype regime for the substrate
to earn its name. Single-measure wins on noisy regimes don't count.

### Prior art (verified by subagent, not invented)

- Levina & Bickel (2004) — MLE intrinsic dimension. NIPS.
- Facco, d'Errico, Rodriguez, Laio (2017) — TwoNN. *Scientific Reports*.
  [nature.com/articles/s41598-017-11873-y](https://www.nature.com/articles/s41598-017-11873-y)
- **Macocco, Glielmo, Laio (2023) — *Intrinsic Dimension Estimation
  for Discrete Metrics*. PRL.** [arxiv.org/abs/2207.09688](https://arxiv.org/abs/2207.09688)
  *The load-bearing citation; ID under Hamming distance.*
- Ansuini, Laio, Macke, Zoccolan (2019) — Applied TwoNN to deep-net
  representations across depth. NeurIPS.
- Naitzat, Zhitnikov, Lim (2020) — Topology of Deep Neural Networks
  via Betti numbers. JMLR.
- Mattia et al. (2025) — Local intrinsic dimensions of contextual
  LLM representations. [arxiv.org/pdf/2506.01034](https://arxiv.org/pdf/2506.01034)
- KV-cache low-dim structure precedents: ACL 2025
  ([aclanthology.org/2025.acl-long.703.pdf](https://aclanthology.org/2025.acl-long.703.pdf)),
  arXiv 2603.04427. **Neither uses Hamming/trit metrics** — that's
  the substrate's claimable novelty.

### Failure modes (pre-registered)

- **M1 null result (d̂_substrate ≈ d̂_{B2}):** the third state is
  decorative. **Highest-information null;** would force a
  substantial vision-claim narrowing.
- **M1 negative (d̂_substrate > d̂_{B2}):** substrate's threshold
  extraction adds noise rather than structure. Forces re-examination
  of τ tuning.
- **M2 k-sensitivity:** clustering coefficient is k-dependent.
  Mitigation: require substrate-vs-baseline ordering stable across
  k ∈ {5, 10, 20, 50}.
- **M3 plateau in both substrate and B2:** cluster structure is in
  the K-values intrinsically, not contributed by threshold
  extraction. The M1 null in different guise.
- **General correlational caveat:** these measures show
  structure-of-representation, not downstream utility. A separate
  test (does substrate routing using these signatures outperform
  B2 on a downstream task?) is what closes the loop on
  "load-bearing vs decorative." Phase α is a diagnostic, not a
  utility experiment.

## What this pre-research commits the project to

Implementing the diagnostic is an **estimated 1-2 day work-unit**:
- Compute K-signature dumps from existing ACTV2 data via
  threshold_extract at τ=5000.
- Build distance matrix.
- Apply M1, M2, M3 to substrate + B1, B2, B3.
- Bootstrap CI.
- Verdict per pre-registered criteria.

The Phase α deliverable would be a journal entry citing this
pre-research and recording per-measure results + bootstrap CIs.

## Discipline notes (load-bearing)

This document **is not** a frozen pre-registration. A frozen
pre-registration would be committed BEFORE implementation, with
FROZEN sections and explicit "modifications require justification"
clauses (per the pattern from `td27_7_phase_a_2026-05-11.md`).

When implementation is undertaken, a follow-on commit will:
1. Copy the FROZEN sections (success criteria, baselines, measures)
   verbatim into a new pre-reg journal.
2. Lock them before the first line of implementation code.
3. Record any modifications with explicit justification.

The split-commit pattern (pre-reg → implementation → result) is the
local antibody for the pre-verdict overclaim pattern caught 5+
times this session.

## Cross-references

- Vision memory: `feedback_pure_ternary_routed_architecture.md`,
  `project_vision.md`, `feedback_coherence_not_bit_parity.md`.
- Substrate-claim scope discipline: `feedback_substrate_claim_scope.md`.
- P0-3 precedent for regime-aware substrate claims:
  `docs/FINDINGS.md` Axis 3.
- Pre-registration pattern: `journal/td27_7_phase_a_2026-05-11.md`.

## Sign-off

This pre-research converts "the audits might have been measuring
the wrong thing" from speculation to a concrete experimental
program with peer-reviewed methodology. The Macocco PRL 2023
citation grounds intrinsic-dimensionality measurement under
Hamming distance — exactly the substrate's metric. The B2 baseline
(sign-only, drops zero state) is the falsification test for the
project's strongest claim ("base-3 carries information base-2
collapses").

Whether the substrate earns the vision claim depends on
implementation results that don't exist yet. This document just
makes the test feasible.
