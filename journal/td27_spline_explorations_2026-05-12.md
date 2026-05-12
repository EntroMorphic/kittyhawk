# Spline / Nyström approximation explorations — pre-research

Records two rounds of spline-based ideas explored 2026-05-12. Status:
**PRE-RESEARCH, no implementation, no commit.** Both rounds tempered
by adversarial math/literature agents. Captured here so future cycles
know what was considered and tied to which validation premise.

## Round 1 — Spline as differentiable relaxation (rejected as step-change)

### Idea A: Smoothstep relaxation of `threshold_extract`

Replace the substrate's step-function `threshold_extract` with a cubic
Hermite spline (`y = 3t² − 2t³`, normalized over `[τ−δ, τ+δ]`) for
differentiable training. As `δ → 0`, recovers the discrete step
function exactly.

**Math verification (subagent):** sound. Cubic Hermite is C¹
continuous (NOT C² — `smootherstep` 6t⁵−15t⁴+10t³ would fix that).
Gradient at midpoint = 3/(4δ), scales as 1/δ. The polynomial is
literally Perlin's 1985 `smoothstep` from graphics.

**Prior art (subagent):**
- IR-Net Error Decay Estimator (Qin et al., CVPR 2020) — anneals
  tanh-based estimator from soft to hard.
- Quantization Networks (Yang et al., CVPR 2019) — sum-of-sigmoids
  quantizer with annealed temperature.
- Soft-then-Hard (Guo et al., ICML 2021) — explicit soft-to-hard
  annealing.
- Learned Step Size Quantization (Esser et al., ICLR 2020) — learns
  step size with custom gradient near transition boundaries.
- BinaryConnect, DoReFa-Net — STE with clipping (the precedent for
  the substrate's current approach).

**Verdict:** mathematically sound, but the "soft-anneal-to-hard"
pattern is heavily prior-arted with **modest, not phase-transition,
gains** over STE. The specific smoothstep polynomial isn't in QAT
literature but is mathematically equivalent in spirit.

**Most important failure mode for Glyph's specific use case:**
- Phase A.1's plateau at variable-N is a **cardinality-gradient
  problem**, not a pointwise-threshold-uncertainty problem.
- Per-element threshold relaxation gives gradient near the threshold
  but doesn't address "select one more / one fewer" — that's a
  competitive operation between candidates.
- **Gumbel-top-k** (Kool et al., 2019) or **SoftSort** (Cuturi
  et al.) are the better-targeted relaxations for Phase A.1.

### Idea B: Learnable spline-parameterized cost matrix

Replace fixed Hamming distance over substrate signatures with a
trainable 3×3 cost matrix `c[a, b]` for `a, b ∈ {-1, 0, +1}`,
parameterized as spline coefficients.

**Math verification (subagent):** under sign-symmetry (which the
problem has) and monotonicity, the 3×3 matrix collapses to **at most
1 free parameter** — the ratio of full-disagreement cost to
half-disagreement cost. "Spline parameterization" over 3 input
points has identical expressive content as 1-2 scalars with a
monotone constraint.

**Prior art (subagent):**
- Ristad & Yianilos, "Learning String Edit Distance" (IEEE PAMI
  1998) — EM-trained finite-alphabet substitution costs. The
  proposed work is a degenerate-3-symbol instance of exactly this
  framework.
- Hamming Distance Metric Learning (Norouzi et al., NeurIPS 2012) —
  learns codes against fixed Hamming (the opposite axis).
- Mahalanobis metric learning (Xing et al., NeurIPS 2002).

**Verdict:** mathematically sound but **largely notation over
existing infrastructure**. The substrate already has
`m4t_route_confidence_weighted_dist` (m4t/src/m4t_route.h:106-139),
a hand-designed instance of this exact family with entries in
{0, 1, 2, 3, 4}. The principled version of this direction is "make
that table trainable" — not a new spline anything.

### Verdict on Round 1

Both ideas:
- Mathematically sound
- Heavy prior art
- Modest absolute impact (single-digit % gains at best in
  precedented work)
- Don't address the specific failure modes the project is currently
  blocked on

**Neither warrants implementation cycles** in their current framing.
Salvageable elements: Gumbel-top-k for Phase A.1 H2 (already
pre-registered as the STE fallback in `td27_7_phase_a_2026-05-11.md`);
making `m4t_route_confidence_weighted_dist` trainable as a focused
~50-line change.

This was the **7th and 8th caught overclaim of the session.**

## Round 2 — Spline / Nyström approximation over the bank

Same-day follow-up after Round 1's tempering. The three ideas share
a common dependency: substrate bank has manifold structure that
landmarks can sparsely cover.

### Idea C: Soft routing via splined Nyström approximation

Replace hard top-k attention selection with soft attention weights
that decay smoothly with Hamming distance, parameterized as a
spline. Substrate angle: integer popcount distances → polynomial
weights, avoiding `exp()` in softmax.

**Pre-existing related work:**
- **Nyströmformer** (Xiong et al., AAAI 2021) — Nyström
  approximation for transformer attention, O(N) cost via
  landmark decomposition.
- **Linear Attention** (Katharopoulos et al., ICML 2020) —
  kernel-feature attention without softmax.
- **Performer** (Choromanski et al., 2020) — random-feature
  softmax approximation.

**Substrate-specific value:** modest engineering win. On ARM NEON,
`exp()` is ~5-10 cycles per element; degree-3 polynomial is 1-2
cycles. Per cost v3 (commit `51a7b53`), attention is 6.1% of
per-token cost at seq_k=1040; removing `exp()` from softmax could
shave maybe 1-2% off total. **Not transformational.**

**Pitfall:** the Z normalization still requires evaluating the
spline at every tile's distance — same O(seq_k) cost as dense
softmax unless we truncate to top-m closest, at which point it's
just soft-top-m attention (also precedented).

### Idea D: Bank interpolation between landmarks

If bank tiles are landmarks on a similarity manifold, splining
allows querying between landmarks for signatures not explicitly in
the bank. Hypothesized benefit: smooth generalization across
equivalence classes.

**Structural pitfall:** in substrate space (`{-1, 0, +1}^D`), the
natural interpolation between two tiles is per-cell averaging —
which lands in `[-1, +1]^D` (continuous), not the substrate's
discrete space. Downstream consumers expecting discrete tiles
can't use the interpolation without re-quantization, which collapses
back to an existing tile.

The "smooth response between equivalence classes" framing works
only if downstream consumes a **soft distribution over tiles**
rather than a single tile — at which point it converges to Idea C.

**Pre-existing related work:**
- VQ-VAE with soft retrieval (Gumbel-VQ, etc.)
- Implicit Neural Representations (NeRF-style continuous queries
  over discrete features)
- Continuous relaxations of categorical codebooks

**Substrate-specific value:** speculative. Depends on whether
interpolation preserves semantic structure in substrate space —
empirical question, not provable.

### Idea E: Bank compression via Nyström landmarks + spline coefficients

Pick `m << N` landmark tiles; express each non-landmark tile as a
spline combination of nearby landmarks. Storage:
`O(m·D + N·coeff_size)` vs `O(N·D)`.

**Math:** standard Nyström decomposition `K ≈ C W⁻¹ Cᵀ` adapted
to discrete bank.

**Concrete numbers for Glyph:**
- BitNet K-cache at seq_k=4096: 32 bytes/tile × 4096 = 128KB per
  layer per kv_head.
- With m=100 landmarks, 4-byte coefficients: 3.2KB + 16KB = ~19KB.
- **Theoretical compression: ~6.7×** on K-cache.

**Pre-existing related work:**
- **Quest** (Tang et al., ICML 2024) — landmark-based K-cache
  subset selection.
- NestedKV, hierarchical KV-cache compression schemes.

**Substrate-specific value:** modest. K-cache is already small
relative to model weights (1.7GB for BitNet 2B); absolute savings
in the low-MB range. Matters more for long-context regimes where
K-cache size becomes the actual bottleneck (seq_k > 8192).

### Verdict on Round 2

All three ideas are **better-grounded than Round 1** but not
step-changes. Common dependency: substrate bank has manifold
structure that landmarks can sparsely cover. **This is exactly the
question Phase α (`td27_geometric_prereg_2026-05-12.md`) was
designed to answer.**

**Implement order, IF pursuing:** Phase α → measure substrate's
intrinsic dimensionality vs B2 (sign-only baseline) → if substrate
shows lower intrinsic dimensionality (manifold structure
validates), then evaluate which of C, D, E is worth pursuing.

If Phase α validates substrate manifold structure:
- C (soft routing) → modest cost savings (~1-2% total)
- D (bank interpolation) → research direction; depends on
  re-quantization preserving semantics
- E (Nyström compression) → modest K-cache compression (~6-7×);
  matters more at very long context

If Phase α fails:
- C, D, E all lose their motivation. The substrate's
  approximation toolkit collapses to "polynomial softmax instead of
  exp() softmax," which is hardware optimization unrelated to the
  substrate's distinctive claim.

## Meta-pattern note

Round 1 and Round 2 together: **second instance of getting excited
about spline-based ideas, both rounds tempered by literature scan.**

Specifically:
- Round 1 ideas: smoothstep / cost matrix — heavily precedented,
  modest gains historically.
- Round 2 ideas: Nyström / soft routing / compression — well-precedented
  (Nyströmformer, Linear Attention, Quest), bounded absolute impact.

**The pattern says: substrate-distinctive value is more likely to
live in the geometric premise itself (Phase α: does substrate have
lower intrinsic dimensionality than binary?) than in any specific
approximation operation built on top of it.** Worth keeping that
orientation when evaluating future spline-or-approximation
proposals: ask the manifold-structure question first, then
evaluate the operation.

## Status

- Round 1 (smoothstep, cost matrix): **NOT PURSUING** in current
  framing. Salvageable elements (Gumbel-top-k for Phase A.1 H2;
  trainable `confidence_weighted_dist`) recorded separately.
- Round 2 (soft routing, interpolation, Nyström compression):
  **DOWNSTREAM OF PHASE α**. If/when Phase α validates substrate
  manifold structure, re-evaluate these three.

No implementation, no commit. This journal entry IS the deliverable
from these explorations.

## Open question worth recording

The user (in the conversation that prompted this journal) framed all
three Round 2 ideas as connected by "bank manifold structure." That
framing **explicitly bridges the empirical-philosophical gap** noted
earlier in the session: if the substrate has manifold structure that
binary projections don't, then approximation operations on the
substrate produce different results than the same operations on
binary — and the "base-3 carries information base-2 collapses"
vision claim has its first operational connection to a measurable
property.

**That's the load-bearing reason Phase α matters more than any of
the spline operations downstream of it.** Phase α validation would
not just enable C, D, E; it would provide the substrate's first
measurable point of distinction from binary on a property the
vision claim names. The spline operations are scaffolding that
becomes interesting only because of Phase α's underlying premise.

Recorded for future-cycle prioritization.
