# Cycle 2 Phase 2.5 probe — methodology findings

Per `journal/cycle2_design.md` Phase 2.5: 4 arms × 6 k values × 2 prompts
to validate methodology before scaling to the full 24-prompt battery.

## What the probe was supposed to validate

- The 4-arm structure (dense / random / routed / oracle) executes cleanly
- The trajectory measurement (k ∈ {128, 64, 32, 16, 8, 4}) produces
  meaningful data at each k
- Pre-commit gates from `partB_experiments_synth.md` are
  measurable

## What the probe actually validated

**Methodology validation: PASSED with three corrections.**

### Finding 1: 4-arm structure works cleanly ✓

All four arms execute. Dense bit-exact unchanged. Random/routed/oracle
produce different outputs at small k, identical to dense at large k.
Smoke verifies the wiring.

### Finding 2: Token agreement is the WRONG metric

The probe initially used "token agreement % vs dense" as the quality
proxy. This is misleading at low k for two reasons:

**(a) Random can match dense by coincidence for a few tokens** before
diverging into degenerate output. Example (factual_capital, random k=4):
- agreement: 3.3% (matches dense for 1 token then diverges)
- text: "The answer is the capital of the answer is the answer is..." ← loop garbage

**(b) Routed often diverges to a DIFFERENT VALID continuation**, getting
penalized in token agreement despite being qualitatively correct.
Example (factual_capital, routed k=4):
- agreement: 20.0% (matches dense for ~6 tokens)
- text: "Answer: Paris. Paris is the capital of France. It is also..." ← coherent prose, correct fact

**Token agreement conflates "matches dense" with "is correct" and
"isn't garbage." Strict-pass classification (manual review of correctness)
is the right metric.**

### Finding 3: Oracle is NOT a true upper bound

"Top-k by |score|" picks the k positions with largest absolute scores.
But the renormalized softmax over those k positions REDISTRIBUTES mass
in a way that doesn't match dense softmax. Specifically:

- Dense softmax gives most mass to the largest POSITIVE score, with some
  mass spread over near-positive and small-negative scores.
- Renormalized top-k softmax puts ALL mass on the top-k subset, even if
  the ideal "match-dense" subset would include some near-zero scores
  the renormalization mishandles.

Empirical: oracle k=8 on math_div produced 40% token agreement while
routed k=8 produced 26.7% — but oracle had a "144 divided by 12 equals
12\n144 divided by 12..." loop while routed had "What is the result of
144 divided by 12?" — different valid output.

The true upper bound on post-hoc top-k would require combinatorial
search over all k-subsets that minimize divergence from dense softmax.
That's intractable; oracle-as-implemented is a strong baseline, not a
ceiling.

**Implication: pre-commit gate "no k value gets routed within 20pp of
oracle" is meaningless if oracle isn't actually the ceiling. Drop that
clause from the falsification gate.**

### The actual signal: routed > random on text quality at small k

Strict-pass classification of the probe data:

| prompt | dense | random k=4 | routed k=4 | oracle k=4 |
|---|---|---|---|---|
| factual_capital | ✓ "Paris. britannica.com..." | ✗ (loop) | ✓ "Paris. capital of France..." | ✓ "Paris. capital of the country" |
| math_div | ✓ " 12\n}\n\nQuestion..." | ⚠ ("That's the answer... 1. However") | ✓ "12...What is the result of 144÷12?" | ⚠ (loop) |

**Routed PASSES on both prompts at k=4. Random FAILS or DEGRADES on both.**
This is a stronger signal than the token agreement numbers suggested.

## Refined methodology for Phase 2.6 (full battery)

1. **Primary metric: strict pass rate** (✓ / ⚠ / ✗) per (arm, k, prompt),
   manually classified after reading the full text.
2. **Secondary metric: loop heuristic flag** (mostly to flag suspicious
   outputs for manual review).
3. **Tertiary metric: token agreement %** (kept for reproducibility but
   NOT used for the gate verdict).

## Refined pre-commit gates

Per the original synth, with the oracle-clause dropped:

**PART-B EVIDENCE if all of:**
- At k=64, routed strict-pass rate within 10pp of dense pass rate
- At k=16, routed strict-pass rate beats random by >10pp
- The gap (routed − random) on strict-pass rate WIDENS as k decreases

**PART-B FALSIFICATION if any of:**
- Routed indistinguishable from random across the trajectory
  (within ±5pp at every k)
- Routed degrades faster than random as k decreases

**INCONCLUSIVE if:**
- Quality varies wildly per prompt and trajectory is noisy
- Wall-clock results contradict expected FLOP savings
  (implementation issue rather than thesis result)

## Scoping for the full battery

Probe took ~10 minutes for 38 runs. Full battery: 24 prompts × 4 arms ×
6 k = 576 runs. At similar pace: ~2.5 hours. Tractable in background
with monitoring.

Run plan:
- 24 prompts from `journal/inference_battery_v2_prompts.tsv`
- Each prompt: 1 dense + 6 random + 6 routed + 6 oracle = 19 runs
- Total: 24 × 19 = 456 runs (not 576 — dense is per-prompt-only,
  k-independent for it)
- Manual classification of ~456 outputs after the run

That's a meaningful amount of manual work. Manageable but not trivial.

## Decision: proceed with full battery using refined methodology

The probe confirms:
- Infrastructure works
- The qualitative pattern (routed > random at small k) appears real
- The 4-arm structure produces interpretable trajectory data
- Methodology refinements (drop token agreement, drop oracle ceiling
  clause) are well-defined

Phase 2.6 launch criteria from cycle2_design.md are met. Proceeding.
