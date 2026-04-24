---
date: 2026-04-24
scope: LMM cycle — substrate_distance_refinement
phase: REFLECT
---

# REFLECT: substrate_distance_refinement

## Core insight

**The base3_go_probe exposed a pathology of int8 trit Hamming that we've been quietly stepping around for a year.** On MNIST and CIFAR we compensated for it implicitly — via gradient channels, multi-scale pyramids, per-region tau. Each of those enrichments is a feature-extraction step that precedes Hamming. We never called them "density-decorrelation," but that's part of what they do: they add features whose distribution is less density-scaling than raw pixel sign.

The Go probe stripped that enrichment away (raw board, nothing but trits) and the pathology surfaced nakedly: density-only beats Hamming on a Hamming-solvable task.

## What this cycle is actually about

Not "make Go work." Not "find a better metric." The real question is **what is the minimum step a ternary substrate must take before Hamming can see class structure?**

MS4 and gradient channels were the step for images. This cycle measures whether a parallel step exists for Go, and by implication for any sparse-discrete domain.

## Why density-normalized Hamming is the first experiment

It's a one-line change. If it fixes the phase-ID failure, the diagnosis is complete: the metric was the problem. We would then be able to retrofit this normalization into direct_lsh, routed cascades, everywhere Hamming is used, with trivial code churn and probable marginal gains.

If it doesn't fix phase-ID, the diagnosis is incomplete: the representation is also insufficient. That's a bigger finding, but one we need to make before committing to any substrate-wide change.

## Why local 3×3 contrast is the second experiment

It's the smallest representation enrichment that captures anything structural. Each cell's feature now depends on its neighborhood, not just its own state. If this works where raw doesn't, we've proven that the substrate's natural shape for discrimination is *local-structural-trits*, not *pixel-trits*. That's a substrate-design claim that's been implicit for a year and never explicitly tested.

## Why same-game retrieval is the third experiment

Phase-ID is a density-correlated task. Even a "fixed" Hamming might score well on phase while still missing actual positional structure. Same-game retrieval is density-controlled by construction (adjacent positions have similar density) so it isolates *positional* discrimination. It's the real test of whether the substrate sees structure.

## What could falsify the cycle's hypothesis

- If density-normalized Hamming and 3×3 contrast BOTH fail on phase-ID AND neither succeeds on same-game retrieval, the pathology is structural, not representational. Two failures of two candidate fixes in two independent dimensions is strong evidence that the substrate's current distance machinery is fundamentally insufficient for structured-discrete data. That would be an important finding — it would kill not just Go but any analogous target. Next cycle would be forced into learned routing or a more elaborate per-cell feature.

- If density-norm works but contrast3 doesn't, the representation is fine as long as the metric compensates for density. Implication: retrofit density-norm widely.

- If contrast3 works but density-norm doesn't, the representation was the problem all along. Implication: invest in local-feature extraction; every future benchmark gets a pre-processing stage.

- If both work, prefer contrast3 (richer, extensible to other features) but keep density-norm as a cheap fallback.

## What the cycle cannot answer

- Whether the same fixes generalize to images. The image pipeline already has MS4+gradient features; re-measuring there would need a controlled "raw only" test not in scope.
- Whether Go is ultimately the right primary benchmark. The `base3_benchmarks` synthesize's Go-first decision was conditional on the probe. Probe was RED. Even a successful distance fix doesn't automatically re-validate the Go commitment — it re-opens it.
- Whether learned routing could close the gap on the original raw-Hamming setup. That's its own cycle.

## NORTH_STAR alignment check

- **§4 (scaffolding sanction)**: density normalization and local-contrast encoding are both consumer-layer scaffolding for measurement — explicitly sanctioned.
- **§12 (no binary float in compute)**: both fixes stay in trit/int space. Density normalization is a single scalar division per comparison — a compute step, not a representation step. Acceptable.
- **§13 (training in consumer)**: no training involved in either fix. Both are substrate-layer candidates; if successful, they might graduate into libm4t primitives.
- **"Routing is essential in base-3"**: not directly at stake. This cycle is about the *distance* that routing would use. If a decent distance exists, routing on top of it becomes worthwhile; if not, routing has nothing to route.

## Anti-patterns explicitly rejected

1. Don't pile "fixes" — measure minimum viable first.
2. Don't re-engineer direct_lsh — keep the probe standalone.
3. Don't add learned components — this cycle is about non-trainable fixes so we can isolate the distance question from the training question.
4. Don't swap benchmarks if things look bad — stay on Go position data so results are comparable to the probe.

## What the cycle's output must be

- A 2 × 2 × 2 = 8-cell results grid across `{raw, contrast3} × {hamming, hamming_norm} × {phase-ID, same-game-retrieval}`, with the most informative 3–4 cells populated.
- A clear go/no-go call on whether the substrate fix candidates work.
- A decision on what the NEXT cycle is: trainer, or another distance-metric attempt, or pivot.
- Not a solution to Go; a characterization of whether a solution is within substrate-level reach.

## Residue for SYNTHESIZE

SYNTHESIZE needs to specify: (a) exact CLI/metric semantics so results are reproducible, (b) the target gates, (c) what the cycle explicitly does NOT try, (d) what each possible outcome implies for the next cycle.
