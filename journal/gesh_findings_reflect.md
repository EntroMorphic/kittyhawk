---
cycle: gesh_findings
phase: REFLECT
date: 2026-05-02
scope: pressure-test the reference frame; surface what the findings hide
companions: gesh_findings_{raw,nodes}.md
status: critical
---

# REFLECT — gesh_findings

The data is honest. The interpretation is more fragile than it reads.

## The benchmark is structurally rigged

**A1 is the load-bearing anchor and the load-bearing problem.**

The synthetic benchmark places K=16 dims of clean signal next to D−K=48 dims of pure uniform-random noise. This is the easiest possible signal/noise structure: the informative axis is *axis-aligned*, the noise is *uncorrelated across dims*, and the optimal projection is *literally a coordinate selection*. Any mechanism with the inductive bias to suppress some dims and emphasize others wins. PCA wins. Random ternary projection wins (by the CLT-style averaging hypothesized in H1). Lattice update wins. **They all win because the benchmark made winning the default.**

This means the comparative claims (C1, C2, C3) are weakly informative about *mechanism quality*. They tell us "lattice update can find the informative axis when the informative axis is unmistakable." They don't tell us "lattice update is the right mechanism for harder problems." The comparison we'd need is real data with correlated noise, structured signal, non-axis-aligned classes — exactly the regime the benchmark deliberately avoids.

Reframing: the +8pp compression gain is not "lattice update beats random projection." It's "lattice update beats random projection *on a benchmark where random projection already does most of the work*." The remaining mechanism difference is small.

## The strongest finding is the one we understand least

C2 — random ternary at sig_dim=D beats identity by +7pp — has the largest signal-to-seed-noise ratio in the sweep (+7.4 / ±2.1 ≈ 3.5σ). It's also the finding whose mechanism is most underspecified. The "implicit denoising" framing is plausible but unmeasured.

This is uncomfortable because **the docs lean on C2 as a substrate-affirming observation**: random ternary projection in the routing-first paradigm has emergent denoising properties identity lacks. If H1 (the proposed mechanism) is wrong, C2 still stands as a *correlation* but loses its *story*. The story is what makes C2 feel like evidence; without the story, it's a measurement that needs re-explanation.

The cheap mechanism test for H1 (~20 lines, see NODES) should run before C2 gets cited as substrate-supporting. Right now C2 is being asked to do work that requires a mechanism, and we haven't shown the mechanism.

## "Multi-seed validation" caught one error class but masks another

The single-seed → multi-seed correction was healthy. It promoted methodology to a project rule. But seed-noise is only one source of variance.

The deeper variance source is **dataset variance**. We ran multi-seed on the *same* training/test split (seeds 0x11111111u and 0x22222222u). A different prototype seed, a different sample seed, a different K, a different per-trit noise rate — all unsampled. The "multi-seed" label is misleadingly broad: we sampled over (init_R, train_batch) pairs, not over dataset realizations.

For the synthetic benchmark this matters less (it's parameterized; we could re-run with different `cfg.seed`). For the substrate-claim benchmark it will matter a lot: real data has one realization, and the substrate-claim is on the hook for performance on that one realization, not on a mean over re-sampled training sets.

This isn't an error in the current findings — they're scoped honestly to the (init, train) seed surface. It's a flag that "multi-seed" needs further granularity in future work: per-dataset-realization, per-init, per-train_batch are distinct variance sources, and which ones matter depends on the claim.

## A2 is a load-bearing constraint, not a hedge

The "Phase A is mechanism-validation, not substrate-claim" framing is sometimes read as throat-clearing — a humility note before the real claims start. It's not. **It's the claim that determines what the next cycle has to do.**

If Phase A were the substrate-claim benchmark, we'd be done: 98.6% at sig_dim=1024 with ±0.5pp seed noise is strong. We'd be writing a paper.

We aren't done because A2 says we aren't. The benchmark doesn't satisfy the substrate-claim's evidence requirements. The 98.6% number doesn't transfer.

Acting on A2 means the next cycle scope is **not "improve the synthetic numbers further"** and **not "add more sig_dims to the sweep"** and **not "test more lattice-update variants."** All three are tempting because they're cheap and they look like progress. They're also off-substrate-claim. The next cycle has to pivot to real data or it's punting on what Phase A was scoped to enable.

## What the data doesn't show that we keep treating like it does

The Phase A.2 docs and CHANGELOG describe findings as if they paint a picture of *the mechanism's properties*. Reading them charitably, they paint a picture of *the synthetic benchmark's properties under this mechanism*. These are different.

Examples of where the gap shows:
- "Lattice update earns its complexity in the compression regime" reads like a property of lattice update. It's a property of lattice-update-on-this-benchmark. Whether lattice update earns its complexity on real data is unmeasured.
- "Random ternary projection beats identity by +7pp via implicit denoising" reads like a property of random ternary projection. It's a property of random-ternary-projection-with-uniform-random-noise-dims. Real data has structured noise; the denoising story doesn't transfer for free.
- "Expansion saturates monotonically through 16× input dim" reads like a property of the expansion regime. It's a property of the expansion regime *given a benchmark whose Bayes-optimal ceiling is hit by sig_dim ≈ 1024 random projections*. Harder benchmarks may have unsaturated expansion regimes; we don't know.

The fix isn't to delete these statements — they're true within the conditions they're predicated on. The fix is to make sure the conditions stay attached to the claims when they get cited downstream. The docs as written do this in the "Conditions" sections of `sweep_dims_results.md`; the CHANGELOG does this less rigorously.

## What surfaces from the wrong reference frame

**Wrong frame:** "Phase A.2 measured the substrate's behavior across compression ratios."
**Right frame:** "Phase A.2 measured the lattice-update mechanism's behavior on a synthetic probe whose ceiling is the per-trit noise floor."

**Wrong frame:** "Random ternary projection has implicit denoising properties."
**Right frame:** "Random ternary projection wins on benchmarks where noise dims are uncorrelated; the mechanism for the win is hypothesized but not tested."

**Wrong frame:** "Lattice update earns +8pp."
**Right frame:** "Lattice update earns +8pp on a benchmark where +8pp is what's left to take after random projection has already taken most of it."

**Wrong frame:** "Phase A complete; ready for Phase B."
**Right frame:** "Phase A's mechanism-validation goals met; the substrate-claim path requires real-data work that has not started."

The reframings aren't critiques of the data. They're critiques of how the data is being used to support claims. Each right-frame statement is supportable; each wrong-frame statement overcommits.

## The benchmark's ceiling is structurally low

H3 in NODES: the Bayes-optimal classifier on this benchmark probably caps near 99–100%. We hit 98.6% with random ternary projection at 16× input dim. **There is essentially no headroom for any mechanism to demonstrate substrate-claim-worthy improvement on this benchmark.** The expansion saturation isn't telling us "the mechanism is great"; it's telling us "the benchmark is solved."

This further argues that next-cycle work has to move benchmarks. Squeezing the last 1.4pp out of the synthetic isn't worth doing.

## What should happen next, in priority order

1. **Real-data probe** — pick one and run it. Per `project_benchmark_pivot`, Go position evaluation is the primary; image canon as regression-guard is the secondary. Either is more informative than further synthetic sweeping.
2. **H1 mechanism test (cheap, parallel)** — measure whether per-output-dim contribution to inter-class Hamming distance correlates with informative-dim weight in R. ~20 lines. Upgrades C2 from correlation to mechanism, or falsifies the story.
3. **Synthetic K-sweep (deferred)** — H2's prediction (peak gain at sig_dim ≈ K) is testable with the existing harness. Worth doing once but not before #1 and #2.

What should NOT happen:
- More sig_dims in the sweep beyond 1024.
- More lattice-update variants on the synthetic.
- Additional seeds on the synthetic.

These are the easy moves. They also don't move the substrate-claim.

## Loop-back triggers from this REFLECT

- **Back to RAW** if a real-data probe reveals data behavior the synthetic findings can't explain (e.g., if real data shows the compression regime *hurting*, or the expansion regime *helping*). The current node set would need new observations.
- **Back to NODES** if H1's mechanism test falsifies "implicit denoising" — C2 becomes a node without a story, and the docs need a rewrite.
- **No loop-back** if the mechanism test confirms H1 and the real-data probe shows qualitatively similar regimes. The synthetic findings then transfer credibly.
