---
date: 2026-04-24
scope: LMM cycle — base-3-native benchmark selection
phase: REFLECT
---

# REFLECT: base3_benchmarks

## The core insight

**We have been asking the wrong question.** "How do we close the SSTT gap on CIFAR-10?" presupposes CIFAR-10 is the right benchmark. It isn't. The substrate we built is routing-first, base-3-first, inspectability-first — and CIFAR rewards none of those three. Every pp we gain on CIFAR is a pp *not claimed by the substrate's properties*; it's a pp won by brute tolerance for a hostile data format.

The correct question is inverted: **given a routing-first, base-3, inspectable substrate, what benchmarks would you choose to demonstrate its worth?** Start from substrate properties, not from the incumbent canon.

## What the RAW and NODES converge on

Three criteria define a base-3-native benchmark:

1. **Ternary-representable input without real information loss.** Go/chess board state, edit tags, Likert-3, DNA mutations, sign-encoded finance signals, categorical columns. Not: RGB images, audio, dense embeddings. The test is: *would a ternary-quantized version of the input give an informed human the same predictive capacity?*
2. **Routing specialization must be structurally rewarded.** The task must require qualitatively different computation for different inputs. Phase-structured games, multi-domain text, long-tail classification, hierarchical labels. Not: 10 uniformly-distributed balanced classes with overlapping signatures (MNIST, CIFAR).
3. **Inspectability credited or at least not penalized.** Bonus, not primary. For the near term, any benchmark that doesn't *actively punish* having a discrete signature (e.g. by rewarding continuous probability calibration exclusively) is fine.

All three must be live for the claim "routing-first base-3 wins" to be decisive. Image-classification canon has zero of the three.

## Why this reframes the last year of work

- `step_change` cycle measured the CIFAR representation tax precisely and concluded: we can't close the gap at the signature layer. True — but that conclusion was always conditional on continuing to chase CIFAR. **If CIFAR is the wrong proving ground, the measured tax is an answer in search of a different question.**
- `routed_autodiff` cycle's expert collapse finding told us routing isn't yet trainable at multi-class scale. Also true — but we discovered this on *yet another binary-legacy benchmark pattern*. Fixing the trainer to win CIFAR reproduces the same error.
- **The last year's accumulated evidence isn't wasted.** It's the measurements that give us the conviction to redirect. We now know: input representation caps us on continuous-image data (measured); routing caps us at multi-class when gate is frozen (measured); therefore the next move is a benchmark where *neither cap is structural* — because the input is already discrete and the routing has room to specialize.

## The actual constraint the cycle reveals

We're not compute-bound or trainer-bound. We're *benchmark-bound*. We've been trying to validate a routing-first substrate on data that doesn't need routing, in a representation that doesn't admit our native format. That's a category error, not a technical problem.

## Reading the RAW's most uncomfortable question honestly

"Should we leave image classification entirely?" The honest answer is: **for the purposes of validating the substrate's core claim, yes.** Image classification can come back later as a *transfer-of-learnings* target once routing + base-3 specialization is proven elsewhere. Trying to validate routing on images first is like trying to validate a new sorting algorithm on single-element arrays.

This doesn't mean we abandon the MNIST/Fashion/CIFAR artifacts — they are useful regression suites and substrate-capability demonstrations. It means we **stop treating CIFAR-gap-closure as a primary metric** and stop optimizing toward it.

## Which direction the substrate properties point toward

Ranking the candidates from NODES against the three criteria:

| Candidate | Ternary input | Routing load-bearing | Current substrate ready |
|---|---|---|---|
| Tabular | Partial (categoricals yes, continuous needs quantization) | Partial (long-tail helps) | YES (loader only) |
| Board-game state (Go/chess) | **YES natively** | **YES (phase structure)** | Mostly (needs value/policy head) |
| Sentiment-with-neutral / finance direction | **YES natively** | Partial | NO (no text embedding) |
| Extreme classification | Maybe (label-space yes) | **YES (required)** | NO (no text embedding) |
| Compositional (SCAN) | Partial | **YES (structural)** | NO (no seq2seq) |
| Custom synthetic | Tunable | Tunable | YES (build it) |

**Board-game state is the uniquely strongest fit on substrate-property alignment AND on current capability.** It has native ternary input (empty/own/opponent), clear phase-based routing load, doesn't require embedding infrastructure we don't have, and comes with a long history of existing datasets (pro game records, engine-generated positions) we can use without building self-play.

**Tabular is the safer second.** It tests the substrate in a genuinely applied domain (credit, medical, insurance) with real long-tail distributions, and costs only a loader to try. But it's a weaker substrate *claim* — XGBoost is itself a form of routing, so "we routed better" competes against a very mature routing incumbent.

**Custom synthetic is worth it only as a diagnostic**, not as a destination. Use it to verify that our understanding of "routing load-bearing" is correct before committing trainer work to board games or tabular.

## Anti-patterns to avoid

1. **Don't pick a benchmark because it's the next logical rung** ("MNIST → Fashion → CIFAR → ImageNet"). Those rungs are part of the wrong ladder.
2. **Don't build a text embedding to unlock NLP benchmarks now.** That's a substrate-extension cycle of its own and postpones the benchmark validation by weeks.
3. **Don't pick a custom benchmark and stop there.** It's a diagnostic tool, not a destination.
4. **Don't pick a multi-benchmark slate without a single primary commitment.** Measuring three things badly is worse than measuring one thing well. Pick the primary; use others as sanity checks.
5. **Don't abandon the image benchmarks as regression suites.** MNIST/Fashion/CIFAR test signature-layer behaviors that are still load-bearing in every future substrate change. Kept as regression, not as north-star.

## The scope this cycle should actually commit to

Not: "switch all work to board games." That's overcommitment before probing.

Is: **run direct_lsh on one tabular dataset and one ternary-state game dataset as zero-training baselines. Whichever produces a surprising baseline with no trainer work selects the benchmark for the next cycle.** This is low-effort (half-day), high-information (measures substrate-task fit directly), and defers the big trainer question until we know which task actually needs it.

Parallel probe, anti-commitment, data picks. Matches working-style memory: "derive from hardware, don't commit before probing."

## What this cycle's output should be

A substrate-choice decision, not a substrate-implementation commitment. Specifically:
- **Primary benchmark direction** (the one we'll build trainers toward).
- **Diagnostic benchmark** (the one we'll use to validate routing claims cleanly).
- **Regression suite** (the image benchmarks, explicitly demoted from primary to regression).
- **An immediate next step**: the half-day probe to check the selected direction is reachable with current tools.

The substrate team reorients; existing work becomes regression. Future cycles produce claims that earn substrate properties, not claims that ignore them.

## Residue for SYNTHESIZE

SYNTHESIZE needs to: (a) commit to the primary direction, (b) specify the diagnostic, (c) name the half-day probe concretely (what dataset, what tools, what outcome decides), (d) frame the demotion of image benchmarks with dignity — they did their job, they're not the enemy, they're just not the proving ground, (e) connect back to NORTH_STAR so the reorientation is justified by principle, not fashion.
