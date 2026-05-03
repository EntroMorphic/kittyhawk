---
cycle: gesh_kmeans_findings
phase: REFLECT
date: 2026-05-02
scope: pressure-test the reference frame; surface what the findings hide
companions: gesh_kmeans_findings_{raw,nodes}.md
status: critical
---

# REFLECT — gesh_kmeans_findings

The data is clean. The interpretation has uncomfortable implications for prior framings, particularly Phase A's headline claim. Let me push.

## C3 (training hurts at high T) inverts a Phase A claim

The Phase A.2 sweep concluded: *"Lattice update earns its complexity in the compression regime — gain plateaus at +8pp at sig_dim ∈ {16, 32}."* That measurement was on the synthetic with single-prototype bank.

This cycle's measurement: at T=80 (k=8 multi-prototype) on MNIST, **lattice update earns negative gain**. −2.7pp.

These are not in conflict in a strict logical sense — different dataset, different bank, different sig_dim regime. But they're in conflict at the **substrate-claim narrative** level, where "lattice update is the right training mechanism for this substrate" was the load-bearing claim. If lattice update hurts at higher bank capacity, the claim narrows: **lattice update earns its place at low bank capacity.** That's a meaningful scope reduction.

The Phase B closeout's Path A recommendation ("richer consumer with multi-table LSH") was predicated on lattice update helping more once the bank had room. The data says the opposite: at richer bank, lattice update helps less or hurts. Path A's framing needs to be revisited — the next-cycle work isn't "lattice update + richer bank"; it might be "richer bank, no training" or "richer bank + different training objective."

## The bank-vs-training framing was wrong

I framed it earlier as "bank is the bottleneck; training adds on top." The data says the relation is non-additive and possibly negative:

| Consumer architecture | Random R | Trained R | Training contribution |
|---|---:|---:|---:|
| T=10 single-proto | 50.0% | 56.8% | **+6.8pp** |
| T=80 k-means | 64.1% | 61.4% | **−2.7pp** |

Training contribution is sign-dependent on bank architecture. **The dominant lever is the bank; training is a small auxiliary that helps in some regimes and hurts in others.**

This is a meaningful reframing. The substrate-claim story has been "ternary projections + lattice update + class-mean bank, end-to-end." The data now suggests "ternary projections + multi-prototype bank, no training" might be the better-performing single-substrate-pure consumer. Lattice update may be optional or even contraindicated at the right bank size.

## The "+14.1pp from bank" finding has a confound I missed

C2 says bank-architecture change is +14.1pp at random R. But the comparison is:
- T=10 single-prototype, **class-mean** bank
- T=80 k-means with **k=8 prototypes**

Two things change simultaneously: the *number of tiles* (10 → 80) AND the *clustering algorithm* (class-mean → k-means). What if k-means at k=1 (which we know reduces to class-mean) gave a different baseline than the class-mean baseline measured separately? It didn't — C4 verifies they're bit-identical.

But there's still a confound: the k-sweep curve goes from k=1 (50.0%) to k=2 (53.4%) — that's a +3.4pp jump from a single algorithmic step (k-means with k=2 instead of class-mean). Whereas going from random R to trained R single-prototype gave +6.8pp. **Even just doubling tiles gets us about half the training gain — for 0.02s of compute vs 88s.**

This isn't a flaw in the data, but it surfaces that the "training adds value" framing was concealing how *cheap* the bank-capacity moves are. We were burning 88 seconds on training for what ~0.02 seconds of doubling tiles delivers.

## The single-seed risk is large here

Per the meta-rule we just promoted, this cycle has multiple single-seed cells:

- The 50.0% / 56.8% / 64.1% / 61.4% / 70.1% comparisons are all single-seed.
- The k-sweep is single-seed.
- The trained-R + k-means is single-seed.

If we believe our own rule, **C3 (training hurts at high T) is currently OUTCOME, not FINDING.** A different seed could show training helps by +3pp instead of hurting by −2.7pp. The −2.7pp magnitude itself could be a 1σ noise excursion.

We should acknowledge this and run multi-seed before promoting C3 to a finding. Otherwise we're failing our own methodology rule.

## The mechanism story is rich but untested

H1 (loss signal disconnects), H2 (k-means R-sensitivity), H3 (per-batch overfitting) all point to plausible mechanisms for why training inverts. None are tested. **Three candidate mechanisms, three different remediations, none falsifiable yet.**

Following the Phase B mechanism-test pattern: if we want C3 to be a finding with a story, at least one of H1/H2/H3 needs a probe that demonstrates or rules out its mechanism. Cheap candidates:
- H1: budget sweep (cheap; 4 runs at 25K/50K/100K/250K).
- H2: frozen-bank variant (cheap; one config flip).
- H3: batch-size sweep (cheap; 3-4 runs).

Without these, C3 is "we observed this; our explanations are guesses."

## Phase A's "lattice update" framing in journals is now misleading

Several documents (Phase A.2 closeout, gesh_findings doc, etc.) say things like "lattice update earns its place" or "training contributes +8pp in compression." These statements were true at the configurations measured. They're now misleading without context, because:

- They imply training is universally helpful.
- The substrate-claim path's reader might infer "we should always use the lattice-update mechanism."
- The MNIST measurements show this isn't true.

**Doc-currency cascade is needed**: the existing narratives need a "scope qualifier" pass. "Lattice update earns its place at single-prototype bank in compression regime" is the more honest version.

## What's the right substrate-claim measurement now?

A2 (MNIST is the regression-guard, not the substrate-claim primary) says we shouldn't over-update on MNIST. But the MNIST data here is informing our entire mental model of how training fits the substrate. That tension should be acknowledged.

If MNIST keeps producing "training hurts" at high T, the Go-positions substrate-claim measurement might also produce this pattern. Then the substrate-claim becomes "ternary projections + multi-prototype bank" and lattice update is a footnote, not a feature.

Alternatively: lattice update may shine on Go positions even though it hurts on MNIST. Different domain structure. **We don't know until we measure.** But the prior assumption that lattice update is generally helpful needs to be downgraded to a hypothesis.

## What surfaces from the wrong reference frame

**Wrong frame:** *"The lattice-update mechanism is the substrate's training story; bank construction is a fixed detail."*
**Right frame:** *"The bank constructor is the substrate's expressivity story; lattice update is an optional refinement that helps at low bank capacity and hurts at high bank capacity."*

**Wrong frame:** *"Training adds value on top of the bank's baseline."*
**Right frame:** *"Training and bank capacity are competing levers; investing compute in one vs the other is a real trade-off, not additive."*

**Wrong frame:** *"The Phase B Path A recommendation (richer consumer + lattice update) is the substrate-claim path."*
**Right frame:** *"Richer consumer (multi-prototype bank, multi-table composition) is the substrate-claim path. Lattice update may or may not contribute at richer consumers — currently negative on MNIST at one configuration."*

Each right-frame statement is supportable from the data; each wrong-frame statement is what the prior journals say.

## Loop-back triggers from this REFLECT

- **Back to RAW** if multi-seed validation of C3 produces a different sign (training helps at +1σ, hurts at −1σ). We'd be in a noise-floor regime and need more data.
- **Back to NODES** if H1/H2/H3 mechanism tests *all* fail — the inversion would have a fourth, currently-unhypothesized cause.
- **No loop-back** if multi-seed confirms C3 (training hurts) and one of H1/H2/H3 demonstrates the mechanism. Then C3 promotes to finding with a story; Phase A narrative needs scope-qualifier rewrites.
