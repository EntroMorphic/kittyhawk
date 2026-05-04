# Reflections: Path Forward After R1 Remediation FAIL

## Core Insight

The eight options for the path forward (A through H) reduce to a single more important question: **what kind of failure was R1?**

Three plausible framings, each implying a different next move:

- **F1: "Wrong rule."** Signature richness was the right axis; the dual implementation is wrong. → Fix by redesign (R1 v2, Option B).
- **F2: "Wrong axis."** Signature richness was never going to address the concerns; the consumer needs MORE CELLS, not RICHER CELLS. → Sweep sig_dim with sign-only as primary (Option D leaning toward "revert + dim sweep").
- **F3: "Wrong layer."** The concerns are substrate-level, not consumer-level. Expression-routing should accept sign-only and move on; vision claim #3 manifests in P1-1 or elsewhere. → Revert and pivot (Option F).

The R1 evidence we have does not distinguish these three framings. Picking A, B, F, or H without distinguishing them is choosing on intuition. The cheapest move that DOES distinguish them is a sig_dim sweep that runs BOTH rules side-by-side at multiple dimensions.

That's Option D, but framed correctly: not "do R3 next per the original plan," but "do R3 explicitly to test which framing of the failure is right."

## Resolved Tensions

**T1 (closure cost vs depth) — RESOLVED.** A cheap revert preserves momentum but locks in a framing without testing it. The right move is a small targeted experiment (3 days) that distinguishes the framings. After that experiment, the choice becomes principled rather than intuited. Depth wins, but only the kind of depth that resolves the actual fork.

**T2 (consumer-level vs substrate-level fix) — RESOLVED by reframing.** F2 and F3 explicitly predict that consumer-level rule changes won't help. F1 predicts they will. The sig_dim experiment tests the prediction:
- F1 prediction: dual at sig_dim=64 outperforms sign-only at sig_dim=64 on inter-class distance and partition-change rate.
- F2 prediction: sign-only at sig_dim=64 outperforms dual at sig_dim=16 on the same metrics; dual at sig_dim=64 doesn't add over sign-only at sig_dim=64.
- F3 prediction: both rules plateau at similar discrimination by sig_dim=64; further dim doesn't help.

The experiment can falsify any of the three. That's the discipline move.

**T3 (vision claim conflation) — RESOLVED by separation.** R-track work addresses vision claim #3 (substrate-distinctness in the consumer). P1 work addresses vision claim #1 (six primitives floor). They're independent. The R1 FAIL doesn't tell us anything about P1-1's prospects. P1-1 can proceed in parallel with any R-track choice. This is also the right framing for concern 1 (scope gap, vision claim #2 scaling): R2 was bundled with R1/R3 in the plan, but R2's prospects aren't gated by R1's verdict — only by R3's outcome (does the consumer have the discrimination capacity to scale).

**T4 (sunk cost vs honest pivot) — RESOLVED by naming the bias.** Three weeks invested in R1 is real but doesn't justify another two weeks on R1 v2 unless the data supports F1 specifically. The sig_dim experiment is the cheapest way to find out without committing to either "save it" or "abandon it" prematurely.

## Challenged Assumptions

**A1: "The closeout's three options (A/B/C) are exhaustive."** False. Options D through H exist. The closeout was written under time pressure right after the FAIL; the options weren't fully canvassed.

**A2: "Per-arity rules are inherently bad."** Partially false. They're bad as architecture; they may be acceptable as targeted optimization once principled. If the sig_dim experiment shows that arity-2 benefits from dual at sig_dim=16 while arity-1 doesn't, per-arity dispatch is a real finding, not an ad-hoc patch.

**A3: "P1-1 is downstream of R-track success."** False. P1-1 is independent. The R1 FAIL doesn't change whether exp/log primitives are needed for vision claim #2 to scale beyond the current vocabulary. If anything, the FAIL frees us to consider P1-1 sooner.

**A4: "The R2 plan's three-track structure is invariant."** False. The plan was sequenced for R1 success. With R1 FAIL, the sequencing has to be reconsidered. R3 with the dual rule alone would be wasted; R3 with both rules is the falsification experiment we need.

**A5: "Vision claim #3 must manifest in the expression-routing consumer."** Probably false. Vision claim #3 says "base-3 carries information base-2 collapses." This is a substrate-level claim. The substrate's affordances (third state for wildcards, dual-threshold magnitude bands, MTFP cross-exponent accumulator) don't have to all be exercised by every consumer. A numerical-computation consumer (P1-1 territory) might exercise them naturally; an equivalence-routing consumer (current track) might not need them.

## What I Now Understand

**The R1 FAIL is a fork in the road, not a setback.** Three framings of the failure imply three different next moves. The choice between them is empirically resolvable by a 3-day experiment (sig_dim sweep with both rules). After the experiment, the right move is determined by data, not intuition.

**The right next cycle is a focused sig_dim experiment, not a full R3 as originally planned.** The original R3 (per `PLAN_EXPRESSION_ROUTING_R2.md`) assumed R1 PASSed — it was meant to calibrate sig_dim for the new rule. The post-FAIL R3 has a different purpose: distinguish F1 from F2 from F3.

**P1-1 (close primitives floor with exp/log) is independent and worth starting in parallel.** The R1 FAIL doesn't change P1-1's prospects. Vision claim #2 cannot scale to "all mathematics" without exp/log in the vocabulary, regardless of which signature rule wins.

**Per-arity rules are an outcome, not a strategy.** If the sig_dim experiment shows the right rule depends on arity, that's a finding to be embraced. If it doesn't, per-arity rules shouldn't be adopted preemptively.

**The Laundry Method (from LMM) applies cleanly here.** Partition the question first, search within. The big partition is F1/F2/F3. Within whichever framing wins, the right move follows. Don't try to pick between A through H without first picking between F1/F2/F3.

## Resolved Choice (Provisional)

The next cycle should be a **focused sig_dim experiment** that tests both rules at sig_dim ∈ {16, 32, 64} on:
- Curated bank inter-class distance (does discrimination scale with dim?)
- Random-bank merger rate (does the equivalence partition stabilize?)
- Subagent-probe match rate (do mathematical-intuition probes still route correctly?)

Pre-committed thresholds (committed before running):
- F1 wins iff dual at sig_dim=64 has strictly better arity-1 inter-class min distance than sign-only at sig_dim=64.
- F2 wins iff sign-only at sig_dim=64 reaches inter-class min distance ≥ 6 AND dual at sig_dim=64 doesn't add ≥ 2 over sign-only at the same dim.
- F3 wins iff both rules plateau at sig_dim=64 with similar discrimination AND neither reaches min ≥ 6.

After the experiment, the choice between A/B/C/D/E/F/H is determined by which framing won.

**In parallel:** P1-1 (close primitives floor with exp/log) can begin design work. It's gated only by owner authorization, not by R-track outcomes.

## Remaining Questions

- For the sig_dim experiment, how should we extend the test-input set beyond 16 inputs? The current 16 were carefully chosen; sig_dim=64 needs 4× more, with care to preserve the symmetric / sign-flip-spanning property.
- Should the experiment also test sig_dim=128 and 256? Probably not for the initial cycle — diminishing returns and longer runtime. Keep to {16, 32, 64} unless results are inconclusive.
- If F3 wins, what does pivoting to P1-1 look like in practice? It's a substantial design cycle on its own.
