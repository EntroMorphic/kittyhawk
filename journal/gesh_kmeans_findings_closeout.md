---
cycle: gesh_kmeans_findings
phase: CLOSEOUT
date: 2026-05-02
scope: post-synthesize observations on the LMM cycle deployed against the multi-prototype bank investigation
companions: gesh_kmeans_findings_{raw,nodes,reflect,synthesize}.md
status: COMPLETE — no mid-cycle reframings; awaits W1/W2/W3 measurements in next cycle
---

# Closeout — gesh_kmeans_findings

The cycle ran cleanly to a SYNTHESIZE commitment. No mid-cycle observations have surfaced that contradict the synthesize phase. Closeout is thin.

## What the LMM cycle revealed that direct analysis hadn't

Direct analysis of the data ("training hurt at high T → maybe overfitting") gave us H1/H2/H3 as plausible mechanisms but didn't surface what the cycle's REFLECT did:

1. **The Phase A "lattice update earns its place" claim was scope-misnamed in the journals.** The MNIST inversion at high T isn't a contradiction of Phase A — it's a scope reduction. Phase A claimed too much; the data was always specific to single-prototype regime. The journals carry the over-broad statement.

2. **The bank-vs-training framing was wrong, not just incomplete.** Earlier I described it as "bank is the bottleneck; training adds on top." The data says the relation is **non-additive and possibly antagonistic**. At T=10 training adds +6.8pp; at T=80 training subtracts −2.7pp. Sign-dependent on bank architecture. **That's not "additive," that's "competing levers."**

3. **The cheap-vs-expensive lever asymmetry is striking.** k=2 doubling gives +3.4pp for 0.02s of compute. Training gives +6.8pp for 88s of compute. The substrate-claim story has been emphasizing the expensive lever (training) when the cheap lever (bank) does most of the work.

These aren't news in a strict logical sense — they were latent in the data. LMM's value here was making the implicit explicit so next-cycle scope (W1/W2/W3) can't drift away from these constraints.

## What did NOT surface

- No "the bank IS the substrate"-level reframe à la Phase A's "the lattice IS the geometry." We had a meaningful framing shift but not a structural one.
- No suggestion that the substrate-claim itself is wrong. The substrate's primitives (ternary projections, packed-trit Hamming, SDOT matmul) all work fine. What's narrower is the *consumer-architecture story* built on top.
- No falsified prior finding. C2 (random > identity) survives, Finding 3 (capacity floor at low sig_dim) survives, the SDOT cleanup is unaffected. What changes is the *interpretation* of "training value-add" at consumers richer than the Phase A class-mean.

## The SYNTHESIZE commitments worth restating

Three workstreams, in priority order:

- **W1**: multi-seed validation of C3 (training hurts at high T). 3 seeds at the existing config; paired-CI verdict.
- **W2**: H2 mechanism test (frozen-bank trained-R). Single decisive run.
- **W3**: doc-currency scope-qualifier pass on Phase A claims (gated on W1 confirming).

## The methodology meta-rule held this cycle

This cycle's REFLECT explicitly applied the recently-promoted "match the scope of evidence to the scope of claim" rule to its own work and downgraded C3 from FINDING to OUTCOME pending multi-seed. That's the rule eating its own dog food. Worth recording.

The pattern: when a cycle measures a directional effect and the magnitude is comparable to single-seed noise (here ≈ 2–3pp), the cycle's own SYNTHESIZE must commit to a multi-seed gate before it can call the effect a finding. **The rule isn't an external auditor; it's a constraint the writer applies to their own cycle.**

## Loop-back triggers from this closeout

Standard set, restated for the next cycle:

- **Back to RAW** if W1's multi-seed verdict introduces effects (e.g., bimodal trained-R distribution across seeds) the current node set can't explain.
- **Back to NODES** if H1/H2/H3 mechanism tests *all* fail. We'd need a fourth, currently-unhypothesized cause.
- **Back to REFLECT** if W1 produces a CONFIRMED verdict but the magnitude is much larger than expected (e.g., −15pp instead of −2.7pp). The reframings would need to extend further.
- **No loop-back** if W1 confirms C3 with mild magnitude and W2 demonstrates one of H1/H2/H3. Then doc-currency cascade (W3) executes and the next cycle moves on.

## Methodology note for future kmeans cycles

If next cycles continue investigating the bank-vs-training relationship, two patterns are worth carrying forward:

1. **Always measure random-R baseline at the same bank shape as trained-R.** The "training contribution" number is meaningful only against the matched random-R baseline. Comparing trained-R-with-bank-X against random-R-with-bank-Y conflates two effects.

2. **Decompose the comparison axis explicitly:** when reporting numbers, name the axis being varied (sig_dim, k, T, top_k, training-on/off, etc.). The MNIST measurements in this cycle used a 5-axis space; only single-axis variation gives clean attribution.

These aren't new rules — they're applications of the meta-rule to this specific problem domain. Worth flagging since the next cycle will hit the same pattern.

## Closeout methodology check

This cycle's structure mirrored the gesh_findings cycle (RAW → NODES → REFLECT → SYNTHESIZE → CLOSEOUT). Each phase produced substantively new content:

- RAW: data dump with explicit "what's NOT measured" section.
- NODES: claims/hypotheses/anchors split, with severity tags.
- REFLECT: pressure-tested the prior framing, surfaced the wrong-frame/right-frame pairs.
- SYNTHESIZE: pre-committed gates with verdict thresholds.
- CLOSEOUT: thin if SYNTHESIZE held; this is thin.

The pattern is mature. Future cycles should follow the same structure unless a structural reframe surfaces (which would warrant a heavier closeout).
