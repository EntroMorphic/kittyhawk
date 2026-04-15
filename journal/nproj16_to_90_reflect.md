---
date: 2026-04-15
scope: LMM cycle — can N_PROJ=16 reach >90%?
phase: REFLECT
---

# REFLECT

## Core insight

**The 16-bit hash has been asked to do two jobs it cannot do simultaneously: filter candidates and classify them. Voting is a classifier on top of the filter — but the filter output is too coarse for voting alone to recover the label.**

The atomic probe showed why: tied sets are small (avg ~4-10), usually contain the correct class (75.85%), but often mix 2-3 classes (avg 2.10 distinct classes per tied set). Voting can extract the dominant class at ~77% accuracy on this partition, which is approximately the Bayes-optimal label rate given the class mixing. **Voting has hit its ceiling.** The remaining gap to 90% is not a voting problem — it's a discrimination problem on the filtered candidate pool.

This reframe unifies every observation:
- Why did rank-wt only gain +0.09% over majority? Because both were at the voting ceiling.
- Why did amplification with pixel k-NN fail globally? Because pixel k-NN is weak on *random* 60K candidates. But on a *filtered* tied set of ~10 candidates, the signal-to-noise is entirely different.
- Why is elsewhere partition so bad (24.65%)? Because voting can never rescue a correct class sitting at rank 7 against a wrong class at rank 1 with 3 prototypes. Only external re-ranking can.

The entire "vote-rule inversion at N_PROJ=16" debate is asking the wrong question. Both vote rules are near-optimal within their scope; neither can break 80%.

## Resolved tensions

**T1 resolved (spirit of N_PROJ=16 vs 90%).** The honest answer: "N_PROJ=16" should mean *16-bit hash as primary index.* A classifier that uses a 16-bit hash to narrow 60K → 10 and then resolves locally is still a "N_PROJ=16 architecture" — the hash does all the heavy work. Cost-accounting backs this: hash is 60K×16 bits of work; resolver is 10×784 bits of work. Resolver is a rounding error. We are not smuggling in N_PROJ=4096 under the table.

**T3 and T4 resolved (partition strategies).** Per-partition adaptive voting gains ≤ 1%; it is not the path to 90%. Cascade is the path. Within cascade, the same partition structure naturally appears: tied-min → re-rank by pixel L1 among tied; elsewhere → use pixel L1 to demote bad top-1. Cascade subsumes partition adaptivity.

**T5 resolved (learned hash).** Parked. Learned hashing is a legitimate research direction but lives outside the current routing-first engineering phase. Cascade is the immediate target.

## Prediction

**Cascade architecture reaches 85-91% at N_PROJ=16.** Derivation:
- 8.53% of queries are unrecoverable (correct not in top-10). Loss: 8.53%.
- Remaining 91.47% has correct class somewhere in top-10. If pixel L1 over ≤10 candidates can pick the right one at 95%+ accuracy (plausible for MNIST — classes are very separable in pixel space over small candidate pools), we get 91.47% × 0.95 ≈ 87%.
- If pixel L1 does near-perfectly (98%), we get 89.6%.
- If the candidate pool is widened slightly (top-20 instead of top-10), ceiling rises to ~95.5%, and at 95% resolver accuracy we'd hit 90.7%.

So 90% is on the edge of cascade feasibility. Top-K selection (K=10 vs K=20 vs K=50) and resolver choice (pixel L1 vs pixel L2 vs centroid distance) are the knobs.

## What the prior work predicted this

The "mechanism-cycle" prediction that rank-weighted would fail in highly-tied regimes — confirmed.
The "amplification negative result" established that pixel fallback fails when the fallback set is random. The missing piece was recognizing that a *filtered* fallback set is a different regime.
The LSH literature is unambiguous: hash-then-resolve is the standard pattern. Glyph has been implicitly using hash-as-classifier.

## The real question for SYNTHESIZE

Not "can we hit 90%" — the answer is "probably yes via cascade." The real question is: **what is the minimum-complexity cascade that gets us there, and how do we stay honest about the architecture?**
