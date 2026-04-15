---
date: 2026-04-15
scope: LMM cycle — can N_PROJ=16 reach >90%?
phase: RAW
---

# RAW: can 16 bits of signature carry 90% of MNIST?

The atomic probe says: 62% aggregate. Adaptive voting predicted 62.75%. The user thinks >90% is on the table. Let me think honestly before reaching for a plan.

First-order ceiling. With 16-bit signatures and k=7 voting over 60K prototypes, the maximum any pure-signature-k-NN strategy can reach is bounded by the partition breakdown: 75.85% tied-min + 15.62% elsewhere-top-10 + 8.53% nowhere. If we perfectly resolved tied-min AND perfectly resolved elsewhere, we'd hit 91.47%. That's the ceiling given the 16-bit hash and prototype set. To reach 90% means extracting 90 / 91.47 = 98.4% of the resolvable signal. Currently rank-wt extracts 67.6%. That's a 30-point efficiency leap on top of a ceiling that's already close to the target. Hard but not impossible.

But "pure-signature-k-NN" is a choice, not a law. If the 16-bit hash is a coarse index that narrows 60K → tied-set (avg ~10 prototypes), and we allow a *fine* resolver over just those candidates, then the question changes: can we pick the right class from a small candidate pool? Pixel L1 over 10 candidates costs 10 × 784 ops ≈ 8K scalar ops — dwarfed by the 60K signature-distance scan we already did. The cascade pattern is almost free.

So the real question: does the user mean "strictly N_PROJ=16 hash alone" or "an architecture whose primary index is N_PROJ=16"? The former is an information-theory question. The latter is an engineering question. I'll write the cycle to honor both readings.

Information-theory read. 16 bits can distinguish 2^16 = 65536 unique signatures over 60K prototypes — enough in principle to give each prototype a unique code, but a *random* projection doesn't achieve near-uniqueness. The probe shows ~52% of query signatures collide exactly with a training signature. So signature space is heavily collapsed — we occupy ~40-50% of the 2^16 codebook. That collapse is why tied sets exist.

What can we do within 16 bits? The atomics give some clues. Tied-min average has 2.10 distinct classes. So the mode is a two-class or three-class discrimination. If we had a sharp per-class prior within the tied set, we could resolve maybe 75-85% correctly — which is roughly what the vote rules already achieve. So pure voting has hit near-ceiling for the tied-min partition: you can't do much better than "which class has the most members in the tied set" when you have no other signal.

Wait. That framing is important. The tied-min partition currently resolves at 77.65% (rank-wt). Is that near-ceiling for what voting alone can extract? What is the ceiling? If the tied set has 60% class A and 40% class B, and the query is equally likely to be A or B a priori, the Bayes-optimal vote picks A and gets 60% right. But we're not reading off just a fraction — we're reading the *dominant* class, and across 7585 queries we get 77.65%. The ceiling is whatever the per-query class-purity-of-tied-set averages out to. If the average tied set has 60% correct-class purity, 77.65% ≈ that ceiling. So majority/rank-wt may already be near-Bayes-optimal for tied-min-only.

Which means: to exceed 77.65% on tied-min, you NEED additional signal beyond the 16-bit signature. That's the cascade argument. Or: you need to USE the signature-bits differently than just popcount.

Where can additional signal come from, staying "at N_PROJ=16" in spirit?
1. **Cascade**: 16-bit index, then pixel L1 on tied set. Not "N_PROJ=16 only" but "N_PROJ=16 as primary index".
2. **Multi-probe / multi-hash**: M independent 16-bit hashes with different projection seeds. Each query does M hashes; candidates are intersection or union. M=4 gives effectively 64 bits but with structural locality.
3. **Secondary 16-bit resolver**: *inside* the tied set, use a different 16-bit projection to resolve. Candidate pool already small, so cost is O(tied_count * 16) — tiny.
4. **Trit values not just Hamming**: the sig is 16 trits. Hamming bucket = Hamming match on bit-pairs. But we could use weighted trit agreement (balanced) instead of popcount. Though at Hamming=0, all trits agree, so this doesn't help tied-min. It MIGHT help elsewhere partition.
5. **Bayesian per-class prior on tied set**: weight each tied candidate by log(class_prior / total_prior). Essentially modifies the vote count. Probably minor.
6. **Learned projection**: replace random ternary projections with ones optimized for class separation. A "learned N_PROJ=16" hash. This is a different problem (training) but technically still 16 bits at inference.

Strategies 1 and 2 most promising. Strategy 3 is elegant because it doesn't require any data not already in the sig index — just reuses a second hash. Strategy 6 is the most interesting if we want to stay true to "16 bits only": could a learned 16-bit hash separate MNIST at 90%? LSH theory says random projections are near-optimal for arbitrary data, but for MNIST specifically — with strong class structure — a *supervised* hash might do much better.

I'm honestly unsure whether a learned 16-bit hash can hit 90%. MNIST has 10 classes with ~log2(10) ≈ 3.3 bits of label entropy per query. 16 bits is ~5x the label entropy. In principle, plenty of room. But random projections waste most of their bits on intra-class variation. A learned hash could concentrate bits on inter-class separation. That's exactly what MNIST-specific hashing algorithms (ITQ, BRE) do.

But training a learned hash is a larger engineering task. The user said "deploy LMM"; that suggests architectural design, not a model training regime. So I lean toward cascade + multi-probe as the natural LMM outputs.

Tensions I'm feeling:
- "Stay at N_PROJ=16" vs "reach 90%" — these may be in direct conflict depending on interpretation.
- Cascade feels like cheating: we said 16-bit, but we're using pixels on the tie-break. Does that count?
- Yet the 16-bit stage *does* all the heavy lifting (60K → 10 candidates). The pixel stage is 10 candidates. So the architecture IS 16-bit dominated.
- Multi-probe feels cleaner: M × 16 bits, all signature-based, but it's actually N_PROJ=64 in disguise.
- Learned hash feels most "pure" but introduces supervised training, which is architecturally heavier.

Gut call: the LMM output should present cascade as the likely winner — the engineering answer — and multi-probe as the second option. Name the learned hash as a third arm but rank it lower because it departs from the routing-first signature philosophy.

One more angle. The elsewhere partition resolves at 24.65% (majority) or 18.76% (rank-wt). Both bad. What's the ceiling there? Correct is at rank 2-10 in this partition. With 7 votes, correct contributes only 1 vote. Majority needs correct to win by plurality over wrong classes that each have ≤ 6 votes. Usually there's one wrong class with 2-3 votes dominating correct's 1. So majority ceiling here might be around 30-40%, and rank-wt penalty drives it lower. To significantly improve elsewhere partition we need to *exclude* wrong-class top-1 prototypes — which means a cascade pixel check could drop them when pixel distance says they're not actually close.

Actually this is beautiful: pixel cascade on top-k candidates re-ranks them. If top-1 is wrong-class but pixel-far, pixel re-ranking demotes it; if rank 7 is correct-class and pixel-close, pixel elevates it. This directly attacks both partition failures.

OK I think the cycle's core insight is: **the 16-bit hash is a CANDIDATE FILTER, not a classifier. Treat it as filter, add a cheap resolver, and 90% becomes reachable. The error we've been making is asking the hash to do both jobs.**

RAW phase long enough. Next: extract nodes.
