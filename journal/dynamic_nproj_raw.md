---
date: 2026-04-16
phase: RAW
topic: Dynamic N_PROJ — resolution-adaptive routing cascade
---

# Dynamic N_PROJ — RAW

Unfiltered thinking. Honest. Doubts included.

---

This idea feels right in a way that radius-aware and density-mixing didn't. Those were trying to extract signal from geometry that was already computed — reweighting the same information. This is about giving the system MORE information when it needs it. The scaling curve already proves that wider projections produce better accuracy. The seed-overlap proves that the system knows when it doesn't have enough. The cascade just connects those two facts.

But I have doubts.

First doubt: the confidence gate. We measured per-table vote agreement on always-right (24.4%) vs always-wrong (15.3%) queries. That's a clean separation in AGGREGATE. But is it clean per-query? A threshold at, say, 20% (13/64 tables) will have false positives (always-wrong queries that happen to get 13+ votes on one seed) and false negatives (always-right queries with only 12 votes on this seed). The seed-overlap showed 5.8% swing queries — those are exactly the queries the gate will misroute. I'm worried the gate is less clean than the aggregate numbers suggest. Need to look at the actual per-query vote distribution, not just the mean.

Second doubt: will N_PROJ=32 actually move CIFAR-10 meaningfully? The scaling curve from the matrix sweep was on MNIST, which is 784-dim grayscale. CIFAR-10 is 3072-dim RGB. The random projection at N_PROJ=32 is still a linear combination of ~1024 random RGB subpixels (at density 0.33). Maybe the problem isn't resolution per-se but that random projections over raw RGB are fundamentally limited regardless of width. Natural images have spatial structure that random linear combinations destroy. If N_PROJ=32 only moves CIFAR-10 from 35% to 38%, the cascade barely helps. We'd need N_PROJ=256 or higher, which makes the cascade more expensive and the "cheap first pass" less valuable.

Third doubt: memory and build time. Three stages × M tables × sig_bytes × N_train is a lot of memory for CIFAR-10 (50K training images). Stage 1 = 50K × 4 × 64 = 12.8 MB. Stage 2 = 50K × 8 × 32 = 12.8 MB. Stage 3 = 50K × 16 × 16 = 12.8 MB. Total ~38 MB — fine for Apple Silicon. But build time: N_PROJ=32 over 3072 dims is ~2× the projection cost of N_PROJ=16, and N_PROJ=64 is 4×. Build time at N_PROJ=16 was already 38 seconds. Full three-stage build would be ~38 + 76 + 152 = ~4.5 minutes. That's a lot of upfront investment per experiment. Not blocking, but worth noting.

Fourth doubt: is the "cheap first pass" actually cheap enough to matter? At M=64 tables, each query probes ~4500 buckets (CIFAR-10 data). The SUM resolver scans ~12K candidates × 64 tables. That's the dominant cost. If Stage 2 needs M=32 tables at N_PROJ=32, the resolver cost is 32 tables × 8-byte sigs instead of 64 × 4-byte sigs — comparable. The "savings" from early exit are real but maybe not 2× — more like 30-40% wall-time reduction for the easy queries.

Fifth doubt: threshold calibration. We need to set K (the confidence threshold) before running the cascade, but the right K depends on the dataset. On MNIST, almost everything clears any reasonable threshold. On CIFAR-10, the threshold must be set low enough that 30-35% of queries clear it. On a new dataset, we don't know in advance. Do we need a calibration pass? That adds complexity.

Now the things I'm more confident about:

The self-similarity is real and beautiful. Every stage is literally the same code with a different n_proj parameter. No special cases, no if-else at the architecture level. This is the kind of design that feels right in the NORTH_STAR sense — it emerges from the primitives rather than being bolted on.

The monotone accuracy property is genuinely valuable. You can't hurt accuracy by adding stages. The worst case is "you spent compute on escalation and got the same answer." The best case is "you fixed a failure that the first pass missed." There's no regression risk.

The fact that it falls out of the existing infrastructure is compelling. We don't need new library modules, new resolver variants, or new primitives. We need uint64 bucket keys (planned anyway) and a confidence gate (already measured). The cascade is 50 lines of tool-level orchestration on top of the existing library.

The confidence metric choice matters. Per-table votes are intuitive but coarse — M=64 bins into integers 0-64. SUM margin is finer-grained but requires tracking the runner-up. I think SUM margin is better because it's continuous and doesn't lose information by discretizing into "which class won this table." But votes are simpler to implement and explain.

One thing I haven't thought about: what happens to the candidate union across stages? Stage 1 builds a union of candidates from N_PROJ=16 signatures. Stage 2 builds a completely new union from N_PROJ=32 signatures. The two unions may overlap (same training prototypes found via different signatures) or not. The Stage-2 union doesn't benefit from Stage-1's filtering — it starts from scratch. Is there a way to warm-start Stage 2 from Stage 1's candidates? Probably not cleanly, since the signature spaces are different. But it's worth thinking about whether the Stage-1 winner (even if uncertain) should be injected into Stage-2's union as a "hint."

Actually, that's interesting. What if Stage 2 doesn't build a full union from scratch, but instead re-scores Stage-1's top-K candidates using the wider signatures? That's much cheaper than a full Stage-2 build — no bucket index needed at N_PROJ=32, just encode the K candidates' wider signatures and re-rank. But this only works if the right answer is IN Stage-1's top-K, which by the oracle data it always is (oracle=100% at M≥8). So the re-ranking variant is feasible and dramatically cheaper.

Wait. Re-ranking Stage-1's union with wider signatures. That's different from the full-cascade design. Let me think about this.

Full cascade: build independent tables at N_PROJ=32, probe from scratch, build new union, resolve. Cost: full build + probe + resolve at wider N_PROJ.

Re-ranking: take Stage-1's ~12K candidates, encode their N_PROJ=32 signatures on the fly (or pre-compute at build time), compute sum_dist at the wider N_PROJ for just those candidates, pick the argmin. Cost: N_candidates × M₂ × popcount_dist at 8 bytes. No bucket build, no probing.

The re-ranking approach is MUCH cheaper. And the oracle data says the correct answer is always in the Stage-1 union. So the wider signatures would just re-order the candidates within the existing union. The question is: does re-ordering with wider signatures produce better accuracy than resolving with the original 16-trit signatures?

I think yes, because the 16-trit resolver is the bottleneck (75% per-table ties). A 32-trit resolver over the SAME candidate set would have far fewer ties and better discriminability. And we don't need to build any N_PROJ=32 tables or buckets — we just need the signature encoders.

This is a simpler, cheaper, and possibly equally effective variant. The full cascade (with independent tables at each stage) is the maximally powerful version. The re-ranking variant is the minimal-infrastructure version that still uses wider projection information.

I should present both in the NODES phase and let REFLECT decide which is the right first step.
