---
cycle: gesh_zero_signal_design (P0-1)
phase: REFLECT
date: 2026-05-02
scope: pressure-test the proposed primitives, the wildcard framing, and the substrate-novelty audit on each
companions: gesh_zero_signal_design_{raw,nodes}.md
status: critical
---

# REFLECT — gesh_zero_signal_design

The proposed primitives all pass the substrate-novelty audit (NODES table). That's necessary, not sufficient. Pressure-testing them on harder questions.

## Q1 — Is wildcard the right semantics for "zero in a tile"?

The wildcard interpretation says tile-zero means "match anything" — this position is class-irrelevant; any query value at this position is a free match. Per P1's cost table: (q=±1, t=0) → cost 0.

But there's an alternative interpretation: tile-zero means "this position averages out across the class" — the class is heterogeneous on this dim, and in some sense *both* +1 and -1 are *partial* matches; neither is a full match. Cost 1 (current ternary Hamming).

A third interpretation: tile-zero means "the class wants zero here" — the class is *characterized* by abstention on this dim. A query that has ±1 here is a *mismatch*, because the query asserts something the class doesn't. Cost 2 (full mismatch).

These three interpretations give *different* per-cell costs:
- Wildcard:        (q=±1, t=0) → 0
- Current Hamming: (q=±1, t=0) → 1
- Class-asserted:  (q=±1, t=0) → 2

**Which is right depends on what zero means in the class signature.** And that depends on how the bank was constructed:

- **Class-mean bank with sign-thresholding**: zero emerges when within-class samples *cancel* — the class is genuinely split. Wildcard interpretation: maybe (it's not that the class doesn't care; it's that the class doesn't agree). Current-Hamming: probably more correct. Class-asserted: too strong.
- **Wildcard bank constructor (P6)**: zero is *deliberately placed* at low-SNR positions. Wildcard interpretation: correct by construction.
- **Sparse-coded bank**: zero means "this dim was pruned." Wildcard: correct (don't penalize queries for asserting at pruned dims).

**Implication:** the wildcard distance kernel (P1) is correct ONLY paired with a wildcard bank (P6). Using P1 against a class-mean bank treats emergent ties as deliberate wildcards, which over-promotes ambiguous matches.

This couples P1 and P6 tighter than NODES suggested. **Cannot ship P1 without P6** (or without explicit consumer documentation that P1 is for use with wildcard-tiled banks). Otherwise we'd have a primitive that operationally distinguishes states based on a semantic interpretation the bank doesn't enforce.

## Q2 — Does the wildcard interpretation actually demonstrate substrate advantage, or is it just a different distance metric?

Push harder. A claim like "wildcard distance is substrate-novel" needs to survive: would base-2 be able to replicate the same routing decisions if it had a separate mask bit?

**Answer:** yes, base-2 with a mask bit *can* replicate the routing. The substrate's advantage is that the mask is *the same bits* as the data — no separate storage, no separate indexing, no separate compute for the mask. The substrate-novelty isn't that wildcards are impossible in base-2; it's that they're *free* in the substrate.

So the substrate-claim measurement for P1 must demonstrate not just "wildcards work" but "**wildcards work at zero additional storage and zero additional compute beyond the substrate's native packed-trit form**." The base-2 comparison has to include the storage and compute of the mask.

**Implication for verification:** the measurement is "compute cost per query × accuracy," not just accuracy. Base-2 with separate masks pays storage and compute overhead the substrate doesn't. If the substrate matches base-2's accuracy at lower compute cost, that's substrate advantage.

## Q3 — P3 (skip_zero_query) has an obvious overhead concern. Does it survive that?

Skipping K-iterations where activation is zero requires either:
- Pre-scanning the activation to build an index of nonzero positions (overhead per activation).
- Branching per iteration on zero-test (branch misprediction cost; defeats SIMD).

Either approach adds overhead. The skip-zero benefit must exceed this overhead.

**Concrete cost analysis at MNIST scale:**
- Activation: 784 trits, ~60% zero → ~314 nonzero positions per query.
- Pre-scan cost: 784 byte loads + zero-tests + index writes ≈ 800 cycles per activation.
- SDOT inner loop cost: ~50 cycles per 16 trits = 50 × 49 = 2450 cycles per query (full).
- SDOT inner loop with skip: ~50 cycles per 16 *nonzero* trits ≈ 50 × 20 = 1000 cycles per query.
- Net savings: 2450 − 1000 = 1450 cycles per query, minus 800 cycles pre-scan = **~650 cycles savings per query**.

Speedup factor: 2450 / (1000 + 800) = ~1.36×. **Not 2.5× as H2 predicted; closer to 1.4×.**

The realistic prediction needs adjustment. **H2 should be revised: skip-zero gives a measurable speedup of 1.3–1.5× at 60% zero density on MNIST shapes.** Below ~50% zero density, the pre-scan overhead may negate the benefit.

**Implication:** P3's substrate-claim relevance is weaker than initially thought. Still positive at MNIST scales (because the substrate quantization produces high zero density by design), but not the dramatic substrate-native speedup we might have hoped. **Revisit P3 priority** based on this.

The alternative is a NEON-vectorized scan-and-skip path that uses `vcgtq_s8(zero_mask)` to identify nonzero lanes and `vcompress` (or equivalents) to pack them. NEON doesn't have a clean compress instruction (ARM SVE does, ARM v8 baseline doesn't). On Apple Silicon (which we target), SVE is not present in the NEON path. This is an architectural constraint we should confirm before designing P3 for it.

## Q4 — Does P2 (zero_alignment) carry information beyond Hamming?

H3 hypothesizes that zero_alignment captures information Hamming distance discards. Is this real or an illusion?

Decompose Hamming over a position pair (q, t):
- Hamming = c[q,t] where c is the cost table.
- The cost depends on the joint state.

Zero-alignment counts only (q=0, t=0) positions. It's literally the count of one specific cell of the joint distribution.

Hamming aggregates costs across all 9 joint states (3 × 3 trit pairs). Zero-alignment isolates one of the 9.

**Is the (0, 0) cell information-distinct from the other 8?** Consider two cases:
- Signature pair A: 50 positions agree-as-zero, 14 positions agree-as-±1, 0 mismatches. Hamming = 0.
- Signature pair B: 0 agree-as-zero, 64 agree-as-±1, 0 mismatches. Hamming = 0.

Both have Hamming = 0. Zero-alignment distinguishes them: A has 50, B has 0. Operationally these are very different signatures — A is mostly silent, B is mostly opinionated.

**For routing decisions, does this matter?** Yes, if the consumer interprets zero density as confidence/specificity. A confidence-aware consumer prefers high-zero-alignment matches when the query is also zero-heavy (matched specificity); prefers high-±1-agreement when both are opinionated.

**Implication:** P2 is real signal but its operational value depends on having a *consumer* that uses it. Currently no consumer does. Building P2 in isolation gives a number nobody acts on. **P2 needs to ship with a consumer integration that demonstrably uses the alignment count.**

## Q5 — What's the right benchmark for P0-1's verification?

The plan calls for a measurement showing substrate advantage. MNIST is base-2 home turf — wildcard semantics may help, but base-2 with masks can match.

**Stronger benchmark options:**

(a) **A task with explicit don't-care structure.** E.g., game-tree position evaluation where many board positions are legal but most are uninformative for the value estimate. The substrate naturally represents "don't care" without spending storage on masks.

(b) **A sparse signal recovery task.** Anomaly detection where most dimensions are normal and only a few signal anomalies; the substrate's structural zero is the natural representation of "this dim is normal, no signal."

(c) **Symbolic computation / decision rules.** Three-valued logic (Kleene logic, Łukasiewicz), where {true, false, unknown} maps directly to {+1, -1, 0}. The substrate is the natural computational substrate for three-valued logical reasoning.

**MNIST as a substrate-claim benchmark is fundamentally limited.** It's a 10-class continuous-image classification task; the don't-care structure isn't natural to the data. Even with wildcard semantics, the gain over standard k-means routing is bounded.

**Implication:** P0-1's verification measurement should include a benchmark that *naturally* exhibits don't-care structure. Constructing one (or finding one in the archive) is part of P0-1's design work, not a separate cycle. **The synthetic prototype benchmark could be modified** to inject structured don't-cares: K informative dims, M never-relevant dims (that should be wildcards in any class signature), N uniform-noise dims.

## Q6 — What about base-2 frameworks that already use sparse / masked attention?

Sparse attention (e.g., Longformer, BigBird) and masked attention (decoder-style with causal mask) use sparsity in base-2. Wildcard semantics in TCAM is decades-old (1990s router hardware). The substrate-novelty claim has to handle these.

The honest framing:

- **Sparse attention** uses sparsity at the *attention pattern* level (which positions attend to which). The substrate's structural zero is at the *signature value* level (which positions have an opinion). Different layers; can compose.
- **TCAM wildcards** are hardware. Storing wildcards efficiently in software needs extra bits or structure. The substrate gives them in normal trit storage — same as having ±1.
- **Masked attention** uses a separate mask tensor. The substrate's mask is the data's third state — no separate structure.

The substrate-claim isn't "ternary discovers wildcards"; it's "ternary makes wildcards free." That's worth measuring against base-2 alternatives that *can* express wildcards but pay storage/compute overhead.

**Implication:** the verification measurement is comparative not absolute. Substrate primitive runtime+storage vs base-2 wildcard implementation runtime+storage at matched accuracy. If substrate is cheaper, claim succeeds.

## Q7 — Are we building primitives that Phase A's bank/forward consumers can even use?

Substrate-novelty rule applied to current consumers:

- `gesh_forward_classify`: takes packed-trit query, computes Hamming dist to each tile, top-k vote. The Hamming step is `m4t_popcount_dist`. **To use P1 (wildcard_dist), we'd add a different forward path or a config flag.**
- `gesh_bank_build_class_mean`: produces emergent zeros from sign-thresholding ties. **To produce wildcard tiles, need P6.**
- `gesh_train_lattice_update`: uses classification error on the current bank. **No direct change needed; the loss flows through whatever distance kernel the forward uses.**

The consumer integration scope:
- A new forward variant that uses P1 instead of `m4t_popcount_dist`.
- The new bank constructor (P6).
- A new probe that measures P1+P6 vs current Hamming+class-mean.

This is consumer-side work that comes AFTER the substrate primitives. Per the P0 protocol, kernel implementation comes after design SYNTHESIZE. So:

1. SYNTHESIZE picks which primitives.
2. Substrate spec amendments.
3. Kernel implementations + tests.
4. Consumer integration.
5. Verification measurement.

**The amount of consumer work is real.** New forward variant + new bank constructor + new probe. Probably as much code as the kernel work itself.

## Wrong frame / right frame

**Wrong frame:** "Add wildcards because TCAM does it."
**Right frame:** "The substrate's structural zero already exists in trit storage; build primitives that operationally use it; demonstrate that base-2 has to pay overhead the substrate doesn't."

**Wrong frame:** "Skip-zero matmul gives 2.5× speedup."
**Right frame:** "Skip-zero matmul gives 1.3–1.5× speedup at MNIST-scale zero density, with overhead that may negate it below ~50% zero density. Worth doing if combined with substrate-shape benchmarks where high zero density is natural; questionable as a standalone primitive."

**Wrong frame:** "Build P1-P6 in parallel; pick winners."
**Right frame:** "P1 is coupled to P6 (must ship together for semantic coherence); P3 is mid-priority and not as transformative as initially thought; P2 needs consumer integration to be useful; P5 is auxiliary; P4 is deferred to P0-4. SYNTHESIZE should commit to P1+P6 as the primary deliverable, with P2 as an optional add-on."

**Wrong frame:** "MNIST is the benchmark."
**Right frame:** "MNIST is a regression-guard. P0-1's verification needs a benchmark with natural don't-care structure to demonstrate substrate advantage; build one (synthetic with explicit irrelevant dims) or identify one. MNIST as a secondary measurement to ensure no regression."

## Loop-back triggers from this REFLECT

- **Back to RAW** if the verification benchmark choice (Q5) requires reframing the substrate-novelty claim itself. We're not there yet but it's possible.
- **Back to NODES** if SYNTHESIZE's chosen primitive set fails the substrate-novelty audit on closer inspection. The audit is the core gate.
- **No loop-back** if SYNTHESIZE commits to P1+P6 with a substrate-shape benchmark and the kernel/consumer work proceeds.

## Carry-forwards to SYNTHESIZE

- P1 and P6 must ship together.
- P3 is real but smaller than initially thought; demote to "phase 2 of P0-1" or fold into P0-4 (where multi-stage activations are naturally sparser).
- P2 needs consumer integration (a forward variant that uses zero_alignment); without it, it's a useless number.
- P5 is auxiliary; ship if cheap, defer if it complicates.
- The verification benchmark needs explicit don't-care structure; modify the synthetic prototype generator to include it.
- Substrate-novelty audit must be present in the verification measurement: "what does base-2 with masks pay that we don't?"
