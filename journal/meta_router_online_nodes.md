---
date: 2026-04-15
scope: LMM cycle — online, inline meta-router
phase: NODES
---

# NODES

## Discrete ideas

1. **Meta-router as k-NN over routing-context signatures.** The natural routing-only instantiation: a growing bank of (query sig, local-state sig, action, outcome) tuples. Inference looks up the query's context in the bank and reads off the recommended action. Online update appends new tuples.

2. **Self-supervised ground truth: ensemble disagreement.** H2 vs H3 disagreement within the filtered top-50 is a label-free estimator of query hardness. Variance in fused distance across committee members is another. These are cheap byproducts of the cascade we already run.

3. **Self-supervised ground truth: H1 margin.** Gap between H1 top-1 and H1 top-K distance. Small gap = many candidates at the same distance = tied-min regime = hash lacks discrimination. Directly computable during the primary pass.

4. **Self-supervised ground truth: OOD via distance to known signatures.** For each query, distance to the nearest prior-query signature in a rolling buffer. Queries that land far from all recent queries are flagged as distributional outliers.

5. **Failure-signature bank as the observer's memory.** A bounded store of signatures from queries that showed high-failure signals. New queries compute Hamming to the bank. Near a bank entry → "similar to a past failure" → flag for escalation. The bank is global state updated online.

6. **Per-region confidence map.** Quantize signature space into a fixed number of regions (e.g., hash-bucket the query sig to a small index). Each region maintains rolling success/fail counters from self-supervised signals. New queries look up their region's confidence and route accordingly. Cheaper than k-NN over a bank, but coarser.

7. **Routing-context signature structure.** A concatenated trit vector: [query H1 sig | H1-top-1 distance quantized | tied-count quantized | disagreement flag quantized]. Gives the meta-router a richer lookup key than just the raw query sig.

8. **Action vocabulary.** Meta-router outputs are not predictions but *decisions*: (a) accept local fusion's top-1, (b) widen K from 50 to 200, (c) escalate to global H1+H2+H3+H4 over all 60K, (d) invoke k-NN majority instead of 1-NN at the resolver, (e) defer to a different hash family. Vocabulary size determines the meta-router's expressiveness.

9. **Cold-start seeding.** Preload the bank from an offline pass on training data: run local + global over training queries, record where they disagreed, record the self-supervised signals at those queries, preload as initial bank contents. Hybrid online-after-offline-seed.

10. **Online update rule.** Candidate rules: (i) append every query's (sig, signals, action-taken) tuple; (ii) append only queries with self-supervised signal above threshold; (iii) replace bank entries via reservoir sampling to keep bounded; (iv) update per-region counters without storing individual entries.

11. **Escalation target must actually be better.** The prerequisite empirical question from the prior turn: `P(global correct | local failed)`. If near 1, observer routing is load-bearing. If near baseline (~50%), observer routing adds cost with no gain. Measure this before building the meta-router.

12. **Closed-loop self-confirmation bias.** Observer flags X, bank updates on X, next similar query also flagged. Without a grounding signal, the loop drifts. Mitigations: (i) periodic decay of bank entries, (ii) force exploration (randomly sample non-flagged queries for bank inclusion), (iii) use only self-supervised signals that are fresh-per-query, not accumulated.

13. **Inline cost budget.** Primary H1 pass = 960K trit-ops at N_PROJ=16. Meta-router lookup must stay under 10-20% of that = 100-200K ops. That allows ~5-10K bank entries × 16 trits per lookup, or 1-2K entries × wider sig. Bounded.

14. **Anticipation vs correction.** Anticipation = meta-router decides the strategy before any cascade runs, based on query sig alone. Correction = meta-router runs after the cascade and re-routes if confidence is low. Correction is strictly cheaper and uses more information. Anticipation is purer but weaker.

15. **Hybrid: observe during, decide after.** Primary H1 runs unconditionally. Local H2+H3+H4 fusion runs. Then self-supervised signals (margin, disagreement, OOD) are computed from the already-gathered data. If signals cross a threshold → meta-router queries its bank and escalates. Default is "local is fine." Escalation is rare → amortized cost stays near cascade baseline.

## Tensions

**T1. Anticipation vs correction.**
Anticipation is architecturally cleaner but practically weaker because it can only use query features, not local-routing state. Correction is practically stronger but requires running the local cascade first. Hybrid (node 15) is the compromise; the meta-router observes the local cascade's computation and decides whether to accept or escalate. This is what "inline" actually means — the observer reads the cascade's intermediate state, not just the query.

**T2. Online learning without labels.**
The only fresh, reliable supervision available online is *self-supervised* (margin, agreement, OOD). These are noisy labels of query hardness, not of correctness. Training a meta-router from noisy labels can still work, but closed-loop drift is a real risk. Resolving this tension probably requires node 9 (offline seed) combined with node 12 mitigations (decay or forced exploration).

**T3. Learning target vs action vocabulary.**
The meta-router's learning is only meaningful if it can output different actions that actually produce different outcomes. If the action vocabulary is tiny (accept vs escalate), the learning is a binary classifier. If it's wider (pick one of several resolver families), the learning is multi-arm bandit-shaped. I lean toward the smaller vocabulary first — binary escalation — because it's easier to measure and easier to debug.

**T4. Self-confirmation vs exploration.**
Closed-loop update on self-supervised signals can drift. The classical fix is exploration: occasionally take the non-recommended action and observe the outcome. In a k-NN routing substrate, "exploration" means running both local and global on a sampled fraction of queries, regardless of the observer's recommendation, to keep the bank calibrated. Exploration costs cycles; too little drifts, too much negates the efficiency win.

**T5. Prerequisite measurement vs forward progress.**
T11 node says we need to measure whether global beats local on local-failures. Until we do, everything here is speculative. The risk is that the cycle produces a beautiful meta-router design that the prerequisite falsifies. Prerequisite first; design second.

**T6. Is this actually different from conditional escalation with a learned threshold?**
If the observer's decision function is "flag when self-supervised-signal > threshold," and the threshold is adaptive based on bank contents, then yes — in practice it's a threshold that moves over time, which is a simple form of online learning. "Meta-routing" may be a grander framing than the underlying mechanism justifies. That's fine as long as the simpler framing does the work; we should not over-engineer.

## Dependencies

- **Prerequisite measurement** (node 11) blocks everything. If it fails, the whole cycle ends as a known-unreachable.
- **Cold-start seed** (node 9) depends on offline access to training labels, which we have on MNIST. Not a blocker.
- **Online update** (node 10) requires a bounded bank data structure. Not hard to implement.
- **Self-supervised signal computation** (nodes 2-4) depends on what the cascade already emits. Most signals are already computed in `mnist_cascade_atomics.c`; we'd reuse them.
- **Routing-context signature construction** (node 7) is a small add on top of the existing signature pipeline — a concatenation of trit sequences.
- **Action vocabulary** (node 8) determines the observer's output shape and the bank schema. Start binary.
