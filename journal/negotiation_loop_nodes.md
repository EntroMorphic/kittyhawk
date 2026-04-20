---
date: 2026-04-20
phase: NODES
topic: LSH ↔ GSH negotiation loop — two instruments working as a team
---

# Negotiation loop — NODES

---

## Node 1 — The missing piece is BIDIRECTIONAL communication

Current: LSH → GSH (one-way). The GSH receives the union and
produces a confidence signal, but the LSH never adjusts its
behavior based on GSH feedback.

Negotiation: LSH ↔ GSH (bidirectional). The GSH identifies
WHICH confusion pair is active. The LSH uses that information
to probe DIFFERENT tables optimized for that pair.

## Node 2 — The GSH's feedback is: "which pair are you confused about?"

The GSH disagrees when the query's routing pattern matches a
different class than the LSH predicted. The TOP-2 classes (LSH
prediction and GSH prediction) define the confusion pair. This
pair IS the feedback signal.

## Node 3 — The LSH acts on the feedback via specialist tables

Per-confusion-pair bucket tables with keys selected by pair-IG.
The key positions are the 16 trit positions most discriminative
for that specific pair. When the GSH identifies the pair, the
LSH probes THOSE tables.

The specialist tables use the SAME training signatures. Only
the key selection differs. No random weights — each key
position has a SPECIFIC MEANING (high pair-IG for that class
pair).

## Node 4 — The oracle is 99.99% — expansion may not be needed

The standard union already contains the correct neighbor almost
always. The specialist tables' value is NOT finding NEW
candidates — it's finding candidates through a DIFFERENT filter
that's optimized for the confusion pair. Those candidates might
be the SAME prototypes found differently, or they might include
a few additional ones.

More importantly: the specialist tables produce a DIFFERENT
union that can be scored INDEPENDENTLY, giving a third opinion.
Three instruments: LSH (broad filter), GSH (confidence), and
specialist (confusion-specific filter).

## Node 5 — The specialist's prediction breaks the tie

LSH says Cat. GSH says Dog. The specialist (for Cat/Dog pair)
finds candidates using the most discriminative positions for
Cat vs Dog, then k-NN on those candidates produces a third
prediction. If the specialist agrees with either LSH or GSH,
that's the final answer. If all three disagree, take the
specialist (it was designed for this exact confusion).

## Node 6 — Only 8 specialist table sets are needed

The top-8 confusion pairs on CIFAR-10 account for ~25% of all
errors:
```
Truck→Auto: 232    Dog→Cat: 185     Deer→Bird: 188
Plane→Ship: 190    Cat→Dog: 174     Truck→Ship: 160
Dog→Bird: 141      Ship→Plane: 141
```

8 confusion pairs × 8 specialist tables = 64 tables. Same build
cost as the standard LSH.

## Node 7 — The key selection is deterministic, not random

For confusion pair (a, b), the 16 key positions are selected by:
1. Compute pair-IG for every trit position d.
2. Rank by pair-IG descending.
3. Take the top 16.

These positions are WHERE the two classes differ most in
trit-space. The key built from those positions produces
buckets that SEPARATE classes a and b maximally.

## Node 8 — The negotiation loop is 2 rounds max

Round 1: standard LSH + GSH → agree or disagree.
Round 2 (only on disagreement): specialist probe → specialist
prediction → combine.

No infinite iteration. Two rounds. The cost of round 2 is one
additional probe of 8 specialist tables on the ~50% of queries
where GSH disagrees. On the agreeing 50%, no additional work.

## Tensions

**T1:** Does the specialist add information the standard LSH
union doesn't have? The oracle is 99.99% — the correct
neighbor is already found. The specialist can only re-rank
within a different subset.

**T2:** Specialist tables duplicate training data (same sigs,
different keys). Is the memory cost (32 MB for 8 pairs)
justified by the accuracy gain?

**T3:** The specialist prediction is a THIRD opinion alongside
LSH and GSH. The combination of three opinions might introduce
more confusion rather than resolving it.
