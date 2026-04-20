---
date: 2026-04-20
phase: RAW
topic: LSH ↔ GSH negotiation loop — two instruments working as a team
---

# Negotiation loop — RAW

---

The user pointed at the one-way communication. The LSH talks to
the GSH but the GSH never talks back. They're not a team — they're
a speaker and a listener.

What would teamwork look like? The GSH evaluates the LSH's
proposal and sends SPECIFIC feedback that the LSH can act on.
Not just "I disagree" but "I disagree because I see Dog, you
see Cat — look harder at Dog/Cat candidates."

The feedback loop:

1. LSH proposes: "I found a union of 1592 candidates. My k-NN
   says Cat (rank 1 by summed Hamming distance)."

2. GSH evaluates: "My routing pattern matches Dog images, not
   Cat images. I disagree."

3. LSH refines: "OK, I'll re-probe specifically for Dog candidates.
   Let me expand the union by probing tables where Dog prototypes
   cluster." Then re-scores the EXPANDED union.

4. New prediction from the expanded union.

Step 3 is the key innovation. Currently the LSH probes ALL tables
uniformly. After the GSH's feedback, it probes SELECTIVELY —
focusing on tables and regions where the specific confusion pair
(Cat vs Dog) is most resolvable.

How does the LSH know where to look for Dog candidates? From the
pair-IG weights! The pair-IG for (Cat, Dog) tells us which trit
positions are most discriminative for that pair. The trit positions
with high pair-IG weight are the ones where Cat and Dog prototypes
DIFFER. If the LSH can filter by those positions, it finds
candidates that are specifically relevant to the Cat/Dog question.

But the bucket key is fixed (hierarchical spatial summary). We
can't change the key to focus on pair-IG positions mid-query.
What we CAN do: lower the vote threshold for the specific
confusion pair's prototypes. Currently all candidates in the
union are treated equally. After GSH disagrees with "Dog not Cat":
re-rank with pair-IG weights (which we already do) BUT ALSO
inject additional Dog prototypes from the training set that the
multi-probe might have missed.

Wait — the oracle is 99.99%. The correct neighbor is ALREADY in
the union almost always. The issue isn't missing candidates — it's
RANKING within the union. The GSH disagreement tells us which
confusion pair to focus on, and pair-IG re-ranks for that pair.
We already do this in the selective scorer.

So what ADDITIONAL action can the negotiation loop take that the
selective scorer doesn't already do?

The selective scorer: GSH disagrees → pair-IG re-rank the SAME
union.

The negotiation loop: GSH disagrees → pair-IG re-rank → if STILL
uncertain (pair-IG margin is low), the LSH re-probes with LOWER
threshold or WIDER radius to find MORE candidates in the
confusion pair's neighborhood → re-rank the EXPANDED union.

The additional step is: RE-PROBE after the first pair-IG
re-ranking doesn't produce a clear winner. It's a second-round
expansion triggered by persistent uncertainty.

But wait — re-probing produces more candidates from the SAME
bucket neighborhoods. If the original probe with radius 2 and
min_cands=50 didn't find discriminative candidates, expanding to
radius 2 with min_cands=200 or max_union=32768 would find MORE
of the same — not necessarily DIFFERENT candidates. The buckets
are keyed on the same spatial summary; expanding just includes
more images from the same spatial neighborhoods.

Unless we change the KEY. What if, for the second round, we key
on DIFFERENT summary trits — specifically the ones with high
pair-IG for the identified confusion pair? Build SPECIALIZED
bucket tables (one set per confusion pair) that key on the most
discriminative trits for that pair. When the GSH identifies the
confusion, switch to the pair-specific tables.

This is the specialist concept from earlier — but now it's TABLES
not WEIGHTS. Each confusion pair has its own set of bucket tables
whose keys are selected by pair-IG to maximize discrimination
for that specific pair.

The per-pair tables use the SAME training signatures (direct
quantization, no random weights). Only the KEY SELECTION differs.
The standard tables key on random permutations of spatial summary
trits. The specialist tables key on the TOP-16 pair-IG-weighted
trit positions for that confusion pair.

Implementation:
- Build 45 specialist table sets (one per class pair).
- Each set has M_spec tables (small, e.g., 8).
- Each table keys on the 16 trit positions with highest pair-IG
  for that pair.
- At query time: if GSH disagrees, identify the confusion pair,
  probe the specialist tables, expand the union with the
  specialist hits, re-rank with pair-IG.

The specialist tables are NO random weights — they key on SPECIFIC
trit positions selected by pair-IG from the training set. Each key
trit has a specific meaning: "this pixel/gradient is discriminative
for this class pair."

This is the structural zero doing its job: the specialist table's
key has zeros at positions that DON'T discriminate the pair, and
±1 at positions that DO. The key itself IS the attention mask for
that confusion pair.

The cost: 45 pairs × 8 tables = 360 additional tables. Each is a
sorted array of (key, proto_idx) pairs = 50K × 8 bytes = 400KB
per table × 360 = ~144 MB. That's a lot but fits in memory on
Apple Silicon.

Actually, we don't need tables for ALL 45 pairs. The top-8
confusion pairs account for 25% of errors. Build specialists
for those 8 pairs only = 64 tables = ~32 MB.

The negotiation becomes:
1. LSH proposes (standard probe, Hamming k-NN)
2. GSH evaluates (structured profile, agree/disagree)
3. If agree → accept LSH
4. If disagree → identify pair → probe specialist tables →
   expand union → pair-IG re-rank → final prediction

This is a routing-native negotiation: the GSH feeds SPECIFIC
information (which confusion pair) back to the LSH (which
specialist to consult), and the LSH acts on it (probes the
specialist tables to find pair-specific candidates).

No random weights anywhere. The specialist table keys are
selected by pair-IG from training data. Each key position has
a specific meaning. The structural zero in the key IS the
attention mask.
