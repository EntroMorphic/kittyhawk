---
date: 2026-04-20
phase: REFLECT
topic: LSH ↔ GSH negotiation loop — two instruments working as a team
---

# Negotiation loop — REFLECT

---

## Core insight

The RAW circled around specialist tables and re-probing. NODES
mapped the architecture. REFLECT finds the structural truth.

**We already HAVE the negotiation loop. We just haven't recognized it.**

The selective scorer in direct_lsh does:
1. LSH produces Hamming k-NN prediction
2. GSH evaluates (agree/disagree)
3. If disagree: pair-IG re-ranks the union for the confusion pair

Step 3 IS the negotiation. The GSH says "you're confused about
Cat vs Dog." The LSH responds by re-scoring with Cat/Dog-specific
weights. The bidirectional communication IS happening — the pair-IG
re-ranking is the LSH's response to the GSH's feedback.

What the specialist TABLES add: a second FILTER stage for the
specific confusion, not just a second SCORING stage. Re-ranking
finds the best candidate in the EXISTING union using pair-IG
weights. Specialist tables find candidates through a filter
DESIGNED for the confusion pair. This could surface candidates
that the broad filter missed.

But the oracle is 99.99%. The correct candidate is already in
the union. A second filter for candidates that are ALREADY FOUND
is redundant. The only value of the specialist filter is if it
produces a SMALLER, CLEANER union for the confusion pair — fewer
confusers, making the pair-IG re-ranking more effective.

## T1 resolved: the specialist adds value through PURITY, not COVERAGE

The standard union has ~1600 candidates. Many are irrelevant to
the Cat/Dog question (they're Airplanes, Ships, etc.). Pair-IG
re-ranks ALL 1600 by Cat/Dog weights — including the irrelevant
ones that add noise. The specialist table, keyed on Cat/Dog-
discriminative positions, would produce a union that's PURE:
mostly Cat and Dog candidates, few irrelevant confusers.

Re-ranking a PURE union (mostly Cat/Dog) is more effective than
re-ranking a MIXED union (Cat/Dog/Airplane/Ship/...).

## T2 resolved: test without the specialist tables first

Before building 64 specialist tables, test whether FILTERING
the existing union to only Cat and Dog candidates (where the
labels are Cat or Dog) improves pair-IG accuracy. This is
free — no new tables, just restrict the re-ranking to candidates
whose labels are one of the two confused classes.

If filtering the union to the confusion pair improves accuracy,
the specialist tables are justified (they do the same filtering
at the INDEX level instead of post-hoc). If filtering doesn't
help, specialist tables won't either.

## T3 resolved: the specialist replaces the tie-break, not adds to it

The combination is not "three opinions." It's:
- If LSH+GSH agree → accept (high confidence, no specialist)
- If disagree → specialist for that pair produces the answer

The specialist IS the tie-breaker. It doesn't add a third
opinion — it RESOLVES the disagreement directly. No voting
between three systems.

## What I now understand

1. **We already have the negotiation loop** in the selective
   scorer. Pair-IG re-ranking IS the LSH's response to the
   GSH's feedback. The communication is bidirectional.

2. **The missing piece is PURITY, not coverage.** The specialist
   filter would produce a cleaner candidate set for the
   confusion pair, making pair-IG more effective. But we should
   TEST this by filtering the existing union first.

3. **The simplest test: restrict pair-IG re-ranking to candidates
   labeled as one of the two confused classes.** If this helps,
   specialist tables are warranted. If not, the union's class
   composition isn't the bottleneck.

4. **The negotiation loop is already built.** The improvement is
   in the PURITY of the re-ranking set, not in the loop
   structure. The architecture is complete — the gain is in the
   details of what gets re-ranked.
