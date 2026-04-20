---
date: 2026-04-20
phase: SYNTHESIZE
topic: LSH ↔ GSH negotiation loop — two instruments working as a team
---

# Negotiation loop — SYNTHESIZE

Executable specification.

---

## What the LMM found

The negotiation loop already EXISTS in the selective scorer.
Pair-IG re-ranking IS the LSH's response to the GSH's feedback.
The bidirectional communication is there — the question is whether
we can make it MORE EFFECTIVE.

The hypothesis: restricting pair-IG re-ranking to only the
candidates labeled as one of the two confused classes (PURITY)
should improve accuracy on disagreement queries because it
removes irrelevant confusers from the re-ranking set.

## What to test FIRST (zero-cost diagnostic)

Before building specialist tables, test FILTERED pair-IG:

When GSH disagrees (identifies pair c1 vs c2):
- Current: pair-IG re-ranks ALL ~1600 candidates
- Test: pair-IG re-ranks ONLY candidates with labels c1 or c2

This is a one-line filter in the pair-IG re-ranking loop. If it
helps, the specialist tables (which produce a pre-filtered union
at the index level) are justified. If it doesn't, the class
composition of the re-ranking set isn't the bottleneck.

## Implementation

In direct_lsh.c's pair-IG re-ranking loop, add:

```c
/* FILTERED pair-IG: only re-rank candidates labeled c1 or c2. */
for (int j = 0; j < st.n_hit; j++) {
    int idx = st.hit_list[j];
    int lbl = ds.y_train[idx];
    if (lbl != c1 && lbl != c2) continue;  /* ← new filter */
    /* ... pair-IG distance computation ... */
}
```

Then compare: filtered pair-IG accuracy vs unfiltered pair-IG
accuracy on the disagreement set.

## If filtered pair-IG helps: build specialist tables

For each of the top-8 confusion pairs:
1. Compute pair-IG for all trit positions.
2. Select top-16 positions by pair-IG.
3. Build 8 bucket tables keyed on those 16 positions (from the
   same training trit signatures, just different key selection).
4. At query time when GSH identifies that pair: probe specialist
   tables → produce specialist union → pair-IG k-NN → prediction.

## If filtered pair-IG doesn't help

The bottleneck is not class impurity in the union. It's that
the Hamming distance and pair-IG scoring are both insufficient
to distinguish the confused classes on these signatures. Further
architectural work is needed at the representation level (richer
block encoding, pattern-level distance).

## Go / no-go

**Go:** filtered pair-IG > unfiltered pair-IG by ≥1pp on the
disagreement set. Purity matters and specialist tables would
help.

**No-go:** filtered pair-IG ≤ unfiltered pair-IG. The confused
class candidates aren't being misranked due to interference from
other classes — they're genuinely indistinguishable at the
per-trit level.

## Estimated effort

- Diagnostic (filtered pair-IG): ~5 lines added to direct_lsh.c
- Measurement: one CIFAR-10 run (~30 seconds)
- If go: specialist tables (~150 lines in a new tool)
- Total: 5 minutes for the diagnostic, 30 minutes for the full
  build if warranted
