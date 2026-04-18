---
date: 2026-04-17
phase: SYNTHESIZE
topic: Reciprocal LSH+GSH learning and the role of W_f[hidden]=0
---

# Reciprocal learning + structural zero — SYNTHESIZE

Executable specification.

---

## The architecture (four instruments)

```
         query
           │
      ┌────┴────┐
      │         │
     LSH       LSH routing
   (broad)     pattern
      │         │
      │        GSH
      │      (deep)
      │         │
      │    ┌────┴────┐
      │    │         │
      │  agree    disagree
      │    │         │
      │    │    confusion ID
      │    │    (top-2 classes)
      │    │         │
      │    │    specialist
      │    │    re-rank
      │    │    (focused)
      │    │         │
      └────┴────┬────┘
                │
           final pred
```

**LSH:** broad random projections. Finds the neighborhood.
**GSH:** routing-pattern hash. Confirms or flags uncertainty.
**Specialist:** informed-zero projections for specific confusion
pairs. Re-ranks the LSH union when the GSH says "uncertain."
**Combination:** agree → accept LSH. Disagree → specialist breaks
the tie.

## What to build

### Step 1: confusion-pair specialist projections

For each of the top-K confusion pairs (from the CIFAR-10 atomics
or from the GSH disagreement set):

1. Generate N_cand=1000 random ternary projection directions
   (same as sig_builder_init — random {-1,0,+1} weights over
   input_dim, density=0.33).

2. For each candidate direction w, compute the ternary projection
   output for all class-i and class-j training prototypes:
   `proj(w, x) = ternary_dot(w, x)` → threshold → trit (+1/0/-1).

3. Fitness: fraction of (class-i, class-j) prototype pairs where
   the trit value DIFFERS. High fitness → the projection
   distinguishes the two classes in trit space.

4. Keep the top S directions (e.g., S=16) for this confusion pair.

5. Encode each training prototype's specialist signature: S trits
   from the S selected directions. Pack as S/4 bytes.

### Step 2: specialist re-rank

At test time, after the LSH + GSH pass:

If GSH agrees with LSH → accept LSH prediction. Done.

If GSH disagrees → identify the confusion pair (LSH's prediction
vs GSH's prediction, or LSH's top-2 candidates). Look up the
specialist projection set for that pair. Encode the query's
specialist signature (S trits). Score each candidate in the LSH
union by Hamming distance on the specialist signatures. The
candidate with the lowest specialist-distance is the specialist's
prediction.

Final prediction:
- If specialist agrees with either LSH or GSH → take the agreed
  answer.
- If all three disagree → take the specialist (it was designed
  for this confusion).

### Step 3: measure

Report:
- LSH standalone accuracy
- GSH standalone accuracy
- Specialist standalone accuracy (on disagreement queries only)
- Combined (LSH + GSH + specialist) accuracy
- Per-confusion-pair rescue rate

### Parameters

| parameter | value | rationale |
|---|---|---|
| K (confusion pairs) | 8 | top-8 cover 25.8% of CIFAR-10 failures |
| N_cand | 1000 | cheap to generate and score |
| S (specialist trits) | 16 | 4 bytes, same as N_PROJ=16 LSH sig |
| density | 0.33 | matched to LSH for comparable fitness |

### Implementation

New tool: `tools/specialist_rerank.c`

1. Build LSH (existing code from layered_lsh.c).
2. Build GSH (existing code from layered_lsh.c).
3. Identify top-K confusion pairs from the GSH disagreement set
   on the training data.
4. For each confusion pair, generate N_cand directions, score
   fitness, keep top S.
5. Encode training specialist signatures for each confusion pair.
6. At test time: LSH → GSH → if disagree, specialist re-rank.
7. Report all accuracy metrics.

The specialist generation is O(K × N_cand × N_class_pair) ternary
projections — cheap. The specialist encoding is O(K × S × N_train)
— also cheap (16 trits per prototype per specialist).

## The role of W_f[hidden] = 0

The structural zero is the mechanism that makes specialists
POSSIBLE. Each specialist projection has ~67% zeros (at
density=0.33). Those zeros are the attention mask — they hide
the pixels that don't help distinguish the target confusion pair.
The ±1 weights are the measurement — they compare the pixels
that DO distinguish.

In the LSH, the zeros are random. In the specialist, the zeros
are SELECTED — the routing fitness function (fraction of cross-
class pairs distinguished) implicitly selects for zero patterns
that hide noise and expose signal.

The zero is not an absence of information. It's a DECISION to
not look. Making that decision informed is how the ternary
architecture learns where to focus.

## Go / no-go

**Go:** combined accuracy ≥ 42% on CIFAR-10 (+5pp over 37%
baseline). The specialist re-rank provides genuine gains on
confused queries.

**Strong go:** ≥ 48%. The specialist mechanism dramatically
improves the confused subset. Full reciprocal loop is
warranted.

**No-go:** ≤ 39%. Either the routing fitness doesn't select
discriminative projections, or the confusion pairs are too
structurally similar for any ternary projection to distinguish.

## Estimated effort

- Specialist generation: ~80 lines (candidate gen + fitness
  eval + selection).
- Specialist encoding + re-rank: ~60 lines.
- Tool orchestration (LSH + GSH + specialist): ~100 lines
  (reuse layered_lsh structure).
- Total: ~250 lines, new tool.
- Measurement runs: ~15 minutes per dataset (including
  specialist generation).

## What the LMM cycle found

The two questions ("can GSH and LSH learn together" and "would
W_f[hidden]=0 help") are the SAME question seen from two angles.
The GSH provides the SIGNAL (where the system fails). The
structural zero provides the MECHANISM (how to focus the
measurement). Together, they form a closed loop where routing
measurements discover informed attention masks.

This is the structural advantage of base-3 that NORTH_STAR
claims: the zero state is not wasted bandwidth — it's the
attention mechanism that enables the routing architecture to
learn where to look. Base-2 projections ({-1, +1}) must look
everywhere. Base-3 projections ({-1, 0, +1}) can choose.
