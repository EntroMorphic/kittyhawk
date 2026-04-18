---
date: 2026-04-17
phase: REFLECT
topic: Reciprocal LSH+GSH learning and the role of W_f[hidden]=0
---

# Reciprocal learning + structural zero — REFLECT

---

## Core insight

RAW found the convergence point. NODES mapped the mechanism. Now
REFLECT needs to find what's really NEW here versus what I'm
re-deriving from known techniques.

The structural zero enabling selective attention is not new — it's
what NORTH_STAR says. Feature selection via routing measurements
is not new — it's what the projection selection LMM proposed.
What's new is the INTERACTION between three things:

1. The GSH identifies WHERE the system fails (which queries,
   which confusion pairs).
2. The structural zero controls WHAT each measurement sees.
3. Routing measurements from the LSH determine which zero
   patterns produce discriminative measurements.

None of these works alone. The GSH without the structural zero
can identify failures but can't fix them. The structural zero
without the GSH is just random filtering. The routing
measurements without the GSH's guidance don't know which
confusions to optimize for.

**The three components form a closed loop where each enables
the next.** This is the "more here" the user sensed.

## T1 resolved: generate and select, not copy

Copying successful random tables (Node 7a) is limited — if none
of the M=64 random tables happen to discriminate Cat/Dog, there's
nothing to copy. Generating N_cand new random candidates and
selecting by routing fitness (Node 7b) explores more of the
projection space and is more likely to find the right zero
pattern.

But the selection criterion must be routing-native. The simplest
routing-native fitness for a specialist targeting confusion
pair (i, j):

For candidate projection w, compute the ternary signature of
every class-i and class-j training prototype. For each class-j
prototype (the "correct" class), find the fraction of class-i
prototypes (the "confusers") that have a DIFFERENT signature.
If the fraction is high, the projection distinguishes the two
classes in signature space. If it's low, the projection
collapses them.

    fitness(w, i, j) = fraction of (class-i, class-j) pairs
                       with different signatures under w

This is computed from trit signatures (m4t_trit_pack) with no
pixel-space math. It measures whether the structural zeros in w
are placed so that the ternary quantization separates the two
classes.

## T2 resolved: always score specialists, weight by GSH confidence

The specialist score is cheap (one popcount_dist per specialist
per candidate). Always compute it. But WEIGHT it by the GSH's
confidence signal:

- GSH agrees with LSH → trust LSH, ignore specialist.
- GSH disagrees → specialist's score breaks the tie.

The GSH is the conductor: it decides when the specialist solos.

## T3 resolved: one iteration first, loop if it works

The reciprocal loop is powerful but complex. Start with one
iteration:

1. Build LSH + GSH (done).
2. Identify top-K confusion pairs from the disagreement set.
3. Build specialists for those K pairs.
4. Measure accuracy with LSH + GSH + specialists.

If one iteration crosses 45% on CIFAR-10, the loop is worth
pursuing. If it stays near 38%, the specialist mechanism doesn't
work and the loop can't help.

## T4 resolved: the fitness function IS routing-native

The fitness function in T1 (fraction of cross-class pairs with
different signatures) is computed entirely from ternary
signatures. The projection generates the signature. The
comparison is bit-level. No pixel-space statistics.

The GENERATION of candidate projections is random ternary
(same as the LSH). The SELECTION is routing-native. The
structural zero pattern is NOT designed from pixel statistics
— it's DISCOVERED by generating random patterns and selecting
the ones that routing measurements validate.

This is the key distinction from PCA/centroid approaches:
those DESIGN projections from pixel statistics. This
DISCOVERS projections by generating ternary candidates and
selecting via routing fitness. The zero pattern emerges from
selection, not from pixel analysis.

## What I now understand

1. **The structural zero is the routing architecture's attention
   mechanism.** In base-2, every pixel contributes to every
   measurement — no attention is possible. In base-3, the zero
   says "don't look here," enabling focused measurement. This
   is a STRUCTURAL advantage of base-3, not just an arithmetic
   one.

2. **The GSH is the attention controller.** It identifies which
   queries need focused attention and which confusion pair
   drives the failure. Without the GSH, the specialist doesn't
   know when to engage.

3. **Reciprocal learning is a closed loop:** GSH identifies
   failures → specialist generation targets failures → improved
   accuracy → updated GSH → smaller failure set → iterate.
   Each component enables the next.

4. **The specialist is a RE-RANK, not a new index.** The LSH
   union already contains the correct answer (oracle=100%).
   The specialist provides a better SCORING FUNCTION for
   specific confusion pairs. No new bucket index needed.

5. **One iteration is the minimum viable experiment.** Build
   specialists for the top-8 confusion pairs. If 8 specialists
   correct half their target failures, CIFAR-10 moves from
   37% to ~45%. That's the go/no-go for the full loop.

## What remains uncertain

- Whether routing-discovered zero patterns actually discriminate
  confusing class pairs. The fitness function (fraction of
  cross-class pairs with different signatures) might not
  correlate with downstream accuracy.

- Whether N_cand=1000 random candidates are enough to find
  good specialist projections for each confusion pair. If the
  discriminative axis is very specific (e.g., "a particular
  edge pattern at pixel 17"), 1000 random ternary directions
  might not include one that's close enough.

- Whether the specialist re-rank compounds with the existing
  multi-resolution re-rank. They could be complementary (multi-
  resolution for general improvement, specialist for confusion-
  specific improvement) or redundant.

- Whether the reciprocal loop converges or oscillates (fixing
  one confusion pair might create new confusions elsewhere).
