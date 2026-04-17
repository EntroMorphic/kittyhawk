---
date: 2026-04-17
phase: REFLECT
topic: GSH + LSH in concert — functions and relationship
---

# GSH + LSH in concert — REFLECT

---

## Core insight

The RAW phase discovered the root failure: the GSH re-projected
a clean signal through random weights, making it a noisy LSH
instead of a different instrument. NODES found the fix: encode
the routing pattern directly as a trit signature without random
projection.

But REFLECT reveals a deeper issue: **the GSH and LSH have
different SCOPE, and the architecture must honor that.**

The LSH asks: "which PROTOTYPES are geometrically close?" It
operates at the level of individual training images. Each query-
candidate comparison is a direct measurement.

The GSH asks: "which ROUTING PATTERNS are similar?" It operates
at the level of BEHAVIORS — how the lattice collectively
responds to an image across M tables. This is a higher-order
measurement: not "is this pixel-signature close?" but "does the
lattice see this image the same way?"

The failure of the one-hot GSH was not just noise from random
projection — it was a SCOPE mismatch. The GSH was trying to
re-do the LSH's job (find nearby prototypes) using degraded
input (labels instead of distances). It should have been doing
its OWN job: finding images with similar routing behaviors.

## T1 resolved: multi-trit categorical encoding

Node 8's encoding is right: each table's vote (0-9) is encoded
as 4 trits (3^4=81 > 10). M tables → M×4 trits. The Hamming
distance between two such signatures counts per-trit
disagreements, which approximates per-table vote disagreements.

Why this works better than one-hot:
- One-hot: 640 binary dims → needs random projection → noise
- Multi-trit: 256 ternary trits → IS the signature → no projection

The multi-trit encoding uses the existing trit infrastructure
(pack, popcount_dist, bucket index, multi-probe) directly on
the routing pattern. The GSH IS an LSH, but the "image" it
hashes is the routing pattern, not the pixels.

At M=64 × 4 trits/table = 256 trits = 64 bytes. Bucket key
on the first 16 trits (4 bytes). This encodes the first 4
tables' votes as the routing address.

## T2 resolved: self-match is handled by the LSH probe

When computing training routing signatures, we probe the LSH
and build a union. The self-match (training image i finding
itself at distance 0) WILL be in the union. We exclude it
when computing per-table labels.

The resulting routing signature is "what does the LSH say
about this image's neighborhood, excluding itself." This is
the correct leave-one-out behavior.

## T3 resolved: agreement-based combination

The combination should honor each instrument's voice:

- When LSH and GSH AGREE: high confidence. Both geometric
  proximity and routing-pattern similarity point to the same
  class. Accept immediately.

- When they DISAGREE: the disagreement itself is information.
  The LSH says "geometrically nearest is Cat" while the GSH
  says "routes like a Dog." This means the query sits at a
  boundary in pixel space (near a Cat prototype) but in an
  unusual region of routing space (its routing pattern
  matches Dog images). In this case, defer to LSH (the
  direct geometric measurement) but FLAG the query as
  uncertain.

Simplest implementation: `combined_pred = lsh_pred` when
they agree OR when GSH k-NN margin is below a threshold.
`combined_pred = gsh_pred` only when GSH margin is high
AND LSH margin is low.

But for V1: just report agreement rate and measure accuracy
conditional on agreement vs disagreement. This tells us
whether the GSH's voice is worth listening to before we
commit to a combination formula.

## What I now understand

1. **The GSH failed because it tried to be a LSH on degraded
   data.** The fix is not better encoding for the same
   approach — it's a fundamentally different approach: treat
   the routing pattern ITSELF as a hashable signature, not
   as input to another random projection.

2. **The GSH's natural distance metric is vote disagreement**
   (how many tables differ), not Hamming distance on random
   projections of one-hot labels.

3. **Multi-trit categorical encoding** maps vote disagreement
   to trit Hamming distance with low distortion, using the
   existing trit infrastructure unchanged.

4. **The combination should measure agreement** before
   committing to a formula. If LSH and GSH agree 95% of
   the time and the 5% disagreement is randomly right/wrong,
   the GSH adds noise. If the 5% disagreement favors the
   GSH, the combination adds value.

5. **The GSH's training routing signatures must come from
   the LSH probe**, not brute-force. The routing pattern IS
   the LSH's output for that image — the GSH routes through
   the LSH's behavior.

## What remains uncertain

- Whether the multi-trit encoding actually produces better
  GSH accuracy than the one-hot encoding.
- Whether the GSH's vote-disagreement distance is
  discriminative enough for k-NN classification.
- Whether the agreement rate between LSH and GSH is high
  enough for the combination to add value.
- Whether the first 4 tables (16 trits = 4 bytes) produce
  a useful bucket key in routing-pattern space.
