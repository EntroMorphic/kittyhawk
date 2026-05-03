---
cycle: gesh_zero_signal_design (P0-1)
phase: RAW
date: 2026-05-02
scope: enumerate every place the codebase currently treats the structural zero as "default" or "tie" — substrate-claim drift that the P0 plan corrects
companions: docs/REMEDIATION_PLAN_P0.md (P0-1 section)
status: capture
---

# RAW — gesh_zero_signal_design

The substrate's three-state alphabet is {-1, 0, +1}. The third state is what base-2 doesn't have. Cataloging where current code uses zero, where current code *should* use zero distinctly, and what zero could uniquely express.

## Where the codebase currently produces zero trits

Direct sources (the operational *origin* of zero state in our pipeline):

1. **`m4t_route_threshold_extract`** at tau=0. Inputs `v` with `v == 0` → output trit 0. Documented in `m4t_route.h` §18 as the "zero" emission state. **Origin source #1.**

2. **Direct ternary quantization** (`image_canon::quantize_unpacked_batch`, `glyph_sig_quantize` legacy). Pixel values within `±tau` of normalized mean → output trit 0. **Origin source #2.**

3. **Ternary projection** (`gesh_project_one_packed`, `gesh_project_batch_unpacked`). The matmul accumulator may equal exactly zero (sums of ±1 terms can cancel); threshold-extract then emits trit 0. **Origin source #3** — though probabilistically rare for high-dim inputs.

4. **Class-mean bank construction** (`gesh_bank_build_class_mean`). Per-class per-dim sums sign-thresholded; zero appears when class samples are evenly split on a dim. **Origin source #4.**

5. **Random ternary projection (balanced)** (`gesh_init_random_projection_balanced`). Init writes ±1 or 0 with equal probability per trit. **Origin source #5** — used in calibration only; the regular `gesh_init_random_projection` writes only ±1.

6. **k-means centroids** (`gesh_bank_build_kmeans_per_class`). Cluster centroid sign-thresholded per dim; zero appears when within-cluster samples split evenly. **Origin source #6.**

## Where the codebase currently *consumes* zero trits — and what it does

For each origin, the downstream consumer treats the zero:

### Hamming distance (`m4t_popcount_dist`)
- Implements ternary Hamming via XOR popcount on packed bytes.
- Per-trit cost lookup:
  - (+1, +1) → 0,  (-1, -1) → 0,  (0, 0) → 0  (any agreement)
  - (0, +1) or (0, -1) or (+1, 0) or (-1, 0) → 1  (one zero, one signed)
  - (+1, -1) or (-1, +1) → 2  (opposite signs)

**Operational role of zero here: zero counts as "halfway agreement."** Two zeros at the same position is a match (cost 0); one zero against a ±1 is partial mismatch (cost 1). That's the standard ternary Hamming semantics — but it's *symmetric*. The kernel treats the zero in either signature equivalently; it doesn't distinguish "zero in the query" from "zero in the bank tile."

### Top-k smallest tile selection
- The forward pass computes Hamming distance to each tile; picks top_k smallest; votes class labels.
- **Operational role of zero here: none distinct.** A tile with many zeros and a tile with all ±1s are compared by aggregate Hamming distance only. The number of zeros in a tile doesn't affect its selection probability beyond what's already captured in Hamming.

### Class vote / argmax
- Per-tile-class vote tally, argmax with lower-class-index tie-break.
- **Operational role of zero here: none.** Bank tile signatures may be zero-laden; vote tally is over class labels, not tile features.

### `m4t_route_apply_signed`
- Accumulates `result[d] += sign * tile_outs[tile_idx * dim + d]` for each (tile_idx, sign) decision.
- Decisions with `sign == 0` are sentinel-skipped.
- **Operational role of zero here: signal-sentinel.** Zero on an *aggregated* decision means "no decision" — that's NOT the structural zero from the *input* signature; it's a different semantic ("decision sentinel" vs "input third state").

### `m4t_mtfp_ternary_matmul_bt`, `m4t_ternary_dot_matmul_bt`, `m4t_mtfp4_sdot_matmul_bt`
- Inner products of ternary × ternary or MTFP × ternary.
- Per-position contribution: `acc += a × b` where `a ∈ {-1, 0, +1}` and same for `b` (or activations in MTFP).
- **Operational role of zero here: ZERO contribution but FULL iteration cost.** A zero trit in the weight skips the negate-and-add via bit-select, but the iteration still loads the activation, evaluates the bit-select, and runs through. SDOT does 16 lanes per cycle regardless of how many lanes are zero.

So the zero is *arithmetically* free (multiplying by zero contributes nothing) but *computationally* paid (iteration runs through it). **The substrate has no primitive that uses zero to skip work.**

### Bank construction (`gesh_bank_build_class_mean`)
- Per-class per-dim sum, sign-thresholded.
- Zero in the result means "no consistent sign across class samples."
- **Operational role of zero here: signature feature.** The zero IS preserved in the tile. But the bank doesn't track *how many* zeros are in each tile, doesn't use it as a confidence signal, doesn't propagate it into routing decisions.

## What current code does NOT do with zero — substrate-claim gap inventory

A1. **No primitive uses zero to gate downstream computation.**
  - SDOT iterates through zero-lanes paying full cost.
  - Threshold-extract emits zeros that are then iterated equally with ±1 by every subsequent consumer.
  - Multi-stage routing (when we build it, P0-4) has no native way to skip dims that the prior stage emitted as zero.

A2. **No primitive measures "agree-as-zero" distinctly from agree-as-anything.**
  - `m4t_popcount_dist` collapses (0,0) and (+1,+1) and (-1,-1) into "cost 0" — they're all "matches."
  - A signature pair (-, 0, +) and (-, 0, +) has Hamming distance 0 — but it's not the same as (-, +, +) and (-, +, +) which is also Hamming 0. The first pair *agrees on a don't-care*; the second pair *agrees on an opinion*. Operationally distinct, currently indistinguishable.

A3. **No primitive interprets zero as wildcard / don't-care.**
  - In ternary content-addressable memory (TCAM) and many classical decision-rule formalisms, the third state is a *wildcard*: a query position vs a wildcard rule position is a free match.
  - Our Hamming costs zero-vs-±1 at 1, treating it as half-disagreement. A wildcard interpretation would cost zero-vs-±1 at 0.
  - This is the simplest substrate-novel routing primitive imaginable, and we don't have it.

A4. **No primitive uses zero count as an information-content signal.**
  - A signature with 95% zeros is much "sparser" than one with 5% zeros — operationally, it claims to have an opinion on only 5% of positions.
  - The substrate could route based on zero count: high-zero queries go to general-purpose tiles, low-zero queries to specialist tiles.
  - We don't track or use zero counts anywhere.

A5. **No bank constructor produces zero-aware tiles by design.**
  - Class-mean: zeros are emergent from sign-thresholding tied sums. No deliberate "this dim is class-c-irrelevant" prior.
  - K-means: same.
  - We could build a bank type where each tile *deliberately* zeros dimensions known to be uninformative for that class. The substrate has the form to express this; the consumer doesn't use it.

A6. **No matmul uses input-side zeros to skip per-cell work.**
  - SDOT iterates the K-loop fully regardless of zero density in the activation.
  - For activations with many structural zeros (post-quantize MNIST has ~60% zeros), SDOT pays 100% cost for ~40% of useful work.
  - A zero-aware matmul could skip K-iterations where the activation trit is zero. Substrate-native sparsity benefit.

A7. **No threshold-extract variant respects an input-side "do not emit anything here" mask.**
  - Current threshold takes a vector and emits a packed-trit. There's no way to say "skip these positions; they were already determined to be zero by an earlier stage."
  - For multi-stage routing (P0-4), this is needed: stage 2 should respect stage 1's zero decisions.

## What zero could uniquely express that base-2 cannot

The structural zero is the substrate's only state that is:

- **Symmetric under sign flip.** ±1 are mirror images; zero is its own mirror. This makes it the natural "sign-agnostic" state.
- **Multiplicative absorbing.** `0 × anything = 0`. In ternary arithmetic, zero propagates exactly; in base-2 there is no element with this property (in {-1, +1}).
- **Free at zero cost in dot products.** No widening, no negate, no add — purely skippable.
- **A natural "abstain" decision.** In voting/routing, zero is "no opinion at this position."
- **A natural "wildcard."** In pattern matching, zero is "match anything."
- **A natural "disabled."** In sparse computation, zero is "this output dim is gated off."

Base-2's substitutes for these:
- "Sign-agnostic" → use sign + magnitude or extra mask bit (doubles storage).
- "Multiplicative absorbing" → no analog (closest is the "off" state in {0,1} but that's not in {-1,+1}).
- "Free at zero cost" → still iterates; can't skip without explicit branching.
- "Abstain" → uniform output or learned skip token (probabilistic).
- "Wildcard" → not natively expressible; must use multiple match rules.
- "Disabled" → mask bit (separate storage) or learned dropout.

The substrate has all six interpretations free in one state. Current consumer code uses *none* of them operationally.

## What the original GESH design said about zero

The closeout for the gesh_design cycle (`journal/gesh_design_closeout.md`) explicitly noted:

> *"In a base-3-native system where:*
> *- The substrate is bit-exact integer arithmetic.*
> *- The projections are ternary from the start (not float-then-quantize).*
> *- The data lives on the trit lattice (not in float space passing through).*
> *- The loss is computable bit-exactly per the verified kernels.*
>
> *...there is no discontinuity to estimate through. STE is solving a problem that doesn't exist."*

This is correct as a critique of STE. But the closeout's positive proposal ("lattice-update coordinate descent") used the trit lattice as a *search space* — flipping trits to optimize a label-shaped objective. **It did not propose using the trit lattice's third state as a structural feature.** The third state was implicit in "the lattice IS the geometry" but not made operational.

Ditto for the design synthesize phase: it focused on the lattice as a discrete optimization space, not as a representational structure with a meaningful zero.

## Free-floating observations

- The "structural zero" terminology was coined in `m4t_route.h::m4t_route_threshold_extract` documentation: *"|value| <= tau → 0 (code 0b00) within neutral band."* The phrase "neutral band" captures the abstain interpretation; the operational consequences were never explored.

- The synthetic prototype benchmark (`synth_proto.c`) generates samples where noise dims are uniform-random ternary. It generates zeros operationally, not structurally — the noise dims have zeros, but those zeros mean "random noise hit zero," not "deliberately uninformative." Different semantics.

- MNIST quantization at density 0.60 produces ~60% zeros. These are *most* of the trit positions. Operating on these zeros as if they're full-information ±1 trits is doing 100% of the work for ~40% of the signal.

- In ternary content-addressable memory (TCAM, used in network routers), the wildcard semantics is the entire point. The substrate has TCAM-shape natively; the consumer doesn't use it.

- The Phase A.2 sweep at sig_dim=2 with C=10 showed 4 distinct trained tile signatures and 6 classes at 0% accuracy (pigeonhole-forced collisions). At sig_dim=2, the signature space is 3² = 9 distinct ternary strings — but all 9 are equally available. **A wildcard-aware bank could treat (+, 0) as a *region* covering both (+, +) and (+, -)** — extending effective coverage without increasing tile count. This wasn't on the table because we never had wildcard semantics.

## What's NOT in this RAW

- Specific kernel API designs. That's NODES territory.
- Comparisons to specific base-2 architectures (sparse attention, masked transformers). That's REFLECT territory.
- Choice of which zero-signal primitive to build first. That's SYNTHESIZE territory.
- Any code. Per the P0 protocol, no code until SYNTHESIZE commits.
