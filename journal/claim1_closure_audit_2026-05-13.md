# Claim 1 closure audit — bridge primitives → elemental floor

Per `glyph_gaps_2026-05-13_synthesize.md`: claim 1's audit was
deferred pending claim 2 bridge outputs. Claim 2's first measurement
landed positive on 2026-05-13 with `experiments/claim2_bridge/`
implementing approach B (routing-derived signatures). This audit
answers: **does the substrate's elemental floor close over the
primitives the bridge uses?**

## Setup

Bridge uses four trit-level routings in `experiments/claim2_bridge/routing.py`:

- `route_add(a, b)` — element-wise saturating ternary add.
- `route_sub(a, b)` — element-wise saturating ternary sub.
- `route_mul(a, b)` — element-wise ternary product.
- `route_neg(a)`    — element-wise trit negation.

Substrate's elemental floor (per `journal/elemental_floor_closeout.md`,
2026-05-04, currently shipped in `m4t/src/`):

- `add`     (`m4t_trit_add`, `m4t_mtfp_add`).
- `neg`     (`m4t_trit_neg`, `m4t_mtfp_neg`).
- `shift3`  (`m4t_mtfp_shift3`).
- `sign`    (`m4t_route_threshold_extract` at τ=0).
- `select`  (`m4t_route_select`, trit-controlled cell-level mux).

## Mapping the bridge to the elemental floor

| bridge routing | composition over the elemental floor |
|---|---|
| `route_neg(a)`   | direct `neg`. ✓ elemental, 1 op. |
| `route_add(a,b)` | per-cell saturating sum on trits — derivable as `select(a, sat(b+1), b, sat(b−1))` where the +/0/− cases of `a` route to `b+1`, `b`, `b−1` (then saturated). Composite from `add` and `select` (plus constants). |
| `route_sub(a,b)` | `route_add(a, route_neg(b))`. Composite from `add`, `neg`, `select`. |
| `route_mul(a,b)` | per-cell trit product. Derivable as `select(a, b, 0, neg(b))`: if `a=+1` route `b`, if `a=0` route `0`, if `a=−1` route `neg(b)`. Composite from `select` and `neg`. |

**Verdict: the elemental floor closes over the bridge's currently-
used routings.** Every primitive the bridge invokes can be expressed
as a composition of `{add, neg, sign, select}`. `shift3` is not
required by the bridge's current ops (would enter for integer-valued
arithmetic where exponent shifts matter — see "what's missing" below).

## Practical implementation note

The bridge's `route_*` routines are implemented as NumPy element-wise
ops on `int8` trit arrays. The corresponding substrate-native
implementations would go through `m4t_trit_*` / `m4t_route_*`
primitives. The composition-over-floor decomposition above is
mathematically correct; whether to actually re-express the bridge
on the substrate kernels is a downstream question (it's research
code right now).

The point of this audit is NOT "rewrite the bridge in C." The point
is: the elemental floor is sufficient. Claim 1's "the six-primitive
floor closes over math" claim is substantiated for the slice of math
the bridge handles.

## What's missing (claim 1's actual gap)

The bridge handles `{+, −, *}` plus identities and zero/one
constants. It does NOT handle:

1. **Exp / log / division.** Required for softmax, sigmoid, log-
   likelihood, and any operation involving multiplicative inverses.
   Per `journal/p1_1_primitives_floor_closeout.md` and elemental-floor
   analysis: exp/log via Taylor requires division, division is
   composite via iterated conditional-sub + `shift3`. The substrate
   currently lacks a kernel for ternary integer division. Implementing
   it is a focused follow-on.

2. **General integer arithmetic** (constant arithmetic). The bridge
   measurement at constant_arithmetic class scored 0% — `s(2) + s(3)`
   is not `s(5)` because constants are SHA-derived signatures, not
   balanced-ternary numeric encodings. Closing this gap requires
   either:
     - balanced-ternary encoding of integers in trits, so
       `route_add` literally performs ternary addition with carry,
       OR
     - the integer values to be represented at the MTFP level
       (using mantissa + exponent), where addition does carry —
       which means routing through the MTFP add primitive, not
       the trit-level routings.
   Both paths are substrate-native. Neither is implemented for the
   bridge yet.

3. **Comparison / max / min / abs / eq.** Composites per the
   elemental-floor analysis. Used downstream in things like
   attention (compare-and-mask) and clipping. The substrate has
   these as composite kernels (`m4t_trit_max`, etc.); the bridge
   doesn't currently invoke them but would if grammar extended to
   piecewise expressions.

## Status of claim 1

After this audit:

- **Substantiated for the slice the bridge currently covers.** The
  elemental floor `{add, neg, shift3, sign, select}` closes over
  `{+, −, *, ¬}` via explicit composition.
- **Open for `exp/log/div`.** Genuinely missing kernels. Listed
  as deferred work in the Glyph gaps synthesis.
- **Open for general integer arithmetic in the bridge.** Bridge
  uses SHA-derived constant signatures; closing the `2+3=5` gap
  requires an integer-encoding decision (balanced ternary vs MTFP).

## What this changes for the vision

This is the **first explicit measurement linking claim 1 and claim
2**. Before this audit:

- Claim 1: 5 elemental primitives shipped (substrate).
- Claim 2: bridge implements routing-derived signatures (research).
- No demonstration that one closes over the other.

After this audit:

- The bridge's currently-used routings are decomposable into the
  elemental floor. **The floor is sufficient for what the bridge
  does.** The vision's "math = routing over primitives" framing
  has its first end-to-end (bridge → floor) trace.
- The gaps (`exp/log/div`, integer arithmetic) are NAMED and
  scoped, not generic "the substrate is incomplete."

## Recommended next steps

In priority order (per the spirit of the gaps synthesis, not as a
deadline):

1. **Balanced-ternary integer encoding for the bridge.** Lets the
   bridge handle constant arithmetic, polynomial evaluation with
   numeric coefficients, and any expression where constants matter.
   ~1 day for the encoding, ~1 day for re-measurement.

2. **Division kernel** at the substrate level (`m4t_mtfp_div` or
   `m4t_trit_div`). Composite from `shift3` and iterated
   `conditional-sub`. Per the elemental-floor analysis. Once
   shipped, exp/log via Taylor become accessible.

3. **Exp / log via Taylor**, running through the substrate
   primitives. Closes a major gap in claim 1's coverage.

4. **Bridge extension to handle exp/log/div in expressions.** Once
   substrate kernels exist, the bridge's grammar grows.

## Files

- `experiments/claim2_bridge/routing.py` — bridge routings under
  audit.
- `m4t/src/m4t_route.{h,c}` — elemental `add`, `neg`, `sign`,
  `select` kernels.
- `m4t/src/m4t_mtfp.{h,c}` — elemental `shift3` plus MTFP add/neg.
- `journal/elemental_floor_closeout.md` — established floor (2026-05-04).
- `journal/p1_1_primitives_floor_closeout.md` — superseded prior
  cycle that named the exp/log/div gap.

## Sign-off

Claim 1 has its first explicit closure measurement. The floor's
five primitives close over the bridge's four routings via explicit
composition. The named gaps (`exp/log/div`, integer arithmetic) are
the substantive content of claim 1's open work. **No new primitive
is needed for the bridge as-is.**
