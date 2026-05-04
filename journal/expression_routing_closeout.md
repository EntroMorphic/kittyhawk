# Closeout: Expression Routing (vision claim #2 P0)

## What ran

All four P0 pieces from `journal/expression_routing_synthesize.md` shipped and ran end-to-end on 2026-05-03:

- **P0-2** (`gesh/src/expr.{h,c}`): expression tree types over the substrate's primitive ops (var, const, neg, add, sub, mul, max, min). exp/log deliberately absent (P1-1 territory). Evaluator returns int64.
- **P0-1** (`gesh/src/expr_signature.{h,c}`): behavior-based signature derivation. Evaluate the expression at fixed test inputs, ternarize via `m4t_route_threshold_extract` at tau=0. Substrate-discipline preserved.
- **P0-3** (`gesh/src/expr_bank.{h,c}`): equivalence-class bank constructor. Detects byte-equal signatures across candidates, picks first-in-order as representative, exposes a candidate→class map.
- **P0-4** (`gesh/bench/expr_routing_probe.c`): the probe binary, with TWO pre-committed verdict gates (easy + hard).

## Verdict

```
EASY  : 60/60 correct (100.0%)  floor=OK  -> PASS
HARD  : 18/18 correct (100.0%)  floor=OK  -> PASS
OVERALL VERDICT: PASS
```

The EASY gate (originally pre-committed) tested algebraic equivalents — expressions byte-equal to bank representatives on the test inputs. PASS criterion: ≥51/60 with per-class floor ≥1/3. Result: 60/60.

The HARD gate (added after the easy gate trivially passed; pre-committed in code BEFORE running) tested near-equivalents — expressions whose nearest bank class was determined by hand-computed signature distance, not byte-equality. PASS criterion: ≥14/18 with per-class floor. Result: 18/18.

## Equivalence-class detection (P0-3 working as designed)

Expected mergers all caught:

**Arity-1 (12 candidates → 10 classes):**
- `x` ≡ `x*x*x` (sign-equivalent on integer test inputs)
- `|x|` ≡ `x*x` (sign-equivalent)

**Arity-2 (14 candidates → 9 classes):**
- `x+y` ≡ `min+max` ≡ `x+(y+0)`
- `min(x,y)` ≡ `min(min(x,y),x)`
- `max(x,y)` ≡ `max(max(x,y),y)`
- `x²-y²` ≡ `(x+y)*(x-y)`

The bank constructor's job — "detect and merge equivalence classes by signature" — is operational.

## Hard-probe highlights (the non-trivial result)

Three probes had non-obvious expected routings that I hand-computed and the system matched:

- `x*(x-1) → |x|`: distance 1 (one extra zero compared to |x|), vs 4+ to other quadratic-shaped classes.
- `(x+y)² → |x-y|`: distance 8 (both are non-negative magnitude functions), vs 12 to `x²-y²`.
- `max(x,y)+5 → |x-y|`: counterintuitive — naively would route to max, but adding 5 flips the negative max-quadrant signs to positive, matching |x-y|'s "all positive except diagonal zeros" pattern at distance 4 vs distance 6 to max.

Every hand-prediction matched routing. The routing is doing principled nearest-neighbor in trit space, not pattern-matching to my expectations.

## What this PASS proves

> **AMENDED 2026-05-03 per M4 in `expression_routing_redteam.md` and closed in `expression_routing_remediation_closeout.md`.** Original framing said "vision claim #2's mechanism is operational." The honest narrower version is below; see the remediation closeout for the corroborating subagent-blind result (29/30 = 96.7%) that earned the broader version.

1. **Behavior-based equivalence-class lookup is operational at toy scale on hand-designed banks.** Sign-equivalent expressions on the test inputs converge to byte-identical signatures, and nearest-tile routing returns the equivalence-class representative deterministically. (The remediation cycle subsequently corroborated this with 96.7% match rate from an independent subagent designing probes from mathematical intuition.)
2. **The bank-as-equivalence-class-lookup framing (LMM REFLECT insight) was the right shape.** No new bank type required; existing `gesh_bank_t` works with reframed label semantics.
3. **The substrate-discipline rule extends to expression code.** Ternarization through `m4t_route_threshold_extract`; no open-coded sign step in any new file.

## What this PASS does NOT prove

1. **No negative controls in the probe set.** Every hard probe had a single clear expected class. Not tested: probes whose nearest class is genuinely ambiguous (multiple classes equidistant). Healthy routing should produce a deterministic-but-noted answer; the test didn't surface this case.
2. **No adversarial probes.** Probes that look syntactically similar to one class but evaluate to a different one — would test whether the signature rule is fooled by syntactic structure. Not tested.
3. **No probes designed without prior knowledge of the bank.** The probe-author (me) computed expected routings from the same signature math the routing uses. A fresh hand designing probes blind would be a stronger test.

## Discipline notes

- All new code under `-Werror` with project standard flags. Builds clean.
- Substrate-discipline check: ternarization through `m4t_route_threshold_extract`; packing through `m4t_pack_trits_1d`; distance through `m4t_popcount_dist`. No open-coded MAC, no open-coded sign step. The rule from the substrate-discipline cleanup extended cleanly.
- Pre-committed verdict gates (both easy and hard) were written into the code BEFORE running. No post-hoc tuning of the gate to fit results.
- The HARD gate itself was added AFTER the EASY gate trivially passed. This is the project's pattern — when a verdict is suspiciously easy, add a tighter test rather than accept the verdict at face value.

## Recommended next moves

Three options, in order of "tightens current evidence" to "extends to new ground":

1. **Negative-control + adversarial probes.** ~1 day. Adds 5–10 probes whose nearest class is genuinely ambiguous, plus 5–10 probes that look like one class syntactically but evaluate as another. Verdict: does the routing return defensible answers under uncertainty?
2. **Blind probes from a fresh hand.** Spawn a subagent to construct probes without seeing the signature math, only the bank's representative names. Tests whether the routing matches a different observer's intuition or only matches my own.
3. **Proceed to P1.** Two paths: P1-1 (close the primitives floor with exp/log) or P1-2 (unified address space across data and expression banks). Both genuinely hard; both gated on this PASS, which we now have.

My read: option 1 is the lowest-cost discipline move; option 2 is the highest-value epistemic move; option 3 is the right one if we're satisfied the P0 verdict is robust enough.
