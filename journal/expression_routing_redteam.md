# Red-Team: Expression Routing P0

Adversarial pass on the PASS verdict from `journal/expression_routing_closeout.md`. The closeout itself flagged several limits; this document is the unsparing version.

Findings labeled C (critical: would invalidate PASS), H (high: significant), M (medium), L (low).

---

## C1 — The probe author designed every piece the verdict depends on

The PASS rests on a single hand designing:
- The bank candidates (which expressions live in the bank).
- The test inputs (the lens through which behavior is observed).
- The signature derivation rule (sign-threshold at tau=0).
- The probe expressions (what gets routed).
- The expected routing for every probe (computed by the same hand that designed the rule).
- The verdict gate (≥14/18 with per-class floor).

This is the textbook "marking your own homework" problem. The PASS verdict is meaningfully closer to "the system computes what I told it to compute" than to "the system independently recovers a routing-as-equivalence claim."

The closeout admits this under "what this PASS does NOT prove" item 3, but the admission is too gentle. **The verdict is not yet evidence that vision claim #2 holds independently of this author's design choices.**

**Recommended:** at minimum, the blind-probes-from-a-fresh-hand option (option 2 in closeout). Properly: a separate cycle whose author has not seen the signature math, only the bank's representative names. That cycle's verdict is the real PASS.

---

## C2 — The HARD gate is largely tautological

The HARD probes had expected routings hand-computed by sign-Hamming arithmetic — exactly what the routing kernel implements. Verifying routing matches my pen-and-paper computation is verifying the kernel implements its specification, which is what the m4t kernel tests already prove.

Three "non-obvious" hard probes (`x*(x-1) → |x|`, `(x+y)² → |x-y|`, `max(x,y)+5 → |x-y|`) felt like they earned the gate, but they only earned it relative to *naive expectation*. Relative to the actual signature distance arithmetic, the answer was determined by computation, not by routing intelligence.

**The HARD gate doesn't measure "does routing recover truth." It measures "does the routing kernel arrive at the same answer as the closed-form Hamming computation that defines what 'closest' means."** The closed-form computation IS the routing rule.

**Recommended:** real HARD probes need *out-of-distribution* expected answers — cases where two different observers, looking at the bank and the probe, would predict different routings. The probe author and the routing kernel should not have access to the same calculation.

---

## H1 — Test inputs were chosen to span sign-flip points of bank candidates

The 16 arity-1 inputs `{-30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30}` include exactly the points where simple polynomials change sign: 0 (sign change for `x`, `|x|`, etc.), ±1 (for `x²-1`), 0 and 3 (for `x*(x-3)`).

This gives the bank classes maximum sign-pattern distinctness *by construction*. A different test set — for example, all positive integers, or 16 random samples from a distribution — would collapse classes that currently look distinct.

The discrimination quality reported in the verdict is **a property of the input set, not a property of the signature rule.**

**Recommended:** sweep over multiple test-input sets, including (a) randomly drawn from a distribution, (b) all-positive, (c) coarsely spaced (e.g., powers of 10), (d) finely spaced near zero. Report which classes survive each lens. Genuinely robust classes survive multiple lenses; lens-dependent ones don't.

---

## H2 — Bank candidates were picked to be pairwise distinguishable

I selected 12 arity-1 candidates and 14 arity-2 candidates that I expected (or knew) would have distinct sign patterns. I did not include adversarial candidates — for example, `x + 0.5` (would be sign-equivalent to `x` on integer inputs only), `x*x*x*x` (would merge with `x²` under sign equivalence), or any pair I expected to collide artificially.

Selection bias: the bank is curated to make equivalence classes look clean. A bank built from a less curated set would surface more mergers and collisions, exposing the rule's actual discrimination power.

**Recommended:** populate the bank from a less-curated source — random expression trees of bounded depth, or a fixed enumeration. Report the merger rate honestly. A high merger rate means the signature rule is coarse; that's information.

---

## H3 — Sign-only ternarization throws away magnitude

The signature is `sign(eval)` at every test input. Two expressions that have the same sign at every input but different magnitudes hash to byte-identical signatures.

Concrete example: `x` and `1000*x` produce byte-identical signatures on this test set. The system would route a query for `1000*x` to the `x` class. **Mathematically these are not the same function.**

The probe set didn't include this case. If it had, the system would correctly route — but to a class that loses real mathematical information about the expression's scale.

This is a known limit of the chosen rule (acknowledged as "compose-equivalence is a feature" decision D5 in the synthesize), but the probe set didn't surface it. A red-team probe set should include `2*x → x`, `100*(x+y) → x+y`, etc., explicitly to expose the collapse.

**Recommended:** add a "scale-collapse" section to the probe set. Report which mergers happen. They will all happen as designed; that's the data, not a failure. The honest framing is "the routing recovers sign-equivalence classes, NOT magnitude-aware functional equivalence." That's a narrower claim than the closeout's framing.

---

## H4 — The HARD gate was defined AFTER seeing the EASY gate pass

The discipline rule says verdict gates are pre-committed. The HARD gate was added as a tightening response to the EASY gate trivially passing. The gate selection was therefore *responsive to what we'd observed*, not blindly committed in advance.

This is not as bad as tuning gate thresholds AFTER seeing results (which I didn't do), but it is in the same family. A cleaner version would have committed both gates in the original synthesize before any code ran.

**Recommended:** for future cycles, pre-commit BOTH the easy and hard gates in SYNTHESIZE before any code lands. The pattern of "easy gate passed → add hard gate" should be flagged as a gate-revision event in the closeout, not as a normal tightening.

---

## M1 — Arity-2 discrimination headroom is small

Sig_dim 16 with 9 classes leaves ~1.78 trits per class on average. The bank constructor reported all 9 classes as distinguishable, but barely — the closest pair is at distance 4 of 16 (one quarter of the signature). Adding any candidate that happens to land within distance 4 of an existing class collides.

Concrete risk: the next 5 candidates added to the arity-2 bank may merge unexpectedly. Discrimination is not a property to take for granted at this dim.

**Recommended:** report inter-class signature distance distribution as part of bank diagnostics. If the minimum inter-class distance is < ⌈sig_dim / (4 × n_classes)⌉, sig_dim is too small.

---

## M2 — No multi-seed; everything deterministic

The project's own multi-seed rule (Phase A.2 red-team) says single-seed configurations don't generalize. The probe is single-config:
- One bank candidate set.
- One test-input set.
- One probe expression set.

The deterministic PASS proves "this specific configuration works." It does not prove configurations of this shape work in distribution.

**Recommended:** parameterize the bank candidate generator, test-input sampler, and probe generator with seeds. Run 5–10 seeds. Report mean and CI on PASS rate. If single-seed PASS is artifact, multi-seed will surface it.

---

## M3 — The substrate's third state is barely used

Vision claim #3 says base-3 carries information base-2 collapses; the §19 zero-state taxonomy operationalizes this. The signature in this work is sign-extract at tau=0 — the zero only appears when an expression evaluates to exactly 0 on a test input. For most expression-input pairs, that doesn't happen.

Effective storage: the signatures are mostly 1-bit-per-position (sign), with rare 2-bit-meaningful-zero. The substrate's distinct capability (wildcards, dual-threshold, confidence-weighted routing) is unused.

This is fine for a P0 demonstration of the basic mechanism, but framing the PASS as "vision claim #2 operational" is an overclaim if vision-claim-#3 affordances aren't being exercised. The system here would behave nearly identically with binary signatures.

**Recommended:** flag this for P1-1 design. When closing the primitives floor, the design should let the third state carry information (e.g., wildcard for "expression undefined at this input," weak/strong magnitude band).

---

## M4 — "Vision claim #2 operational" is overclaim

The closeout writes: "Vision claim #2's mechanism is operational." Read literally, this is true for the toy scale tested. Read as the audience would read it ("vision claim #2 is now established"), it overclaims.

What is established: a 10-class equivalence-class lookup over hand-designed expressions works deterministically. What vision claim #2 actually requires: open-ended math, infinite equivalence classes, real algebraic structure routing through real signature space.

The closeout's own "what this PASS does NOT prove" section walks this back, but the headline framing remains.

**Recommended:** reframe the closeout's lead sentence. "The mechanism for behavior-based equivalence-class lookup is operational at toy scale on hand-designed banks." That's the honest version.

---

## M5 — Compose-equivalence (D5) creates a hidden cost-blindness

D5 declared "routing `exp(x)` and its Taylor truncation to the same address is a feature." True for equivalence-recognition. But: if any downstream consumer treats the bank's representative as the canonical implementation AND uses the representative's evaluation cost as the operation cost, it will be wrong for any query whose actual structure is more expensive.

P0 didn't deploy any cost-aware consumer, so the issue is latent. P1 work that uses the bank for anything beyond identity-lookup needs to address this.

**Recommended:** when P1-1 lands exp/log, document the cost-blindness explicitly. Either carry cost as metadata on each candidate AND on each query, or restrict the bank to identity-lookup with no cost interpretation.

---

## L1 — `tile_idx` vs `class_idx` conflation in the probe

The probe uses `route_signature` which returns `best_t` (a tile index). It compares this to `bank.candidate_to_class[expected_candidate]` (a class index). They happen to be equal because `expr_bank_build` packs classes into tiles 0..n_classes-1 in order.

If a future bank constructor permuted tile order (e.g., for cache-line alignment, or to put high-frequency classes first), the comparison would silently break.

**Recommended:** add a `tile_to_class[]` indirection in `expr_bank_t`. Even if it's the identity today, the indirection documents the semantic distinction.

---

## L2 — Equivalence-detection uses `memcmp`, not a substrate kernel

`expr_bank.c` detects equivalence via `memcmp` on packed signatures. That's stdlib, not libm4t. The substrate-discipline rule was about MAC and sign-threshold, not byte equality, so this isn't a violation — but the substrate has no kernel for "are two trit signatures equal." Worth considering whether this should be one (`m4t_route_signature_equal`).

**Recommended:** if any future consumer needs to compare signatures in a hot path, add `m4t_route_signature_equal` as a substrate primitive. For P0 the memcmp is fine.

---

## L3 — Per-class floor in HARD gate has no teeth

HARD gate: ≥14/18 AND per-class floor of ≥1/3. With most classes having only 1–2 hard probes, the 1/3 floor is met by any single correct answer. The floor is structural at this probe count, not actually testing anything.

**Recommended:** at the 18-probe scale, the per-class floor adds no information. Either drop it (and report the per-class breakdown directly) or scale up the probe count until the floor has teeth (≥6 probes per class would make the 1/3 floor meaningful).

---

## L4 — Constants in probes introduce implicit assumptions

Probes use `K(1)`, `K(5)`, `K(10)` etc. The constant magnitudes interact with the test-input range: `K(5)` shifts a sign-flip point onto a test input; `K(50)` would shift it past the input range and become invisible.

The probe set effectively assumed "constants close to test-input scale matter; constants far from it don't." That's true on this set, but it's a hidden coupling between probe construction and input-set choice.

**Recommended:** when building probes for any future cycle, document the scale-coupling explicitly. A more robust probe construction would use constants drawn from a distribution matched to the input set's scale.

---

## Summary

| ID | Severity | Status |
|----|----------|--------|
| C1 | Critical — single hand designed everything | PASS verdict not independent of author |
| C2 | Critical — HARD gate is largely tautological | Doesn't measure what it claims to |
| H1 | High — test inputs chosen to favor bank | Discrimination is artifact of choice |
| H2 | High — bank candidates curated for distinctness | Selection bias |
| H3 | High — sign-only signatures throw away magnitude | Real equivalence-collapse, untested |
| H4 | High — HARD gate added after EASY gate passed | Gate selection was responsive |
| M1 | Medium — arity-2 discrimination headroom small | Risk for future bank growth |
| M2 | Medium — no multi-seed | Doesn't generalize |
| M3 | Medium — third state barely used | Vision claim #3 affordances unexercised |
| M4 | Medium — closeout headline overclaims | Reframe |
| M5 | Medium — cost-blindness latent in D5 | P1 needs to address |
| L1 | Low — tile/class conflation | Add indirection |
| L2 | Low — memcmp not a substrate kernel | Future primitive candidate |
| L3 | Low — per-class floor lacks teeth | Drop or scale up |
| L4 | Low — constant/input scale coupling | Document |

## What this red-team changes about the verdict

The original PASS stands as "the wiring works at toy scale on hand-designed banks." It was always that, even before red-team — the closeout said so. The red-team sharpens *how narrow* that claim is and identifies the specific work needed to extend it:

1. **C1, C2 must be addressed** before the PASS can be cited as evidence for vision claim #2 in any external context. Closeout option 2 (blind probes from a fresh subagent) is the minimum next step.
2. **H1, H2** suggest the bank-curation pattern needs to be replaced or supplemented with random/enumerated bank construction.
3. **H3** suggests a mandatory probe section: scale-collapse cases, with the honest result reported (not as a failure, but as a documented limit of sign-only signatures).
4. **H4** is a methodology note for future cycles, not actionable on this one.
5. The M findings shape P1's design.
6. The L findings are housekeeping.

**Recommendation: do not cite the P0 PASS as evidence for vision claim #2 outside this project until at least C1 + H3 are addressed.** Internally, the PASS is fine as a "the basic plumbing works" milestone; it is not yet a substrate-claim measurement.
