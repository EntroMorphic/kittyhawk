# Phase α red-team — VALIDATED downgrades to MIXED, with one robust finding

This is the red-team pass on `td27_phase_alpha_synthesis_2026-05-12.md`
(commit `309fed0`). The original verdict was "VALIDATED 2/3 measures
clear." On adversarial inspection, **M3 doesn't actually pass any
meaningful test, the M2 fail is harder than originally stated, and
the M1 41% gap depends on unit-of-measure choices.** The honest
verdict is **MIXED (1/3 measures clear, with one robust finding and
two real fails).**

This journal does not retract the commit — it records the red-team
discoveries the original commit didn't acknowledge with enough force.
The corrections below should be the basis of any external citation
of this work.

## Critical finding 1 — M3 "pass" is an artifact, not a result

The original verdict said: "M3 PASS (degenerate)."

The numbers:

| representation | longest_bar | second_bar | p95 | n_merges |
|---|---|---|---|---|
| substrate | **1.000** | 1.000 | 0.000 | 2399 |
| B1_raw    | 90583 | 31463 | 208.48 | 2399 |
| B2_sign   | **1.000** | 1.000 | 0.000 | 2399 |
| B3_sign   | 5.000 | 1.000 | 0.000 | 2399 |

Substrate's longest persistence bar is **1.0**. B2's is **also 1.0**.
They are IDENTICAL on the metric the M3 verdict cares about.

The FROZEN verdict rule says: "substrate's longest β_0 persistence
bar > 2× B3 null threshold." B3's bar_p95 = 0, so 2× = 0, and any
positive number passes. Substrate's 1.0 > 0.0 = trivially true.

But the same test would say B2 also passes the M3 criterion (1.0 >
0.0). And B3 itself passes with 5.0 > 0.0. The test discriminates
**nothing** — it's "is your data non-trivially clustered, ever?" The
answer at integer-Hamming distances is always yes.

**Correct read: M3 does NOT distinguish substrate from B2.** They
are equal on the metric. **M3 should be marked FAIL, not PASS.**

This was caught by the spot-check-before-verdict discipline that has
already fired three times this session sequence. I didn't apply it
to my own verdict here.

## Critical finding 2 — M2 fails BOTH criteria, not just one

The original verdict said: "M2 FAIL — substrate kNN reciprocity LOWER
than B2's."

The FROZEN spec required TWO things to hold across ≥3 of 4 k values:
(a) substrate's reciprocity > B2's by ≥ 5 pp,
(b) substrate's degree Gini < B2's by ≥ 0.05.

Both directions fail, at every k:

| k  | recip_s | recip_B2 | gap     | gini_s | gini_B2 | drop    |
|----|---------|----------|---------|--------|---------|---------|
|  5 | 0.749   | 0.770    | −0.021  | 0.124  | 0.118   | −0.006  |
| 10 | 0.646   | 0.802    | −0.156  | 0.170  | 0.105   | −0.065  |
| 20 | 0.478   | 0.807    | −0.329  | 0.219  | 0.095   | −0.124  |
| 50 | 0.355   | 0.811    | −0.456  | 0.243  | 0.087   | −0.156  |

- **Reciprocity gap is negative** at every k (substrate LOWER), and
  the gap grows with k: substrate's kNN graph gets less reciprocal
  faster than B2's as k increases.
- **Gini drop is negative** at every k (substrate MORE hub-dominated,
  opposite of predicted). Gap grows with k too.

The substrate kNN graph is **systematically more asymmetric and more
hub-dominated than B2's** — not just at one k, but increasingly so
across the full range. This is a strong direction-of-effect signal.
The substrate's metric concentrates "nearness" on a few hubs and
loses reciprocity at scale.

**Implication for downstream sigdist / KV-eviction work:** the
substrate's kNN structure is LESS suited to symmetric-neighbor
operations than binary Hamming. Anyone planning to use substrate for
graph-based eviction or routing should treat the substrate as
asymmetric-by-default and not assume reciprocal-neighbor structure.

## Critical finding 3 — The M1 41% gap depends on unit choice

The M1 verdict rule is "absolute d̂ relative gap ≥ 20%." Under that
rule, M1 passes at 41%. But the rule itself is contestable:

- Substrate ambient D = 128 trits, d̂ = 78.9 → d̂/D = 0.617
- B2 ambient D = 203 bits, d̂ = 133.8 → d̂/D = 0.659
- **Normalized gap: 4.2 percentage points, not 41%.**

The 41% absolute gap is dimensionally correct under the FROZEN rule,
but the right intuitive read is: **substrate fills ~62% of its
ambient capacity, B2 fills ~66% — a modest difference, not a
dramatic one.**

Two ways to handle this:

1. **Stick with the FROZEN rule.** It was pre-registered before data
   was seen. The rule says ≥20% absolute gap, we got 41%. Pass.
   (This is the journal's current position.)

2. **Acknowledge the rule was wrong.** Equal-bits B2 was designed to
   match information capacity; comparing absolute d̂ across different
   ambient spaces re-introduces the ambient-dim asymmetry the
   equal-bits design tried to eliminate. The intellectually honest
   metric is d̂/D (or some other dim-normalized quantity).

I think (2) is the stronger position. The pre-reg was good
discipline but the d̂-comparison framing was a gap the pre-reg
didn't catch. M1 still has a real direction-of-effect (substrate IS
more compressed), but "41% gap" overclaims it.

## Critical finding 4 — M3 close regime contradicts pooled

Pooled M3 (claimed pass):
- substrate longest_bar = 1.0
- B2_sign longest_bar = 1.0

Per-layer M3 (N=80 per layer):
- Layer 0: substrate=6, B2=20 → substrate **shorter**, opposite direction
- Layer 14: substrate=6, B2=13 → substrate shorter
- Layer 29: substrate=4, B2=20 → substrate shorter

At every layer-stratified scale, substrate's longest persistence bar
is *shorter* than B2's. The pooled "tie" at 1.0 is the artifact of
integer-distance discretization at large N (most merges happen at
the same distance value, killing bar length resolution).

**The per-layer view says substrate has LESS topological clustering
than B2** — opposite of the spec's prediction. Even M3 fails in the
right reading.

## Critical finding 5 — Calibration was on uniform data; K-cache is not uniform

The Macocco estimator assumes **uniform local density on a d-dim
manifold**. The calibration used uniform random ternary vectors
embedded in higher D — by construction, locally uniform.

Real K-cache data has training-induced structure: 54% nonzero cells
under τ=5000, not 67% (uniform-ternary nonzero rate). The
distribution of distances is shaped by the model's learned
representations. The estimator is being applied **off-distribution**
relative to what it was validated on.

This doesn't automatically mean the K-cache d̂ is wrong, but it
means the calibration result ("within 1% on synthetic d ≥ 10")
doesn't transfer directly. The actual bias on K-cache could be
larger.

**Mitigation:** run calibration on **structured** synthetic data
(generated to match K-cache's marginal distribution, then projected
to a known-d manifold) before trusting the d̂ values numerically.
Not done; this is a methodology gap.

## Critical finding 6 — N=2400 from a single prompt setup

The ACTV2 dumps cover 8 (prompt, position) combos across **what is
effectively one or two prompts** (the "dump" and "multitoken"
prefixes). The substrate-distinctive claim needs to hold across
*diverse* prompts; we've shown it on a narrow input distribution.

This is testable by rerunning the harness with 10+ different
prompts and repeating Phase α. **Not done.** The current result is
"on this corpus, substrate showed lower M1 d̂ than equal-bits
binary." Extending to "on language model K-caches in general"
requires broader sampling.

## Critical finding 7 — τ = 5000 is hard-coded and not sensitivity-tested

The substrate signature uses `threshold_extract` with τ = 5000,
yielding 54% nonzero cells. This is one point in threshold space.
At τ = 10000, occupation would be lower (more zeros); at τ = 2000,
higher (more ±1). The d̂ estimate is sensitive to occupation rate.

**No sensitivity sweep was run.** The "VALIDATED" verdict applies at
τ=5000 only. The substrate-claim is robust if and only if the M1 gap
persists across a range of τ values. This is a one-day experiment
that wasn't done.

## Critical finding 8 — B2 random projection is a weak baseline

B2 is sign of a random Gaussian projection from R^128 → R^203. By
construction, this is **maximally uninformative** — a random hash of
the K vector. d̂_B2 ≈ 134 says "this random hash fills two-thirds of
its capacity."

The substrate-claim's interesting form is not "substrate beats
random binary hash" — almost any structured representation will. The
load-bearing comparison is **substrate vs a STRUCTURED binary
representation of the same data.** Candidates not tested:

- PCA to k dims, then sign-of-projection (preserves linear structure)
- Spherical LSH (preserves angular similarity)
- Substrate-but-binary: same threshold_extract but with τ chosen so
  every cell is ±1 (no zeros) — the "binary substrate" ablation

Without one of these, the substrate's distinctive claim reduces to
"better than random," which is a low bar.

## Honest revised verdict

| measure | original verdict | revised verdict | basis |
|---|---|---|---|
| M1 | PASS (41% gap) | **PASS (with caveats)** | direction is robust; magnitude overstated by unit-of-measure choice; gap remains ~4pp normalized |
| M2 | FAIL | **FAIL (harder)** | both criteria fail at every k; direction growing with k |
| M3 | PASS (degenerate) | **FAIL** | pooled tie with B2 + per-layer contradiction; original "pass" was a thresholding artifact |

**Revised count: 1/3 measures clear → MIXED, not VALIDATED.**

Per FROZEN spec: "MIXED iff 1 of 3 measures clears."

The substrate's distinctive claim has **one robust finding** (M1
direction: substrate has fewer effective dimensions than equal-bits
sign-only at the same information capacity, normalized gap ~4pp) and
**two real fails** (M2 graph is more asymmetric/hubbed; M3 has no
topological advantage and at some scales has less clustering).

## What's still load-bearing

The M1 direction is the substrate's first measurement-grounded
distinctive property. Specifically:

> **Substrate K-signatures at equal information capacity occupy a
> lower-dimensional manifold than random-projection binary
> signatures of the same K data.** Effect size: ~4 percentage points
> in d̂/D, CI-significant via bootstrap. Limited to:
> - The K-cache corpus from one prompt setup (N=2400)
> - τ=5000 substrate threshold (no sensitivity sweep)
> - B2 random-projection baseline (not structured binary)

That's a much narrower claim than "VALIDATED" implies, and it's the
claim the data actually supports.

## What does NOT survive the red-team

- The "41% gap" framing — it's a unit-of-measure artifact at this magnitude.
- The "VALIDATED 2/3" label — M3 is FAIL on honest reading.
- Any claim that substrate has *topologically distinctive* kNN
  structure — M2 and M3 (per-layer) both say no.
- Any claim that substrate provides *more reciprocal* or *less
  hubbed* kNN graphs — strongly contradicted at every k.

## Implications for downstream

The spline-explorations journal (`td27_spline_explorations_2026-05-12.md`)
deferred Round 2 (soft routing, bank interpolation, Nyström compression)
to Phase α's outcome. With the verdict revised to MIXED:

- **Soft routing (Idea C):** the manifold-structure premise is
  *partially* supported. The substrate has fewer effective dims at
  the same capacity (M1), so Nyström-style attention could compress
  by some factor. But the kNN graph is more hubbed (M2 fail) and
  topologically less clustered (M3 fail per-layer), so any
  graph-based routing operation will be more dominated by hubs and
  less stable than the binary baseline.
- **Bank interpolation (Idea D):** still speculative; M3's per-layer
  failure is a negative signal — substrate K-vectors are LESS
  cluster-like than binary at the per-layer scale.
- **Nyström compression (Idea E):** weakly supported by M1; the
  ~6.7× compression target is plausible but not guaranteed.

The downgrade from VALIDATED to MIXED matters for prioritization:
these are now "possible payoffs, with two predictions already
falsified," not "validated next steps."

## Discipline log

This is the **10th caught overclaim** of the session sequence —
caught by my own red-team after committing the original "VALIDATED"
verdict, but only after the user prompted me to red-team.

The lesson: **the pre-reg's verdict rule is itself a thing that
needs red-teaming.** I followed the FROZEN spec literally and got a
verdict, but the spec had gaps (M3's 2× B3.p95 rule is degenerate at
integer-Hamming distances; the d̂ absolute gap is unit-dependent at
mismatched ambient D). Pre-registration prevents one class of
overclaim (post-hoc rule-shopping) but doesn't prevent another
(pre-registering a rule with a hidden edge case).

The spot-check-before-verdict discipline (memory item
`feedback_spot_check_before_verdict.md`) caught this on the second
look. **I should run it on my own verdicts, not just on agents'
verdicts.**
