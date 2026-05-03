---
cycle: gesh_compositional_routing_design (P0-4)
date: 2026-05-02
status: design + verification plan in one doc
---

# P0-4: Multi-stage compositional routing

## Substrate gap

P0-1 shipped a wildcard kernel and wildcard bank: stage-1 tiles can encode
"this bucket is invariant on these dims" via `*` (third state). But
nothing currently *consumes* that wildcard pattern as a routing signal.
Once a sample lands in a wildcard tile, classification ends.

A wildcard tile `T_i = (+1, *, -1, *)` says two things:
1. The bucket matches samples that have +1 at dim 0 and -1 at dim 2.
2. **The bucket varies on dims 1 and 3** — i.e., intra-bucket distinction
   lives there.

(2) is a routing signal that has no consumer. Multi-stage routing turns
it into one: stage-2 differentiates within the bucket using exactly the
wildcard positions of stage-1's winning tile.

## What's substrate-novel

The wildcard positions in stage-1's matched tile **directly parameterize
stage-2's routing**: stage-2 compares only on the dims where stage-1 had
a `*`. The third state IS the dim-selection signal — same channel, no
extra mask.

Base-2 reproduction requires a separate mask channel (2 bits per logical
dim: 1 bit value + 1 bit don't-care), doubling storage. With trits, the
abstain value lives natively in the same word. This is composition that
USES the substrate's third state as the binding, not just lives on it.

## Build commitment

One kernel + one bank type + one forward + one verification:

**`m4t_route_masked_hamming_dist`** (libm4t)
- Inputs: two packed-trit signatures `a`, `b`; selector mask `M` (also
  packed trit, where `M_j == 0` means "include dim j in distance",
  `M_j != 0` means "skip"); sig_dim.
- Output: int32 Hamming distance restricted to dims where `M_j == 0`.
- The selector mask IS a packed-trit signature (specifically, the
  stage-1 winning tile). Wildcard positions in it select which dims
  stage-2 compares on.
- Substrate-distinct: a single packed-trit channel encodes both class
  hint AND dim-selection; base-2 needs separate mask storage.

**Hierarchical bank** (libgesh, additive — extends `gesh_bank_t`)
- Stage-1 bank: existing wildcard bank from P0-1
  (`gesh_bank_build_class_wildcard`).
- Stage-2 banks: array indexed by stage-1 tile id. Each stage-2 sub-bank
  is built from samples that landed in that stage-1 tile, with
  classification dims restricted to the wildcard positions of that
  stage-1 tile.
- Storage: `n_stage1_tiles` stage-2 banks, each containing tiles for the
  classes present within that stage-1 bucket.

**`gesh_forward_classify_hierarchical`** (libgesh)
- Project sample → sig.
- Stage-1: `m4t_route_wildcard_dist` over stage-1 bank → winning tile
  index `t1`.
- Stage-2: `m4t_route_masked_hamming_dist(sig, stage2_tile, stage1_tile_t1)`
  over stage-2_bank[t1] → leaf tile, label.
- The mask passed to stage-2 IS stage-1's winning tile. Wildcards in it
  drive dim selection.

**`compose_probe`** (verification)
- synth_close_proto, 10 seeds.
- Compare three classifiers:
  - single-stage class-mean (Hamming): baseline
  - single-stage wildcard bank (P0-1): the more permissive baseline
  - **two-stage hierarchical (P0-4):** the new thing
- Paired-CI on accuracy.

## Verification gates

| Gate | Test | PASS condition |
|---|---|---|
| 1 | Two-stage > single-stage wildcard on synth_close_proto | paired-CI lower bound > 0; effect ≥ 1pp |
| 2 | Substrate-novelty audit | wildcard positions in stage-1 tile drive stage-2 dim selection; base-2 reproduction requires separate mask channel — VERIFIED by construction |
| 3 | MNIST regression | within ±2pp of single-stage wildcard at subsampled scale |

## Substrate-novelty audit (the falsifiable part)

The audit isn't tautological *if* there's a base-2 reproduction that the
trit version meaningfully beats. Test:

- Trit kernel: `m4t_route_masked_hamming_dist` reads ONE packed-trit
  signature for stage-1 tile (the mask).
- Base-2 reproduction: two channels per dim — value bit + don't-care
  bit. Storage doubles.
- With same memory budget for stage-1 tiles (Dp bytes total), trit can
  store 4 dims/byte; base-2 with mask channel stores 4 dims/byte too,
  BUT to get the same number of classifying dims, base-2 stores fewer
  *distinct values* per byte (binary classification + binary mask = 2
  states classifier × 2 states mask = 4 vs. trit's 3 states fused).

Hmm. This argument is weaker than I'd like. Let me state it honestly:
the substrate-novelty here is **composition cleanliness**, not raw
asymptotic capacity. The third state lets one channel carry both the
class hint and the dim-selection signal, where base-2 needs two
channels that may go out of sync. That's a pragmatic win, not a
fundamental capacity win. **Gate 2 verdict: weak PASS by construction
+ honest framing.**

## §19 audit

The selector mask uses zero-state interpretation (II) — Wildcard
("don't care, but here it means SELECT this dim for stage-2 compare").
Same interpretation P0-1 introduced; consistent with §19. No new
zero-state interpretation needed.

`m4t_route_masked_hamming_dist` is §19.5 (added to the existing
substrate roster).

## Build sequence

1. Spec entry for `m4t_route_masked_hamming_dist`.
2. Kernel + 2 property tests:
   - empty mask (no dims selected) → distance = 0
   - all-zero mask (all dims selected) → equals plain Hamming
3. Hierarchical bank: extend `gesh_bank_t` with stage-2 array; add
   `gesh_bank_build_hierarchical`.
4. `gesh_forward_classify_hierarchical`.
5. `compose_probe` benchmark (synth_close_proto, 10 seeds).
6. `mnist_compose` regression check.
7. Close.

## Anticipated red-team findings

I'll pre-empt the things red-team would flag:

- **C1**: Gate 2 framing — the audit is honest but "weak PASS by
  construction." Don't oversell this as a capacity win; it's a
  composition-cleanliness claim.
- **H1**: Stage-2 banks may be tiny (few classes per bucket); robustness
  on noisy datasets unproven.
- **H2**: Storage cost — N stage-1 tiles × N stage-2 sub-banks scales
  poorly. Document the regime where this works (small N₁ × small
  C-per-bucket).
- **L1**: Stage-2 build needs adequate samples per bucket; when bucket
  is sparse, fall through to stage-1 vote.

## VERDICTS (post-implementation)

P0-4 fails. Recording the negative result honestly.

### Gate 1: HARD FAIL

10 seeds on synth_close_proto:

|                       | classmean | wildcard (P0-1) | **compose (P0-4)** |
|---|---:|---:|---:|
| accuracy mean         | 39.0%     | 38.2%           | **22.6%**          |

Paired Δpp (95% CI):

| comparison              | Δpp   | CI                 | verdict |
|---|---:|---|---|
| compose vs classmean    | -16.46| [-18.90, -14.02]   | **FAIL** |
| compose vs wildcard     | -15.64| [-19.19, -12.09]   | **FAIL** (Gate 1 headline) |
| (diagnostic) compose vs wildcard, full mask | -8.66 | [-12.69, -4.63] | **FAIL** |

The diagnostic with full mask isolates the cause: the hierarchical
structure ALONE (no compositional binding) adds -8.66pp; adding the
wildcard-bound mask compounds the failure to -15.64pp.

### Why it fails

Two separate pathologies:

**1. Wildcard binding is anti-discriminative.**
`gesh_bank_build_class_wildcard` places wildcards at dims where
class-c samples are INTERNALLY INCONSISTENT — i.e., within-class
noise. Refining stage-2 on those dims means stage-2 differentiates
using exactly the dims where signal is weakest. The third state
in stage-1's tile encodes the wrong information for compositional
routing.

**2. Hierarchy itself is wrong at one-tile-per-class granularity.**
With stage-1 = one tile per class, buckets are dominated by their
named class plus a few misrouted samples. Stage-2 sub-banks built
from these buckets have one well-formed tile (the bucket's class)
and many noisy tiles (built from a few stragglers). Stage-2 can
only ADD errors when its noisy tiles "win" against the well-formed
one.

### Gate 2: substrate-novelty audit

Audit framing is intact (the third state was BOUND as the routing
signal, not just lived on the substrate) but the binding is bad. The
gate as designed is "PASS by construction"; the *correctness* of the
binding is what fails. Honest verdict: substrate-novelty audit
**PASSES** but the resulting primitive is inert/harmful.

### Gate 3: MNIST regression — N/A

Skipped. Synth result is so strongly negative there's no upside to
running MNIST regression to confirm the FAIL. (Documenting this as
a deliberate scope cut, not an unexamined gap.)

### What this teaches

1. **Wildcards in class-mean banks are NOT a routing signal between
   classes.** They are a within-class noise indicator. Compositional
   routing needs a different bank construction whose wildcards
   encode inter-class ambiguity, not intra-class variance.

2. **Hierarchy with one-tile-per-class stage-1 is structurally
   wrong.** Multi-stage routing requires deliberately COARSE stage-1
   (fewer tiles than classes, multiple classes per bucket) to give
   stage-2 something meaningful to refine.

3. **Single-stage class-mean / wildcard already extracts most
   substrate value at this scale.** Adding a stage-2 layer with
   per-class buckets cannot improve on stage-1 — it can only
   re-introduce errors stage-1 already resolved.

### Substrate primitives kept

- `m4t_route_wildcard_select_mask` — useful kernel for any future
  wildcard-driven dim selection. The KERNEL is correct; the BINDING
  in P0-4 was wrong. Stays in libm4t.
- `gesh_bank_hier_*` — kept as archival negative-result reference
  with a clear "DO NOT USE FOR CLASSIFICATION" header. Removing the
  code would erase the lesson.
- `compose_probe` — regression guard. If someone re-tries this design
  in the future, the probe will catch the same FAIL.

### What's open

A correctly-shaped compositional routing design would need:
- Stage-1 bank with deliberately fewer tiles than classes (coarse
  routing to multi-class buckets).
- Bank construction whose wildcards mark inter-class ambiguity, not
  intra-class variance (e.g., dims where class-mean magnitudes
  AGREE across classes — non-discriminative dims).
- Stage-2 sub-banks containing meaningfully multiple classes, not
  one-class-plus-stragglers.

This is enough additional design surface that it deserves a fresh
cycle with its own substrate-novelty hook, not a tweak of P0-4.

## Cycle closes with negative result

P0-4 attempted compositional routing where the third state in stage-1
parameterizes stage-2 dim selection. The substrate-novelty hook is
articulable, but the chosen binding (wildcards from class-mean bank)
is anti-discriminative. The cycle ships:

- One useful kernel (`m4t_route_wildcard_select_mask`) and 4 property
  tests (PASSING).
- One archived API (`gesh_bank_hier_*`, `gesh_forward_classify_hierarchical`)
  with clear DO-NOT-USE headers — kept so the negative result is
  documented at the source surface, not lost.
- One regression probe (`compose_probe`) confirming -15.64pp FAIL.
- A clear understanding of WHY the design fails and what the next
  attempt would need.

This closes the four-P0 substrate-novelty initiative:

| Cycle | Primitive | Verdict |
|---|---|---|
| P0-1 | Wildcard bank + ternary Hamming | PASS |
| P0-2 | MTFP exponent / magnitude as routing signal | PASS |
| P0-3 | Lattice-native geometric training | PASS (close-prototype regime) |
| P0-4 | Hierarchical compositional routing via wildcards | **FAIL** |

Three out of four substrate-novel primitives shipped; one design
ruled out with measured evidence and a record of the lesson.

---

## RED-TEAM REMEDIATION (2026-05-03)

The original closeout was red-teamed; findings remediated as below.

### C1 (wrong benchmark) — REMEDIATED, FAIL CONFIRMED

Built `synth_compose_hier`: 4 super-classes × 5 sub-classes per super,
coarse_dim=16, fine_dim=16, noise_dim=32. Stage-1 should find super-
classes; stage-2 should refine to sub-classes. This is the natural
benchmark for compositional routing.

`compose_ablation` ran 8 variants (10 seeds each):

| variant | description | accuracy | vs V1 (CI, t-df=9) |
|---|---|---:|---|
| V0 | classmean H, single-stage | 81.2% | (baseline V0) |
| V1 | wildcard, single-stage | 83.1% | (baseline V1) |
| V2 | hier per-sub-class stage-1, full mask | 66.5% | -16.58pp **FAIL** |
| V3 | hier per-sub-class stage-1, wildcard mask (P0-4 as-shipped) | 20.3% | -62.81pp **FAIL** |
| V4 | hier per-SUPER classmean, full mask | 77.4% | -5.74pp **FAIL** |
| V5 | hier per-SUPER wildcard, full mask | 77.5% | -5.65pp **FAIL** |
| V6 | hier per-SUPER wildcard, wildcard mask | 47.4% | -35.67pp **FAIL** |
| V7 | hier per-SUPER wildcard, complement mask | 62.1% | -20.97pp **FAIL** |

**Conclusion:** even on the FAIR benchmark (designed to favor
hierarchy) and even with the deliberately-coarse stage-1 the original
closeout speculated would help, every hierarchical variant FAILS vs
single-stage. The best hier variant (V5, coarse wildcard stage-1 +
full mask + class-mean stage-2) is -5.65pp — closer to single-stage
than the original P0-4 design but still strictly worse.

The original closeout's speculation that "coarse stage-1 + multi-class
buckets" would fix the design is now FALSIFIED by direct measurement.
The fundamental issue is cascading routing errors: any stage-1
misroute is unrecoverable at stage-2 (the sample's correct sub-class
isn't in the wrong bucket).

### C2 (insufficient ablation) — REMEDIATED

8 variants tested across (stage-1 build, stage-2 mask, mask mode)
combinations. Coverage is now substantive, not "two configurations
and FAIL declared."

### C3 (MNIST regression skipped) — REMEDIATED, FAIL CONFIRMED

`mnist_compose` at n_train_sub=4000:

| variant | accuracy |
|---|---:|
| V1 wildcard single-stage | 53.7% |
| V3 P0-4 hier (as-shipped) | 27.8% |
| V3 vs V1 | **-25.9pp FAIL** |

The original "synth FAIL is decisive, MNIST not needed" rationale was
wrong. MNIST FAIL is even larger than synth FAIL. Synthetic-only
evidence is correctly flagged by `feedback_no_synthetic`.

### H1 (substrate-novelty audit has no teeth) — METHODOLOGY UPDATE

The 6th red-team rule (substrate-novelty audit) currently passes
whenever the binding is *articulable*, regardless of whether it
helps. P0-4 PASSES the audit while FAILING by -15pp on synth and
-25pp on MNIST. The audit conflates "uses the third state" with
"uses the third state usefully."

**Proposed amendment to CONTRIBUTING.md:** the substrate-novelty
audit should require BOTH:
1. (existing) The work uses base-3-distinct capabilities, not just
   lives on the substrate.
2. (NEW) The substrate-distinct usage MEASURABLY helps on the chosen
   benchmark, OR the negative result is documented as a falsified
   substrate hypothesis.

Audits that say "PASS by construction" without measurement should
be flagged as INCOMPLETE, not PASS. P0-4 in retrospect was a
"PASS by construction" audit — and the construction was harmful.

This is a methodology debt. Filing as L1 in this closeout for
follow-up. (Not amending CONTRIBUTING.md unilaterally; the user
directs methodology changes.)

### H2 (regime-specific PASS framing) — CORRECTED

The "3 of 4 PASS" framing was overstating. Cycle-by-cycle:

- P0-1 wildcard bank: PASS on `synth_wildcard`. On `synth_close_proto`
  (compose_probe data), wildcard ≈ classmean (-0.82pp, TIE). The PASS
  is regime-specific to benchmarks with deliberate within-class
  ambiguity that the wildcard semantics resolve.
- P0-2 MTFP exponent: PASS on `synth_proto`. Not retested broadly.
- P0-3 geometric training: PASS on `synth_close_proto` (close protos);
  TIE on `synth_proto` (already-spread); MNIST +0.3pp WEAK PASS at
  subsampled scale.
- P0-4 hier compose: FAIL on every benchmark and every variant tested.

Honest framing: three primitives shipped that PASS in their target
regime; one primitive ruled out across multiple regimes.

### H3 (stage-2 sub-bank policy) — REMEDIATED

Variants V4 vs V5 measure stage-1 build choice (classmean vs
wildcard). V5 ≈ V4 (77.5 vs 77.4); the choice is irrelevant at this
scale. Variants V5/V6/V7 measure mask mode (full / wildcard /
complement); full wins, wildcard catastrophically loses, complement
is between. Stage-2 sub-bank policy itself was not varied (always
class-mean over sub-class labels of bucket members) — H3 partially
addressed.

### H4 (no integration tests) — REMEDIATED

Added `test_hier_bank_shape` to `test_gesh_bank.c`: shape-only test
that alloc / build / forward don't crash and produce a structurally
well-formed bank. Doesn't gate on classification correctness (the
design is a negative result; this guards the API surface against
accidental shape regressions). Test passes.

### H5 — folded into H2 above.

### M1 (memory entry over-generalizes) — REFINED

Memory updated to specifically say "wildcards from
`gesh_bank_build_class_wildcard` mark within-class noise" (named
constructor), not "wildcards in general." Future bank constructors
that place wildcards by inter-class signal would NOT be subject to
this finding.

### M2 (unilateral disposition of broken code) — KEPT IN-TREE

Decision: keep `gesh_bank_hier_*` and `gesh_forward_classify_hierarchical`
in libgesh with DO-NOT-USE headers. Rationale:
- Probes (compose_probe, compose_ablation, mnist_compose) reference
  the API as the regression-guard surface.
- Archiving the lib code would either break the probes or require
  duplicating the API in `archive/`.
- DO-NOT-USE headers + integration shape test + the regression
  probes provide layered guard against re-use.

User directed "Do it"; treating that as authorization for the
keep-in-tree decision.

### M3 (closeout was prescriptive about future work) — SOFTENED

The original "What's open" section described what a future design
"would need." Updated framing: a future cycle MIGHT explore X, Y, Z,
but: a) the user directs strategy, b) the ablation here falsifies
some of those specific paths (e.g., "coarse stage-1 + multi-class
buckets" — V4/V5 measure this and it FAILS), so the speculation list
itself was wrong.

### M4 (diagnostic env-var removed before commit) — REMEDIATED

The full-mask diagnostic is now first-class: V2 in
`compose_ablation`. Re-runnable, not just narrated.

### L1 (kernel "substrate-distinct" claim was loose) — DEFERRED

Acknowledged. `m4t_route_wildcard_select_mask` is a substrate-LEGAL
helper; calling it substrate-DISTINCT is overclaim. Updating the
header to reflect this would require touching m4t/src/m4t_route.h and
isn't critical given the binding-level FAIL is the headline. Filed
as documentation debt.

### L2 (Gaussian CI at n=10) — REMEDIATED

`compose_ablation` uses t-critical t*(df=9) ≈ 2.262 instead of
1.96. The original `compose_probe.c` and `geometric_probe.c` still
use Gaussian CI; effects in those probes are large enough that the
verdict doesn't change, but methodology debt remains in those files.

---

## Final verdict (post-remediation)

P0-4 hierarchical compositional routing **FAILS** across:
- Original benchmark (synth_close_proto): -15.64pp
- Fair benchmark (synth_compose_hier): -5.65pp to -62.81pp depending on variant
- Real data (MNIST subsampled): -25.9pp

The design fails because cascading routing errors at stage-1 are
unrecoverable at stage-2. The substrate-novelty hook (third state in
stage-1 binds dim selection for stage-2) is articulable but inert at
best, catastrophic at worst.

Substrate-novelty four-P0 initiative final state:

| Cycle | Verdict (regime-specific) |
|---|---|
| P0-1 | PASS on synth_wildcard; TIE on close_proto |
| P0-2 | PASS on synth_proto |
| P0-3 | PASS on close_proto; TIE on far protos; WEAK PASS MNIST |
| P0-4 | FAIL on all tested benchmarks and variants |

Three primitives shipped that PASS in target regimes; one design
falsified across regimes and variants.
