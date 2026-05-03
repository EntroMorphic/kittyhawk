---
status: P0 — central directive 2026-05-02
authority: owner directive, full project priority
scope: restore substrate-claim novelty along four axes the work has neglected
supersedes: nothing — adds priority order over all in-flight cycles (gesh_kmeans_validation, substrate purification, etc.)
---

# Remediation Plan — Substrate-Novel Capabilities (P0)

## Diagnosis: how this happened

The post-2026-05-01 rebuild started with sound substrate primitives (Tier 1/2/3 kernels, cross-exponent accumulator, SDOT path, ternary matmul) and a stated substrate-claim ("base-3 routing-first matches base-2 attention"). What followed was technically clean but framing-narrow:

1. **The substrate-claim is parity-shaped, not advantage-shaped.** "Matches base-2" is the ceiling, not the floor. To exceed base-2, we have to use something base-2 doesn't have. The whole rebuild has been building infrastructure that demonstrates competence (forward, train, bank, sweep) without using the substrate's distinct capabilities.

2. **Methodology rules accumulated; substrate-novelty audit didn't.** The five red-team rules (multi-seed, multi-config, kernel-use, in-scope-kernel, scope-of-evidence) all catch *correctness drift*. None catch *capability drift* — "are we using what only this substrate can do?"

3. **Benchmark drift toward base-2 home turf.** MNIST is a 10-class continuous-image task. The synthetic prototype benchmark has uniform-random noise dims. Both are base-2-friendly. We've been measuring on tasks where the best the substrate can do is *match* what base-2 does well. The substrate-claim's strongest evidence would come from a task where ternary structure is *natively* advantageous — and we never built one.

4. **LMM cycles converged on "next: more measurement."** Each cycle's SYNTHESIZE committed to the next measurement. The cumulative trajectory was *engineering refinement*, not substrate-design exploration. The cycles' REFLECT phases never asked "is the work using the substrate's distinct capabilities?"

5. **The original GESH design's three Gs got collapsed.** The design closeout reduced "Three Gs (Global, Geometric, Gradient)" to "two mechanisms (Global, Lattice-Geometric)" then dropped "Global" as deferred. We never built Global. We never built genuinely lattice-Geometric (we built classification-error training, which is label-Geometric). What we have is the *Gradient* G in disguise (lattice update is gradient descent on a discrete surface; the move from STE to "the lattice IS the geometry" was supposed to fix this, but the loss function we minimized stayed label-shaped).

The same drift led to the prior cycle's archive. The archive's failure was different in detail — too many engineering directions, not enough convergence — but identical in shape: building things that ran rather than building things that *demonstrated the substrate*.

This plan exists to interrupt the drift by hard-prioritizing four substrate-novel capabilities. The four are P0. Everything else (purification, validation cycles, MNIST tuning) is paused until each P0 has at minimum a design cycle complete.

## The four P0 capabilities, in priority order

The order is **dependency-ordered**, not impact-ordered: each P0 builds on the prior. P0-3 (lattice-native geometry) requires P0-1 and P0-2 to be substrate primitives before it can be a loss function. P0-4 (multi-stage routing) requires P0-1, P0-2, P0-3 because each stage's behavior depends on those primitives.

### P0-1: Structural zero as a first-class routing signal

**The substrate gap.** Ternary's three states are {-1, 0, +1}. Currently 0 is treated as "default" or "tie" by every consumer:

- `m4t_route_threshold_extract` produces 0 when `|v| ≤ tau` — but downstream consumers treat the 0 as just another trit.
- `m4t_popcount_dist` already weights "this position has a 0 vs a ±1" at half the cost of "0 vs the opposite ±1" (per ternary Hamming) — that's correct, but it's a measurement consequence, not a routing semantics.
- `m4t_route_apply_signed` skips decision.sign==0 (sentinel) but doesn't propagate signature-zeros into a sparsity decision for the next stage.

Base-2 has no third state. Every position is informative. To approximate "uninformative" base-2 must use masks, gating, or learned dropout. **The substrate's structural zero is a free signal that base-2 has to spend computation to fake.** We've never used this advantage.

**What "fixed" looks like.** At least three substrate primitives that *use zero as a sparsity signal* in operational distinction from ±1:

1. **`m4t_route_apply_signed_sparse`** — same shape as `m4t_route_apply_signed` but skips not just sentinel decisions but also ones with `sign==0` (already the case) AND propagates "this dim is uninformative" through to the next-stage's input by writing the structural zero to the next stage's signature.

2. **`m4t_route_zero_alignment`** — measures *how many positions two signatures agree as 0*. Distinct from Hamming distance: two signatures (-1, 0, +1) and (+1, 0, -1) have full Hamming dissimilarity on positions 0 and 2 but agree-as-zero at position 1. The agree-as-zero count is a "shared uninformativeness" signal that bank construction or routing decisions can use.

3. **`m4t_mtfp_ternary_matmul_bt_skip_zeros`** — a matmul variant that detects zero rows in the projected output and *propagates the zero forward without computing the threshold step*. Currently sign-extract operates on every output; with this variant, output dims that the matmul will threshold to 0 (because |acc| ≤ tau) skip the next stage's computation entirely.

**Required design work.** One LMM cycle (`gesh_zero_signal_design`) that:
- RAW: dump every place the codebase currently treats zero as "default" or "tie."
- NODES: name the operational distinctions zero should make (sparsity, alignment, propagation, gating).
- REFLECT: pressure-test against base-2 — what does base-2 use to approximate this? Cost? Are we measuring the *right* base-2 baseline (sparse attention with masks, not dense)?
- SYNTHESIZE: commit to which of (1)/(2)/(3) above (or all) gets built first, with measurement gates.

**Required code work.**
- Three new substrate primitives (or fewer, if SYNTHESIZE selects).
- Tests: each primitive needs a property test demonstrating zero-state operational distinction, not just bit-equivalence to a reference.
- Consumer integration: at least one consumer (gesh_forward, a new consumer, or a benchmark) USES the zero signal to do something operationally distinct.

**Verification.** A measurement that demonstrates substrate advantage:

- A task where ternary signatures with structural zeros achieve **the same classification accuracy at lower compute cost** than dense base-2 alternatives. The compute cost difference must be attributable to zero-skipping, not to other architectural choices.
- Or: a task where the zero-alignment signal carries information that ±1-only routing can't recover. Demonstrate by ablation (turn off zero-alignment → accuracy drops; turn on → recovers).

**Anti-pattern.** Building a primitive named "skip zeros" that runs the same number of ops as the dense version (no actual sparsity benefit). The "advantage" must be measurable, not just nominal.

### P0-2: MTFP exponent as a routing signal

**The substrate gap.** MTFP19 is mantissa + per-block exponent. The mantissa carries data; the block exponent carries scale. Currently:

- Every matmul outputs MTFP19 mantissas at a *single global scale* (M4T_MTFP_SCALE = 3^10).
- We never compute or track per-block exponents in routing.
- The block exponent's role in the substrate spec (§7) is "sidecar metadata" — currently sidecar to nothing.

Consequences:
- Routing decisions are sign-only. A query whose projection accumulator is +5 and another whose accumulator is +500 produce identical ternary signatures (both +1) and identical routing. The substrate has the *resolution* to distinguish them; we discard it.
- Bank construction collapses to ternary signs. A class with strong per-dim signal and a class with weak per-dim signal produce equivalently-sized tile signatures.
- No primitive emits "confidence" alongside "decision."

Base-2 attention has continuous activations carrying magnitude. Base-2 has to convert magnitude to softmax probabilities to use it for routing. **The substrate has magnitude as a free per-block exponent that base-2 doesn't have at this granularity.** We've never used it.

**What "fixed" looks like.** Two substrate primitives plus one consumer integration:

1. **`m4t_threshold_extract_with_exponent`** — output is `(trit, exp_class)` where `exp_class ∈ {weak, mid, strong}` based on |acc| / global_scale. Five-state effective output: {strong-, weak-, zero, weak+, strong+}. Or richer if the exponent has more bits.

2. **`m4t_route_apply_confidence_weighted`** — accepts (decision, confidence) pairs and emits a magnitude-aware output. When two queries vote for different tiles, the higher-confidence query's vote dominates.

3. **A bank constructor** that stores per-class *signature + confidence vector* (per-dim exp_class). Forward classification uses both signature Hamming and confidence agreement.

**Required design work.** One LMM cycle (`gesh_exponent_signal_design`) that:
- RAW: enumerate every place magnitude is *measurable but discarded* in current code.
- NODES: distinguish "magnitude as a measurement output" from "magnitude as a routing input."
- REFLECT: pressure-test substrate-claim. The mantissa-exponent split is base-3-native (block exponents are powers of 3, mantissas are base-3 trit-counts). Base-2 has the same separation conceptually but at much coarser granularity (float32 is one global scale). Is the substrate's per-block exponent operationally distinct?
- SYNTHESIZE: choose the primitive set to build first.

**Required code work.**
- New threshold-extract variant (3-bit or 4-bit output per cell).
- New routing primitives.
- New bank constructor.
- Tests: confidence-weighted routing must demonstrate that high-confidence queries route differently than low-confidence ones in cases where ternary-only routing collapses them.

**Verification.** A measurement showing that magnitude information beyond sign improves routing in a way base-2's quantize-to-trit-then-route can't (the comparison: base-2 mantissas → ternary quantize → forward-pass vs. our base-3 mantissas → exponent-aware routing).

**Anti-pattern.** Adding "confidence" as a multiplier on the existing scalar voting. The confidence has to alter the routing topology, not just the vote magnitude.

### P0-3: Lattice-native geometry as the training objective

**The substrate gap.** `gesh_design_closeout.md` declared: *"the lattice IS the geometry; lattice update walks it directly."* What we then built minimizes **per-batch classification error** — a label-shaped loss using the lattice as a search space. That's gradient descent on a discrete grid; it's not lattice geometry.

Genuinely lattice-native training would optimize a *geometric* property of the trit lattice:

- **Inter-class Hamming margin.** Maximize the minimum pairwise Hamming distance between class-mean signatures. The lattice's natural distance metric.
- **Within-class signature compactness.** Minimize the Hamming variance of within-class samples around their centroid. Manifold-shaped: classes should be local in lattice space.
- **Ratio (Hamming Fisher).** within-class variance / between-class variance — the discrete analog of LDA. Lattice-native.
- **Polytope volume.** Maximize the volume (in trit-count) of the polytope spanned by class-mean vertices. Geometric.
- **Topology preservation.** Trit-lattice neighbors of training samples should approximately respect class membership.

None of these need a held-out validation set, none need per-batch sampling, none care about classification error directly. They are *geometric* objectives over the bank, not *predictive* objectives over labels.

**Consequence of fixing this.** The training signal becomes substrate-native:

- No more "training hurts at high T" pathology — that came from per-batch classification noise interacting with k-means refresh. A geometric loss is dataset-wide, not batch-wide; refreshes don't move the target.
- The lattice update's flips become flips toward geometric optima, not toward batch-dependent label fits. Single-flip evaluations cost more (whole-bank Hamming margin compute) but the signal is much higher quality.
- Substrate-claim narrative regains coherence: "the lattice IS the geometry" matches what training optimizes.

**What "fixed" looks like.** At least one new training mode + at least one new measurement:

1. **`gesh_train_lattice_margin`** — trains R via lattice update on a margin-maximization objective. No labels-as-loss; only "does this flip increase the minimum inter-class Hamming distance?" computed over the current bank. Bank refresh is part of the cycle but the loss never references the labels except via the bank's class-grouping.

2. **A measurement comparing classification-error training vs margin training.** Same R init, same data, same budget, same bank type. Hypothesis: margin-trained R generalizes as well or better, AND no longer shows the high-T inversion.

3. **A measurement of the geometric metric itself across training.** Track Hamming margin over training; verify it monotonically increases with margin training (sanity check that the optimization is doing what its name claims).

**Required design work.** One LMM cycle (`gesh_geometric_training_design`) that:
- RAW: enumerate the geometric properties of the trit lattice that could serve as training objectives.
- NODES: distinguish "geometric metrics" (computable from bank only) from "geometric losses" (defining a gradient direction for R-flips).
- REFLECT: pressure-test against the existing classification-error loss. What measurements would show the geometric loss is genuinely substrate-native vs. just a different loss function shape?
- SYNTHESIZE: pick the geometric loss to build first; pre-commit measurement gates.

**Required code work.**
- Geometric loss kernel (e.g., min-pairwise Hamming, computed via existing kernels).
- Lattice update variant that optimizes against this loss.
- Comparison harness against classification-error training.

**Verification.** Two measurements:

- Margin-trained R + same bank type produces *higher* test accuracy than classification-error-trained R, AND/OR the margin-trained run does not show the training-hurts-at-high-T inversion that classification-error training shows. (Either condition would demonstrate the substrate-native loss is preferable; both is the strongest evidence.)
- The geometric metric (Hamming margin) is monotonically improved by the training. If it is, "the lattice IS the geometry" is no longer just a slogan; it's a demonstrated invariant.

**Anti-pattern.** Building a "geometric loss" that's secretly just classification error in different notation. Test by inspection: does the loss reference any sample's *label* directly (other than via bank grouping)? If yes, it's classification-shaped. If no — only references signatures and bank tiles — it's geometric.

### P0-4: Multi-stage compositional routing

**The substrate gap.** Single-stage Gesh: x → R·x → top-k → vote → label. One projection, one bank, one routing decision per query. The original GESH design had multi-stage routing (the "Global" G in three-Gs framing), which got dropped from Phase A as deferred. **It was never built.**

Multi-stage routing is what gives the substrate compositional power. Each stage emits a ternary signature; subsequent stages route on the *signature*, not on the original input. The substrate's structure (ternary trits as a geometric primitive) means that signatures *are* substrate-native objects — they can be consumed by the same kernels that consume original inputs. Base-2's continuous activations don't have this discrete, stable, geometric form.

Possible multi-stage architectures:

- **Hierarchical bank.** Stage 1 routes to a *coarse bucket* (one of T1 broad regions). Stage 2 routes to a fine prototype within that bucket (one of T2). Total bank: T1 × T2 logical tiles, but only T2 are computed per query (the rest are skipped — substrate-native sparsity, see P0-1).
- **Multi-table parallel.** Multiple R matrices route the same input through independent ternary sub-spaces; each emits a signature; aggregate via vote or signature concatenation. (The archive's `mnist_routed_bucket_multi M=32 SUM` is this shape — but built without the substrate-native primitives we're now committing to.)
- **Sequential refinement.** Stage 1 produces a coarse signature; stage 2 *modifies* the signature based on stage 1's confidence (P0-2 dependency). High-confidence stage-1 outputs route immediately; low-confidence ones get a stage-2 refinement pass.
- **Multi-scale self-routing.** Stage 1 emits a signature; that signature is the input to stage 2's R, producing a coarser signature; recursion. The signatures form a hierarchy that the bank tile structure should mirror.

**What "fixed" looks like.** A multi-stage consumer that:

1. Demonstrably uses signatures from a prior stage as substrate-native inputs to a subsequent stage.
2. Achieves accuracy that single-stage cannot reach at comparable parameter count.
3. Exhibits *compositional* properties — e.g., changing R2 while holding R1 fixed shifts performance in predictable ways consistent with hierarchy/composition.

**Required design work.** One LMM cycle (`gesh_multi_stage_design`) that:
- RAW: enumerate the multi-stage architectures the substrate naturally supports.
- NODES: separate them by "what each stage's input/output is" and by their compositional properties.
- REFLECT: pressure-test against multi-table parallel (the archive's path). What does sequential give that parallel can't? What does hierarchical give that flat doesn't?
- SYNTHESIZE: pick one architecture; commit to measurement gates that demonstrate composition (not just "more parameters helps").

**Required code work.**
- Stage-2 routing primitive.
- Multi-stage bank constructor.
- Forward pipeline that propagates ternary signatures through stages.
- Training pipeline that handles multi-stage R updates (per stage, jointly, or alternating — design choice).

**Verification.** Two measurements:

- A multi-stage consumer reaches accuracy at least competitive with the archive's `mnist_routed_bucket_multi M=32 SUM` (97.24%) on MNIST.
- Compositional ablation: holding stage 1 fixed and varying stage 2 shows accuracy patterns that single-stage parametrization cannot reproduce.

**Anti-pattern.** Building "multi-stage" as just "two consecutive matmuls without anything between them" — that's parameter scaling, not composition. The hallmark of composition is that earlier stages' *decisions* are inputs to later stages, not just earlier stages' *activations*.

## Sequencing & dependencies

P0-1 → P0-2 → P0-3 → P0-4. Each builds on the prior:

- P0-2 (exponent signal) needs the threshold-extract semantics of P0-1 to interoperate. Building P0-2 first would produce primitives that don't respect the zero-state operational distinction we add in P0-1; we'd have to redo them.
- P0-3 (geometric loss) operates on signatures emitted by primitives from P0-1 + P0-2. A geometric loss without zero-state and exponent-state semantics is a degraded measurement of geometry.
- P0-4 (multi-stage) requires all of P0-1, P0-2, P0-3 to be substrate-native or stages just compose impoverished signals.

**Each P0 follows the same protocol:**

1. **LMM design cycle** (RAW → NODES → REFLECT → SYNTHESIZE → CLOSEOUT). No code until SYNTHESIZE commits.
2. **Substrate-spec amendment** (M4T_SUBSTRATE.md update for new primitives) per principle 7.
3. **Kernel implementation + property tests** with substrate-novelty audit (the new test category, see below).
4. **Consumer integration** (gesh side or new bench tool).
5. **Verification measurement** with the pre-committed gates.
6. **Closeout** updating the project narrative with the substrate-native capability now in evidence.
7. **Red-team pass** before declaring the P0 complete.

Don't proceed to P0-(N+1) until P0-N's verification gate has produced a verdict.

## The substrate-novelty audit (new test category)

Every red-team going forward includes a substrate-novelty check, alongside the existing five rules:

> **Substrate-novelty audit.** Does this work USE the substrate's distinct capabilities, or just live ON the substrate? For every new measurement, every new claim, every new primitive: name what only-base-3-can-do property is being exercised. If the work would produce identical results on a base-2 substrate (with appropriate quantization), it's not substrate-claim work — it's correctness work, which is fine but should be labeled as such, not framed as substrate-claim evidence.

Promote to `CONTRIBUTING.md` as the sixth red-team rule alongside the prior five (multi-seed, multi-config, kernel-use, in-scope-kernel, scope-of-evidence). This is the rule that catches the drift this plan is correcting.

## Pause-and-reset protocol

If at any point during P0 execution, the substrate-novelty audit catches drift back into "doing classifier work that doesn't use substrate distinct capabilities":

1. **Stop the in-progress P0 immediately.** Don't push code that fails the audit.
2. **Open a journal entry** (`journal/p0_drift_<date>.md`) recording what drifted and why the audit caught it.
3. **Return to the P0's design cycle** — the SYNTHESIZE wasn't tight enough; revise.
4. **Resume only after the design cycle's SYNTHESIZE explicitly addresses the drift mode.**

This is the protocol that would have caught the drift this remediation plan is correcting, had it existed earlier.

## What's paused

Until each P0 has at least its design cycle complete:

- `gesh_kmeans_validation` cycle (W1/W2/W3) — paused. The k-means measurements are correctness-level; substrate-novelty is the gate now.
- Substrate purification (task #14) — paused. The 100%-on-substrate cleanup remains the right work, but its priority drops below substrate-novelty work; the existing scalar paths don't break correctness, they just defer perfection.
- Phase B Path A (multi-table LSH consumer) — paused. Building "richer consumer" that just emulates the archive's M=32 SUM doesn't advance the substrate-claim. After P0-4's multi-stage design, multi-table can resume IF it serves the multi-stage architecture.
- All MNIST tuning (sig_dim sweeps, k_per_class sweeps, top_k variants, training budget tuning) — paused. These are tuning-shaped work; the substrate-claim isn't a tuning problem.

## What's still active

- The substrate kernels we have (m4t_*, gesh_project, gesh_bank class-mean) are correct and stay; new primitives layer on top.
- The methodology meta-rules (multi-seed, multi-config, etc.) apply to P0 work too; P0 doesn't escape them.
- The journal/CHANGELOG discipline continues; P0 cycles are documented like any other LMM cycle.
- The bit-equivalence and substrate-discipline cleanup direction is preserved as a quality bar; P0 work follows it.

## Verification of the plan itself

The plan succeeds if, after all four P0s land:

- The substrate's claim has shifted from "matches base-2 attention" to "uses zero/exponent/lattice/composition in ways base-2 can't, demonstrating qualitative advantage on at least one task."
- The novel capabilities have measurements, not just primitives — substrate-claim evidence, not just substrate-claim infrastructure.
- The next benchmark target is chosen to *exercise* the substrate-native capabilities, not to *match* base-2 on its home turf.

The plan fails if:

- Each P0 ships primitives but no measurement demonstrates substrate advantage.
- The work moves the codebase toward "more capable classifier" without moving the substrate-claim narrative.
- The drift recurs (substrate-novelty audit catches the same pattern again post-P0).

## What this plan does not commit to

- A specific timeline. P0 work is substantive and deserves the time it deserves.
- A specific benchmark beyond MNIST as a regression-guard. Substrate-native benchmarks are partly P0-1's design work to identify.
- A specific architecture for multi-stage routing — that's P0-4's design cycle.
- A guarantee that all four P0s succeed at their verification gates. Some may surface mechanism gaps that require further design work.

This plan commits to *the work that has to happen before the substrate-claim has step-change potential* — not to the specific outcomes of that work.

## Appendix: how the four P0s map to the original three-Gs framing

The original GESH design had three Gs: Global, Geometric, Gradient. The Phase A closeout collapsed this to two (Global, Lattice-Geometric), then dropped Global as deferred.

| Original G | Phase A status | Restored by |
|---|---|---|
| Global (multi-scale routing) | Deferred, never built | **P0-4** (multi-stage routing) |
| Geometric (PCA + manifold-aware projections) | Replaced by lattice update | **P0-3** (lattice-native geometric loss) |
| Gradient (STE-based end-to-end) | Subsumed into lattice update | (Now subsumed via P0-3 properly) |

P0-1 (structural zero) and P0-2 (exponent signal) are *preconditions* the original design didn't explicitly name but that the substrate makes available. They're the "what does ternary uniquely give us at the kernel level" that the original Gs were structured around without naming.

The four P0s collectively restore the design's substrate-claim ambition.
