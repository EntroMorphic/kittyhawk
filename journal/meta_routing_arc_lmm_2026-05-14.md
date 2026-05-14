---
cycle: meta-routing arc closeout LMM
phase: ALL (raw + nodes + reflect + synthesize)
date: 2026-05-14
scope: synthesize the meta-routing arc findings (iters 1–8 + δ + red-team
       + metric refinement) into a coherent picture and a principled
       next step. The architecture worked AS an epistemic mechanism;
       the empirical claim refuted; the response surface revealed
       sharp structure (death zone) and overfit traps (anisotropic
       refinement). What does this MEAN, and what should we do?
companions: meta_routing_arc_closeout_2026-05-14.md (closeout journal);
            meta_routing_arch_proposal_2026-05-14.md (initial proposal);
            experiments/phase_zeta/results/meta_iterate/anchors.json (13 anchors).
---

# Meta-routing arc — LMM cycle

## RAW

What I actually think, unfiltered:

The architecture works in a way I didn't fully appreciate until iter
8 α′ — the kernel + structural wall isn't just a "cleaner" predictor
than regression, it's a predictor with HONEST FAILURE MODES. When
the regression missed, it gave a confident, wrong, large-magnitude
number. When the kernel misses, the disagreement signal is loud and
the prediction was usually conservative — the architecture KNOWS it
extrapolated and the miss is signal, not catastrophe. That's a real
property of the structural separation, not a happy accident.

But I also have to admit: I was too confident about the anisotropic
refinement. The LOO improvement (4.52 → 3.59pp MAE) was real but
small, and I treated it as validation rather than as a hypothesis
that needed an out-of-sample test. Iter 8 α′ wasn't designed to be
that test — Tripp proposed running it, I designed it as confirmation,
and then it became the falsification. That's lucky; if Tripp hadn't
pushed for "α′ and δ in parallel," I'd have iterated under the
anisotropic kernel and quietly accumulated more retrofitting.

The death zone is the most interesting finding and I almost missed
it. Iter 4 was a 20.9pp miss under the regression; I attributed it
to "the regression model is bad" and moved on. The kernel rebuild
absorbed iter 4 as one outlier anchor; the kernel still couldn't
predict it from neighbors, but I read that as "the metric needs
work" rather than "there's a structural pattern here." Only when
iter δ confirmed (1,-1,0) at -13.67pp — 0.2pp from (0,-1,0) — did
the regional pattern crystallize. Two cells in the same region
agreeing within noise is a sharp signal; I needed Tripp's prompt to
run that probe.

The empirical claim ("Layer 3 finds a champion") was always weak.
The 4 original anchors were 4 of 27 cells, and the 3 anchored cells
above 0 (qsigdist, sigdist, fifo) covered the response surface
poorly. The regression model produced a confident wild prediction
(+13.3pp on (0,-1,1)), it was wrong, and that should have been my
cue that the program family + anchor budget combination was too
sparse to predict anything reliable. I kept iterating because Tripp
said "we are not here to rest" and I read that as "keep generating
predictions" instead of "keep learning about the space."

What I'm avoiding looking at:

- I never explicitly considered whether the program family was the
  right family. The score function `w_r·age + w_kk·KK + w_qk·QK` is
  ADDITIVE over three features, all of which are computed against
  the SAME reference (current K position). Multiplicative or
  conditional terms — e.g., "use KK_sim only if QK_sim is below
  threshold" — could carve the space differently. The death zone
  exists in part because there's no QK signal in w_qk=0 cells; a
  different parameterization might never have a "no QK signal at all"
  region.

- I never asked whether qsigdist itself is at the ceiling or whether
  a parameterization-richer policy could exceed it. The answer in
  this 27-cell family is no; in any family it's not known. We don't
  have a substrate-derived upper bound on KV eviction quality at
  window=16.

- The "structural wall" framing came from Tripp's article. I
  interpreted it as "no fitted parameters in the predictor." But it
  could be interpreted differently — e.g., "no information flow
  between successive predictions except through the anchor store."
  Under that reading, the LOO MAE sweep IS a violation: it touched
  the predictor's hyperparameters using accumulated data. The
  refutation of the anisotropic refinement is in some sense a
  vindication of a stricter reading of the principle.

- The Reflex repo Tripp shared validates a different scale of the
  same architectural commitments (ternary, no FP, on ESP32-C6).
  Their CMD 4/5 distillation methodology and NSW retrieval
  architecture have direct analogs to what we just built. I noted
  this earlier but didn't seriously consider whether to import any
  of their methodology — e.g., their three-seed-with-stddev
  discipline would have caught the (1,-1,1) "near zero" predictions
  having anomalously high noise.

- I'm using "n=100 prompts" as if it gave clean signal. The CIs on
  individual cells are wide (qsigdist Δ = +6.4 with 95% CI [+1.7,
  +11.2]). A 5pp HIT tolerance is roughly the half-width of those
  CIs. Several of my "HIT" calls (especially iter 6 at 3.0pp err)
  may be inside the noise floor of the measurement, not evidence
  of architectural calibration.

## NODES

Extracted tensions and constraints:

**N1 — Architecture vs. application.** The kernel + wall architecture
clearly works as an epistemic mechanism (mean error 4.3pp vs
regression 14.1pp on the same anchor budget). It also clearly
FAILED to find a policy beating qsigdist. These are different
questions; I conflated them throughout the arc.

**N2 — Calibration vs. measurement noise.** The kernel's HITs (iter
5: 0.8pp, iter 6: 3.0pp, iter 7: 5.1pp) are within the same
order as the per-cell measurement CI (~±5pp half-width). I can't
distinguish "the kernel architecture is calibrated" from "any
predictor within ±5pp would HIT-rate similarly" without paired
comparisons against alternative predictors.

**N3 — Sharp structure can't be predicted from L1-trit neighbors.**
The death zone (w_kk=-1, w_qk=0, two cells at -13.7pp ±0.2) is
sharp on the w_qk axis. No linear-distance metric over coordinates
captures this; the response surface has regions, not gradients.

**N4 — In-sample MAE is misleading.** The anisotropic sweep dropped
LOO MAE 1pp and was REFUTED on the next out-of-sample cell. The
"improvement" was the predictor learning to memorize the smooth
qsigdist-family gradient harder, at the cost of generalization
elsewhere.

**N5 — Disagreement signal vs. measurement noise.** The "structural
separation" principle was supposed to make the disagreement signal
informative. With ±5pp per-cell CI, a 3pp disagreement is barely
distinguishable from sampling noise. The principle's value is
clearest at LARGE disagreements (iter 1: 16.3pp, iter 4: 20.9pp),
where it correctly flagged "the predictor doesn't know this region."
At small disagreements, the signal isn't doing much.

**N6 — The program family was never justified.** I inherited
`score = w_r·age + w_kk·KK + w_qk·QK` from the proposal as
parameterized over the existing four hand-coded modes. I never
asked whether THIS family was the right family to search, or
whether a richer/different family would have a champion.

**N7 — Goalpost migration risk.** I notice I'm tempted to reframe
the arc as "architecture validated" — sliding from the original
"find a champion" claim toward "the architecture is a clean
predictor." That's a confirmation-bias pattern, same family as the
Phase 2 wu1.8 goalpost migration documented in lmm_post_redteam.md.
The original empirical claim was refuted; that should be the
headline, not a side note.

**N8 — Resource budget exhausted on this family.** Each iteration
costs ~22 min of wallclock and ~$0 of user attention. With 14 of 27
cells unanchored, fully mapping the family is another ~5 hours
wallclock. That budget is better spent on a different family (β:
add a feature) or a different problem (γ: ship and move on) than
on filling in the response surface I've already characterized
qualitatively (smooth qsigdist family + death zone + intermediate
diffuse).

## REFLECT

Structure, assumptions to challenge, leverage points:

**Structure.** Two distinct dimensions of "the meta-routing arc
worked" that I've been collapsing:

1. **Epistemic mechanism quality**: does Layer 3 produce calibrated,
   falsifiable predictions with honest failure modes?
   *Yes, after the rebuild* — 4/5 directionally correct under
   kernel+wall vs 2/4 under regression. Errors at the same order as
   measurement noise.

2. **Empirical search quality**: does Layer 3 find a champion in
   the chosen family?
   *No* — the family caps at qsigdist; no untested cell is predicted
   to exceed it; observations confirm.

These are different bars. Confusing them was my error. The
architecture is good even though the search failed; the family was
too small/wrong-shaped to contain a champion.

**Assumptions to challenge:**

1. **"The linear-score family is the right place to look."** It
   isn't, necessarily. Additive over three features computed against
   the same reference (current K) is one parameterization out of
   many. Conditional rules, multiplicative interactions, or richer
   features (slot age × attention mass, hit-count, etc.) might
   carve the space such that the death zone is unreachable and the
   ceiling is higher.

2. **"qsigdist is the upper bound."** It's the upper bound IN THIS
   FAMILY. It's not a theoretical bound on KV eviction quality at
   window=16. We have no upper-bound argument. A different family
   could exceed it; we just haven't tried.

3. **"5pp HIT tolerance is meaningful."** It's the half-width of
   per-cell measurement CIs, so HITs at 3pp may be noise. The
   "calibration" claim should be qualified: the kernel is
   calibrated *within measurement resolution*, not below it.

4. **"The architecture is application-agnostic."** It might be. But
   we tested it on ONE application (KV eviction in {-1,0,+1}³). One
   data point is weak evidence for a general claim. The architecture
   would deserve a different application's worth of validation
   before being framed as a generic principle.

5. **"Sweep-tuned anisotropic metric is in the spirit of the
   structural wall."** It isn't, in retrospect. The wall principle
   says prior holder and evidence reader are wired apart. Tuning
   the kernel's bandwidth using LOO over the anchor store *is*
   using accumulated evidence to modify the predictor's behavior —
   exactly the antipattern the wall is supposed to prevent. The
   refutation of the anisotropic refinement is partly a vindication
   of a stricter wall reading.

**Leverage points** (where small changes have large effects):

L1. **Different program family** (β). A 4th feature (slot
hit-count, attention mass over time, recency since last attended)
expands the search from 27 to 81 cells AND probably changes the
response-surface topology (death zone might not exist when there's
always SOME informative signal). This is the most interesting next
step. Cost: ~30 min harness modification + new iteration arc.

L2. **Stricter wall.** Forbid hyperparameter touches that read the
anchor store. The kernel's α and metric are FROZEN at architecture
time, not tuned. This forces the predictor to do worse on existing
data but generalize honestly to new data. The earlier anisotropic
overfit would have been prevented structurally.

L3. **Multi-seed measurement discipline** (from the Reflex repo's
methodology). Each iteration runs N=100 prompts with M=3 different
RNG seeds for the random-mode comparison baseline; the per-cell Δ
estimate is the mean ± std. This shrinks the per-cell CI and makes
the HIT tolerance meaningful.

L4. **Honest framing in the writeup**. The arc's substantive output
is (a) the structural separation principle empirically working at a
relevant scale, (b) the death zone observation, and (c) the
anisotropic-LOO overfitting trap. Frame these as findings; frame
the failed champion search as the test that produced them.

L5. **Stop iterating in this family.** Remaining cells are all
predicted either negative or near-mean by the kernel; no champion
hides in them. Mapping them is busywork.

## SYNTHESIZE

Concrete actionable output:

### What the meta-routing arc actually delivered

1. **A working three-layer architecture** (Layers 1–3 wired through
   the harness; CLI for inspection and iteration; append-only anchor
   store with provenance).

2. **A refuted strong claim and a validated weak claim.** Layer 3
   did NOT find a champion beating qsigdist in the linear-score
   family. Layer 3 DID produce calibrated predictions consistent
   with measurement resolution, with disagreement signals that
   correctly flagged uncertain regions.

3. **Two transferable findings:**
   - Structural separation (prior holder ≠ evidence reader, both
     literally distinct components) is a real engineering pattern
     that improves predictor honesty.
   - L1-trit distance over coordinates is not sufficient for
     response surfaces with sharp regimes (death zones).

4. **A trap documented for future arcs:** in-sample MAE improvement
   from a hyperparameter sweep is NOT metric validation; the
   anisotropic refinement was overfit. Saved as memory
   `feedback_in_sample_overfit_trap.md`.

### What β (next exploration) should do

1. **Enrich Layer 2 with a 4th feature**. Candidates: slot
   hit-count (how many times this slot won argmax-attention since
   it was written), recency-since-last-attended, attention-mass
   accumulator. Hit-count is the simplest to plumb through the
   harness and least likely to interact pathologically with
   existing modes.

2. **Re-anchor the existing fixed modes in the 81-cell space.**
   Their Δs don't change but their position does (the 4th weight
   is 0 for them). Anchor count starts at 4 instead of bootstrapped
   from prior arc — the 13 existing meta-mode anchors live in the
   3-feature subspace and CAN be projected, but their Δs were
   measured under no 4th-feature contribution; they may not be
   identical to running the 81-cell harness at the same trit
   coordinates with weight=0. Re-measure to be honest.

3. **Apply the lessons.** Frozen kernel bandwidth (no LOO sweep).
   Three-seed measurement discipline borrowed from the Reflex
   methodology. Explicit out-of-sample validation BEFORE claiming
   any refinement works.

### What NOT to do

- Map the remaining 14 cells of the 27-cell space. The kernel
  already says "no champion here" and the cost is ~5h for no new
  architectural insight.
- Tune the kernel's hyperparameters again on the existing data.
  Overfit trap; documented.
- Reframe the arc as "architecture validated." The strong
  empirical claim was refuted; that's the headline. The
  architecture insights are findings produced along the way.
- Run another iter under the anisotropic kernel.
- Treat measurement CIs as if they were smaller than they are.
  The per-cell N=100 has ±5pp half-width on Δ; HIT tolerance
  should be reported with that context.

### Honest framing for forward communication

The meta-routing arc tested a strong empirical claim (Layer 3
discovers a layer-2 program beating qsigdist) and refuted it. The
refutation produced two transferable findings: the structural
separation principle works empirically at the scale we tested,
and the response surface has sharp structure that ternary-coordinate
distance can't model. The arc's epistemic discipline (append-only
anchors, explicit disagreement signals, red-teamed harness)
performed as designed; the empirical bet didn't pay off in this
particular program family.

Forward: β (richer program family) is the natural next step. γ
(this arc as written-up) is the present output. The substrate's
six-primitives floor is untouched; META-mode is built on top of
the existing primitives via the harness.

## Status

LMM cycle complete. The meta-routing arc's actual capability and
limits are documented without overclaiming or underclaiming. The
forward plan distinguishes between (a) writing up what was just
done, (b) exploring the next program family, and (c) closing on the
substrate's six-primitives floor — none of which require further
iteration in the 27-cell family.
