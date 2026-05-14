# Path Forward — LMM (Quickstart, end of 2026-05-14)

**Method:** Lincoln Manifold canonical (`anjaustin/lmm`), Quickstart
30-min discipline. Single file; four phases inline.

**Scope:** With today's arc closed (per-prompt routing reframe →
qsig_filter K=1 integration win → workload sensitivity → Trit
Lattice LSH FFN synthetic validation), what's next?

---

## Phase 1: RAW (8 min)

We have more open path-forward options than at any previous point
in this project, and that itself feels like a sign we should be
careful about how we choose. The qsig_filter arc closed cleanly: a
deployable per-workload recommendation backed by stat-sig data
(tech-heavy +9.21pp, CI excludes 0). The LSH FFN synthetic gave a
strong yes (in/cross collision ratio 25,600x at k=8, 99.5% bucket
purity). Both arcs point to genuinely new architectural territory;
neither is forced.

What concerns me is the temptation to keep adding wins to the pile
without consolidating any of them into something durable. Tripp
asked twice today for the path forward, both times via reframes
that opened new arcs. The reframes were correct in each case but
the SUM of unfinished arcs is now: meta-routing closeout (sealed),
qsig_filter integration (deployable but not deployed), LSH FFN
(synthetic-validated, no real-data check yet, no harness integration),
and four un-tested integrative architectures still on the menu
(consensus, handoff, soft eviction, multiplicative).

What scares me: the substrate-vision foundation (ternary, routed,
non-dense, six primitives) is still where Glyph's value lives. All
of today's wins are at the application/research layer ON TOP of
that foundation. None of them changed the substrate. If I keep
chasing application wins, the substrate's known gaps (per-tensor
bx, A8 recipe alignment, training pipeline) get older without
attention.

What I keep avoiding: the question of what Glyph IS, on a
six-month horizon. Today's qsig_filter win is real but the gain is
measured against `no_evict` match-rate at window=16 — an internal
metric. To translate to "noticeably better generated text" is a
separate gap (coherence-vs-bit-parity from memory). The LSH FFN
prototype, if it works, would be similarly internal-metric for a
while.

The naive approach now: pick the most exciting thread (LSH FFN,
because the synthetic ratio was huge), and dive. The probable
problem with that: if real activations don't cluster as the
synthetic suggested, we burn a few days. Cheap real-activation
check first is the right move; I had this in the previous response
but rushed past it.

Open questions:
1. Should we operationalize qsig_filter (workload-aware K config)
   or move on to the next arc?
2. Does LSH FFN work on REAL BitNet activations the way it does on
   synthetic clusters?
3. Are we accumulating arcs because each one IS the right move, or
   because I'm pattern-matching "what's interesting" instead of
   "what's load-bearing"?
4. What's Glyph's six-month direction — substrate maturation,
   research arcs, or product applications?
5. Does the day's discipline (journal+LMM after each finding) scale
   if we open 5 simultaneous arcs?

---

## Phase 2: NODES (7 min)

Applying the Laundry Method: partition path-forward into buckets
first; the delta lives at the boundaries.

### Bucket A: Consolidate today's wins

**Node A1 — qsig_filter K=1 deployment.** The +9.21pp tech-heavy
win is shippable. Operationalizing means: env-var docs, workload-
detection heuristic (or explicit deploy-time flag), entry in
README/CHANGELOG. ~1-2 hr.

**Node A2 — qsig_filter validation expansion.** N=200 prompts
(another +100 prompts) would tighten CI from sqrt-noise. ~22 min
harness × 1 mode (the K=1) + qsigdist re-run. Diminishing returns
unless we suspect the workload-sensitivity finding is fragile.

**Tension A1↔A2:** A1 ships at single-seed N=100; A2 hardens
the evidence first. If we ship and the +9.21pp doesn't survive
in deployment, that's an embarrassing reverse. If we expand and
the finding tightens, we ship a stronger story.

### Bucket B: Continue LSH FFN arc

**Node B1 — Real-activation clustering analysis (P1).**
Instrument harness to dump FFN inputs from prompts; run synthetic
prototype's analysis on real data. Tells us if real activations
cluster as synthetic suggested. ~2 hr.

**Node B2 — LSH FFN drop-in prototype (P2).** Replace BitNet's
dense FFN with hash→tile→matmul. Initialize tiles from dense by
routing prompts. Measure quality vs dense baseline. ~5-6 hr.

**Tension B1↔B2:** B1 is the cheap validation that should precede
B2 (per "validate input before mechanism" memory). B2 alone risks
building on an unvalidated assumption.

### Bucket C: Other integrative architectures

**Node C1 — Multi-policy consensus.** Each of {qsigdist, sigdist,
fifo, K=1} proposes top-K victims; vote. ~2 hr harness mod +
battery.

**Node C2 — Conditional handoff.** Route by signal-stat (e.g., Q
variance across slots): high → qsigdist; uniform → sigdist or
fifo. ~3 hr.

**Tension C↔A,B:** Other integrations might independently improve
OR stack with K=1. But pursuing them WITHOUT first validating
LSH FFN's mechanism (B1) means we don't learn whether the
integration framing transfers across layers/problems. C tests the
SAME problem (KV eviction); B tests a DIFFERENT layer (FFN).

### Bucket D: Substrate maturation

**Node D1 — Per-tensor bx tracking.** Phase 2 wu2 priority. Closes
the ACT_BX/FFN_BX/GATE_ACT_BX sweep dependency.

**Node D2 — A8 recipe alignment with bitnet.cpp.** Closes the 1.3%
trit mismatch.

**Node D3 — Training pipeline.** Big lift; would unlock "Glyph as
learning substrate" claim.

**Tension D↔A,B,C:** Substrate work has long-term leverage but
slower rate of visible wins. Today's research arcs all run on top
of unchanged substrate.

### Bucket E: Different problem entirely

**Node E1 — Embedded inference port.** Reflex repo proves it works
at scale; Glyph could too. Concrete deliverable.

**Node E2 — NSW retrieval system.** Build the NSW-as-learning
prototype I described. Tests the architecture I claimed transfers.

**Node E3 — Trit-routing applied to math expressions (claim 2).**
Already partly built; could be productized.

### Delta (Laundry Method — boundaries between buckets)

**Δ1 — Where qsig_filter meets LSH FFN.** Both are integrative
architectures using signature-based routing. Validating LSH FFN on
real activations would IMPLICITLY validate the assumption that
qsig_filter's K-K-similarity protection works because real K
signatures cluster meaningfully. **B1 also validates A.**

**Δ2 — Where LSH FFN meets substrate maturation.** LSH FFN as
described uses ONLY existing primitives. But to be production-
quality it would need bucket-tile data structures, append-only
update primitives — exactly the gap Glyph's substrate has on
"sequence-level data structures built on the primitives" (per
NSW discussion). **B2 reveals what substrate primitives are
missing.**

**Δ3 — Where consolidation meets new arcs.** Today's discipline
(journal+LMM after each finding) was sustainable for ONE serial
arc. With 5 candidate threads, it doesn't scale. The delta: when
do we STOP iterating and commit to a vector?

---

## Phase 3: REFLECT (8 min)

### Core insight

**Today's "discoveries" are all instantiations of one underlying
move: replace LEARNED routing with NATIVE-GEOMETRY routing.**
qsig_filter integrates K-K-similarity STRUCTURALLY (filter, not
learned weight). LSH FFN integrates input-signature STRUCTURALLY
(hash, not learned router). Both win where competition fails by
sidestepping the routing-confidence problem entirely.

This is the substrate vision in action ("base-3 IS the graph,"
"math as signatures via routing"). Today proved the move works on
two different applications. The "path forward" question reduces
to: which application's win generalizes most, and to what?

### Resolved tensions

**Tension A1↔A2 resolved:** Ship qsig_filter as a documented
config option AND run the N=200 expansion. They're independent;
the doc work is short, the expansion runs unattended. Both can be
done in parallel without architectural conflict.

**Tension B1↔B2 resolved:** B1 (real-activation analysis) MUST
precede B2 (prototype). The synthetic was strong but the
"validate input before mechanism" memory has a tax of TWO journals
of mechanism on gibberish-prompt data. P1 first.

**Tension C↔A,B partly resolved:** C tests SAME problem, B tests
DIFFERENT layer. B has higher generalization payoff per hour. C
is interesting but lower priority.

**Tension D↔A,B,C resolved by re-reading scope:** Today's wins
sit ON TOP of unchanged substrate. The substrate is solid enough
to support more research arcs without immediate maturation work.
Defer D until either (a) a research arc surfaces a substrate gap,
or (b) we declare "research arcs done, ship phase begins."

**Tension E↔others:** Different problems are good if they validate
the architectural pattern (NSW retrieval would test "native-
geometry routing" outside KV/FFN). Embedded port is product work
that could come later. Math expressions arc was already partly
shipped; could be productized but isn't on critical path.

### Hidden assumptions

- **Assumption:** Tripp wants to keep pushing forward research
  arcs vs consolidating. **Challenge:** They asked for path
  forward but might also be ready to ship and pivot to product
  work. Should ask, not assume.

- **Assumption:** The qsig_filter K=1 finding will hold up in
  deployment. **Challenge:** It's tested on N=100 single-seed at
  one batch size; CI on the headline still spans 0. The
  workload-stratified +9.21pp on tech is the claim that survives.

- **Assumption:** LSH FFN's synthetic strength predicts real-data
  strength. **Challenge:** The "validate input before mechanism"
  memory says check this BEFORE building.

- **Assumption:** "Native-geometry routing" is a generalizable
  insight. **Challenge:** Two data points (qsig_filter, LSH
  synthetic). Strong but n=2.

- **Assumption:** Five open arcs is unsustainable; need to commit.
  **Challenge:** Maybe. But explicit task scheduling could let two
  arcs run in parallel while one consolidates. The discipline cost
  isn't unbounded.

### What I now understand

The path forward isn't a single ranked next-step; it's a SET of
moves with clear scoping. The Laundry-Method delta (Δ1, Δ2)
reveals dependencies that change priorities:

- B1 (real-activation analysis) is the highest-leverage cheap move
  because it validates BOTH the LSH FFN arc AND implicitly the
  qsig_filter mechanism (K-K signatures cluster meaningfully).
- A1 (qsig_filter ship-doc) is parallelizable trivial work.
- A2 (N=200 expansion) is parallelizable harness time.
- C and E are explicitly NOT priority until B1's outcome is known.

---

## Phase 4: SYNTHESIZE (7 min)

### Architecture

Two parallel tracks for next session:

1. **Validation track (B1):** Real-activation clustering analysis.
   Instrument BitNet harness to dump FFN inputs from N=20-50
   diverse prompts. Run the synthetic prototype's measurement
   protocol on real activations. Outputs:
   - Per-layer cluster purity at k∈{6, 8, 10}
   - Bucket utilization distribution
   - Comparison to synthetic baseline

2. **Consolidation track (A1):** qsig_filter ship-doc. Update:
   - `gesh/bitnet/bitnet_harness.c` — add header comment with
     workload guidance
   - `CHANGELOG.md` — entry for the new mode + workload
     recommendation
   - Memory: save "Trit Lattice integration > Routing" as a
     transferable feedback memory

### Key decisions

1. **B1 before B2.** Real-activation validation precedes prototype
   per "validate input before mechanism" memory.

2. **A2 deferred.** N=200 expansion would tighten CI but isn't
   load-bearing for the workload-stratified claim. Run only if a
   downstream consumer needs the headline-mean tightened.

3. **C and E explicitly deferred.** Other integrative architectures
   and different-problem arcs are open menu items. Don't open them
   until B1 either confirms the LSH path (then prioritize B2) or
   refutes it (then C becomes more interesting).

4. **D held.** Substrate maturation is a deliberate hold. Reopen
   when a research arc surfaces a substrate gap, or when the
   ship-phase begins.

5. **Recursive Manifold trigger:** If B1's result is mixed (real
   activations cluster but with different sweet-spot k, or with
   skewed bucket distribution), run a sub-LMM on the LSH FFN
   architecture choices.

### Implementation spec for next session

**Step 1 — A1 (consolidation, ~1 hr):**
- Edit `bitnet_harness.c` header comment for `BITNET_KV_EVICT_QSIG_FILTER`
  to include workload recommendation table.
- Add CHANGELOG entry: "qsig_filter K=1 mode for technical workloads
  (+9.21pp on tech-heavy, CI [+1.97, +18.42])."
- Save memory: "Native-geometry routing > learned routing for
  Glyph substrate (qsig_filter and LSH synthetic both validate)."

**Step 2 — B1 (validation, ~2 hr):**
- Modify `bitnet_harness.c` to dump FFN-input activations
  (post-attention, pre-FFN) with env-var
  `BITNET_DUMP_FFN_INPUTS=path/to/dir`.
- Run on N=20 prompts; collect per-layer activation tensors.
- Adapt `experiments/phase_eta/lsh_ffn_synth.py` to operate on
  real activations: threshold-extract → trit signature → bucket.
- Measure: bucket purity (using prompt-level labels OR k-means
  pseudo-clusters), utilization, in/cross collision rates.
- Output: `experiments/phase_eta/lsh_ffn_real_activations.json`
  + brief journal entry.

**Step 3 — Decision branch:**
- **If B1 confirms** (purity ≥ 0.8, in/cross ratio ≥ 100x): proceed
  to B2 (LSH FFN drop-in prototype). Estimate 5-6 hr.
- **If B1 partially confirms** (purity 0.5-0.8 or skewed
  utilization): run sub-LMM on architecture choices (k, tile size,
  cold-bucket policy).
- **If B1 refutes** (purity < 0.5 or in/cross < 10x): pivot to C
  (multi-policy consensus on KV eviction) — the next integration
  test on the same problem.

### Success criteria

- [ ] qsig_filter K=1 documented as a deployable mode with
  workload guidance (A1).
- [ ] Real-activation clustering measurement complete on N=20
  prompts (B1).
- [ ] Decision branch executed: either B2 launched OR sub-LMM
  initiated OR pivot to C committed.
- [ ] No more than ONE new arc opened this session (forces
  consolidation discipline).
- [ ] Journal + LMM at session close, not after every finding
  (the per-finding LMM cadence overwhelmed today; once-per-
  session is the new rhythm to test).

### What we're NOT doing (explicit)

- Not running multi-seed L3 (doesn't help K=1 vs qsigdist CI).
- Not running N=200 expansion (deferred per A2).
- Not opening multi-policy consensus (C deferred).
- Not opening conditional handoff (C deferred).
- Not opening NSW retrieval prototype (E deferred).
- Not starting embedded port (E deferred).
- Not touching substrate maturation (D held).

The discipline is: VALIDATE the current arc before spawning more.

---

## Loop-back signals to watch

- **Back to RAW** if next-session work reveals a fifth/sixth arc
  worth opening AND we feel pressure to open it. The path forward
  was meant to be SCOPED, not extended.
- **Back to NODES** if B1's outcome doesn't cleanly fit any
  bucket (the delta gets bigger, not smaller).
- **Back to REFLECT** if the synthesis above feels arbitrary in
  retrospect — meaning the Phase 3 resolutions were premature.
- **New full cycle** if B1 + A1 surface a NEW core insight that
  reframes the day's work the way Tripp's two reframes did
  during the day.

---

*"Chop to see the dullness, map the grain, sharpen with reflection,
and the wood cuts itself."*

The day's wood was the meta-routing arc; the chop revealed three
reframes (per-prompt routing, integration vs competition, native-
geometry routing). The grain runs from substrate-vision down to
specific architectural moves. Tomorrow's clean cut: validate the
LSH FFN path on real data while shipping the qsig_filter result.
