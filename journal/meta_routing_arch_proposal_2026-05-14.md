# Meta-routing as a three-layer architecture — prototype + empirical test

User proposal (conversation 2026-05-14): a three-layer routing
architecture where layer 1 is the substrate primitives, layer 2 is
compositions of those primitives (programs), and layer 3 is
meta-routing — a search over layer-2 programs evaluated against a
loss signal. The user framed layer 3 as "temporary scaffolding" that
enables learning without violating the six-primitives floor.

The claim being tested: layer 3 can DISCOVER layer-2 programs that
beat hand-coded baselines. Concretely, for the KV-eviction problem,
layer 3 should find a policy that beats qsigdist (+6.4pp over random,
the best hand-coded mode known as of `td28_phase_zeta_n100_closeout_*`).

## Prototype (`experiments/phase_zeta/meta_routing.py`)

Three concrete layers:

- **Layer 1**: trit primitives (multiply + saturating add) from
  `routing.py`.
- **Layer 2**: parameterized eviction policy with score function
    score(slot) = w_r·age + w_kk·KK_sim + w_qk·QK_sim
  Evict slot with LOWEST score. The 4 existing hand-coded policies
  map to fixed points in {-1, 0, +1}^3:
    fifo     = (-1,  0,  0)
    sigdist  = ( 0, +1,  0)
    qsigdist = ( 0,  0, +1)
    random   = ( 0,  0,  0)
- **Layer 3**: additive linear model fit from N=100-pinned anchors;
  predicts Δ vs random for all 27 ternary-weight combinations.

Fitted layer-3 coefficients:
  β_0 = 0.0,  β_r = +5.5,  β_kk = -7.0,  β_qk = +6.4  (all pp/unit).

Top non-anchor candidates predicted:

| rank | (w_r, w_kk, w_qk) | predicted Δ | trit-distance from anchor |
|---|---|---|---|
| 1 | (+1, -1, +1) | +18.8pp | 2 (aggressive extrapolation) |
| 2 | ( 0, -1, +1) | +13.3pp | 1 (closest non-anchor to qsigdist) |
| 3 | (+1, -1,  0) | +12.5pp | 2 |

## Pre-tests (prototype script)

- **A (synthetic recovery)**: ground-truth additive function with
  known coefficients. Layer 3 recovered β's exactly and predicted all
  27 points within 1e-9. **PASS.**
- **B (anchor rank-ordering)**: predicted anchor ordering matches
  observed (qsigdist > random > fifo > sigdist). **PASS.**
- **C (leave-one-out)**: diagnostic only — with 4 anchors × 4 params,
  no anchor is redundant; LOO predictions degenerate to ~0 along
  uncovered axes. This identifies the data-budget limit, not an
  architecture failure.

## Empirical test (this journal)

**Test:** run policy (0, -1, +1) — the least-extrapolative candidate
— across the same 100 prompts at window=16, gen=24. Compare observed
Δ vs random to the predicted +13.3pp.

**Mode implementation:** added `BITNET_KV_EVICT_META` to
`bitnet_harness.c` with env vars `BITNET_KV_EVICT_W_R`,
`BITNET_KV_EVICT_W_KK`, `BITNET_KV_EVICT_W_QK`. Smoke test verified:
- meta(0, 0, +1) reproduces qsigdist exactly (bit-identical tokens).
- meta(0, -1, +1) produces distinct outputs from qsigdist when
  eviction triggers (verified on a 24-token prompt).

**Semantic interpretation of the candidate:** score = -KK_sim + QK_sim.
Evict lowest = LOW KK_sim AND HIGH QK_sim = "slot similar to current
K (redundant) AND dissimilar to current Q (irrelevant to attention)."
A semantically defensible heuristic.

## Results — MISS

The architecture made a falsifiable prediction, and reality falsified
it. This is the right outcome epistemically; it just means the
specific layer-3 model (additive linear over trit weights) is too
simple for this problem.

```
Layer-3 prediction:           +13.3pp Δ vs random
Empirical (N=100, window=16):  -3.0pp Δ vs random   95% CI [-7.3, +1.4]
Error:                          16.3pp; direction wrong
```

Paired against qsigdist on the same prompts:
```
qsigdist Δ (same prompts):    +6.4pp   CI [+1.6, +11.3]
candidate − qsigdist:         -9.3pp   CI [-14.1, -4.8]   **SIGNIFICANT**
wins / ties / losses:         24 / 23 / 53 of 100
```

The candidate is **strictly worse** than qsigdist with high confidence
(paired CI excludes 0 by 4.8pp on the upper bound).

## What the miss tells us

**1. Additive linear is too weak.** The model assumed effects of
`w_kk`, `w_qk` decouple. They don't. Empirically, combining w_qk = +1
(qsigdist-like) with w_kk = -1 (anti-sigdist-like) doesn't sum the
contributions — it INTERFERES.

**2. Plausible mechanism for the interference (semantic guess, not
proven).** Under GQA (20 Q-heads ÷ 5 KV-heads), the current K and the
current Q are highly correlated direction-wise. Sigdist (-7pp) failed
because its K-K proxy doesn't approximate Q-K direction, but the K-K
similarity ITSELF is also indirectly correlated with Q-K similarity
when Q ≈ K_current. So "evict slots like current K" implicitly
evicts slots Q attends to — undoing qsigdist's signal. The additive
model can't see this coupling because it treats w_kk independently.

**3. The "ternary weights are tunable knobs" framing is too simple.**
The trit space has STRUCTURE — there are interactions between axes.
Layer 3 needs to model interactions OR sample anchors on the
interactions directly (more anchors).

## What this validates AND refutes

**Architecture is sound (validated):**
- Layer 3 produced a falsifiable prediction (good).
- The prediction was clearly testable in <2h (good).
- The test fired and told us something concrete (good).
- "Temporary scaffolding" framework still holds: we ran layer 3 once,
  got a wrong answer, learned something — layer 3 isn't a permanent
  dependency.

**Specific layer-3 model is wrong (refuted):**
- Additive linear over 4 anchors can't generalize to the 27-cell
  space along axes where interactions matter.
- The N=100 result is real signal; the prediction error of 16.3pp
  is well outside the random fluctuation in a 100-prompt CI.

**The eviction problem has structure the layer-3 model missed:**
- qsigdist remains the best known policy (+6.4pp, CI [+1.6, +11.3]).
- The 4 anchors don't isolate the interaction terms.

## Next steps

The honest path forward is one of:

**A. More anchors.** Sample 2-3 additional weight combinations
empirically as 5th-7th anchors, then fit a model with interaction
terms. Cost: ~50min per anchor on N=100. With 7 anchors and
quadratic-in-trits features (3 main + 3 pairwise = 6 free params),
we'd have 1 degree of freedom and can test for interactions.

**B. Sample a different family of layer-2 programs.** The score-based
policy might just be the wrong parameterization. E.g., a SOFT
combination (softmax over slots weighted by feature distances)
instead of hard argmax. Adds parameters but in a structure that may
match the actual response surface better.

**C. Constrain layer 3 to staying close to anchors.** The candidate
was 1 trit away from qsigdist; observed +6.4pp − 9.3pp = -2.9pp. The
linear model overestimated by 16pp because it extrapolates with no
prior. A Gaussian-process-style model with a kernel that respects
trit distance would refuse to predict so far from anchors.

**D. Accept the negative result and document.** qsigdist is the best
hand-coded policy. Meta-routing's first probe didn't beat it. The
architecture had a real chance and missed; the floor (six primitives)
is undisturbed.

I'd lean A — the architecture deserves a second chance with a richer
model, and the cost is bounded (a few harness runs). But it's a real
choice; refusing to keep iterating is also a valid call.

## What's in the codebase now (post-experiment)

The artifacts ship regardless of the outcome:
- `experiments/phase_zeta/meta_routing.py` — prototype with 3 layers
  and 3 tests (synthetic recovery PASS, anchor rank-order PASS, LOO
  diagnostic).
- `experiments/phase_zeta/meta_policy_battery.py` — empirical runner
  + analyzer.
- `gesh/bitnet/bitnet_harness.c` — new `BITNET_KV_EVICT_META` mode
  with parameterized weights. Existing modes still recoverable as
  fixed points in the weight space (smoke-tested: meta(0,0,+1) ≡
  qsigdist bit-identically).
- `experiments/phase_zeta/results/meta_policy_battery/` — N=100 raw
  trial data for meta(0,-1,+1) on the same prompt set.

Even if the architectural test is a miss, the parameterized harness
mode is itself useful — future meta-routing experiments can vary the
weights without touching C code.

## Files

- `experiments/phase_zeta/meta_routing.py` — prototype (layers 1, 2,
  3) + tests A, B, C.
- `experiments/phase_zeta/meta_policy_battery.py` — empirical runner
  + analyzer.
- `experiments/phase_zeta/results/meta_policy_battery/` — raw trial
  data.
- `gesh/bitnet/bitnet_harness.c` — added `BITNET_KV_EVICT_META` mode
  with `_W_R`, `_W_KK`, `_W_QK` env vars.
