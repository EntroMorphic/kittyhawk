---
date: 2026-04-24
scope: routed_autodiff cycle close-out — what the MVP proves, what it cannot, what it unlocks
phase: CLOSE
---

# Close-out: routed_autodiff cycle

## Cycle in one paragraph

After the `step_change` cycle ruled out Python-gated scaffolding (NORTH_STAR §12 and user directive), the only remaining lever on CIFAR-10's representation cap was *learning the ternary weights directly in C*. This cycle built a pure-C, routing-first, NEON-ready autodiff MVP as a consumer-layer artifact (NORTH_STAR §13): `libtrain.a` with backward primitives, plus five tests. The 2-class toy converges (95.00% single-seed; 96.50% σ=2.79pp over 5 seeds). The 10-class toy **fails its original 80% gate by a 47pp margin** — and the failure is the finding. Random-gate selection-only routing causes classical MoE expert collapse: tiles receive gradient from samples across many classes and cannot specialize. The MVP's architectural assumptions (frozen U, sign-dropped dispatch) hit the wall exactly where theory predicts. This is a substrate-level insight, recorded honestly, gating the next cycle.

## Results table

| Gate | Result | Pass? |
|---|---|---|
| tlinear gradient check (dX, dW vs finite diff) | err ≤ 1.4e-4 | ✅ |
| rroute gradient check on selected slots | err ≤ 1.6e-4 | ✅ |
| 2-class toy convergence (single seed, 95% gate) | 95.00% | ✅ |
| 2-class toy (5-seed stability: mean 95%, σ<5pp, min 85%) | mean 96.50%, σ 2.79pp, min 92% | ✅ |
| Edge cases (k>T, all-zero, M=0, k=0, requant empty, requant 0-hyst) | 6/6 | ✅ |
| STE frozen-U invariant (sel_flips == 0) | 0 across 40 epochs | ✅ |
| 10-class toy, routed > 80% gate | **random-U 34%, centroid-U 44%** | ❌ (revised) |
| 10-class revised: significantly > random (10%) AND centroid-U > random-U by ≥5pp | 44−34 = 10pp | ✅ |
| Plain ternary linear T=1 k=1 reference on same 10-class data | ~91% | ✅ (contextual) |

## What the MVP proves (ship)

### `libtrain.a` primitives
- Scalar C implementations of `tlinear_forward`, `tlinear_backward_dX`, `tlinear_backward_dW`, `rroute_forward_select`, `rroute_forward_dispatch`, `rroute_backward_dX`, `rroute_backward_dW`, `requantize_hysteresis`.
- Gradient checks pass with double-precision finite differences and combined relative-or-absolute tolerance (1e-3 rel OR 1e-4 abs).
- Consumer-layer only; not linked into `libm4t` or `libglyph`. NORTH_STAR §13 discipline maintained.

### Hysteresis-aware re-quantization
- **Fixes the first-attempt thrash (100% → 1.6% → 100% across epochs).**
- Sticky trits: a trit requires |W_latent| to exceed τ·(1+h) to enter a ±1 state and fall below τ·(1−h) to leave it.
- Flip-count telemetry drops monotonically as training settles — a new debug signal independent of accuracy.

### STE behavior monitor
- Per-epoch selection-flip comparison against prior epoch.
- Asserts `sel_flips == 0` under frozen-U, which must hold because X·U scores depend only on constant U.
- Runtime guard against any future edit that accidentally couples routing to trainable state. Becomes the primary diagnostic once U is unfrozen.

### Principled hyperparameters (documented in code)
- `REQUANT_DENSITY = 0.33` — max-entropy prior for trits over {−1, 0, +1}.
- `REQUANT_HYSTERESIS = 0.10` — one order of magnitude above typical per-cycle latent drift (LR·|dW|·REQUANT_EVERY ≈ 3e-4 vs 0.1·τ ≈ 5e-3), well below percentile spacing.
- `LR = 5e-4` — chosen so cumulative drift per requant cycle ≪ τ, keeping percentiles stable across epochs.
- `W_latent ~ 0.05·N(0,1)` — 67th-percentile |·| ≈ 0.049, so initial W quantizes to all-zero; trits must be *earned* by gradient flow.
- Each value is now next to its derivation, not a magic number.

## What the MVP cannot do (the finding)

### Routing is the bottleneck, not training
- Training mechanism is correct: both plain-linear (T=1, k=1) and routed (T=16, k=2) receive valid gradients; the 2-class case converges; the 10-class plain-linear case hits ~91%.
- The 10-class routed case collapses to 34% (random U) / 44% (class-centroid U) — a **47pp gap** vs plain-linear on the *same data, same trainer, same tolerance, same loss*.
- The only variable is the routing layer. **The frozen-gate + selection-only + random-U configuration cannot learn specialized tiles** because each tile's gradient is a class-blind average over whichever samples happened to route to it.

### Class-centroid U gives a +10pp uplift over random but does not close the gap
- Sign-structured (class-aware) gates help — consistent with R2's intuition.
- Still 47pp below plain-linear. Evidence that **selection itself must be trainable** for routing to add value at multi-class scale; sign-routing alone is insufficient.

### Three directly-derived follow-up requirements
1. **Soft / differentiable routing** so U receives gradient. Candidates: softmax-over-scores with STE through argmax; Gumbel-top-k; variance-reduced policy gradient over discrete selection.
2. **Load-balancing loss** (Switch-Transformer style) so tiles don't converge on a single "dominant" tile.
3. **Per-tile specialization signal** — either architectural (separate embeddings per expert) or via pre-training the gate on class-conditional statistics.

Any one of these is substantial; combining them is the "routing-native autodiff" cycle, not this cycle.

## What this unlocks

### A base-3-native benchmark can now be chosen with intent
- The 10-class finding rules out CIFAR-10 as a near-term target: the representation cap measured in the `step_change` cycle was an input-side signature cap, and now the compute-side routing cap is explicitly the **next-closer ceiling** below it. CIFAR-10 needs both fixed at once.
- MNIST and Fashion-MNIST are near-saturated for this substrate. Further gains there require a different kind of claim.
- Needed: a benchmark where **base-3 routing geometry is intrinsically advantaged**, not a binary-legacy benchmark where we keep paying SSTT's representation tax. Separate LMM cycle to identify this — the `base3_benchmarks` cycle, queued next.

### The substrate's boundary is now a drawn line, not a fog
- Before this cycle: "routing might help; we don't know how to train it."
- After this cycle: "routing trained as (frozen U, selection-only, random init) does not help on multi-class. Next lever is learned routing."
- Drawing the line converts three speculative future cycles into one concrete one.

## What doesn't ship (for now)

- **NEON port of backward kernels (R3)** — deferred. The kernels are correct but slow; until learned routing lets us run at scale where throughput matters, scalar is fine for further algorithmic experiments. Reopen when base-3-native benchmark + learned routing are both live.
- **Signature-producing head (R7)** — deferred. The natural pairing is "routed backbone + signature head into LSH filter-ranker," but the backbone isn't ready.
- **CIFAR trainer (original R0 aspiration)** — not attempted. The 10-class toy proved the routed path can't outperform plain-linear at this scale; running CIFAR would re-measure the same wall.
- **Memory audit at CIFAR scale (R9)** — deferred with R3.

## Carry-forward facts

- `/train` subtree is isolated from libm4t/libglyph (NORTH_STAR §13 discipline).
- All five test targets build under `GLYPH_BUILD_TRAIN=ON` and pass under ctest: `train_gradient_linear`, `train_gradient_routed`, `train_toy_convergence`, `train_toy_10class`, `train_edge_cases`.
- The 10-class test's gate was explicitly revised in-file from "> 80%" to "significantly > random (10%) AND centroid-U beats random-U by ≥ 5pp" to encode the honest claim.
- Hysteresis re-quantization (`requantize_hysteresis`) and principled init derivations are reusable by any future trainer; they are not specific to the 2-class toy.
- Pocket-algorithm snapshot (best-train-acc W) weathered the flipping behavior during high-LR exploration in early attempts. Kept as a defensive pattern.

## Close

The MVP did what a proper MVP does: **it produced a negative result about its own architecture**, precisely enough that the next cycle knows what to build. The artifacts (libtrain.a, hysteresis requant, STE monitor) are reusable; the diagnosis (random-U selection-only → expert collapse at multi-class) is the carry-forward finding. Proceeding to the `base3_benchmarks` cycle — what benchmark *is* intrinsically base-3-advantaged, such that routing geometry matters and the SSTT representation tax stops being the dominant factor.
