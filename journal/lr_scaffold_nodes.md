---
date: 2026-04-21
scope: LMM cycle — would logistic regression benefit Glyph, and if so, where?
phase: NODES
---

# NODES

## Discrete ideas

1. **LR violates three Glyph invariants as classically framed.** Binary float weights, gradient-descent training, dense matmul at inference. Any LR integration has to dispose of all three — not just one.

2. **Pair-IG is already learned weights.** `direct_lsh::build_pair_ig` derives integer per-dim weights from class-conditional frequency statistics on the training set. If these count as "structurally meaningful" under the NO RANDOM WEIGHTS rule, then other data-derived integer weights (including LR-quantized) also count.

3. **Two integration slots, very different blast radius.** Slot A replaces pair-IG's entropy-derived `pw[d]` with LR-derived per-dim class-pair weights (drop-in, one function swap, same data flow). Slot B replaces the Hamming k-NN classifier with a direct LR head scored via SDOT over per-class weight vectors (architectural swing, exercises the unused SDOT kernel).

4. **SDOT has no current consumer.** `m4t_mtfp4_sdot_matmul_bt` was built for SDOT-native routing but no production consumer calls it. The profile shows it sustains 55–60 Gops/s — the fastest primitive on the substrate. Slot B would give it a first production consumer.

5. **Pair-IG's CIFAR-10 contribution is the entire production delta.** +1.95pp from 44.68% Hamming to 46.63% Selective. This means the re-ranking stage *is* doing work on CIFAR-10, so a better re-ranker (Slot A) could, in principle, squeeze more out — unless pair-IG is already at ceiling.

6. **The CIFAR-10 gap may be representational, not scoring.** `cifar10_nproj_ceiling.md` argues the 46% vs SSTT's ~53% gap is in the input representation (per-trit Hamming vs pattern-level block scoring), not the scorer. If true, Slot A hits a scoring ceiling quickly.

7. **Training cannot live in libm4t or libglyph.** The substrate is inference-only by contract (M4T_SUBSTRATE.md §12). Training would live in a one-shot consumer tool (e.g., `tools/lr_train.c`) or be offloaded entirely outside the repo (dump quantized weights as .c or .bin). The pair-IG LUT build is a precedent: startup float, runtime integer.

8. **Three training paths, three tradeoffs.** (a) External Python/numpy train → dump weights. Keeps C-only repo clean but breaks "no Python in toolchain." (b) C-side float gradient descent at startup → quantize → free float. Matches the `m4t_lut_gen.c` shape but adds float at consumer startup (the §12 fourth exception; precedent exists now with pair-IG). (c) Integer-only perceptron-style updates. Respects the strictest reading but loses some training accuracy.

9. **"NO RANDOM WEIGHTS" is the key interpretive question.** If the rule reads "no meaningless weights," pair-IG and LR-learned both pass (each weight has a derivation story). If it reads "each dim must have a structural prior meaning set by the designer," pair-IG already violates, so the rule is effectively already bent. The precedent matters.

10. **Base-2 LR adapted to ternary is scaffolding.** NORTH_STAR §4 sanctions this explicitly: "we may temporarily model some base-2-native ML systems in base-3 … This is scaffolding." So a quantized LR head is a legitimate scaffolding experiment — but §4 also warns "scaffolding must not become the building."

11. **There may be a base-3-native LR shape.** A classifier with per-class trit weights `W_c ∈ {-1, 0, +1}^D`, scored as `s_c = Σ_d W_c[d] × sig[d]`, is structurally base-3-native — it's exactly the SDOT kernel's shape. Trit weights carry sign-zero-sign trichotomy first-class. This is not "adapt LR"; this is "what does LR want to be when the substrate is base-3."

12. **Pair-IG is one-shot bounded cost; LR training is unbounded.** Pair-IG is O(n_train × total_dim × N_CLASSES²) for a single pass over counts. LR training is epochs × n_train × total_dim × N_CLASSES. Gradient descent is iterative and may not converge quickly at int8 precision.

13. **A perceptron update rule is integer-native and one-shot-friendly.** `W[d] += y × sig[d]` for misclassified examples is pure integer arithmetic. Converges in O(n_train × k_epochs) passes where k_epochs is small for well-separated problems. May not converge on CIFAR-10 but is the right first shape to try for MNIST / Fashion-MNIST.

14. **Structured_lsh and structured_gsh already live in the "each trit has a specific meaning" camp.** Bolting an LR head on top reintroduces learned-weight semantics at the scorer level. Question: is that a regression or an orthogonal addition? "Each input trit has a specific structural meaning; each *weight* is a learned scalar on that structural dimension" is internally coherent.

15. **Falsification target must be measurable.** If we run the experiment, we need a null hypothesis ("LR-derived weights do not beat pair-IG on CIFAR-10 by more than 0.5pp") and a metric that clearly resolves it. Without that, we learn nothing. Existing `direct_lsh` already reports both Hamming and Selective — a new resolver flag would wire in LR cleanly for a direct comparison.

16. **Information-theoretic sanity check.** pair-IG weights carry log(2) × 4 = 4 bits per dim (integer in [1,16]). LR weights quantized to the same range carry the same ceiling. If pair-IG is already extracting all 4 bits of discriminative signal, LR can't beat it. If pair-IG is leaving signal on the table (because entropy is a *statistic* of the marginals, while LR fits the *joint*), LR might help — but only if the joint contains bits entropy missed.

## Tensions

- **T1 (Node 2 vs Node 9).** Is pair-IG a learned weight or not? If yes, the "no learned weights" rule is already bent and LR is a small incremental step. If no (pair-IG is "derived statistics" not "learned weights"), then LR crosses a new line. This is the interpretive fulcrum for the whole question.

- **T2 (Node 5 vs Node 6).** Does scoring have remaining headroom on CIFAR-10? Node 5 says yes (pair-IG is already +1.95pp, so scoring is active). Node 6 says maybe not (the gap is representational). These can only be resolved by measurement — e.g., oracle over candidate union gives the absolute scoring ceiling.

- **T3 (Node 3 vs Node 11).** Scaffold LR or find base-3-native LR? Slot A/B both treat LR as a base-2 design to be ported. Node 11 asks "what's natively ternary that serves the LR-shaped purpose?" The scaffolding path is cheaper and immediately comparable; the native path is the thesis-aligned work but ill-defined upfront.

- **T4 (Node 7 vs Node 12).** Where does training live, and is it one-shot? If training is an external Python step, the C repo stays clean but the project-scope rule ("no Python in toolchain") is broken. If training is at startup in C, the float exception list grows (§12 site #5). If training is integer-only perceptron, it's cleanest but may underfit.

- **T5 (Node 10 vs NORTH_STAR §3 rule 2 "rage against the trodden").** LR is the definition of comfortable base-2 ergonomics. Even as scaffolding, it imports the gradient-descent cross-entropy pattern. The §3 rule warns this pull is always present and always misleading. Is there a non-comfortable, substrate-native formulation that would teach us more?

- **T6 (Node 11 vs Node 14).** If structured_lsh already assigns meanings and works, what does adding learned weights on top teach? Either the learned weights amplify the structural ones (multiplicative: "this structural dim matters 3x for class 7"), which is consistent, or they replace them (weight subsumes structure), which erodes the structured discipline.

## Dependencies

- Any Slot A experiment depends on: `direct_lsh` architecture remaining stable (it is), a new `build_pair_lr` function, a CLI flag to select pair-IG vs pair-LR, measurement harness (already exists — `direct_lsh` reports both pure Hamming and Selective).
- Any Slot B experiment depends on: all of Slot A's prerequisites, plus a per-class weight vector of size `total_dim × N_CLASSES × cell_width`, plus a decision to use MTFP4 or MTFP9 cells for the weights (so SDOT applies), plus a new consumer tool `lr_classifier` or a flag on `direct_lsh`.
- Any base-3-native formulation (Node 11) depends on: a definition of what "ternary LR" means structurally — which is itself a research question, not a decidable spec.

## Open questions

- Q1 (measurement before build): what's the oracle-over-union accuracy on CIFAR-10 in the existing direct_lsh sweep? If oracle is 46.63% + ε, pair-IG is near ceiling and Slot A won't win.
- Q2 (classification scope): is this about chasing CIFAR-10 numbers, or about finding a new primitive class the substrate should support? If the former, run Slot A and stop if it doesn't win. If the latter, Node 11 is the more interesting deliverable.
- Q3 (consumer rule): does training-at-startup fit the "one-shot float" exception pattern (§12 sites 3–4), or is gradient descent qualitatively different (because it's iterative)?
