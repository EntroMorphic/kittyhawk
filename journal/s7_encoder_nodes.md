---
date: 2026-04-22
scope: LMM cycle — S7 thesis-calibration scaffold
phase: NODES
---

# NODES

## Discrete ideas

1. **Experimental question is binary.** "Substrate-bounded or encoding-bounded?" The cycle's deliverable is a single number-plus-interpretation, not a production tool. Success criterion is clarity, not accuracy.

2. **Decoupling encoder from downstream is the key design choice.** The encoder produces a per-image packed-trit signature on disk. `direct_lsh` loads signatures from file via a new flag and runs the normal filter + resolver pipeline. This isolates the ONE variable: the encoding. Every other component — bucket build, pair-IG, Selective — is held constant.

3. **Glyph is C-only per project memory. S7 needs float training.** Two paths: (a) external Python trainer, output binary trit files, rule bent as sanctioned scaffolding per NORTH_STAR §4; (b) in-repo C trainer with hand-rolled autodiff, keeping discipline at cost of 2-week project. Path (a) is the pragmatic choice for this calibration experiment.

4. **Three architecture tiers, increasing cost.** Linear encoder (W × x → sign), small MLP (linear → GELU → linear → sign), patch CNN (3×3 conv → pool → conv → sign). Each produces a fixed-dim trit signature. Linear is cheapest and most directly comparable to direct quantization.

5. **Straight-through-estimator for quantization is standard but not unique.** Naive STE (forward: sign; backward: identity, clipped) trains adequately. More sophisticated: stochastic quantization, tanh-based surrogate, clipped STE with learned bounds. Naive STE is the minimal sufficient choice.

6. **Signature dimension is a variable with semantic weight.** Same dim as MS4+R4 (~11232) → direct comparison of encoding quality at fixed budget. Smaller dim (e.g., 1024) → tests whether expressive learning does more with less. Both are valuable but first-pass should be same-dim for the head-to-head.

7. **Loss function is a design choice.** Classification loss (softmax + cross-entropy on class labels) is standard. Prototype loss (minimize distance from correct-class prototype signature) is more direct. Contrastive loss (same-class sigs close, different-class far) aligns with the downstream k-NN task. Classification loss is cheapest; contrastive is probably best for this specific downstream.

8. **Training duration is bounded by the scaffolding framing.** This is a calibration experiment, not a production model. Train to convergence on a modest number of epochs (50-100), single GPU/CPU, no hyperparameter sweep. Accept whatever accuracy results; don't chase marginal gains.

9. **Lower bound: random ternary projection.** If S7 underperforms a plain random-projection LSH at the same dim, the encoder is trivially adding no value. This bounds what "encoder worked" means.

10. **Upper bound: SSTT's 53%.** SSTT uses a learned ternary transformer with block-pattern scoring. If our simple learned encoder reaches 53%, we've closed the gap without SSTT's downstream. If we plateau below, either our encoder is underpowered OR the downstream (filter + pair-IG) is a ceiling.

11. **Oracle-in-union says the filter stage has headroom.** P1 measurements showed oracle ~99.99% on CIFAR-10 at M=64. The filter isn't losing signal. So any gap between S7 Selective and SSTT-level accuracy is in: (a) the encoder, (b) the signature distance metric, or (c) pair-IG re-rank expressiveness. S7 tests (a) while holding (b) and (c) constant.

12. **The experimental contrasts.** Running all three (direct, random-projection, S7) at the same dim directly measures what each adds. Direct is the current baseline. Random-projection is the classical LSH baseline (Glyph's legacy). S7 is the learned baseline.

13. **Python script scope: 200-300 lines max.** Load CIFAR-10 (torchvision), define encoder (linear or MLP), train with cross-entropy + STE, generate signatures for all 60k train + 10k test, pack to trit bytes, write to disk. No data augmentation initially (matches Glyph's direct-quantization precedent).

14. **On-disk format: packed-trit binary files.** Same 2-bit packing as `glyph_sig_quantize` outputs. Read by direct_lsh via `--sigs_from_file train.bin test.bin`. Format: `[n_train × sig_bytes][n_test × sig_bytes]` or two separate files. Separate files are cleaner.

15. **Ideas out of scope for this cycle.** Data augmentation, learned post-encoding distance metric, learned pair-IG weights, end-to-end differentiation through bucket lookup. All interesting; all expansive. Keep the cycle tight.

## Tensions

- **T1 (rule violation vs calibration value).** Running external Python training violates "no Python in toolchain." But the NORTH_STAR §4 scaffolding sanction explicitly allows this for calibration experiments. Tension is whether the user reads this as sanctioned scaffolding or as discipline drift. Default: ask explicitly; default to disabled if uncertain.

- **T2 (architecture: simple vs sophisticated).** Linear is the minimum viable encoder and most directly answers "does learning help." MLP/CNN is what practitioners would actually use. If linear underperforms SSTT significantly, we don't know if it's "learning doesn't help" or "linear is underpowered." Resolution: run linear first (cheap), escalate to MLP only if linear shows encoding-bound signal.

- **T3 (same-dim vs sweep).** Same-dim is the clean comparison. Sweep is more informative but 2-3× the compute. Resolution: same-dim first; sweep if result is interesting.

- **T4 (loss function).** Classification loss is standard. Contrastive is more aligned with k-NN downstream. Prototype loss is simplest. Resolution: classification first (easiest), contrastive only if classification plateaus.

- **T5 (training quality vs cycle scope).** A weak-trained S7 underestimates learned encoding's ceiling. A well-trained S7 is a full week-plus project. Resolution: accept that a 1-2 day training run might underperform SSTT by X pp due to training-compute deficit, and report "S7 at reasonable training effort reaches Y%" as the answer. X is an acceptable unknown.

- **T6 (what counts as "substrate-bounded" vs "encoding-bounded").** Not a clean binary. If S7 reaches 50%, 51%, 52%, 53% — each tells a different story about the gap composition. Resolution: pre-declare interpretation thresholds.

## Dependencies

- Python 3 + PyTorch for the trainer. Available on the dev machine.
- CIFAR-10 torchvision download or reuse of existing `data/cifar10/*.bin` files.
- A binary trit-packing utility (either in the Python script, using numpy; or in a small C helper the Python script calls).
- `direct_lsh` new flag `--sigs_from_file` + matching loader.

## Open questions

- **Q1: Is external Python training acceptable for this calibration experiment?** Per scaffolding sanction (NORTH_STAR §4), yes. Per strict scope rule ("no Python in toolchain"), no. Default: frame as external, not in toolchain; artifacts committed as binary data only; trainer script lives outside the repo or in a `tools/experimental/` subdir marked as such.

- **Q2: What pre-declared gate makes this cycle informative?** Proposal:
  - **S7 ≥ 51% Selective on CIFAR-10:** encoder materially helps; +3pp over MS4+R4 baseline = substrate is not the ceiling.
  - **S7 within ±1pp of MS4+R4 baseline (47-49%):** inconclusive; encoder roughly matches direct quantization on this downstream; more training or richer architecture needed before strong conclusion.
  - **S7 below 47% (MS4+R4 baseline by >1pp):** encoder has bug OR under-trained OR linear is too weak. Inconclusive until escalated.
  - **S7 ≥ 53% on CIFAR:** matches SSTT with simpler downstream; thesis fully calibrated — encoding-bound confirmed.

- **Q3: What training hyperparameters?** CIFAR-10 on linear encoder at dim=11232: likely ~10 epochs with Adam, lr=1e-3, batch=256 — takes ~1 hour on CPU, minutes on GPU. Plus signature export. 1-day cycle feasible.

- **Q4: Should the encoder condition on pixels, or on intensity + gradients features (like Glyph's current pipeline)?** Pixels gives a cleaner encoder-vs-direct comparison. Features gives a "which is better at same input" comparison. Pixels first.

- **Q5: Should we also run S7 on MNIST and Fashion as reference?** Cheap to do (both converge fast). Useful as "what's the headroom on each dataset?" sanity.

- **Q6: What if the user says no to Python?** Fallback: in-C centroid-based encoder (per-class prototype, no training). It's not really "learned" in the gradient sense, but answers a weaker version of the question. Much cheaper scope. Could be the first run.
