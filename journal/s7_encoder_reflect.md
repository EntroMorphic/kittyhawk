---
date: 2026-04-22
scope: LMM cycle — S7 thesis-calibration scaffold
phase: REFLECT
---

# REFLECT

## The core insight

**S7 is not about whether "learning works." It's about whether Glyph's downstream pipeline (bucket + pair-IG + Selective) can USE an arbitrarily good signature to reach SSTT-level accuracy on CIFAR-10, or whether the pipeline itself is the ceiling.**

The framing "substrate-bounded vs encoding-bounded" was close but not quite the right partition. The actual question has three parts:

1. **Encoding ceiling:** what's the best ternary signature any encoding can produce for CIFAR-10 at ~11k dim?
2. **Downstream ceiling:** given a near-optimal ternary signature, what accuracy can Glyph's bucket + pair-IG + Selective deliver?
3. **Gap:** where is the 5pp gap to SSTT living?

S7 partially pins down (1) by training a strong encoder. If S7 reaches 53%, the gap is (2) — Glyph's downstream is fine. If S7 reaches only 48%, the gap is distributed — encoding AND downstream both contribute.

## Why this reframes the experiment design

The reflection sharpens the gate. It's not enough to "measure S7 Selective accuracy." We need to compare:

- **S7 Selective** — learned encoder + Glyph downstream.
- **S7 raw-signature nearest-neighbor** — learned encoder + BRUTE-FORCE 1-NN over all training samples in trit space, skipping the bucket/pair-IG stages.

If the raw-signature 1-NN also plateaus at 48%, the encoding is the ceiling. If raw-signature 1-NN reaches 53%+, the downstream IS the ceiling and S7-in-Glyph-downstream's 48% tells us Glyph's filter+resolve stages are lossy.

**This is a big insight.** Without the raw-signature 1-NN control, a 48% S7-Selective would be ambiguous. With it, we can decompose the gap cleanly.

## Second reframe: Python is the right choice here

The NORTH_STAR §4 scaffolding sanction is explicit. This is a calibration experiment, not a production path. Artifacts (trained weights → binary trit signatures) are committed; training script is committed as experimental. Repository discipline: `tools/experimental/s7_train.py` with a README header marking it non-production and not-on-the-C-build-path. Clean separation.

The in-C-autodiff alternative would be purism for its own sake. NORTH_STAR §3 rule 1: "uncertainty leads; the pull toward the familiar is always present and always misleading." The pull toward "C-only toolchain at all costs" here IS familiar-pattern-matching — we're comfortable with integer kernels, so we force everything into that shape. For a scaffolding calibration experiment, that's exactly the wrong pull.

Python is the right tool for the one-shot training. Go.

## Resolved tensions

**T1 (Python vs no-Python):** resolved in favor of Python for this calibration cycle, with discipline:
- Script in `tools/experimental/s7_train.py`, not on the C build path.
- Output is binary trit signature files (same packing as `glyph_sig_quantize`).
- README in `tools/experimental/` explains the sanctioned-scaffolding framing.
- The artifact committed to repo is the SIGNATURES, not the weights. Only the signatures are consumed by `direct_lsh`.
- If the user doesn't approve, fall back to in-C class-centroid encoder (weaker answer but still informative).

**T2 (linear vs sophisticated):** start linear. If linear under-performs SSTT significantly but also under-performs direct quantization, the encoder's linear expressive power is the bottleneck, not Glyph's downstream. Escalate to small MLP. If even MLP underperforms SSTT, the gap is either architectural (our downstream) or training-compute-bounded (SSTT trained longer).

**T3 (same-dim vs sweep):** same-dim (11232) first. Lets us say "same budget, different encoding, result changes by X." Sweep deferred.

**T4 (loss function):** start with softmax classification loss. Cheapest, most common. If classification plateaus, consider contrastive (aligns with k-NN). Almost certainly a second-order effect.

**T5 (training quality underestimating ceiling):** accept. Report as "S7 at reasonable training effort reaches Y%". If Y is below the SSTT baseline by more than 3pp, document as "possible under-training; SSTT ceiling may be reachable with more compute but wasn't verified in this cycle."

**T6 (interpretation thresholds):** pre-declare and stick to them.

## Hidden assumptions I was making

- **Assumption: Glyph's downstream handles any sig.** Probably true but untested. The filter stage (bucket + multi-probe) was tuned for direct-quantization signatures. Multi-scale signatures worked because they were appended in the same space. Learned signatures might have different structure (e.g., more entropy per trit), and the bucket filter could behave differently. **Raw-signature 1-NN control tests this.**

- **Assumption: S7 with same dim is a fair comparison.** Maybe not. A learned encoder with 11k dim is much more expressive than direct-quantization 11k dim because each learned trit aggregates over many input pixels. Same dim comparison might UNDERstate S7 (which could do more with less). The right comparison might be "smallest learned dim that matches direct quantization" — but that's a bigger sweep.

- **Assumption: a naive linear encoder reaches "reasonable" performance.** SSTT uses attention + blocks + pattern scoring. A single-layer linear encoder is dramatically simpler. It might land at 45% and tell us "single-layer can't match SSTT" without answering whether encoding per se is the ceiling. Escalation to MLP is likely required.

## What I now understand

1. **The raw-signature 1-NN control is essential.** Without it, S7 Selective results are ambiguous. With it, the gap (encoder vs downstream) decomposes.

2. **Python is the right tool here.** NORTH_STAR §4 explicitly sanctions. Discipline enforced by scope: `tools/experimental/` subtree, binary artifacts, non-production flag.

3. **Linear encoder is the starting point, not the end point.** If linear lands at 48%, that's one data point. If MLP also lands at 48%, that's stronger evidence. If CNN lands at 53%, that's a different story. Cycle gate: linear first; MLP if linear is inconclusive; don't escalate to CNN within this cycle.

4. **Pre-declared interpretation is load-bearing.** The outcome space is continuous but the lessons aren't. Threshold table:

   | S7 Selective on CIFAR-10 | Interpretation |
   |---|---|
   | ≥ 53% | Encoding-bounded confirmed. SSTT gap lived in the encoder. Substrate is fine. |
   | 50–52% | Encoding matters but downstream is also lossy. Mixed story. Need MLP escalation. |
   | 48–49% | Inconclusive — learned encoder roughly matches direct quantization. Linear may be too weak; try MLP. |
   | 45–47% | Learned encoder loses to direct quantization. Either training is broken, encoder is too weak, or direct quantization was actually near-optimal. |
   | < 45% | Clear bug; debug before drawing conclusions. |

   Similarly for raw-signature 1-NN (no Glyph downstream):

   | Raw 1-NN S7 sig | Interpretation |
   |---|---|
   | ≥ 53% | Downstream IS lossy — Glyph's filter+resolve stages bottleneck at 48%. Future downstream work. |
   | 48–52% | Downstream is fine; encoder is near-ceiling at this spot. Look for different encoding paradigms. |
   | < 48% | Encoder itself is the bottleneck at this architecture; need better architecture/training. |

5. **Two numbers, clean decomposition.** The cycle's deliverable is the 2×5 interpretation grid with actual measured values placed on it.

## Open residuals

- **R1 (cycle scope).** Training script, `--sigs_from_file` loader in direct_lsh, raw-1-NN control (maybe a separate micro-tool or flag). Probably 1-2 day implementation.

- **R2 (MNIST/Fashion reference runs).** Same training pipeline on those datasets. Probably cheap once the pipeline exists. Worth including for cross-dataset sanity.

- **R3 (what if the user declines Python).** Fallback: in-C class-centroid encoder. Write the Python track as primary; queue the centroid fallback as a smaller experiment that can run in parallel on a different architecture branch if needed.
