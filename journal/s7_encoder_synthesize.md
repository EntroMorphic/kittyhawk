---
date: 2026-04-22
scope: LMM cycle — S7 thesis-calibration scaffold
phase: SYNTHESIZE
---

# SYNTHESIZE: S7 learned ternary encoder with raw-1-NN control

## The reframe

The seed question ("substrate-bounded or encoding-bounded?") was close but asks a binary where the answer is distributed. The reflection sharpens it into two measurements that cleanly decompose the remaining 5pp CIFAR gap:

1. **S7 Selective (learned encoder + Glyph downstream):** where the gap lives when signatures are learned but the pipeline is Glyph's.
2. **S7 raw 1-NN (learned encoder only, brute-force 1-NN classifier, no Glyph downstream):** what the encoder alone can reach, holding downstream constant at the simplest possible classifier.

The (S7-Selective, S7-raw-1NN) pair locates the 5pp gap on a 2D grid — between encoder quality and downstream quality. That grid is the cycle's deliverable.

## Decision

**Run S7 as an external Python calibration experiment per NORTH_STAR §4 scaffolding sanction.** Scope-bound:

- Trainer in `tools/experimental/s7_train.py` (not on the C build path).
- Output: binary packed-trit signature files (`train_sigs.bin`, `test_sigs.bin`) loaded by `direct_lsh --sigs_from_file`.
- Committed artifacts: the Python script + the binary signature files. Not the float weights.
- Linear encoder first; escalate to MLP only if linear is inconclusive.

**Also implement a raw-1-NN control inside `direct_lsh`:** new flag `--brute_1nn` that bypasses bucket + multi-probe + pair-IG and classifies each query by nearest-neighbor class over all training signatures.

**Gate by pre-declared thresholds** (listed below). After measurement, place the (S7-Selective, S7-raw-1NN) pair on the interpretation grid.

## Success criteria

**Cycle-level:**
- [ ] Linear S7 trained, signatures exported, loaded by `direct_lsh --sigs_from_file`, measured on all three datasets.
- [ ] Raw 1-NN control measured on the same S7 signatures.
- [ ] Interpretation grid populated with the measured pair.
- [ ] Cycle report: closed with substrate-vs-encoding verdict. No need for S7 to "win" — the gate is the clarity of the decomposition.

**Interpretation grid (pre-declared):**

| S7-Sel on CIFAR | S7-raw-1NN on CIFAR | Interpretation |
|---|---|---|
| ≥ 53% | ≥ 53% | Encoding+downstream both fine; Glyph was just using wrong encoding. Mass win. |
| ≥ 53% | < 53% | Glyph's downstream helps on S7 sigs beyond raw 1-NN. Downstream is an ASSET, not a bottleneck. |
| 48–52% | ≥ 53% | Downstream is LOSSY on S7 sigs. Learned encoder unlocks signal that filter+pair-IG can't fully exploit. |
| 48–52% | < 53% | Encoder and downstream both near their ceiling at this budget. ~50% is plausibly the honest ternary-at-11k-dim CIFAR ceiling; gap to SSTT lives elsewhere (SSTT uses bigger models, more training, etc). |
| < 48% | any | Linear encoder under-performs direct quantization. Need MLP escalation before concluding. |

## Implementation specification

### Python trainer (`tools/experimental/s7_train.py`)

```python
# Inputs: CIFAR-10 (load from data/cifar10/*.bin via numpy).
# Architecture: nn.Linear(3072, 11232) + sign quantization via STE.
# Loss: cross-entropy with an auxiliary classifier head (nn.Linear(11232, 10))
#       applied to the UN-quantized pre-sign activations during training.
#       At inference, we quantize and export the sign pattern.
# Optimizer: Adam, lr=1e-3, batch=256, epochs=10-30.
# Output: packed-trit binary files for train_sigs and test_sigs.
```

Why linear + sign with an auxiliary classifier head: quantization through sign is non-differentiable; training the PRE-sign activations for class discrimination, then quantizing once trained, produces a reasonable signature. This is the simplest approach that works. Naive STE through the sign is the alternative; auxiliary classifier is simpler and more stable.

File format (matching Glyph's packed-trit convention):
- Header: `uint32 n_images` `uint32 n_trits`
- Data: `n_images * ceil(n_trits / 4)` bytes, 2-bit trit encoding per `m4t_trit_pack.h`.

### `direct_lsh` new flags

```
--sigs_from_file TRAIN_SIG_PATH TEST_SIG_PATH
    Skip direct_lsh's signature building. Load pre-computed packed-trit
    signatures from disk. Uses the loaded signatures for the rest of the
    pipeline. Validates header matches n_train/n_test.

--brute_1nn
    Replace filter+resolver with a brute-force 1-NN over all training
    signatures. For each query, compute popcount_dist against every
    training sig and classify by nearest-neighbor label. Reports a
    single accuracy number.
```

### Experimental matrix (minimum)

| Config | Encoder | Signature source | Classifier |
|---|---|---|---|
| baseline MS4+R4 | direct quantization (existing) | Glyph built-in | Selective |
| S7-Sel | S7 linear | --sigs_from_file | Selective |
| S7-raw | S7 linear | --sigs_from_file | --brute_1nn |
| baseline raw | direct quantization (existing) | Glyph built-in | --brute_1nn |

Four numbers per dataset × 3 datasets = 12 numbers. Cycle deliverable.

### Training hyperparameters (for replicability)

- Architecture: `nn.Linear(3072, 11232, bias=True)` with auxiliary `nn.Linear(11232, 10)` classifier head on the PRE-sign activations.
- Normalization: per-image zero-mean unit-variance (match Glyph's `--normalize`).
- Loss: cross-entropy on auxiliary classifier + L1 regularization on activations (keep sparse signatures).
- Optimizer: Adam, lr 1e-3, weight decay 1e-4.
- Batch size: 256. Epochs: 20 (CIFAR), 10 (Fashion/MNIST).
- Quantization at export: `sign(activations)` converts pre-sign float to {-1, 0, +1}. Tau chosen as per-dim density percentile to match Glyph's `--density 0.395` (so the trit population densities are comparable).

### Gate for escalation

If linear S7-Sel on CIFAR lands < 47% (below MS4+R4 by > 1pp), **do not proceed to MLP in this cycle**. Document as "linear encoder under-performed direct quantization; MLP escalation deferred to a follow-up cycle." Cycle still produces useful information: "linear learned encoding doesn't beat direct at this budget."

If linear S7-Sel on CIFAR lands ≥ 47% but < 51%, escalate to one-hidden-layer MLP (linear 3072 → 512 → linear 512 → 11232 → sign). Same training loop. Measure again.

If linear S7-Sel on CIFAR lands ≥ 51%, stop and report — we have the answer.

## Handling the major tensions

- **T1 (Python tolerable?):** resolved. External Python in `tools/experimental/`, artifacts binary, sanctioned per NORTH_STAR §4.
- **T2 (linear vs MLP):** linear first. MLP only if linear is inconclusive (47-51%).
- **T3 (same-dim):** yes, same dim as MS4+R4 for direct comparison.
- **T4 (loss):** classification loss via auxiliary head. Simplest, most stable.
- **T5 (training compute):** accepted. 20 epochs = 1-2 hours on CPU. If we under-train, the gate "≥ 51%" says to escalate. Accept that SSTT's 53% may need more compute than this cycle allots.
- **T6 (threshold interpretation):** pre-declared grid above.

## Quality check

- **Could someone else execute this?** Yes. Python script, C flag, four measurements, interpretation grid. Total implementation: ~1-2 days.
- **Does it address all major tensions?** All six resolved or accepted with pre-declared paths.
- **Is it simpler than the starting point?** RAW had three architecture options, three training location options, five loss function options, five scope questions. Synthesis is one architecture, one training location, one loss, one dim, with a single escalation trigger.
- **Surprised?** Yes — entered expecting "train a network, measure CIFAR accuracy." Left with "the raw-1-NN control is the load-bearing addition that makes the measurement interpretable." Without it, we'd have one ambiguous number.

## Immediate next actions

1. Confirm Python path is acceptable to user (per scaffolding sanction). If no, fallback to in-C class-centroid encoder.
2. Implement `direct_lsh --sigs_from_file` loader (~50 lines).
3. Implement `direct_lsh --brute_1nn` classifier (~30 lines).
4. Write `tools/experimental/s7_train.py` (~200 lines).
5. Run S7 training on CIFAR-10, export signatures.
6. Measure 4-config experimental matrix on CIFAR-10.
7. Extend to Fashion/MNIST for reference.
8. Populate interpretation grid; write cycle close-out.

Total estimated time: 2-3 days of implementation + measurement + writeup.

## What this cycle produces regardless of outcome

A 2×5 interpretation grid with measured values, decomposing the CIFAR 5pp gap-to-SSTT into (encoder contribution, downstream contribution). Whether S7 reaches SSTT or not, the cycle's deliverable is the decomposition — the clean partition that tells us where to invest next.

The reframe — "raw-1-NN control is load-bearing" — is the cycle's design contribution regardless of numbers. Future S7-like calibration experiments should always pair with a downstream-free control.
