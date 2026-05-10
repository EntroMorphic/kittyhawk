# math_div atomics + gate1+fudge2 — TD-22 closure

Date: 2026-05-10. The hp sweep (`journal/hp_sweep_2026-05-10.md`) recovered
4 of 5 substrate-specific failures via `BITNET_GATE_ACT_BX = 1`; math_div
remained the holdout (TD-22). User asked to dig into the atomics of that
failure.

## Phase 1 — find divergence position

Substrate (gate1) and HF (bf16, greedy) on the math_div prompt
("144 divided by 12 equals", 7 tokens including BOS):

| pos | substrate | HF | flag |
|---|---|---|---|
| 0 | 1148 ` what` | 1148 ` what` | ✓ |
| 1 | 5380 `?\n` | 5380 `?\n` | ✓ |
| 2 | 16533 `Answer` | 16533 `Answer` | ✓ |
| 3 | 25 `:` | 25 `:` | ✓ |
| 4 | 220 ` ` (single space) | 2057 ` To` | **✗** |

First divergence at gen position 4. Both have the **identical** 11-token
shared prefix at this point; substrate's last-position argmax is the
single-space token (220), HF's is ' To' (2057). Same context → different
argmax → noise difference at the logits level.

## Phase 2 — capture activations on the shared prefix

Substrate run with `--prompt-tokens "128000,8929,18255,555,220,717,17239,1148,5380,16533,25" --dump`
produces per-(position, layer) ACTV2 dumps. HF run with the same 11 tokens
under forward hooks captures fp32 activations at matching sites
(`x_norm_input`, `q_pre_rope`, `k_pre_rope`, `v`, `attn_sub_norm`, `x_norm`,
`gate_pre_relu2`, `up`, `ffn_sub_norm`).

## Phase 3 — per-layer ε at the last position

Best-scale-fit ε at position 10 (the next-token-prediction position):

| Layer | x_norm_input | q/k/v | attn_sub_norm | x_norm | up | ffn_sub_norm |
|---|---|---|---|---|---|---|
| L0 | **0.007** | 0.04-0.09 | 0.24 | 0.14 | 0.13 | 0.25 |
| L1 | 0.22 | 0.09-0.18 | 0.46 | **0.84** | 0.77 | 0.82 |
| L2+ | 0.82+ | 0.4-0.6 | 0.85+ | 0.85+ | 0.65-0.95 | 0.85+ |

Note: the `gate` site comparison is misleading because substrate captures
post-relu² while the HF hook captures pre-relu². Excluded from the analysis.

## What this is — and isn't

**Not a single-kernel bug.** Unlike the RMSNorm bug (where ε amplified 80×
in one specific kernel call), here ε grows roughly **5× per layer
transition** through compound noise. There is no localized jump where one
kernel's output is anomalously bad. The L0 input_layernorm output is clean
(ε = 0.65%); BitLinear outputs are 4-9% (small); attention chain grows ε
to ~24%; one layer of residual+norm+attention amplifies that to ~84% by
L1; from L2 onward the activations are essentially uncorrelated with HF
(ε ≈ 0.85, fit_s well below 1).

This is the **accumulated quantization noise regime**: MTFP19 vs bf16 logit
ε reaches a threshold that flips a tight-margin argmax. Not fixable by
patching a single kernel.

**The fix came from revisiting the prior sweep, not from the atomics.**
TD-22 had recorded that `gate1+fudge2` was an untested combination;
gate1 alone had been the clear single-knob winner. Tested the combination:

```
math_div on gate1+fudge2: " 12\n}\n\nQuestion: How many times does
                          144 divided by 12 equals 12?\n\nSolution:\n..."
```

**First generated token is " 12" — direct correct answer.**

The atomics investigation is what gave us *confidence* that no single-kernel
fix was warranted; the actual recovery came from a knob combination the
prior sweep had identified but not tested together.

## Full 24-prompt battery on `gate1 + fudge=2`

Manual classification (loop heuristic alone is unreliable for borderline cases —
verified by reading the full text on every loop-flagged result):

| Verdict | gate1 | gate1+fudge2 |
|---|---|---|
| ✓ correct/strong | 19 | **22** |
| ⚠ degraded | 5 | 2 |
| ✗ broken | 0 | 0 |
| Strict pass rate | 79% | **92%** |

Improvements (3):
- **`factual_hamlet`**: "(Hint: It's a famous play..." ⚠ → **"Answer: William Shakespeare wrote Hamlet. He was an Englis..."** ✓ (the gate1 regression is recovered)
- **`math_div`**: "12 × 12 = 144" ⚠ → **"12"** ✓ (the holdout is recovered)
- **`def_ml`**: vague tautology ⚠ → "of artificial intelligence that involves the development of" ✓

Apparent regressions caught by loop heuristic but not real on inspection:
- `code_python`: both algorithms correct; g1f2 is more compact (`if n < 2: return 1`)
- `edge_question`: both coherent paragraphs; g1f2 has slightly more repetition character
- `code_comment`: gate1 produces Python `def sort_array`, g1f2 produces JavaScript
  `function sortArray(arr) { let sortedArr = arr.sort((a, b) => a - b); return sortedArr; }` —
  different valid continuation, not a regression

## Mechanism (hypothesis)

`fudge` is the extra right-shift applied to attention scores before softmax,
softening the distribution. Larger fudge → flatter attention weights → less
spiky attention pattern.

For prompts where the correct continuation has tight logit margins between
plausible alternatives (like `144 ÷ 12 = ?` where "12", "12.144", "what"
are all model-plausible next tokens), a softer attention distribution may
help by:

- Reducing the variance amplification through deep layers (each attention
  layer contributes less to magnitude growth in the residual stream)
- Letting more of the prompt's content tokens contribute to the
  next-token prediction, biasing toward the "obvious" continuation

This is hypothesis, not a confirmed mechanism — would need targeted
experiments to verify.

## Closure

- TD-20 → fully closed. All 5 substrate-specific failures from the v2
  battery now produce coherent + correct output.
- TD-22 → closed. The `gate1+fudge2` combination was the productive one;
  `gate1+ffn8` and `gate1+fudge2+ffn8` not tested but no longer needed
  given the strong gate1+fudge2 result.
- Default updated: `score_shift += 2` (was `+= 1`).

## Methodology lift

When two single-knob optima are identified independently, **test the
combination before declaring either one the winner**. Combinations can
recover failures neither knob recovers alone. The prior sweep (Phase A)
correctly identified gate1 and fudge2 as the two strongest knobs but
didn't test their combination — TD-22 captured that gap and this cycle
closed it.

The atomics investigation also did real work even though it didn't
localize a kernel bug: it produced **negative evidence** that the math_div
failure was NOT a single-bug, which is what gave us confidence to try the
hyperparameter combination instead of continuing to dig kernel-by-kernel.
