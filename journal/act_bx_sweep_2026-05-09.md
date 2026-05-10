# BITNET_ACT_BX sweep — closes TD-19

Date: 2026-05-09. Per `journal/saturation_audit_2026-05-09.md` red-team
finding: late-layer `block_output` saturation (1–9 cells per position at
L24–L29) was newly visible after the RMSNorm `gamma_bx > target_bx` fix.
TD-19 proposed sweeping `BITNET_ACT_BX ∈ {6, 7, 8}` to find the bx that
minimizes saturation while preserving coherence.

## Method

For each `BITNET_ACT_BX ∈ {6, 7, 8}`:

1. `sed` the `#define` in `gesh/bitnet/bitnet_harness.c`
2. `cmake --build build --target bitnet_harness`
3. Forward pass on the 8-token canary prompt with `--dump`; count cells
   `|mantissa| ≥ MTFP19_MAX = 581_130_733` across all 30 layers × 8
   positions × 12 captured sites.
4. Run all 8 battery prompts (`journal/post_rmsnorm_fix_battery_2026-05-09/`)
   with `--gen 30`. Decode and label coherence (heuristic: ≥4 occurrences
   of any 8/12/20-char window = "loop"; else "ok").

## Saturation results

| BX | `gate` | `block_output` | OTHER | TOTAL |
|---|---|---|---|---|
| 6 | 17 | **0** | 0 | **17** |
| 7 | 18 | **0** | 0 | **18** |
| 8 (current) | 18 | 209 | 0 | 227 |

Lowering BX **fully eliminates** block_output saturation at both 7 and 6.
The `gate` saturations (post-relu² outliers) are nearly bx-invariant — they
depend on `BITNET_FFN_BX = 6`, which the sweep didn't touch.

## Coherence results

### BX = 6 (8/8 ok by loop heuristic, 7/8 substantively correct)

- `factual_capital`: "The answer is: Paris..." ✓
- `factual_who`: "Shakespeare wrote Hamlet..." ✓
- `definition_photosynth`: correct definition ✓
- `continuation_once`: "young girl named Alice..." ✓
- `math_simple`: **"20"** ❌ — wrong by one (12 + 7 = 19)
- `reasoning_color`: "blue... color of the ocean is blue..." ✓ (drifts off-topic but coherent)
- `reflective`: cog-sci-flavored answer ✓
- `translate_hello`: doesn't translate but coherent ✓

The math regression is the load-bearing finding. Losing 1 trit of fractional
precision per cell tightens the resolution between adjacent integer tokens
(19 vs 20) enough to flip the argmax.

### BX = 7 (7/8 ok by loop heuristic)

- All except `reasoning_color` ok
- `reasoning_color` loops: "blue... The color of the sky is blue, and the
  color of the sky is blue..."
- Math correct (12 + 7 = 19) ✓

### BX = 8 (8/8 ok, current default — post-RMSNorm-fix)

- All 8 coherent
- Math correct ✓
- Best prose (`factual_capital` gives the fullest answer)

## Conclusion

**Keep BITNET_ACT_BX = 8.** TD-19 closed as "current setting is the right
balance."

The 209 block_output saturations are 0.034% of the 30 × 8 × 2560 = 614,400
residual-stream cells in the full sweep. The substrate's downstream
operations (RMSNorm, softmax) absorb the loss — the saturations were the
symptom of headroom-tightness, but the network is robust to that fraction
of capped cells. End-to-end quality is unaffected.

Lowering BX recovers headroom but at a cost paid elsewhere:

- **BX = 7** trades 209 saturations for one looped output (`reasoning_color`).
  Net negative.
- **BX = 6** also fixes saturation but introduces a math regression
  (12 + 7 = 20). The 1-trit precision loss matters when the model's logits
  are tight between adjacent semantically-distinct tokens. Net negative.

This recapitulates the original tuning rationale at line 44 of
`bitnet_harness.c`: BITNET_ACT_BX was tuned empirically to be the best
trade-off, and that tuning still holds post-RMSNorm-fix. The saturation
counted post-fix is genuine residual-stream overflow, but the consumer
(BitNet's normalization layers) handles it gracefully.

## Methodological note

The TD-19 hypothesis ("lower ACT_BX to fix saturation") was correct in the
narrow sense — saturation goes to zero at BX ≤ 7. But the broader question
("does fixing this saturation improve end-to-end quality?") was answered NO
by the battery. Useful pattern: when a substrate metric improves but the
end-to-end metric doesn't, the substrate metric wasn't load-bearing for
quality. The metric that matters is downstream.
