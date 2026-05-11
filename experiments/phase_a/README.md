# Phase A — substrate-routed attention training (PASS)

Per `journal/td27_7_phase_a_2026-05-11.md` pre-registration.

## Files

- `task.py` — sequence copy data generator. Fixed-length N=8 (see
  amendment in journal). Vocab 64 (32 symbols + BOS/SEP/PAD/reserve).
- `model.py` — TinyGPT (1 layer, 4 heads, head_dim 16, model_dim 64,
  FFN inner 128, ~51K params). Two attention variants:
  - `DenseAttention`: standard scaled dot-product.
  - `SubstrateRoutedAttention`: top-k=4 selection via sign-based
    signature distance, then sparse softmax + V combine. STE is
    implicit via PyTorch's `gather` (gradients flow through selected
    positions only). Weights use `BitLinear` (b1.58-style ternary
    QAT with sign-STE).
- `train.py` — AdamW + cosine schedule + eval every 100 steps. Stops
  at first eval ≥ 95% test accuracy.
- `logs/` — per (variant, seed) JSON output.

## Run

```sh
for seed in 42 43 44; do
  for variant in dense substrate; do
    python3 train.py --variant $variant --seed $seed \
      --max-steps 7000 --log logs/${variant}_${seed}.json --device cpu
  done
done
```

CPU on M-series Mac: ~5 seconds per 1000 steps. Full 3-seed × 2-variant
experiment: ~1 minute wall-clock.

## Result

| seed | dense pass | substrate pass | ratio | dense acc | substrate acc |
|------|-----------|---------------|-------|-----------|---------------|
| 42 | 700 | 900 | 1.29× | 0.950 | 0.952 |
| 43 | 900 | 1100 | 1.22× | 0.998 | 0.977 |
| 44 | 800 | 1100 | 1.38× | 0.989 | 0.971 |
| **mean** | **800** | **1033** | **1.29×** | **0.979** | **0.967** |

All 3 seeds: substrate pass-step ≤ 2× dense pass-step (pre-registered
success criterion). Mean ratio 1.29× — substrate trains in 29% more
steps than dense on this task.

**Phase A: PASS.** The substrate's discrete top-k attention selection
is trainable via implicit STE on a tiny GPT with ternary weights.
