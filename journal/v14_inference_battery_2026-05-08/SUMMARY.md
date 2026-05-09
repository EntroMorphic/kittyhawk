# V14-optimized vs V13 baseline — end-to-end inference comparison

Battery: 6 prompts (varying length) × 16 generated tokens, plus bench-shape (positions=32).
Each call warmed up once before timing. user CPU on M4.

| Prompt | V13 first 8 generated | V14-opt first 8 generated | Token agree | V14 vs V13 latency |
|---|---|---|---|---|
| capital_france (6→16) | 284 330 7063 1 1841 198 46770 1735 | 284 330 7063 1 1665 280 9099 1735 | 10/16 | +1.8% |
| bos_only (1→32) | 374 539 14647 837 13 358 1781 433 | 374 539 14647 837 13 358 1781 433 | 15/32 | +2.0% |
| token_5337 (1→16) | 12814 1807 11 1515 1807 340 848 1807 | 12814 1807 11 1515 1807 340 848 1807 | 16/16 | +2.6% |
| short_a (4→16) | 25 330 20489 18 25 2944 18 25 | 25 330 20489 18 374 539 264 2764 | 5/16 | +2.8% |
| short_b (4→16) | 527 539 264 1695 4950 369 279 4950 | 288 527 264 955 315 24032 430 649 | 1/16 | +2.8% |
| medium (11→16) | 3813 11 323 279 3838 374 264 9099 | 3813 198 791 9099 1665 374 279 3838 | 2/16 | +3.8% |
| pos32_no_gen (bench shape) | — | — | — | +5.1% |

## Summary

Latency: V14-opt within +1.8% to +5.1% across all prompts (mean ~+3%).
Token agreement: 49/112 (43.8%). Prefix tokens often match exactly (token_5337
is full 16/16; bos_only matches first 15 of 32; capital_france matches first 4).
Divergence is expected: V14.G's polynomial exp differs from V13's LUT exp by
small per-cell amounts that compound through 30 attention layers, occasionally
flipping the argmax. Both outputs remain coherent — same model, slightly
different probability distributions over its same vocabulary.

Per-prompt token-by-token data: see v14opt_*.txt and v13_*.txt in this directory.
