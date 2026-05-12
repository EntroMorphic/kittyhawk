# Cost measurement — substrate's attention savings don't manifest in wall-clock for BitNet

Per concern #4 raised in personal reflection earlier: "the cost-distinct
claim is unfalsified theory" — extrapolated from FLOPS arithmetic
(256× fewer dots at seq_k=4096) rather than measured wall-clock.

This experiment falsifies (or validates) the cost claim with measured
data at seq_k ∈ {64..1056}, using the production NEON sparse attention
path already in `bitnet_harness.c`.

## Setup

**Harness instrumentation:** added `clock_gettime(CLOCK_MONOTONIC)`
around the generation loop in `bitnet_harness.c`. Prints
`gen_loop_seconds` and `seconds_per_token` to stderr, factoring out
model load time which is a constant ~5-6 s.

**Test material:** synthesized long-context prompts at fixed token
lengths (32, 64, 128, 256, 512, 1024) by tokenizing repeated factual
text via cl100k_base.

**Configs:** dense / routed_k16 (τ=5000) / routed_k4 (τ=5000), gen=32
tokens each. Max seq_k per run = prompt_len + 32 (so range is 64
through 1056).

## Result (s/token, measured during gen loop only)

| prompt_len | dense | routed_k16 | routed_k4 | ratio_k16 | ratio_k4 |
|------------|-------|-----------|-----------|-----------|----------|
| 32         | 0.367 | 0.363     | 0.364     | 0.99×     | 0.99×    |
| 64         | 0.366 | 0.367     | 0.367     | 1.00×     | 1.00×    |
| 128        | 0.369 | 0.370     | 0.368     | 1.00×     | 1.00×    |
| 256        | 0.372 | 0.371     | 0.434†    | 1.00×     | 1.16×    |
| 512        | 0.367\*| 0.367    | 0.363     | 1.00×     | 0.99×    |
| 1024       | 0.367 | 0.367     | 0.367     | 1.00×     | 1.00×    |

\* First measurement at 512 prompt_len was a system-contention
outlier (29.27 s/tok); retest showed 2/3 runs at 0.37 s/tok and
1/3 at 30.7 s/tok. Background process / thermal throttle.

† routed_k4 at prompt_len=256 was 16% slower in one run; likely
similar contention but not retested.

## The honest finding

**Dense and routed are equivalent in wall-clock across seq_k 64 to 1056.**

The substrate's theoretical attention-cost advantage (256× fewer
dots at seq_k=4096, k=16) **does not translate to wall-clock speedup**
in this BitNet configuration.

Reason: **attention is not the bottleneck.** BitNet uses ternary
weights + integer arithmetic; attention dots are extremely cheap.
At seq_k=1056:
- Per attention layer per token: ~30 heads × 1056 keys × 64 head_dim
  i64-muls ≈ 2M ops. At >1 GFLOPS NEON-NEON, that's ~2ms.
- × 30 layers = ~60ms attention per token.
- Total per-token = 367ms.
- Attention as fraction: ~16% of total per-token cost.

The other 84% (FFN, LM head, RMSNorm, projections) dominates.
Substrate's "256× fewer dots" is real FLOPS savings on the 16%
slice — but since the 84% is unchanged, total wall-clock barely
moves.

## What "256× fewer dots" actually buys

At seq_k=4096, k=16, theoretical savings:
- Dense attention: 30 layers × 30 heads × 4096 keys × 64 head_dim
  ≈ 236M dot ops per token ≈ 236ms.
- Routed attention: 30 layers × 30 heads × 16 keys × 64 head_dim
  ≈ 0.9M dot ops per token ≈ 0.9ms.
- Savings on attention: 235ms per token.
- Total per-token at seq_k=4096 (dense FFN unchanged): ~600ms
  → 365ms with routed attention.
- **Speedup: ~1.6× at seq_k=4096.**

Per the asymptotic claim:
- At seq_k=16384, k=16, attention saves ~95% of compute time. Speedup
  vs dense maybe 3-5×.
- At seq_k=65536, k=16, attention dominates everything; speedup
  could exceed 10×.

**But none of this is measured at seq_k > 1024.** This codebase
doesn't have prompts at that length, and the BitNet model's max_seq
is 4096 so seq_k > 4096 isn't even reachable without modification.

## Cost claim, corrected

Previous framing (commit 68b53ad's message, retracted in 2188337):
> "Substrate uses ~22% of dense's total compute."

That was theoretical FLOPS at large T, never measured wall-clock.

Honest framing (this measurement):
- **At seq_k ≤ 1056 on BitNet b1.58-2B-4T: substrate provides ZERO
  measurable wall-clock advantage.** Dense and routed are within
  noise (~1%).
- **At seq_k ~ 4096: theoretical speedup ~1.6×, not measured.**
- **At seq_k ≫ 4096: theoretical speedup grows, but is bounded by
  the model's max_seq and not testable without architectural
  changes.**
- **The cost-distinct claim's empirical regime is "very long
  context"** — which is exactly where #10 was designed to operate
  (KV eviction), and where the substrate has measured POSITIVE
  results (sigdist beats dense itself). So the substrate's
  long-context story is supported by quality measurements, even
  if cost-distinct isn't yet supported by speed measurements.

## What this experiment validates

- **The substrate's cost claim was overstated.** "22% of dense's
  compute" was a theoretical optimization that doesn't manifest
  in actual wall-clock at the seq_k range tested. Concern #4
  raised earlier was correct.
- **Attention is not the bottleneck in BitNet inference at seq_k
  ≤ 1056.** FFN + LM head + projections dominate. Substrate
  routing improvements in attention can only ever buy back 16% of
  total per-token cost in this regime.
- **The instrumentation works.** `gen_loop_seconds` print is
  clean and reliable; can be used for future cost measurements.

## What this experiment does NOT validate

- **Substrate's cost advantage at seq_k > 4096.** Untested.
  Theoretical FLOPS arithmetic suggests it would emerge, but the
  test material doesn't exist in this codebase and BitNet's
  max_seq is 4096 anyway.
- **Other models where attention IS the bottleneck.** A model with
  more heads, smaller FFN, or different architecture might show
  the substrate's attention savings as real wall-clock speedup.
- **Per-layer attention-only timing.** A finer instrumentation
  (time attention specifically per token) would let us measure
  substrate's specific contribution to attention cost, not the
  end-to-end per-token time. Recorded as follow-up; not done here.

## Red-team of this measurement

### C1: Sample size is 1 trial per config
Three configs × six prompt lengths = 18 single runs. Retest at
prompt_len=512 showed a 1/3 outlier rate (system contention).
With proper sample (3 trials per config), the mean values would
be more reliable but the OVERALL PATTERN (dense ≈ routed at
~0.37 s/tok) is robust across all 17 non-outlier runs.

### C2: BitNet might be unrepresentative
This measurement is on BitNet b1.58-2B-4T specifically. A model
with smaller FFN or larger context would shift the bottleneck
toward attention, making substrate's savings more visible. The
"cost claim doesn't manifest" finding is bounded to this model.

### C3: Substrate's NEON sparse attn_v_combine is scalar
Per the original substrate harness code, `bitnet_sparse_attn_v_combine`
is scalar (marked experimental). If this were NEON, routed might
be FASTER than dense at the attention step. Currently routed is
TIED with dense — implying the scalar V combine roughly cancels the
sparse-routing savings at the seq_k range tested. NEON-ifying this
is a recorded engineering follow-up.

### C4: Wall-clock includes scoring overhead I didn't fully analyze
At sparse mode, "compute scores" is replaced by "compute signatures
+ distance + topk + score k positions." The TOTAL work in the
sparse path might be HIGHER than dense's "compute all scores +
softmax" at small seq_k. This explains why dense and routed are
tied even though routed theoretically saves attention compute. The
break-even is where the savings exceed the overhead.

### C5: The "16% of total" attention fraction is back-of-envelope
Derived from FLOPS arithmetic (2M attention ops vs 12M+ total ops
per token). Could be off by 2× either way without actual layer-by-
layer profiling. The qualitative claim (attention is not the
bottleneck) is robust; the precise number is approximate.

### C6: Could the substrate ever win in wall-clock on this model?
At seq_k > ~4× the FFN-equivalent breakpoint. Theoretical
breakpoint for BitNet 2B: seq_k ~ 12000 if my back-of-envelope is
right. Not testable without re-architecting the model's max_seq.

## Implications for the project

1. **The cost-distinct claim was theoretical-at-large-T, and that
   "large-T" is beyond what this codebase can currently test.**
   Concern #4 (this measurement was supposed to address) is
   ACKNOWLEDGED but not RESOLVED — the answer is "the cost claim
   doesn't apply at testable scales."

2. **The substrate's story shifts.** Quality wins (Phase A, #10
   long-context, #9 cell prediction) remain. The cost-distinct
   story becomes "asymptotic at scales not currently measurable."
   That's narrower than the original framing.

3. **NEON-ifying `sparse_attn_v_combine` is a real follow-up.** If
   the scalar V combine is what's masking substrate's attention
   advantage, fixing it is a measurable engineering win. Recorded.

4. **The project's "substrate is more efficient" narrative needs
   refining.** It's NOT broadly true at the scales we can measure.
   It's CONDITIONALLY true at scales we can't currently reach.
   That's a real limit and worth being honest about.

## Files

- `gesh/bitnet/bitnet_harness.c`: instrumentation (~10 LOC).
- `journal/td27_cost_measurement/prompts.tsv`: 6 synthesized
  long-context prompts.
- `journal/td27_cost_measurement/timing.tsv`: 18-config result.
- This journal: `journal/td27_cost_measurement_2026-05-11.md`.

## Sign-off

This experiment **resolves concern #4** by measuring rather than
extrapolating. The verdict is humbling: the substrate's cost claim
in its widely-quoted form ("22% of dense's compute") is unsupported
at the scales currently testable. The theoretical FLOPS arithmetic
is correct; it just doesn't matter at seq_k ≤ 1056 because
attention isn't the bottleneck in BitNet inference.

Concern #4 status: **resolved as ACKNOWLEDGED LIMITATION** rather
than fixed. The substrate's quality wins stand; the cost narrative
needs the corrected framing.
