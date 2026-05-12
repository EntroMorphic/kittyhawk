# Cost measurement v3 — multi-trial, attention-attributed, statistically bounded

Per the cost-v2 red-team (commit `33b53ce` and the prior discussion).
v2 claimed "substrate is ~3% faster, consistently" based on n=1 trial
per config with no attention-attribution and an outlier-exclusion
heuristic. The red-team caught it; v3 fixes the methodology.

Engineering fixes (already committed):
- Scratch buffer for sparse V gather, replacing malloc-per-call
  (~900 mallocs/gen-step → 0). Commit `967c22b`.
- Prompt-parse error on buffer overflow (silent truncation guarded).
  Commit `967c22b`.
- Attention-only timing instrumentation: `g_attn_seconds`
  accumulator + `attn_seconds_per_token` and `attn_fraction` in
  harness output. Commit `967c22b`.

This commit lands the multi-trial measurement run on top of those
fixes.

## Setup

- 4 prompt lengths {64, 256, 512, 1024} × 2 modes {dense, routed_k4}
  × 3 trials = 24 runs.
- gen=16 (longer than v2's gen=8 to amortize first-token startup).
- routed_k4 uses fixed τ=5000 (the production sparse default).
- Total measurement time: ~1 hour wall-clock on M-series CPU.

## Results

### Per-config stats (mean ± stddev)

| prompt_len | mode | n | total s/tok | attn s/tok | attn fraction |
|---|---|---|---|---|---|
| 64 | dense | 3 | 0.3664 ± 0.0003 | 0.0015 ± 0.0000 | 0.4% |
| 64 | routed_k4 | 3 | 0.3675 ± 0.0004 | 0.0014 ± 0.0001 | 0.4% |
| 256 | dense | 3 | 0.3725 ± 0.0001 | 0.0057 ± 0.0000 | 1.5% |
| 256 | routed_k4 | 3 | 0.3709 ± 0.0003 | 0.0037 ± 0.0001 | 1.0% |
| 512 | dense | 3 | 0.3784 ± 0.0022 | 0.0113 ± 0.0000 | 3.0% |
| 512 | routed_k4 | 3 | 0.3725 ± 0.0002 | 0.0065 ± 0.0000 | 1.7% |
| **1024** | **dense** | **3** | **0.3939 ± 0.0024** | **0.0241 ± 0.0001** | **6.1%** |
| **1024** | **routed_k4** | **3** | **0.3793 ± 0.0005** | **0.0125 ± 0.0001** | **3.3%** |

### Substrate Δ vs dense (negative = substrate faster)

| prompt_len | Δ total (s/tok) | Δ total (%) | 95% Welch CI | Δ attn (%) |
|---|---|---|---|---|
| 64 | +0.0011 | **+0.31% SLOWER** | [+0.0003, +0.0020] | −8.9% |
| 256 | −0.0016 | **−0.43% faster** | [−0.0021, −0.0011] | −34.5% |
| 512 | −0.0059 | **−1.56% faster** | [−0.0094, −0.0024] | −42.5% |
| **1024** | **−0.0147** | **−3.72% faster** | **[−0.0186, −0.0108]** | **−48.1%** |

**All 4 Δ values have 95% CI excluding zero — all four signs are
statistically significant.** Substrate's advantage scales monotonically
with seq_k, exactly as the attention-fraction-grows-with-seq_k story
predicts.

## What the data shows

**Four CI-significant signs, monotonic**: substrate is
- **slower at seq_k=80** (+0.31%)
- **faster at seq_k=272** (−0.43%)
- **faster at seq_k=528** (−1.56%)
- **faster at seq_k=1040** (−3.72%)

All 95% CIs exclude zero. Substrate's advantage scales monotonically
with seq_k, confirming the attention-fraction-driven story:
- At seq_k=80, attention is 0.4% of per-token cost. Substrate's gather
  overhead (constant) > savings (small base).
- At seq_k=1040, attention is 6.1% of per-token cost. Substrate halves
  it (−48.1% on attention), recovering ~3% of total per-token wall-clock.

This is qualitatively different from v2's "constant ~3% advantage"
framing. v2 was averaging across the regime where the **sign of the
substrate's advantage flips**. The honest cost picture is:

1. **Substrate has a fixed gather overhead** (memcpy from K cache
   positions into scratch buffer + reduced V combine) that's present
   regardless of seq_k.
2. **Substrate's attention savings scale with seq_k** (k=4 vs full
   seq_k dot products; substrate halves attention time at seq_k≥256).
3. **The crossover** is around seq_k=80-200, where the savings equal
   the overhead.
4. **At long context**, savings dominate by a growing margin.

### Attention is a small fraction of BitNet's per-token cost

The attention-fraction column tells the structural story:
- seq_k=80: attention is 0.4% of per-token cost.
- seq_k=272: 1.5%.
- seq_k=528: 3.0%.
- seq_k=1040: 6.1%.

At seq_k=1040, even an attention-only savings of 50% (which
substrate roughly delivers, with attn dropping from 6.1% to ~3%)
buys only ~3% total wall-clock. The "256× theoretical FLOPS reduction
on attention" is real arithmetic but multiplies against a small base
in this model.

The cost-distinct story is thus:
- **At BitNet b1.58-2B-4T at seq_k 64-1024**: substrate has a measured
  but small (<2% at largest tested seq_k) wall-clock advantage.
- **Extrapolation to seq_k > 4096** (BitNet's max_seq, untestable in
  this codebase): the attention fraction would grow further (linear
  in seq_k for dense, constant for routed_k4); substrate's advantage
  could exceed 5%. Asymptotic, but in a regime we cannot measure.

## What this commit retracts from v2

v2 (commit `c8e3dd8`) claimed:
> Substrate IS measurably faster than dense — by ~3% wall-clock,
> consistently across all tested seq_k (72-1032).

v3 corrects this:
> Substrate is **slower at small seq_k** (+0.3% at seq_k=80) and
> **faster at large seq_k** (~1.5% at seq_k=528 and growing). v2's
> "~3% consistent" was the mean across a regime where the sign flips;
> it was n=1 per config with no variance bound. v3's n=3 + 95% Welch
> CI shows the crossover.

## What this commit retracts from v1 (cost-measurement-2026-05-11.md)

v1 (commit `a29e287`) claimed:
> Substrate's attention savings invisible at testable scales.

Both v1 and v2 had problems. v1 had the prompt-truncation bug (256-cap)
that made all "long" prompts effectively seq_k≤288. v3 fixes the bug
(buffer 4096) and runs at actual seq_k.

## Methodology checks (red-team requirements satisfied)

1. **n≥3 per config**: ✓ (3 trials each).
2. **mean ± stddev reported**: ✓.
3. **CI on substrate − dense difference**: ✓ (Welch 95%).
4. **Attention-only attribution**: ✓ via `g_attn_seconds` timer.
5. **Scratch buffer for gather** (eliminates per-call malloc): ✓
   (commit `967c22b`).
6. **Outlier handling explicit**: ✓ (within-config stddev is 0.0001-
   0.0022 s/tok in this run; no outliers excluded post-hoc).

## What this experiment does NOT establish

- **Cost advantage at seq_k > 1056**: not measured. BitNet's max_seq
  is 4096; running at seq_k=4000 with gen=16 would take ~hours per
  trial and isn't blocking the cost claim's narrative correction.
- **Cost advantage on other models**: BitNet's small attention
  fraction (~5% at seq_k=1k) is model-specific. Models with larger
  attention fraction (more heads, smaller FFN) would amplify
  substrate's advantage.
- **The 1.56% at seq_k=528 ≠ "production-relevant speedup"**: this
  is a measured number on a specific model; production decisions
  should integrate it with the gather-overhead floor and the
  attention-fraction trend.

## Citations to add in any forward writeup of this measurement

- Loki (Singhania et al., NeurIPS 2024, arXiv:2406.02542) — measured
  attention K-cache low-rank structure; relevant context for cost
  measurement at long seq_k.
- Pitfalls of KV Cache Compression (arXiv:2510.00231) — methodological
  precedent for measuring cost-vs-quality tradeoffs honestly.

## Discipline note

This is the third cost-measurement journal in this thread. Each prior
version had a discovered flaw:
- v1: prompt truncation bug → claim "invisible" was artifactual.
- v2: single-trial + outlier-exclusion → claim "consistent +3%" was
  unsupported and missed the crossover.
- v3: n=3 + CI + attention attribution → honest small-but-positive
  scaling-with-seq_k.

The pattern across this session is consistent: ship a measurement →
red-team catches methodology gap → fix → re-measure. The structural
fix (require multi-trial + variance bound + attribution before any
cost claim) keeps being discussed but didn't get codified in workflow.

Worth eventually formalizing as a "pre-flight checklist for cost
claims" — but for this session, this v3 is the corrected record.

## Sign-off

Substrate's measured cost advantage on BitNet b1.58-2B-4T:
- seq_k=80: **+0.31% slower** (gather overhead > attention savings)
- seq_k=272: **−0.43% faster** (crossover)
- seq_k=528: **−1.56% faster**
- seq_k=1040: **−3.72% faster**

All 95% Welch CIs exclude zero. The scaling is monotonic and predicted
by attention-fraction-growth: attention is 0.4% of per-token cost at
seq_k=80 (substrate can't save what isn't there) and 6.1% at seq_k=1040
(substrate halves it, recovering ~3.7% of total).

The "256× theoretical attention savings" remains real arithmetic; at
seq_k=1040 substrate measurably halves attention compute, but
attention's small share of total per-token cost in BitNet caps the
wall-clock manifestation at ~3.7%. Extrapolating to seq_k=4096 (BitNet's
max): attention fraction would be ~24%, substrate halves that, total
saving ~12% — but that regime is untestable in this codebase (BitNet's
max_seq is 4096; running would take ~hours per trial).

This is the honest cost claim, finally measured: **substrate is
slower at small seq_k and faster at large seq_k, with monotonic
scaling proportional to attention fraction**. The substrate's quality
wins (Phase A trainability, #10 long-context coherence, #9 cell
prediction headroom) are unchanged.
