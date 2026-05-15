# Red-team of Step 1 / Step 1.5a — fallback confound found

**Date:** 2026-05-15
**Companions:** `journal/lsh_ffn_real_b1_2026-05-14.md` (B1 dispatch
validation); commits cda2303 → 7dd0536 (Step 1 PoC, Step 1.5a
hybrid). Today's red-team scrutinizes those commits at 100/100
intensity.

## What was claimed yesterday

Yesterday's commits:
- Step 1 PoC: harness integration of routed FFN. End-to-end measurement
  at L15-only gave 0.348 mean match-rate vs dense.
- Step 1.5a: n_min sweep showed mean match jumped to 0.596 at n_min=10.
  Framed as: "Hypothesis CONFIRMED. Per-bucket data was the bottleneck."

Today's red-team examines six angles. The mechanical correctness
findings hold; the quality framing was overclaimed.

## Red-team scorecard

| # | Test | Verdict |
|---|---|---|
| 1 | Fallback path bit-identical to dense | **PASS** |
| 2 | Routing counter works; reports routed-vs-fallback fraction | **PASS** |
| 3 | Dense baseline reproducible bit-identically | **PASS** |
| 4 | Python ↔ C lookup math bit-identical (9/9 buckets) | **PASS** |
| 5 | Per-prompt attribution — is the 0.596 fallback or routing? | **PARTIAL** — fallback-dominated |
| 6 | Routing quality vs routing fraction correlation | **NEG** — r = -0.455 |

## RT#1 — Fallback bit-identicality (PASS)

Built an empty dict (`n_min=99999` → 0 recipes for all 608 buckets at
L15). Ran harness with L15 active. Generated tokens bit-identical to
dense across multiple prompts and at L2/L15/L27 simultaneously.

→ When recipe_len == 0, harness falls through to dense FFN
correctly. The fallback path is verified.

## RT#2 — Routing counter (PASS)

Added `g_lsh_routed_count` / `g_lsh_fallback_count` increment in the
harness; prints per-run summary line. Verified counters are
correctly maintained:
- n_min=1 dict (all populated buckets in dict): 96% routed on a real prompt
- n_min=10 dict: 13.8% routed (most positions hash to sparse-bucket → fallback)
- empty dict: 0% routed

## RT#3 — Dense reproducibility (PASS)

Same prompt, same env, three runs → bit-identical generated tokens.
qsigdist eviction is deterministic; no hidden seed entropy.

## RT#4 — Python ↔ C consistency (PASS)

Built Python replica of harness's runtime routed FFN compute (bucket
hash + recipe lookup + prediction reconstruction). Loaded the dict
in Python, dumped post-routed `s->x` from harness on 9 different L15
positions during a routed run, compared:

```
9 (input, output) pairs at L15:
  pos=0000 bucket=301: max_abs_diff=0, n_disagreeing_dims=0/2560
  pos=0001 bucket=233: max_abs_diff=0, n_disagreeing_dims=0/2560
  ...
  pos=0008 bucket=565: max_abs_diff=0, n_disagreeing_dims=0/2560
```

Predictions bit-identical across all 9 buckets / 2560 dims each.
Bucket assignment math correct (Python and C agree on trit
boundaries: v > tau / v < -tau / else). Atom × scale reconstruction
correct (manual verification of first 4 dims for one bucket
matched automated prediction byte-for-byte).

→ The harness lookup computes EXACTLY what Python calibration
intended. No numerical drift, no encoding error.

## RT#5 — Per-prompt attribution (PARTIAL)

Re-ran N=20 prompts × {dense, n_min=1, n_min=10}. Captured per-prompt:
- routing fraction (from counter)
- match-rate vs dense

**n_min=10 (headline config):**

```
Mean routing %: 13.4
Mean match: 0.596
Correlation routing% vs match: r = -0.455
```

Negative correlation. More routing → worse match. Two prompts that
got perfect 1.000 match scored only 4.1% (poetry_blake) and 4.8%
(code_python_fn) routed — essentially dense fallback.

Stratified:
- 8 prompts with routing < 10%: mean match **0.667**
- 12 prompts with routing >= 10%: mean match **~0.547**

The 0.596 mean was inflated by low-routing prompts trivially matching
dense.

**n_min=1 (all-routed comparison):**

```
Mean routing %: 96.2
Mean match: 0.354
Correlation: r = -0.174
```

When routing fires on nearly every position, mean match is only
0.354 — most generated tokens differ from dense.

## RT#6 — Routing quality itself (NEG)

The negative correlation in RT#5 says: per-prompt, more routing
means worse match. Adding the n_min=1 vs n_min=10 comparison
quantifies the gap:

- 96% routed (n_min=1):  mean match 0.354
- 13% routed (n_min=10): mean match 0.596
- 0% routed (dense):     mean match 1.000

The difference between n_min=10 (0.596) and n_min=1 (0.354) of
+0.242 has two contributions:
- **Selecting good buckets** (real signal): n_min=10 only routes
  buckets with enough calibration data
- **Routing less** (fallback win): n_min=10 routes 13% vs 96%

These aren't separated. But the negative correlation within
n_min=10 alone (r = -0.455) tells us: even AMONG well-trained
buckets at n_min=10, when routing fires it hurts more than it
helps.

## Honest conclusion about Step 1.5a

**Mechanical correctness: VERIFIED.** RT#1-#4 all pass. The LSH
dispatch + atom-composition lookup + fallback mechanism is
correctly implemented end-to-end.

**Quality claim: PARTIALLY REFUTED.** Yesterday's framing —
"Hypothesis CONFIRMED. Per-bucket data was the bottleneck." — was
overclaimed. The 0.596 headline at n_min=10 was confounded with
fallback dominance. The architecture's TILE FIDELITY (atom-comp
M=16 K=4) doesn't preserve generation match-rate well enough; even
at 13% routing the per-prompt correlation with quality is negative.

The honest reframe:
- The architecture WORKS MECHANICALLY (dispatch + lookup).
- The TILE CONTENT (atom-composition with 16 atoms × 4 coefficients)
  produces predictions whose accumulated error breaks downstream
  token generation.
- Higher tile fidelity is needed before the architecture has a
  real shot at preserving dense-equivalent inference quality.

## Methodical remediation — three actions

### 1. Update yesterday's CHANGELOG entry with the red-team finding

The CHANGELOG entry from `cda2303` framed Step 1 as a positive
PoC. The CHANGELOG should reference this red-team and note that
the architecture is mechanically validated but quality-bounded by
tile fidelity.

### 2. Add a "Step 1.5a errata" entry to the journal arc

This file IS that errata. The Step 1.5a journal stands as a
historical record but is amended by this red-team.

### 3. The architectural path forward changes

**Variant B2-b (per-bucket subset of dense FFN)** was previously
described as "an alternative architecture." After today's red-team
it becomes the ONLY architecturally viable path forward for
generating-quality-preserving routed FFN.

Why: variant B2-b's tile content is a SUBSET of the dense FFN's
exact compute, not a statistical approximation. The selected cells'
contribution to output is computed exactly via the substrate's
existing primitives (gate/relu²/up/sub_norm/down on the subset).
Unselected cells contribute zero. The fidelity loss is therefore
bounded by the selected cells' coverage of the total FFN output —
a controllable parameter (subset size), not a statistical
estimation error.

If we select K=512 of 6912 intermediate cells per bucket, the
tile preserves ~7% of dense FFN's compute. If that 7% covers the
dominant output components, generation quality should be much
closer to dense than the atom-comp prediction error allows.

## What to do next

**No more atom-comp iterations.** The atom-comp tile design is
data-bound AND fidelity-bound — both bottlenecks compound. Even
with infinite calibration data, the M=16 K=4 approximation can't
reach dense fidelity. And scaling M, K dramatically (M=256, K=64)
makes the dict 100x bigger AND still doesn't guarantee dense match.

**Variant B2-b becomes Step 1.5b.** Scope:
1. Instrument harness to dump per-(layer, position) gate_act
   activations (need new dump path)
2. Per-bucket: identify top-K cells by mean |gate_act| across
   bucket members (= "the cells that fire on this bucket's inputs")
3. Build a sparse-FFN compute path: gate/up/down restricted to
   the selected cells per bucket
4. Quality measurement with the same N=20 protocol

Estimated cost: ~5-7 hr (gate_act dump + sparse-FFN compute in C +
calibration script for cell selection + battery).

## Files

- `experiments/phase_eta/redteam_python_c_consistency.py` — RT#4
- `experiments/phase_eta/redteam_attribution.py` — RT#5+#6 + counter use
- `gesh/bitnet/bitnet_harness.c` — added routing counters,
  fixed output-dump location, fallback consistency verified
- `experiments/phase_eta/results/lsh_ffn_dict_nmin{3,5,10}.bin` — built
  via `--n-min` flag added to build_lsh_dict.py
- `/tmp/redteam_attribution.log` — raw RT#5+#6 output (5642s of harness time)

## What I'm NOT walking back

The B1 / B1.5 / B1.6 results (LSH dispatch IS faithful, addresses
data geometry) remain correct. Those measured the dispatch in
isolation, not end-to-end. The dispatch mechanism's faithfulness
to activation geometry is a real architectural property.

What I AM walking back: the Step 1.5a framing that the n_min sweep
"confirmed the per-bucket data hypothesis." It demonstrated the
n_min knob exists and reduces routing fraction, but didn't isolate
"good routing on well-trained buckets" from "less routing falls
back to dense."

The substrate vision is unchanged. The next concrete step (B2-b)
is sharper-defined as a result of this red-team.
