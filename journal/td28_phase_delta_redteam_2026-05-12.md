# Phase δ red-team — direction survives, magnitude and recommendation revised

User directive: "red-team with extreme prejudice." Three issues tested
directly; five more flagged. The δ-1 KV-eviction direction survives.
The δ-2 magnitude is partly estimator offset. The "production should
switch to L1" recommendation was an overclaim — it's not supported by
this measurement alone.

## RT-A: prompt-clustered bootstrap

The δ-1 flat-trial bootstrap used n=2250 trials but the trials are
correlated — 7 source prompts × ~50 (layer, kv_head, position) trials
per prompt × strong intra-prompt correlation. Effective sample is closer
to the prompt count than the trial count. Re-bootstrap by resampling
prompts:

```
                       flat (n=2250)        prompt-clustered (n=4)   leave-one-prompt
  k_frac=0.25         CI=[+0.008, +0.020]   CI=[+0.003, +0.021]      range=[+0.007, +0.019]
  k_frac=0.50         CI=[+0.026, +0.038]   CI=[+0.014, +0.052]      range=[+0.022, +0.037]
  k_frac=0.75         CI=[+0.027, +0.035]   CI=[+0.021, +0.037]      range=[+0.029, +0.034]
```

**Verdict: direction survives, CIs are 2-4× wider than reported.** The
flat-bootstrap CIs in the δ-1 commit were too tight. The clustered
bootstrap shows the L1 > Hamming finding holds (all CIs above 0, all
LOO subsets positive), but the magnitude uncertainty is larger.

Honest restatement at k_frac=0.5: **L1 beats Hamming by +1.4 to +5.2pp
recall@k**, depending on which prompts dominate the corpus. The 95%
interval is wide because n_prompts is small.

Only 4 unique prompts contribute (multitoken, p1, p4, p5) — three of
the 7 corpus prompts had insufficient cache size at k_frac levels
tested. This is a real limitation: **the result is measured on 4
prompts**, not 7.

## RT-C: estimator offset on uniform random data — partial falsification

The δ-2 finding of 300/300 paired comparisons with substrate < B4 was
attributed to substrate's structural advantage. But the Macocco
estimator might just systematically report lower d̂ for L1-on-ternary
than for Hamming-on-binary, irrespective of data.

Test: generate uniform random K-cache (no manifold structure; true
intrinsic dim = ambient cell count = 128), measure all three reps:

```
    N    sub_L1_d̂   B0_Ham_d̂   B4_PCA_d̂
  200      129.4       99.8      146.0
  500      128.6       98.7      142.5
 1000      127.8       98.6      141.6
```

On uniform random data, substrate_L1 d̂ ≈ 128 (close to true), but
B0_Ham underestimates (~99, expected for Hamming-on-substrate
since substrate has 60% nonzero → Hamming sees ~60% of cells as
"active") and B4_PCA overestimates (~142-146, since random Gaussian
projection of random ternary is ~maximally uninformative).

**Substrate_L1 d̂ is ~14 LOWER than B4 d̂ on uniform data with no
structural advantage.** This is **estimator offset**.

Implication for δ-2's "300/300 paired substrate < B4" finding:
- On real K-cache: substrate=70, B4=190 → gap=120
- On uniform random: substrate=128, B4=142 → gap=14
- **Structural component: gap_real − gap_uniform = ~106 d̂ units**

The substrate IS structurally compressed relative to B4 on K-cache
data, but by less than the raw 120 suggested. ~14 of the 120 is just
the estimator's L1-vs-Hamming offset; ~106 is real structure.

Also: substrate < B0_Ham on real K-cache (98 < 105), but substrate
> B0_Ham on uniform data (128 > 99). **The L1 metric IS doing real
work over Hamming on substrate signatures** — the direction reverses
on uniform data, meaning real K-cache has structure that the L1
metric exploits and the Hamming metric misses.

## RT-E: L1 kernel cost is much higher than Hamming

The "production should switch to L1" recommendation didn't consider
that L1 is a fundamentally more expensive operation. Measured cost
in Python/NumPy reference (production NEON would differ, but the
asymptotic gap remains):

```
    N    D    Ham_ms    L1_ms    L1/Ham
  200  128      0.22     1.79     8.0x
 1000  128      2.34    67.75    29.0x
 4096  128     63.93  1169.00    18.3x
```

L1 is 8-29× more expensive than Hamming at production-relevant N.
Hamming has a hardware popcount fast-path; L1 requires byte
subtraction + abs + horizontal sum.

**The +3pp recall lift from L1 is paid for with 10-30× more compute
per distance comparison.** At seq_k=4096 (BitNet's typical context),
this could be 1+ second per attention step just for the eviction
computation — likely net-negative versus the recall lift.

The honest version of the recommendation: **L1 needs a NEON-optimized
implementation and end-to-end cost-benefit measurement before any
production switch.** Until then, L1-substrate KV-eviction is a
research finding, not a shippable architecture change.

## Untested concerns (acknowledged)

The following red-team concerns were not directly addressed in this
pass; recording them as known gaps:

1. **Q-head averaging.** δ-1 averages the 4 Q-heads sharing each
   kv_head before computing the oracle. Real attention computes
   per-Q-head separately. The averaged oracle is not the model's
   actual attention pattern. Could shift the result.

2. **Tiny cache sizes.** The dumps cover prompts of 1-11 positions;
   δ-1 measures eviction at cache_size ≤5 (k_keep=2 at k_frac=0.5).
   Real KV-eviction matters at seq_k > 1000. The geometric advantage
   at very short context may not extrapolate.

3. **Recall@k is not softmax-mass preservation.** Attention output
   depends on softmax-weighted sums. Preserving high-rank K's that
   have low absolute attention mass doesn't help generation quality.
   The right downstream metric is softmax(Q·K)·V preservation, not
   rank overlap.

4. **No generation-quality validation.** The acid test is "does
   L1-substrate eviction produce more coherent inference than
   Hamming-substrate?" That requires running the full harness with
   each policy on a real prompt battery and measuring generation
   metrics. Not done.

5. **P3 robust finding is a marginal-statistics effect.** Per Phase γ's
   shuffled-K control (γ-F), P3 survives shuffling that destroys
   learned structure. The "centrality of 0 is geometric" framing is
   really "more-common cells as center make for compact metrics on
   asymmetric marginals." Real but mechanistically pedestrian.

## Revised verdict on Phase δ

**δ-1 direction:** L1 beats Hamming on KV-eviction recall@k. **Robust
to prompt-clustered bootstrap, robust across all 4 LOO subsets.** The
direction-of-effect is the strongest finding of the entire arc.

**δ-1 magnitude:** the +3pp recall lift at k_frac=0.5 is correct on
the measured data but the CI is wider than reported (clustered
[+0.014, +0.052] not flat [+0.027, +0.039]). Range: +1.4 to +5.2pp.

**δ-2 magnitude:** the 120-d̂-unit gap between substrate and B4 on
real K-cache includes ~14 d̂ units of estimator offset (per RT-C on
uniform data). Real structural component is ~106 d̂ units — still
substantial.

**δ-3:** confirmed, biases are asymmetric. Reinforced by RT-C.

**Production recommendation:** the "switch to L1 in production" line
from the δ commit was **an overclaim.** Justifications still needed:
- NEON-optimized L1 kernel timing (RT-E showed L1 is 10-30× slower
  in reference impl; NEON might close some but not all of the gap).
- Generation-quality test (does L1 eviction produce more coherent
  output than Hamming on a real prompt battery?).
- Long-context validation (does the +3pp recall lift hold at
  seq_k=1000+?).

Restated: **L1-substrate KV-eviction is a promising research direction
with one positive measurement against one quality proxy on a narrow
corpus. It is not yet production-ready as an architecture change.**

## Discipline log

16th caught misalignment of the arc. Pattern: I keep claiming positive
verdicts based on a single measurement, and red-team keeps deflating
the strong form. The deflations don't kill the direction; they
narrow the magnitude and qualify the recommendation. Each pass
preserves the core finding while shedding the overclaim layer.

**The core finding survives:** the substrate's L1 cell-graph metric
captures structure that categorical Hamming on the same signatures
does not, and this translates to a real (small, paid-for) advantage
on a KV-eviction-relevant metric. **Magnitude is methodology-bounded,
direction is robust, application impact requires more work to
characterize.**

## Files

- `experiments/phase_delta/redteam_checks.py` — RT-A, RT-C, RT-E
  implementations.
- `experiments/phase_delta/results/redteam.json` — numeric results.
- `experiments/phase_delta/results/redteam_log.txt` — archived output.
