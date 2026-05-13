# c_dump_v3 provenance audit (Track C1 + C5)

Per `glyph_gaps_2026-05-13_synthesize.md` Track C1, verify whether
the Phase α-ε oracle activations (data/c_dump_v3) were generated
from natural-language or gibberish prompts.

## Method

The c_dump_v3 files are activation dumps (ACTV2 format) keyed by
(prompt, position, layer). They are not in git, and the script
that generated them is also not in git (data/ is gitignored).

Direct test: compare the layer-0 `x_norm_input` field of
`long64.pos0.layer0.bin` against fresh harness dumps run with
two candidate BOS tokens — `1` (the gibberish-tokenizer BOS used
by the original 5-prompt eviction battery) and `128000` (the
correct LLaMA-3 tokenizer BOS).

## Result

```
c_dump_v3 long64 pos0 x[0..7]: [-155, -1, 40, 0, -8, -1, 178, 0]
BOS=1            pos0 x[0..7]: [-155, -1, 40, 0, -8, -1, 178, 0]
BOS=128000       pos0 x[0..7]: [-2, -1, 17, 0, -26, -1, -73, -1]

L1(c_dump_v3, BOS=1)      = 0       ← bit-identical
L1(c_dump_v3, BOS=128000) = 137916
```

All 5 c_dump_v3 prompts (long64, long_a, long_b, long_c, long_d)
share an identical layer-0 pos-0 vector with each other (pairwise
L1 = 0), confirming they share the same BOS.

**c_dump_v3 was generated with BOS=1, NOT BOS=128000.** The Phase
α through ε oracle measurements were on the same gibberish-
tokenizer regime that the plan B red-team R-B4 already
demonstrated produces semantically nonsensical inputs ("capital_france"
decoding to `'" car<p {\n minorobject'`).

## Codebase audit (Track C5)

Files using BOS=1 with hard-coded gibberish-tokenizer token IDs:

- `experiments/phase_alpha/regenerate_dumps_v2.sh` — generates
  data/c_dump_v2 from the 5 gibberish prompts (token IDs 1,1841,...).
- `experiments/phase_zeta/eviction_battery.py` — original 5-prompt
  harness battery. Gibberish.
- `experiments/phase_zeta/eviction_battery.sh` — bash sibling of
  the above. Gibberish.
- `experiments/phase_zeta/perstep_probe.py` — per-step probe used
  the same 5 prompts. Gibberish.

Files using BOS=128000 with proper tokenization:

- `experiments/phase_zeta/redteam_b_harness.py` — 20-prompt
  natural-language battery added 2026-05-13.
- `experiments/phase_zeta/tokenize_prompts.py` — tokenizer script.

The gibberish usage is concentrated in early-arc files (Phase α,
Phase ζ original). All post-2026-05-13 work uses natural language.

## What this invalidates

The following oracle numbers from prior journals are on
**out-of-distribution gibberish inputs** and need re-validation
before further claims rest on them:

1. **Phase α `td28_phase_alpha_methodology_pivot`** — methodology
   pivot was correct in spirit; numbers were on gibberish.
2. **Phase β `td28_phase_beta_synthesis`** — "M1 estimator
   reverses under L1 distance." On gibberish.
3. **Phase γ `td28_phase_gamma_robustness`** — robustness battery
   findings. On gibberish.
4. **Phase δ `td28_phase_delta_application`** — substrate-L1
   eviction +3.2pp recall@k=0.5. On gibberish.
5. **Phase ε `td28_phase_epsilon_application`** — "L1 reduces
   attn-output L2 error by 38-62% vs Hamming, 35× vs random."
   **THE HEADLINE NUMBER OF THE ENTIRE ARC IS ON GIBBERISH.**
6. **Plan A red-team `td28_phase_zeta_planA_redteam`** — "K-K
   eviction is 2× worse than random in single-shot L2." On
   gibberish c_dump_v3 activations.
7. **Plan B sanity `qsigdist_oracle_sanity.py`** — "qsigdist is
   10× better than random in single-shot oracle." On gibberish.
8. **Plan B trajectory `redteam_b_trajectory.py`** — "qsigdist
   cumulative L2 is 6.8× smaller than random." On gibberish.

## What survives

The only natural-language territory data is from
`redteam_b_harness.py`: the N=20 battery at window=16 showed:

- qsigdist match% = 57.1%, Δ vs random = +6.0pp (CI [-5.6, +18.1])
- sigdist match% = 50.8%, Δ vs random = -0.2pp
- fifo match% = 52.5%, Δ vs random = +1.5pp

This is the ONLY substrate-eviction finding currently grounded on
natural-language inputs. It is the ONLY one that should be cited
in future synthesis without an "on gibberish, needs re-validation"
caveat.

## What this means for the gaps plan

`glyph_gaps_2026-05-13_synthesize.md` predicted this branch:
"If gibberish: flag Phase α-ε oracle numbers for re-validation."
The branch fires.

This does NOT change the synthesis's priority ordering. Track A
(claim 2 bridge) is still the highest-leverage move; the substrate
oracle's gibberish-ness reinforces that "more eviction work" is
diminishing returns. The natural-language N=20 result stands as the
arc's actual closing position on the eviction territory.

The next eviction-related work should NOT cite Phase ε's 38-62%
number as motivation. The motivation is the +6pp natural-language
trend, which is statistically inconclusive but real.

## Action items

1. ~~Decode c_dump_v3 BOS to confirm regime~~ — done (BOS=1, gibberish).
2. ~~Audit other codepaths for BOS=1 usage~~ — done; 4 files identified.
3. **Update `feedback_proxy_to_territory_pattern.md`** to note the
   territory-validation also requires input validation.
4. **Update the journal INDEX (Track C2)** to flag retracted claims
   #1-#8 above with this provenance note.
5. **Decide:** regenerate c_dump_v3 with natural-language prompts,
   or accept that the oracle infrastructure is no longer cited in
   substrate-claim synthesis. Recommendation: **accept**. The
   oracle was an intermediate measurement; the territory test
   (redteam_b_harness) is what matters for the substrate-claim
   verdict.

## Discipline

The synthesis's "validate before mechanism" memory entry was
written 2026-05-13 morning based on the 5-prompt-battery gibberish
discovery. This audit shows the gibberish reach was broader: the
ORACLE activations themselves were on gibberish. Two layers of the
measurement chain were on OOD inputs, not one.

Updating `feedback_validate_input_before_mechanism` to emphasize
**every** input layer needs validation, not just the final
benchmark.
