---
date: 2026-04-24
scope: LMM cycle — base-3-native benchmark selection
phase: SYNTHESIZE
---

# SYNTHESIZE: pivot from image-classification canon to base-3-native benchmarks

## The reframe

We were measuring the substrate on the wrong data. The MNIST → Fashion → CIFAR ladder was chosen because it is universal, not because it validates a **routing-first, base-3, inspectable** substrate. All three benchmarks have continuous float inputs (representation tax), balanced uniform classes (no routing pressure), and top-1-accuracy-only evaluation (no inspectability credit). The substrate's signature properties compete on zero axes there.

NORTH_STAR asserts routing is essential *in base-3*. Image-classification canon is base-2-float canon. Validating a base-3 claim on base-2 data has been the category error driving the last year's struggles against SSTT.

**This cycle commits to a pivot**: primary benchmark direction is where the substrate's properties are the *reason* the benchmark is chosen. Image benchmarks remain as regression suites only.

## Decision

### Primary direction: **ternary-state board-game position evaluation**

Go position state — `{empty, own, opponent}` — is natively ternary, with 361 trits per 19×19 board and zero quantization loss. Phase structure (opening / middlegame / endgame) makes routing load-bearing by construction — different tiles should fire for fuseki vs life-and-death vs yose. Existing datasets (KGS/OGS game records, pro tournament positions, engine-generated positions) remove the need for self-play infrastructure.

This is the substrate-claim-richest candidate that is also reachable with current capabilities. A value net or policy classification head on pre-ternarized positions can run the existing `direct_lsh` pipeline today with only a loader.

### Diagnostic: **custom synthesized benchmark with tunable routing load**

Controlled-property dataset where we can *turn routing pressure up and down* (parameter: fraction of inputs needing specialized tile) and *turn ternary-information completeness up and down* (parameter: what fraction of signal survives trinarization). Used to isolate whether claims about routing/base-3 advantage are mechanism-true or dataset-accident. Not a destination benchmark; an instrument.

### Regression suite (demoted, not abandoned)
MNIST, Fashion-MNIST, CIFAR-10 stay in the test matrix as "no substrate change should regress these." They no longer drive the work; they guard it.

### Explicitly deferred
- Tabular classification — good second direction, but weaker substrate claim (XGBoost is its own routing incumbent). Revisit once games prove the thesis.
- NLP / extreme-classification / compositional — require embedding or seq2seq infrastructure we don't have. Separate cycles.

## The half-day probe (anti-commitment gate)

Before committing trainer effort to Go, run this probe:

1. **Acquire Go position dataset** — pro games from public KGS archives, converted to per-move board states. Target ~50k positions across all game phases. Label: next-move-is-winning (binary) OR phase identification (3-class: opening/middle/end).
2. **Trinarize positions** — direct mapping `{empty, own, opponent} → {0, +1, -1}`. No quantization step, no signature extraction parameters to tune. This is the substrate-native input it always should have been given.
3. **Run direct_lsh with the ternary positions as signatures**, no trainer, no MS4, no R4. Vanilla filter-ranker with Hamming distance, k=50-200 neighbors, Selective aggregation.
4. **Baseline expectations**:
    - Phase identification (3 classes): > 60% suggests the substrate handles board structure naturally. < 40% suggests positional similarity is not Hamming-discriminable and we need different tooling.
    - Next-move-wins (binary): > 55% is surprising-good (random is 50%); > 60% is substrate-is-real territory.
5. **Decision rule**:
    - If probe lands in "surprising-good" territory: Go becomes the primary benchmark, next cycle is `routed_go_classifier` or `routed_go_phase`.
    - If probe lands near random or badly: re-check the RAW's second-place candidate (tabular) with the same probe. If tabular also fails, the benchmark problem is harder than this cycle can resolve, and we revisit infrastructure choices.

## Success criteria for this cycle

**Cycle-level (this document is the deliverable):**
- [x] RAW surveys all candidate directions without pre-filtering.
- [x] NODES decomposes "base-3 native" into ternary-input / routing-load / inspectability criteria.
- [x] REFLECT names the category error (validating base-3 claims on base-2 canon).
- [x] SYNTHESIZE commits to Go as primary direction + custom synth as diagnostic + image canon as regression only.

**Execution-level (gated on the probe):**
- [ ] Go probe executed within 2 working days of this cycle's close.
- [ ] Outcome logged (`journal/base3_go_probe.md`).
- [ ] If green: `routed_go_classifier` LMM cycle kicks off.
- [ ] If red: fall through to tabular probe; if also red, infrastructure-review cycle.

## Why Go and not tabular, specifically

Both are legitimate, but Go wins on *substrate claim purity*:

- **Ternary input at input**: Go state has no continuous-to-ternary conversion; the substrate consumes it as-is. Tabular needs continuous-column quantization (we know how, but it's conversion work that reintroduces a representation parameter).
- **Routing load is structural**: Go strategy is literally phase-routed cognition in both human and engine play. Tabular routing benefits from long tails but most tabular benchmarks have moderate class balance.
- **No incumbent that IS routing already**: XGBoost beats us to the tabular-routing punch in the claim-space. Go is currently dominated by dense policy/value nets (AlphaZero descendants) that do NOT route explicitly.
- **Inspectability payoff is high**: a Go move recommendation with "these tiles fired because they match fuseki patterns in the corner" is a real product story in a way that "this credit default probability is 0.73" isn't for interpretability claims.

## How to frame the demotion of image canon

MNIST/Fashion/CIFAR did exactly what they were supposed to do:
- They forced us to build the substrate end-to-end (signature → filter-ranker → re-ranker → metrics).
- They let us measure the representation tax precisely (via `step_change`).
- They gave us multi-seed stability evidence across a year of cycles.
- They remain sensitive regression tests: any substrate change that regresses MNIST > 0.3pp is probably a bug.

They stop being our north star because they aren't the right proving ground for the claim we're actually making. That's not their failing; it's a clarification of what we're trying to prove.

## NORTH_STAR alignment

- **§4 (scaffolding sanction)**: the probe tooling (dataset loader, Hamming runner) is scaffolding for measurement — explicitly sanctioned.
- **§13 (training artifacts in consumer)**: a Go trainer lives in `/train` or a new `/train_go`, not in `libm4t`. Same discipline as `routed_autodiff`.
- **"Routing is essential in base-3"**: Go phase structure is *exactly* where this claim lives. Picking it puts measurement where the claim lives.
- **"End-game unknowable"**: committing to Go doesn't mean committing away from everything else. It means picking the next load-bearing probe for substrate claims. Results may redirect.

## Deliverables

1. This synthesize doc.
2. Closeout of current cycle (`base3_benchmarks_closeout.md` — short, since the cycle is framing-only until the probe runs).
3. Concrete probe recipe with: data source URL, loader spec, expected output format, decision thresholds.
4. Updated task list: Go probe as top-priority pending task; image-canon trainer work demoted.
5. No code in this cycle. Code lives in the probe and the subsequent cycle.

## One-line summary

**Stop validating a routing-first base-3 substrate on continuous-image benchmarks; commit to Go position evaluation as the primary claim surface, gated by a half-day probe that checks substrate-task fit before trainer investment.**
