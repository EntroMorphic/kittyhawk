# Output-quality rubric for sparse / eviction probes

Defined per the #10 amendment red-team finding M1: "first-30-token
agreement with dense" is a divergence metric, not a quality metric.
Two outputs can both be coherent and disagree with dense in different
directions. This rubric provides a defensible quality measure
that doesn't conflate divergence with worse output.

## The 0-4 scale

Each generated output gets ONE score per evaluator, applied to the
*first 30 generated tokens* (matching the existing dense-agreement
window). Decode tokens to text before scoring.

| Score | Label | Description |
|-------|-------|-------------|
| 0 | incoherent | Repeated tokens, gibberish, no grammatical structure |
| 1 | partial loop | Coherent start (≤10 tokens), then degenerates into repetition |
| 2 | coherent off-topic | Grammatical English; drifts entirely off the prompt's subject |
| 3 | coherent on-topic | Grammatical English; stays related to the prompt's subject |
| 4 | coherent + correct | Grammatical English; on-topic; factually/logically correct (where applicable) |

## Application protocol

1. **Decode tokens to text.** Use the tokenizer (no shortcuts —
   token IDs alone don't reveal grammatical structure).
2. **Read the first 30 tokens of output.** Score blind to the
   configuration name (oracle vs random vs sigdist) where feasible —
   reduces self-reinforcing bias.
3. **Apply the rubric.** Pick the integer score; document a one-line
   rationale per output in the journal.
4. **Aggregate by config.** Report mean ± stddev (for multi-seed
   conditions) and median per condition.
5. **Spot-check disagreements.** If two evaluators (or two runs)
   disagree by ≥2 on the same output, escalate to a deeper read.

## What this does NOT replace

- **Loop heuristic.** Keep running it alongside; it cheaply catches
  catastrophic failures (8-gram repeat ≥ 3 = LOOP). A 0-rated
  output should also trigger LOOP; if it doesn't, both metrics are
  worth recording.
- **Dense-agreement.** Keep reporting it, but **label it as
  divergence**, not quality. Useful for ranking how closely a
  config tracks dense (a separate dimension from quality).

## When to use this rubric vs the heuristics

- **For verdicts (positive, negative, mixed):** USE THIS RUBRIC.
  Heuristics alone are insufficient per the #10 amendment.
- **For ranking many configs cheaply:** loop heuristic +
  dense-agreement is fine for first-pass ranking. The rubric is
  then applied to the top-K and bottom-K to confirm the ranking.
- **For comparing very close configs:** the rubric won't distinguish
  configs that all score 3. Use additional measures (factual
  correctness, response length, structural variety).

## Self-evaluator bias

When the same evaluator scores all outputs, there's bias toward
internal consistency over absolute scale. Mitigations:
- **Score in randomized order** (don't go config-by-config).
- **Rate the LOOP-flagged outputs LAST** to avoid anchoring.
- **Cross-reference with dense reference** for each prompt: if dense
  itself is rated 3, then 2 is "as good or worse than dense"; if
  dense is rated 4, then 3 is "decent but not as good."

## How this maps to the #10 corrected rerun

The corrected rerun uses configs:
- `dense` (baseline)
- `fifo`, `random×3 seeds`, `sigdist M={1, 4, 8}`
- at windows `{16, 32, 128}`
- on long-context prompts (item 4)

For each (config, prompt) pair, generate output, decode, rate per
the rubric, aggregate. The verdict on "did sigdist beat random?"
follows from the rubric ratings, NOT from dense-agreement.
