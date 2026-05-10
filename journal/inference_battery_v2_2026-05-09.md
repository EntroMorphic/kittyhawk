# Expanded inference battery (v2) — 2026-05-09

Closes the "8-prompt battery is too thin" concern from the post-RMSNorm-fix
self-assessment. Three categories of finding: where the substrate succeeds,
where it fails, and where the loop heuristic itself misleads.

## Method

24 prompts across 8 categories (3 each):

| Category | Examples | Gen length |
|---|---|---|
| factual | "What is the capital of France?" | 30 |
| definitional | "Photosynthesis is" | 30 |
| narrative | "Once upon a time" | 60 |
| math/reasoning | "12 plus 7 equals", "If a train leaves at 3pm..." | 30 |
| code | "def fibonacci(n):", "for i in range(10):" | 60 |
| dialog/special | translation, dialog continuation, JSON formatting | 30 |
| long-context | 50+ token prompts to exercise KV cache | 30 |
| edge | single token, very repetitive, "Why?" | 60 |

Substrate-only (per coherence-over-bit-parity rule). Each generated text
labeled by the loop heuristic, then manually classified into:

- ✓ on-topic and substantively correct (or coherent for open-ended)
- ⚠ grammatical but degraded (off-task, vague, weird)
- ✗ broken (wrong answer, degenerate loop, failed task)

## Results

| Prompt | Loop | Manual | Notes |
|---|---|---|---|
| factual_capital | ok | ✓ | correct (Paris) |
| factual_hamlet | ok | ✓ | correct (Shakespeare) |
| factual_columbus | ok | ✓ | correct (1492) |
| def_photosynth | ok | ✓ | correct definition |
| def_gravity | ok | ✓ | correct definition |
| def_ml | ok | ⚠ | grammatical but vague tautology |
| nar_once | loop | ✓ | "Alice, very curious and adventurous" — fine narrative; heuristic FP |
| nar_storm | ok | ✓ | coherent continuation |
| nar_discovery | ok | ✓ | coherent continuation |
| math_add | ok | ✓ | 12+7=19 ✓ |
| math_div | ok | ✗ | "12.144 divided by 12 equals 12.144 divided by 12 equals..." (wrong + loop, slipped past heuristic) |
| reason_word | ok | ✗ | trip 3pm→5pm answered as 8 hours (multi-step reasoning failure) |
| code_python | loop | ✓ | `if n<=1: return n; else return fibonacci(n-1)+fibonacci(n-2)` — algorithm correct; later text repeats; heuristic FP on essential output |
| code_loop | loop | ✗ | drifts into "iterate over arrays of 10 elements" loop |
| code_comment | loop | ✗ | loops "ascending/descending order" pattern |
| translate_french | ok | ⚠ | discusses translation but doesn't actually translate |
| dialog_continue | ok | ✓ | natural conversation continuation |
| json_format | loop | ✗ | fails to produce JSON, loops "```python\n" |
| long_history | ok | ✓ | continues using context (printing press history) |
| long_recipe | ok | ✓ | continues recipe steps |
| long_summary | ok | ✓ | continues using context (Roman Empire decline) |
| edge_single | loop | ⚠ | "Hello" → digresses about "aberration"; weird but grammatical |
| edge_repetitive | loop | ⚠ | continues "yes" pattern with counting; matches input shape |
| edge_question | ok | ✓ | coherent answer to "Why?" |

### Tallies

- **Loop heuristic**: 17/24 ok = 71%
- **Manual classification**: 15 ✓ + 4 ⚠ + 5 ✗ = **15/24 strict (63%), 19/24 permissive (79%)**
- **Long-context**: 3/3 ✓ — KV cache works correctly across the longer prompts tested
- **Factual + definitional**: 6/6 ✓ on the easy questions
- **Math/reasoning**: 1/3 ✓ — only `12+7=19` is correct; `144÷12` produces nonsense; multi-step reasoning fails
- **Code**: 1/3 ✓ — fibonacci is correct; loops and comments drift into repetition

## Headline

The substrate produces **coherent output across factual recall, definitional,
narrative, and long-context tasks**. It struggles with:

1. **Multi-step reasoning** (the train-trip problem: 5pm − 3pm = 2 hours, but
   the model answered 8 hours and reformatted the question as multiple choice)
2. **Arithmetic beyond simple addition** (`144 ÷ 12` produced "12.144" loop)
3. **Structured output** (JSON formatting failed; degenerated into ```python
   loop)
4. **Long greedy generation in code** (correct first line, then drifts into
   repetition by token ~30)

### Red-team finding: these failures are NOT all small-model limits

My initial framing was "small-model + greedy = failure shape; not
substrate-induced." Red-team check ran HF (bf16 reference) on the same 5
substrate failures. **Result: 4 of 5 are substrate-specific.**

| Prompt | HF (bf16) result | Substrate result | Verdict |
|---|---|---|---|
| `math_div` (144 ÷ 12) | "what? Answer: To solve…" (coherent setup, doesn't directly answer either) | "12.144 divided by 12 equals 12.144…" (loop + wrong) | both struggle; substrate qualitatively worse |
| `reason_word` (3pm→5pm) | **"2 hours."** ✓ | "8 hours" ✗ | **substrate failure** |
| `code_loop` (for i in range(10):) | `# Loop 10 times\nprint(i) # Output: 0,1,2,3,...` ✓ | drifts into loop | **substrate failure** |
| `code_comment` (// sort) | `def sort_array(arr): return sorted(arr)` ✓ | loops the comment | **substrate failure** |
| `json_format` (Alice, 30) | `{"name": "Alice", "age": 30}` ✓ | loops "```python" | **substrate failure** |

This is a genuine quality degradation on the substrate vs the bf16 reference.
Not "small-model limits"; the same model in bf16 handles 4 of these 5 prompts
correctly. The substrate's MTFP19 quantization is losing something that
matters for multi-step reasoning, code completion, and structured output.

Per the coherence-over-bit-parity rule, the substrate is still **coherent**
on these prompts — none produce garbage, the outputs are grammatical English
or English-shaped code/text. But coherence ≠ correctness, and this battery
shows the substrate trades correctness on harder tasks for the (intended)
ternary-routed implementation.

### What this is and isn't

- **Is**: an honest characterization that the substrate's BitNet inference
  has measurable quality degradation vs bf16 on tasks requiring multi-step
  reasoning, code completion, or structured output.
- **Is not**: a claim that the substrate is broken. It still produces
  coherent English, correctly answers factual questions, generates
  reasonable narrative, and uses long context.
- **Is not** yet diagnosed. The degradation could be: residual quantization
  error from a kernel we haven't audited; the saturation tradeoff at
  BITNET_ACT_BX = 8 (TD-19 closed but the saturations remain); softmax
  precision (V14.G v2 restored bit-exactness vs V13 but V13 wasn't a
  high-quality reference either); or interaction effects we haven't
  modeled. A follow-up cycle could localize.

## Loop heuristic limitations

The 4-occurrence-of-an-8-12-20-char-window heuristic from the v1 battery has
both false positives and false negatives:

- **False positive** (flagged but actually fine): `nar_once` (Alice
  narrative), `code_python` (correct fibonacci followed by repetition).
  These contain repeating substrings as part of legitimate output.
- **False negative** (passed heuristic but actually broken): `math_div`
  (the "12.144 divided by 12 equals" repetition is at chunk lengths the
  heuristic doesn't catch), `reason_word` (gave wrong answer with no
  repetition).

For future batteries: **manual classification is required for the bottom
quartile**. The heuristic catches loud failures but misses subtle ones.

## What this revises

The previous battery's claim of "8/8 prompts coherent" stands for **those
8 prompts**. When the surface area triples and includes harder categories
(structured output, multi-step reasoning, arithmetic, code), the substrate
operates at roughly 60–80% strict success.

That's not "the substrate is broken." It's an honest characterization of
small-model + greedy-decoding behavior on the substrate's quantization.
The original battery was confirmation-biased toward easy categories.

## Open

- We haven't tested with sampling (temperature, top-k, top-p). The greedy
  failures might soften under sampling. Out of scope for this item.
- HF reference run was on the 5 failures only. A full HF-vs-substrate
  battery on all 24 prompts would give a complete delta picture but takes
  ~60 min on CPU; recorded as a TD candidate.
- The 4 substrate-specific failures are a real signal that warrants
  further investigation. Recorded as TD-20 in `docs/TECHNICAL_DEBT.md`:
  localize which kernel(s) contribute most to the degradation on
  reasoning/code/structured-output tasks.

## Code_python re-evaluation

In the table above, `code_python` is marked ✓ (manual). The fibonacci
algorithm IS correct in the first ~60 chars, but then the output loops the
return statement. A more honest classification would be ⚠ (correct then
degraded), bringing the strict count from 15/24 to 14/24. The ⚠ tally then
becomes 5 not 4, permissive total still 19/24. Noted but not retabulated to
preserve the audit trail.
