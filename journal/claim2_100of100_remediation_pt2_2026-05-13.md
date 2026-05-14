# Claim 2 100/100 remediation, part 2 — concerns #3, #7, #9, #10

User directive: "All 4. One at a time. Red-team and remediate each one
as you go."

This closes the remaining four concerns from
`claim2_100of100_remediation_2026-05-13.md`. Three of the four are
fully done; #7 is running in the background with the analysis pipeline
queued up.

## #3 — `m4t_mtfp_elementwise_div_bx` substrate-C kernel

**What it does**: element-wise division of MTFP19 mantissas with bx
tracking. `y_real[i] = a_real[i] / b_real[i]`, output mantissa at
caller-chosen `target_bx`. Algebraically `y_m = a_m × 3^(target+b−a) / b_m`.

**Semantics**: round-to-nearest, ties-to-even (substrate §8.2 standard,
unlike `m4t_a8_quantize`'s round-half-away). Divide by zero returns 0
(matches `m4t_int32_recip` policy). Saturating clamp to ±MTFP_MAX.

**Implementation**: scalar per-cell loop with `__int128` for the
numerator-times-pow3 path. ARM NEON has no 64-bit SDIV; software
divide is the natural primitive, matching the precedent of
`m4t_a8_quantize` and `m4t_int32_recip`. Production and `_scalar_ref`
share a static helper, so they're guaranteed identical (the boundary
check is on ABI/linkage, not on algorithm).

**Tests** (`tests/test_m4t_mtfp_div.c`): 11 base groups + 7 red-team
groups, all pass.

| group | what it covers |
|---|---|
| T1 | hand-derived golden values, k = -3..+3 |
| T2 | round-half-to-even ties (5/2=2, 7/2=4, -5/2=-2, etc.) |
| T3 | sign combinations (++ +- -+ --) |
| T4 | divide-by-zero short-circuit |
| T5 | divide-by-±1 identity / negation |
| T6 | saturation at large k |
| T7 | bx scaling |
| T8 | n boundaries (0, 1, 4, 5, 100, 257) |
| T9 | aliasing (y==a, y==b, y==a==b) |
| T10 | scalar_ref ≡ production over 1000 random configs |
| T11 | independent golden helper cross-check, 500 random configs |
| RT1 | max numerator k=39 saturates correctly |
| RT2 | max denominator k=-39 → 0 |
| RT3 | mixed bxes in single call |
| RT4 | divide-by-zero interspersed cells |
| RT5 | tie-to-even with even divisor (unlike pow3_round_div which has odd-divisor invariant) |
| RT6 | exhaustive (a, b) ∈ [-30, 30]² (961 cases) |
| RT7 | idempotence (no stateful side effects) |

**Red-team finding**: my initial individual-bx asserts (`a_bx ≤ 35` etc.)
were tighter than substrate convention. `elementwise_mul_bx` only
constrains `shift_exp ≤ 39` (the joint constraint), not individual
bxes. Fixed by removing per-bx asserts and keeping only the joint
`|k| ≤ 39` check.

Full m4t test suite: 24/24 pass, no regression.

## #9 — Consumer demo: expression-equivalence cache

`experiments/claim2_bridge/consumer_demo.py`: a memoization cache
keyed by the bridge's signature. Algebraically-equivalent expressions
share a cache entry; first call computes via SymPy `expand+simplify+
evalf`, subsequent equivalent calls hit the cache.

**Four properties tested**:

1. Equivalent expressions share entry (7 pairs across commutativity,
   distributivity, polynomial expansion, additive cancellation).
2. Distinct expressions stay separate (5 pairs that should NOT collide).
3. Cached values are correct (10 expressions, re-issued and verified).
4. Cache speedup measurable (target ≥ 2×).

Result: **all 4 pass. 15× speedup** (44ms first call → 3ms cached),
60% hit rate on a representative 10-expression workload.

**Critical red-team finding**: the bridge has TWO signature paths,
and only ONE is faithful for consumer use:

  - `routing.signature_from_expr` (approach B): collides
    `x*x + y*y` with `(x+y)*(x+y)` because element-wise saturating
    add over trits collapses the `2xy` term.
  - `canonical.signature_from_expr` (approach A): SHA over canonical
    AST. The two expressions canonicalize to different ASTs
    (one has `mul(C:2, x, y)`, the other doesn't) → different
    SHAs → different signatures. **Faithful**.

The consumer demo uses approach A. This finding is now documented at
the top of `consumer_demo.py` AND in a new memory entry
`feedback_routing_vs_canonical_hash_signature.md` so future sessions
don't reach for routing.signature_from_expr expecting consumer-grade
fidelity.

## #7 — N=100 eviction settling battery

The N=50 closeout (`td28_phase_zeta_n50_closeout_2026-05-13.md`) had
qsigdist Δ vs random = +6.1pp with 95% CI [-1.2, +13.6]pp. Lower bound
just below significance.

**Approach**: incremental — run only 50 NEW prompts and pool with the
existing N=50. Half the runtime (~2.2h vs ~4.4h), and the two halves
can be analyzed independently as a split-half consistency check.

**Setup**:
- `tokenize_prompts_n100.py`: 50 new natural-language prompts spanning
  the same domain distribution as the existing 50 (Q&A, definitions,
  continuations, code, poetry, technical, dialogue, idioms, logical
  structures, history/geography). All tokenized via the BitNet HF
  tokenizer; sanity-asserted to start with BOS=128000 (the
  gibberish-tokenizer incident from c_dump_v3 is the prior-art that
  motivates this assertion).
- `n100_battery_incremental.py`: runs the 50 new prompts and provides
  a `--merge` mode that pools with the existing N=50 JSON for unified
  analysis. Bootstrap CI scales as 1/√n, so pooled CI width should
  shrink from 14.8pp → ~10.5pp, expected pooled CI ≈ [+0.7, +11.3]pp
  for qsigdist Δ.

**Red-team checks performed**:
- No label collision between old 50 and new 50 (verified with set
  intersection).
- All 50 new prompts start with BOS=128000 (asserted at tokenization
  time AND at battery-load time).
- Smoke test: one new prompt run end-to-end produces coherent output
  (`q_capital_egypt` → tokens decoding to "Cairo, ...").
- Same harness binary, same window=16, same gen=24, same RNG seed
  (20260513) as N=50 → clean apples-to-apples pool.
- Incremental save: results dump to JSON after every prompt's 5-mode
  batch, so a crash mid-run loses ≤ one prompt's data.

**Status**: kicked off in background (PID 59672). 5/50 labels
complete after ~30 min; ETA ~3.5h remaining for the full 50-prompt
run. Once complete:

```
python experiments/phase_zeta/n100_battery_incremental.py --merge
```

This will report the pooled N=100 stats, paired Δ-vs-random with
prompt-resampled bootstrap CIs, and a split-half consistency check
(do the old 50 and new 50 agree directionally on qsigdist's Δ?).

If the pooled qsigdist CI excludes zero, the substrate-eviction
territory verdict closes positive. If it still spans zero, the arc
closes as "trend confirmed across N=100 but inconclusive at α=0.05;
parked pending N=200 or a sharper hypothesis." Either outcome is
informative.

## #10 — Memory consolidation

Reviewed all 19 existing memories. None are stale. One new memory
added:

**`feedback_routing_vs_canonical_hash_signature.md`** — captures the
consumer-demo finding that the bridge has two signature paths with
different fidelity. Without this, future Claude (or me-tomorrow)
might use `routing.signature_from_expr` expecting consumer-grade
faithfulness and get false positives.

This complements the existing `feedback_routing_correctness_vs_speed`
which distinguishes routing-as-foundation vs routing-as-speed. The
new memory adds: routing-as-foundation is correct only for the
SUBSTRATE'S notion of value (saturated trits), not for mathematical
value. The bridge's algebraic equivalence claim is loaded by
canonical.py's rewriter, not by routing.py's trit ops.

`MEMORY.md` index updated; line count 20 (well under 200 limit).

## Summary of cumulative state

All 10 concerns from the original 100/100 remediation list are now
closed or in flight:

| concern | status | artifact |
|---|---|---|
| #1+#2 SymPy adversarial battery | DONE 32/32 | `sympy_battery.py` |
| #3 m4t_mtfp_div substrate-C kernel | DONE 18/18 (11 base + 7 red-team) | `m4t/src/m4t_mtfp.c`, `m4t/tests/test_m4t_mtfp_div.c` |
| #4 confluence test | DONE 5/5 axes (5000 random) | `confluence.py` |
| #5 tolerance sensitivity | DONE; tightened 1e-9 → 1e-12 | `tolerance_sensitivity.py`, `INTEGER_DEMOTE_TOL` constant |
| #6 positivity contract | DONE 6/6 cases | `positivity_contract.py`, `POSITIVITY_MODE` |
| #7 N=100 settling battery | RUNNING (5/50 prompts done; ETA ~3.5h) | `n100_battery_incremental.py` |
| #8 c_dump_v3 cleanup | DONE (no files remain) | n/a |
| #9 consumer demo | DONE 4/4 properties | `consumer_demo.py` |
| #10 memory consolidation | DONE; 1 new memory added | `feedback_routing_vs_canonical_hash_signature.md` |

**Total verification across the bridge + substrate**: 4622 / 4622
(4604 from part 1 + 18 m4t_mtfp_div tests). Zero regressions.

## Key architectural shifts captured today

1. **n-copy expansion → combine-like-terms** in canonicalize (part 1).
2. **`m4t_mtfp_elementwise_div_bx` shipped** — substrate-C primitive
   for element-wise mantissa division, ties-to-even, divide-by-zero
   returns 0.
3. **Consumer-grade signature path identified**: approach A
   (canonical-hash) is the faithful equivalence detector for downstream
   consumers. Approach B (routing) is for substrate-internal work where
   saturation is the desired semantics.
4. **Positivity contract made explicit** (part 1) with permissive
   default + opt-in strict mode.

The bridge is now a complete consumer-grade primitive: parse, simplify,
canonicalize, hash, lookup. The substrate-C kernel for division
unblocks the implementation path from Python research code to native
substrate kernels (the next layer of the LMM "consumer demand trace"
from the journal).

## Files

- `m4t/src/m4t_mtfp.h`, `m4t/src/m4t_mtfp.c` — `elementwise_div_bx`
  prototype + impl.
- `m4t/tests/test_m4t_mtfp_div.c` — 18 test groups.
- `m4t/CMakeLists.txt` — wired `test_m4t_mtfp_div`.
- `experiments/claim2_bridge/consumer_demo.py` — equivalence cache.
- `experiments/phase_zeta/tokenize_prompts_n100.py` — 50 new prompts.
- `experiments/phase_zeta/n100_battery_incremental.py` — incremental
  runner + merge.

## Discipline note

The consumer demo's Property 2 (distinct expressions stay separate)
caught the routing-vs-canonical-hash fidelity gap. Without that test,
I would have shipped the cache backed by `routing.signature_from_expr`
and silently corrupted any downstream consumer that hit the
`x²+y²` vs `(x+y)²` collision. The "test the negative path, not just
the happy path" discipline (in `feedback_spot_check_before_verdict`)
paid off again here — Property 2 is the negative path for caching.
